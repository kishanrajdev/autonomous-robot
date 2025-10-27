"""
ctot_full_analysis.py

Compute per-episode metrics for multiple algorithms (one jsonl per algorithm),
and produce:
 - CTOT_avg (mean time to first-clean across dirty tiles)
 - CTOT_max (time to last-clean i.e., max)
 - C_t vector (cumulative cleaned count per step) for plotting
 - AUC (numeric sum of C(t) over horizon; optionally normalized)
 - reward/movement_cost (if present in JSON)
 - paired tests and effect sizes vs a chosen baseline algorithm
 - plots: CTOT curves (mean +/- CI), boxplots/violin for CTOT metrics, bar chart for AUC

Each JSONL record should include (example keys):
  - "steps" (int)   # total steps in episode
  - "episode" or "seed" (int/str)  # episode id
  - "dirty_tile_steps" (dict)  # mapping coord string -> step cleaned (or null)
  - optionally "reward" or "movement_cost" etc.

Usage:
  python ctot_full_analysis.py \
    --files boustrophedon.jsonl spiral.jsonl random_walk.jsonl proposed.jsonl \
    --labels boustrophedon spiral random_walk proposed \
    --outdir results_ctot \
    --baseline boustrophedon

"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from statsmodels.stats.weightstats import DescrStatsW
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Utility functions
# -------------------------
def safe_float(x):
    return float(x) if x is not None else None

def compute_episode_metrics(rec, treat_uncleaned_as_steps=True, normalize_auc=False):
    """
    Given a single record (dict), compute metrics.
    Returns dict with:
      - episode (seed)
      - steps (S)
      - n_dirty
      - ctot_avg (mean first-clean time normalized by steps)
      - ctot_max (max first-clean time normalized by steps)
      - ctot_avg_raw (mean first-clean time in steps)
      - ctot_max_raw (max first-clean time in steps)
      - C_t (np.array length S+1 where C_t[t] = #dirty cleaned by step t; index 0 = 0)
      - AUC (sum C_t over t=1..S) optionally normalized by (n_dirty * S)
      - reward / movement_cost if present
      - censored_count (# dirty tiles never cleaned)
    """
    S = int(rec.get("steps"))
    episode = rec.get("episode", rec.get("seed", None))
    dirty = rec.get("dirty_tile_steps", {})
    # convert values: None -> None, else int/float
    cleaned_map = {k: (None if v is None else int(v)) for k, v in (dirty.items() if isinstance(dirty, dict) else [])}
    n_dirty = len(cleaned_map)
    cleaned_times = []
    censored_count = 0
    for v in cleaned_map.values():
        if v is None:
            censored_count += 1
            if treat_uncleaned_as_steps:
                cleaned_times.append(S)
            else:
                # ignore this tile in avg calculation (will reduce n_dirty_effective)
                pass
        else:
            cleaned_times.append(v)

    n_effective = len(cleaned_times)

    if n_effective == 0:
        # nothing cleaned (or all excluded) -> set NaNs / zeros
        ctot_avg_raw = np.nan
        ctot_max_raw = np.nan
    else:
        ctot_avg_raw = float(np.mean(cleaned_times))
        ctot_max_raw = float(np.max(cleaned_times))

    # Normalize CTOTs by S to get fraction of episode
    ctot_avg = (ctot_avg_raw / S) if (not np.isnan(ctot_avg_raw)) else np.nan
    ctot_max = (ctot_max_raw / S) if (not np.isnan(ctot_max_raw)) else np.nan

    # Compute C(t) vector: cumulative cleaned count per step (t from 0..S)
    C_t = np.zeros(S + 1, dtype=int)  # C_t[0] = 0
    # If treating uncleaned as S, then cleaned_map with None -> S already considered above
    # Create list of cleaning times for included tiles (only those we considered)
    clean_times_for_ct = []
    for v in cleaned_map.values():
        if v is None:
            if treat_uncleaned_as_steps:
                clean_times_for_ct.append(S)
            else:
                # skip
                pass
        else:
            clean_times_for_ct.append(int(v))
    # Count cumulative
    for t in clean_times_for_ct:
        # if t < 0 or t > S, clip
        if t < 0:
            t = 0
        if t > S:
            t = S
        C_t[t] += 1
    # cumulative sum over t
    C_t = np.cumsum(C_t)
    # AUC: numeric sum of C_t[1..S] (exclude index 0)
    AUC_raw = float(np.sum(C_t[1:]))
    if normalize_auc:
        denom = n_dirty * S if n_dirty > 0 else 1
        AUC = AUC_raw / denom
    else:
        AUC = AUC_raw

    # optional fields
    extras = {}
    for key in ("reward", "movement_cost", "cost", "total_reward"):
        if key in rec:
            extras[key] = rec[key]

    return {
        "episode": episode,
        "steps": S,
        "n_dirty": n_dirty,
        "n_effective": n_effective,
        "censored_count": censored_count,
        "ctot_avg": ctot_avg,
        "ctot_max": ctot_max,
        "ctot_avg_raw": ctot_avg_raw,
        "ctot_max_raw": ctot_max_raw,
        "C_t": C_t,         # numpy array
        "AUC": AUC,
        "AUC_raw": AUC_raw,
        **extras
    }

# -------------------------
# IO: load jsonl file into dict episode->metrics
# -------------------------
def load_jsonl_metrics(path, treat_uncleaned=True, normalize_auc=False, id_key_order=("episode","seed")):
    path = Path(path)
    results = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            mid = None
            for k in id_key_order:
                if k in rec:
                    mid = rec[k]
                    break
            if mid is None:
                # fallback: use line number? but prefer error
                raise ValueError(f"No episode/seed key found in record: {rec.keys()}")
            metrics = compute_episode_metrics(rec, treat_uncleaned_as_steps=treat_uncleaned, normalize_auc=normalize_auc)
            results[mid] = metrics
    return results

# -------------------------
# Aggregate across algorithms
# -------------------------
def build_table(algo_metrics_map):
    """
    Input: dict label -> {episode_id: metrics dict}
    Output:
      - per_episode_df: DataFrame index episode_id, columns multiindex (algo, metric)
      - C_t_dict: dict (algo -> dict episode_id -> C_t numpy array)
    """
    episodes = set()
    for m in algo_metrics_map.values():
        episodes.update(m.keys())
    episodes = sorted(episodes, key=lambda x: (int(x) if str(x).isdigit() else str(x)))
    rows = []
    C_t_dict = {}
    for algo, emap in algo_metrics_map.items():
        C_t_dict[algo] = {}
    for ep in episodes:
        row = {"episode": ep}
        for algo, emap in algo_metrics_map.items():
            if ep in emap:
                met = emap[ep]
                row[f"{algo}::steps"] = met["steps"]
                row[f"{algo}::n_dirty"] = met["n_dirty"]
                row[f"{algo}::n_effective"] = met["n_effective"]
                row[f"{algo}::censored_count"] = met["censored_count"]
                row[f"{algo}::ctot_avg"] = met["ctot_avg"]
                row[f"{algo}::ctot_max"] = met["ctot_max"]
                row[f"{algo}::ctot_avg_raw"] = met["ctot_avg_raw"]
                row[f"{algo}::ctot_max_raw"] = met["ctot_max_raw"]
                row[f"{algo}::AUC"] = met["AUC"]
                row[f"{algo}::AUC_raw"] = met["AUC_raw"]
                # optional extras
                for k in ["reward", "movement_cost", "cost", "total_reward"]:
                    if k in met:
                        row[f"{algo}::{k}"] = met[k]
                C_t_dict[algo][ep] = met["C_t"]
            else:
                # missing: set NaNs
                for col in ["steps","n_dirty","n_effective","censored_count","ctot_avg","ctot_max","ctot_avg_raw","ctot_max_raw","AUC","AUC_raw"]:
                    row[f"{algo}::{col}"] = np.nan
                C_t_dict[algo][ep] = None
        rows.append(row)
    per_episode_df = pd.DataFrame(rows).set_index("episode")
    return per_episode_df, C_t_dict

# -------------------------
# Plotting helpers
# -------------------------
def plot_ctot_curves(C_t_dict, outpath, labels=None, ci=0.95, show_mean=True, figsize=(8,5), normalize_by_n_dirty=False):
    """
    Plot mean +/- CI bands across episodes for each algorithm.
    C_t_dict: algo -> {episode: C_t numpy array}
    If arrays have different lengths, we will pad/truncate to the common max length.
    If normalize_by_n_dirty: convert C_t to fraction cleaned (divide by n_dirty per episode) before aggregating.
    We'll compute bootstrap percentile CI across episodes at each timepoint.
    """
    plt.figure(figsize=figsize)
    max_len = 0
    algo_episode_ct = {}
    for algo, epmap in C_t_dict.items():
        arrs = [v for v in epmap.values() if v is not None]
        if len(arrs) == 0:
            algo_episode_ct[algo] = np.zeros((0,1), dtype=int)
            continue
        max_len = max(max_len, max(len(a) for a in arrs))
    t = np.arange(0, max_len)  # includes t=0

    # For each algo: build matrix episodes x time
    for algo, epmap in C_t_dict.items():
        arrs = []
        for ep, a in epmap.items():
            if a is None:
                continue
            # pad to max_len with final value (cumulative stays at last)
            if len(a) < max_len:
                pad = np.full(max_len - len(a), a[-1], dtype=int)
                arr_p = np.concatenate([a, pad])
            else:
                arr_p = a[:max_len]
            arrs.append(arr_p)
        if len(arrs) == 0:
            continue
        M = np.vstack(arrs)  # shape (n_eps, max_len)
        # Optionally normalize each episode by its last value (n_dirty_effective)
        if normalize_by_n_dirty:
            last_vals = M[:, -1].astype(float)
            # avoid divide by zero
            last_vals[last_vals == 0] = 1.0
            M = (M.T / last_vals).T

        # Compute mean and bootstrap CI (percentile)
        mean_curve = M.mean(axis=0)
        # bootstrap percentile CI
        n_boot = 2000
        rng = np.random.default_rng(0)
        boot_samples = []
        for _ in range(n_boot):
            idx = rng.integers(0, M.shape[0], M.shape[0])
            sample = M[idx, :].mean(axis=0)
            boot_samples.append(sample)
        boots = np.vstack(boot_samples)
        lower = np.percentile(boots, (1 - ci) / 2 * 100, axis=0)
        upper = np.percentile(boots, (1 - (1 - ci) / 2) * 100, axis=0)

        # Plot
        label = algo if labels is None else labels.get(algo, algo)
        plt.plot(t, mean_curve, label=label)
        plt.fill_between(t, lower, upper, alpha=0.2)

    plt.xlabel("Step")
    plt.ylabel("C(t) (cumulative cleaned tiles)" + (" (fraction)" if normalize_by_n_dirty else ""))
    plt.title("CTOT curves: mean ± {:.0%} CI".format(ci))
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_boxplots(df_long, metric_key, outpath, ylabel=None, figsize=(7,5)):
    """
    df_long: columns ['episode', 'algo', 'value']
    metric_key: string for filename / title
    """
    plt.figure(figsize=figsize)
    sns.boxplot(x="algo", y="value", data=df_long, showfliers=False)
    sns.stripplot(x="algo", y="value", data=df_long, color="k", size=4, jitter=True, alpha=0.6)
    plt.ylabel(ylabel or metric_key)
    plt.title(metric_key)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_bar_with_error(bars_df, outpath, value_col="mean", err_col="se", ylabel=None, figsize=(7,5)):
    """
    bars_df should contain columns ['algo', 'mean', 'se'] or similar
    """
    plt.figure(figsize=figsize)
    x = np.arange(len(bars_df))
    plt.bar(x, bars_df[value_col], yerr=bars_df[err_col], align="center", alpha=0.8, capsize=6)
    plt.xticks(x, bars_df["algo"])
    plt.ylabel(ylabel or value_col)
    plt.title(f"{value_col} per algorithm")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

# -------------------------
# Stats: paired tests, bootstrap CI, effect sizes
# -------------------------
def paired_stats(df, algo_a, algo_b, metric_col, alpha=0.05, n_boot=2000):
    """
    Perform paired comparison (a vs b) on metric_col which exists as f"{algo}::metric" in df.
    df indexed by episode.
    Returns: dict with n_pairs, mean_diff, se_diff, ttest/wilcoxon result, p-value, chosen_test, bootstrap_ci, cohens_d_paired
    """
    col_a = f"{algo_a}::{metric_col}"
    col_b = f"{algo_b}::{metric_col}"
    if col_a not in df.columns or col_b not in df.columns:
        raise KeyError(f"Columns {col_a} or {col_b} missing")
    sub = df[[col_a, col_b]].dropna()
    n = len(sub)
    if n == 0:
        return {"n": 0}

    a = sub[col_a].values.astype(float)
    b = sub[col_b].values.astype(float)
    diff = a - b
    mean_diff = np.mean(diff)
    se = stats.sem(diff, nan_policy="omit") if n > 1 else np.nan

    # Normality of differences
    normal = (n >= 3) and (stats.shapiro(diff).pvalue > alpha)
    if normal:
        tstat, pval = stats.ttest_rel(a, b, nan_policy="omit")
        test_used = "paired_t"
    else:
        # Wilcoxon signed-rank test (requires n>=1 and non-zero differences)
        # If all diffs zero or all identical sign small n, handle gracefully
        try:
            wstat, pval = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        except Exception:
            # fallback to permutation test using numpy
            pval = np.mean([np.mean(np.random.choice(diff, size=n, replace=True)) >= mean_diff for _ in range(1000)])
            wstat = np.nan
        test_used = "wilcoxon"

    # bootstrap CI for mean difference
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(np.mean(diff[idx]))
    boot_low, boot_high = np.percentile(boots, [2.5, 97.5])

    # Cohen's d for paired samples
    sd_diff = np.std(diff, ddof=1)
    if sd_diff == 0:
        cohens_d = np.nan
    else:
        cohens_d = mean_diff / sd_diff

    return {
        "n": n,
        "mean_diff": mean_diff,
        "se": se,
        "test_used": test_used,
        "p_value": float(pval),
        "boot_ci_low": float(boot_low),
        "boot_ci_high": float(boot_high),
        "cohens_d_paired": float(cohens_d)
    }

# -------------------------
# Main routine
# -------------------------
def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    files = args.files
    labels = args.labels
    baseline = args.baseline
    treat_uncleaned = args.treat_uncleaned
    normalize_auc = args.normalize_auc

    if labels is None:
        labels = [Path(f).stem for f in files]
    if len(labels) != len(files):
        raise ValueError("labels length must match files length")

    print("Loading data...")
    algo_metrics_map = {}
    for f, lab in zip(files, labels):
        emap = load_jsonl_metrics(f, treat_uncleaned=treat_uncleaned, normalize_auc=normalize_auc)
        algo_metrics_map[lab] = emap
        print(f"  {lab}: {len(emap)} episodes loaded")

    # Build table
    per_episode_df, C_t_dict = build_table(algo_metrics_map)
    csv_table = outdir / "ctot_per_episode_table.csv"
    per_episode_df.to_csv(csv_table)
    print(f"Saved per-episode table to {csv_table}")

    # Long-form df for plotting metrics like CTOT_avg, CTOT_max, AUC
    long_rows = []
    for algo in labels:
        for ep, row in per_episode_df.iterrows():
            long_rows.append({"episode": ep, "algo": algo, "ctot_avg": row.get(f"{algo}::ctot_avg"),
                              "ctot_max": row.get(f"{algo}::ctot_max"), "AUC": row.get(f"{algo}::AUC")})
    df_long = pd.DataFrame(long_rows)

    # Plot CTOT curves (absolute C(t))
    plot_ctot_curves(C_t_dict, outdir / "ctot_curves_absolute.png", labels={k:k for k in labels}, ci=0.95, normalize_by_n_dirty=False)
    # Plot CTOT curves normalized by n_dirty (fraction)
    plot_ctot_curves(C_t_dict, outdir / "ctot_curves_fraction.png", labels={k:k for k in labels}, ci=0.95, normalize_by_n_dirty=True)

    # Boxplots / violin for CTOT_avg and CTOT_max
    df_ctot_avg_long = df_long.dropna(subset=["ctot_avg"])[["episode","algo","ctot_avg"]].rename(columns={"ctot_avg":"value"})
    plot_boxplots(df_ctot_avg_long, "CTOT_avg", outdir / "boxplot_ctot_avg.png", ylabel="CTOT_avg (fraction)")
    df_ctot_max_long = df_long.dropna(subset=["ctot_max"])[["episode","algo","ctot_max"]].rename(columns={"ctot_max":"value"})
    plot_boxplots(df_ctot_max_long, "CTOT_max", outdir / "boxplot_ctot_max.png", ylabel="CTOT_max (fraction)")

    # Bar chart for AUC (mean ± se)
    auc_stats = df_long.dropna(subset=["AUC"]).groupby("algo")["AUC"].agg(["mean","count","std"]).reset_index()
    auc_stats["se"] = auc_stats["std"] / np.sqrt(auc_stats["count"].replace(0,np.nan))
    auc_stats.to_csv(outdir / "auc_summary.csv", index=False)
    plot_bar_with_error(auc_stats.rename(columns={"mean":"mean"}), outdir / "auc_bar.png", value_col="mean", err_col="se", ylabel="AUC")

    # Paired tests vs baseline
    print("\nPaired tests vs baseline:", baseline)
    tests = []
    for other in labels:
        if other == baseline: continue
        for metric in ("ctot_avg", "ctot_max", "AUC"):
            try:
                res = paired_stats(per_episode_df, baseline, other, metric_col=metric, n_boot=2000)
            except KeyError:
                res = {"n": 0}
            res_row = {"baseline": baseline, "other": other, "metric": metric, **res}
            tests.append(res_row)
    pd.DataFrame(tests).to_csv(outdir / "paired_tests_vs_baseline.csv", index=False)
    print("Saved paired tests to paired_tests_vs_baseline.csv")

    # Save a minimal summary for inclusion in paper
    summary = {}
    for algo in labels:
        s = {}
        arr = per_episode_df[f"{algo}::ctot_avg"].dropna().values
        if len(arr) > 0:
            s["ctot_avg_mean"] = np.mean(arr)
            s["ctot_avg_median"] = np.median(arr)
            s["ctot_avg_sd"] = np.std(arr, ddof=1)
            s["n"] = len(arr)
        else:
            s["n"] = 0
        arr2 = per_episode_df[f"{algo}::ctot_max"].dropna().values
        if len(arr2) > 0:
            s["ctot_max_mean"] = np.mean(arr2)
            s["ctot_max_median"] = np.median(arr2)
            s["ctot_max_sd"] = np.std(arr2, ddof=1)
        arr3 = per_episode_df[f"{algo}::AUC"].dropna().values
        if len(arr3) > 0:
            s["AUC_mean"] = np.mean(arr3)
            s["AUC_sd"] = np.std(arr3, ddof=1)
        summary[algo] = s
    pd.DataFrame(summary).T.to_csv(outdir / "summary_for_paper.csv")

    print("All outputs saved to", outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True, help="list of jsonl files (one per algo)")
    parser.add_argument("--labels", nargs="+", required=False, help="labels matching files (same order)")
    parser.add_argument("--outdir", default="ctot_results", help="output directory")
    parser.add_argument("--baseline", required=True, help="baseline algorithm label for paired tests")
    parser.add_argument("--treat_uncleaned", action="store_true", help="treat uncleaned dirty tiles as cleaned at final step (conservative)")
    parser.add_argument("--normalize_auc", action="store_true", help="normalize AUC by n_dirty * steps (gives fraction area)")
    args = parser.parse_args()
    main(args)
