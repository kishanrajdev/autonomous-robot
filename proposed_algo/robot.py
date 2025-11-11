import numpy as np
import copy

# from monte_carlo_simple_cleaning import observe


class Robot:
    def __init__(self, env, belief, n, m, start=(0, 0), move_cost=0.1, reward_scale=1.0):
        self.env = env
        self.belief = belief
        self.x, self.y = start
        # after self.belief is set
        self.n, self.m = self.belief.alpha.shape  # single source of truth
        self.move_cost = move_cost
        self.reward_scale = reward_scale
        self.total_reward = 0.0
        self.total_steps = 1
        self.last_move = (0, 0)

    def select_action_ucb(self):
        ucb_values = np.zeros((self.n, self.m))
        for i in range(self.n):
            for j in range(self.m):
                mean = self.belief.expected_dirtiness(i, j)
                count = self.belief.visit_count(i, j)
                # To avoid log(0) and division by zero
                count = max(count, 1e-5)
                ucb = mean + np.sqrt(2 * np.log(self.total_steps + 1) / count)
                ucb_values[i, j] = ucb
        idx = np.unravel_index(np.argmax(ucb_values), ucb_values.shape)
        return idx

    def select_best_tile_ucb(self):
        best_utility = -np.inf
        best_tile = (self.x, self.y)
        for i in range(self.n):
            for j in range(self.m):
                mean = self.belief.expected_dirtiness(i, j)
                count = self.belief.visit_count(i, j)
                count = max(count, 1e-5)  # avoid division by zero
                # UCB: mean + exploration bonus (sqrt term)
                ucb_score = mean + np.sqrt(2 * np.log(self.total_steps + 1) / count)
                move_distance = abs(self.x - i) + abs(self.y - j)
                utility = self.reward_scale * ucb_score - self.move_cost * move_distance
                if utility > best_utility:
                    best_utility = utility
                    best_tile = (i, j)
        return best_tile, best_utility

    def move_and_clean_best(self):
        (nx, ny), utility = self.select_best_tile_ucb()
        move_distance = abs(self.x - nx) + abs(self.y - ny)
        self.x, self.y = nx, ny
        obs = self.env.observe(nx, ny)
        self.belief.update(nx, ny, obs)
        if obs == 1:
            self.env.clean_tile(nx, ny)
            immediate_reward = self.reward_scale
        else:
            immediate_reward = 0
        move_penalty = self.move_cost * move_distance
        self.total_reward += immediate_reward - move_penalty
        self.total_steps += 1
        return nx, ny, obs, immediate_reward, move_penalty, utility

    def move_and_observe_ucb(self):
        x, y = self.select_action_ucb()
        self.x, self.y = x, y
        obs = self.env.observe(x, y)
        self.belief.update(x, y, obs)
        self.total_steps += 1
        if obs == 1:
            self.env.clean_tile(x, y)
        return x, y, obs

    def monte_carlo_action(self, horizon=5, n_sim=50, discount=0.95):
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        best_return = -float('inf')
        best_action = (self.x, self.y)
        n, m = self.n, self.m

        for dx, dy in actions:
            nx, ny = self.x + dx, self.y + dy
            if not (0 <= nx < n and 0 <= ny < m):
                continue  # skip invalid moves
            returns = []
            for _ in range(n_sim):
                # Start fresh for each simulation
                sim_x, sim_y = nx, ny
                sim_belief = copy.deepcopy(self.belief)
                cum_reward = 0.0
                curr_discount = 1.0

                for t in range(horizon):
                    # At each step, select a random move
                    valid_moves = []
                    for adx, ady in actions:
                        nnx, nny = sim_x + adx, sim_y + ady
                        if 0 <= nnx < n and 0 <= nny < m:
                            valid_moves.append((nnx, nny))
                    if not valid_moves:
                        break  # reached edge

                    # For the first step of simulation, use the chosen initial move
                    if t == 0:
                        nnx, nny = sim_x, sim_y  # already applied
                    else:
                        nnx, nny = valid_moves[np.random.choice(len(valid_moves))]

                    # Use belief for expected reward
                    expect_dirt = sim_belief.expected_dirtiness(nnx, nny)
                    reward = self.reward_scale * expect_dirt
                    move_penalty = self.move_cost

                    cum_reward += curr_discount * (reward - move_penalty)
                    # Simulate cleaning: pretend now we know it's clean
                    # (this is optional and can be omitted for pure open-loop planning)
                    # sim_belief.alpha[nnx, nny] += 0
                    # sim_belief.beta[nnx, nny] += 1  # bias: "saw clean" in sim
                    sim_x, sim_y = nnx, nny
                    curr_discount *= discount

                returns.append(cum_reward)
            mean_return = np.mean(returns)
            if mean_return > best_return:
                best_return = mean_return
                best_action = (nx, ny)
        return best_action, best_return

    def monte_carlo_action_ucb(self, horizon=5, n_sim=50, discount=0.95, c_ucb=1.5):
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        n, m = self.n, self.m
        action_stats = []  # [(nx, ny, mean_return, std_return, ucb_score)]

        total_actions = 0  # count sims performed across all actions

        for dx, dy in actions:
            nx, ny = self.x + dx, self.y + dy
            if not (0 <= nx < n and 0 <= ny < m):
                continue
            returns = []
            for _ in range(n_sim):
                sim_x, sim_y = nx, ny
                sim_belief = copy.deepcopy(self.belief)
                cum_reward = 0.0
                curr_discount = 1.0

                for t in range(horizon):
                    valid_moves = []
                    for adx, ady in actions:
                        nnx, nny = sim_x + adx, sim_y + ady
                        if 0 <= nnx < n and 0 <= nny < m:
                            valid_moves.append((nnx, nny))
                    if not valid_moves:
                        break
                    if t == 0:
                        nnx, nny = sim_x, sim_y
                    else:
                        nnx, nny = valid_moves[np.random.choice(len(valid_moves))]
                    expect_dirt = sim_belief.expected_dirtiness(nnx, nny)
                    reward = self.reward_scale * expect_dirt
                    move_penalty = self.move_cost
                    cum_reward += curr_discount * (reward - move_penalty)
                    sim_x, sim_y = nnx, nny
                    curr_discount *= discount
                returns.append(cum_reward)
                total_actions += 1
            mean_return = np.mean(returns)
            action_sims = len(returns)
            # (Optional) add standard deviation for reference
            std_return = np.std(returns)
            ucb = mean_return + c_ucb * np.sqrt(np.log(n_sim + 1) / (action_sims + 1e-5)) # used total_actions instead of n_sim earlier
            action_stats.append((nx, ny, mean_return, std_return, ucb))

        # Pick the action with the HIGHEST UCB score
        best = max(action_stats, key=lambda t: t[4])  # t is ucb
        best_action = (best[0], best[1])
        best_return = best
        return best_action, best_return

    def monte_carlo_action_ucb_2(self, horizon=5, n_sim=50, discount=0.95, c_ucb=1.5):
        """
        Monte Carlo action selection with UCB exploration for a grid-based cleaning robot.
        Ensures fair exploration of all valid moves, including corners and edges.
        """
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        n, m = self.n, self.m
        action_stats = []  # [(nx, ny, mean_return, std_return, ucb_score)]

        total_actions = 0  # Total rollouts performed

        for dx, dy in actions:
            nx, ny = self.x + dx, self.y + dy
            # Skip invalid moves
            if not (0 <= nx < n and 0 <= ny < m):
                continue

            returns = []

            for _ in range(n_sim):
                sim_x, sim_y = nx, ny
                sim_belief = copy.deepcopy(self.belief)
                cum_reward = 0.0
                curr_discount = 1.0

                for t in range(horizon):
                    # Compute valid moves from current position
                    valid_moves = []
                    for adx, ady in actions:
                        nnx, nny = sim_x + adx, sim_y + ady
                        if 0 <= nnx < n and 0 <= nny < m:
                            valid_moves.append((nnx, nny))
                    if not valid_moves:
                        break

                    # Weighted sampling: prioritize tiles with higher expected dirtiness
                    dirt_probs = np.array([sim_belief.expected_dirtiness(x, y) + 1e-5
                                           for x, y in valid_moves])
                    dirt_probs /= dirt_probs.sum()
                    sim_x, sim_y = valid_moves[np.random.choice(len(valid_moves), p=dirt_probs)]

                    # Compute expected reward
                    expect_dirt = sim_belief.expected_dirtiness(sim_x, sim_y)
                    reward = self.reward_scale * expect_dirt
                    cum_reward += curr_discount * (reward - self.move_cost)

                    # Optional: update simulated belief (assume tile cleaned)
                    # sim_belief.update_tile(sim_x, sim_y, observed=False)

                    # Apply discount
                    curr_discount *= discount

                returns.append(cum_reward)
                total_actions += 1

            mean_return = np.mean(returns)
            std_return = np.std(returns)
            n_a = len(returns)
            ucb_score = mean_return + c_ucb * np.sqrt(np.log(total_actions + 1) / (n_a + 1e-5))
            action_stats.append((nx, ny, mean_return, std_return, ucb_score))

        # Select the action with the highest UCB score
        best = max(action_stats, key=lambda t: t[4])
        best_action = (best[0], best[1])
        return best_action, best

    def monte_carlo_action_ucb_3(self, horizon=5, n_sim=50, discount=0.95, c_ucb=1.5, alpha_visit=1.0):
        """
        Returns the best next move (nx, ny) for coverage + expected reward
        """
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        n, m = self.n, self.m

        action_stats = []  # [(nx, ny, mean_return, ucb_score)]
        total_actions = 0  # total simulations across all first-step actions

        # Initialize visit count if not done
        if not hasattr(self, "visit_count"):
            self.visit_count = np.zeros((n, m), dtype=int)

        # Ensure last_move exists for tie-break
        if not hasattr(self, "last_move"):
            self.last_move = (0, 0)

        # Count how many times each first-step action has been simulated
        n_a_dict = {}  # (nx, ny) -> number of simulations
        for dx, dy in actions:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < n and 0 <= ny < m:
                n_a_dict[(nx, ny)] = 0
                total_actions += n_sim  # each valid action will run n_sim rollouts

        for dx, dy in actions:
            nx, ny = self.x + dx, self.y + dy
            if not (0 <= nx < n and 0 <= ny < m):
                continue

            returns = []
            for _ in range(n_sim):
                # --- start of rollout ---
                sim_x, sim_y = nx, ny
                sim_belief = copy.deepcopy(self.belief)  # local copy; global state untouched
                cum_reward = 0.0
                curr_discount = 1.0

                for t in range(horizon):
                    # 1) ARRIVAL: sample observation from decayed belief (computed on-the-fly)
                    p_dirty = float(sim_belief.expected_dirtiness(sim_x, sim_y))
                    p_dirty = max(0.0, min(1.0, p_dirty))
                    dirty_obs = (np.random.rand() < p_dirty)

                    # 2) Reward for this step: cleaning if dirty, minus move cost
                    step_reward = (self.reward_scale if dirty_obs else 0.0) - self.move_cost
                    cum_reward += curr_discount * step_reward

                    # 3) Belief update at this tile (rollout-local)
                    #    NOTE: update() already does current_step += 1 and stamps last_visit
                    sim_belief.update(sim_x, sim_y, 1 if dirty_obs else 0)

                    # 4) Pick next move (if any) — use UPDATED belief
                    valid_moves = []
                    for adx, ady in actions:
                        nnx, nny = sim_x + adx, sim_y + ady
                        if 0 <= nnx < n and 0 <= nny < m:
                            valid_moves.append((nnx, nny))
                    if not valid_moves:
                        break

                    # Weighted by current expected dirtiness (after the update)
                    #  should simulate time? time progression
                    probs = np.array([sim_belief.expected_dirtiness(x, y) for (x, y) in valid_moves], dtype=float)
                    probs = np.clip(probs, 0.0, 1.0)
                    s = probs.sum()
                    if s <= 1e-12:
                        probs = np.full(len(valid_moves), 1.0 / len(valid_moves))
                    else:
                        probs /= s
                    sim_x, sim_y = valid_moves[np.random.choice(len(valid_moves), p=probs)]

                    curr_discount *= discount
                # --- end of rollout ---

                returns.append(cum_reward)  # ✅ record rollout return
                n_a_dict[(nx, ny)] += 1  # ✅ record that we simulated this first action

            mean_return = np.mean(returns) if returns else -np.inf

            # Visit bonus for coverage
            visit_bonus = alpha_visit / (1 + self.visit_count[nx, ny])

            # Proper UCB for this first-step action
            cb_term = c_ucb * np.sqrt(np.log(total_actions + 1) / (n_a_dict[(nx, ny)] + 1e-5))
            ucb = mean_return + cb_term + visit_bonus

            # add straight-line bias (optional)
            # if (dx, dy) == self.last_move:
            #     ucb += 0.05

            action_stats.append((nx, ny, mean_return, ucb))

        # Guard: in case no actions (shouldn't happen if grid valid)
        if not action_stats:
            return (self.x, self.y), (self.x, self.y, 0.0, 0.0)

        # Pick the action with highest UCB
        best = max(action_stats, key=lambda t: t[3])

        # Collect all actions within epsilon of best_score
        eps = 0.02
        candidates = [a for a in action_stats if abs(a[3] - best[3]) < eps] or [best]

        # Tie-breaker: prefer action in same direction as last_move
        chosen = candidates[0]
        for a in candidates:
            if (a[0] - self.x, a[1] - self.y) == self.last_move:
                chosen = a
                break

        best_action = (chosen[0], chosen[1])

        # for straight line bias memory
        self.last_move = (best_action[0] - self.x, best_action[1] - self.y)

        return best_action, chosen

    def monte_carlo_action_ucb_3_fast(
            self,
            horizon=5,
            n_sim=50,
            discount=0.95,
            c_ucb=1.5,  # ignored now (kept only to preserve signature)
            alpha_visit=1.0,
            rng=None
    ):
        """
        Fast equal-rollout planner (UCB removed).
        Score = mean_return + alpha_visit / (1 + visit_count[nx,ny]).
        Returns: ((nx, ny), (nx, ny, mean_return, score))
        """
        import numpy as np

        H, W = self.belief.alpha.shape
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        if not hasattr(self, "visit_count"):
            self.visit_count = np.zeros((H, W), dtype=int)
        if not hasattr(self, "last_move"):
            self.last_move = (0, 0)

        # reproducible RNG if not provided
        if rng is None:
            rng = np.random.RandomState(
                (int(self.belief.current_step) + 7)
                ^ (self.x * 5011) ^ (self.y * 6971)
            )

        # neighbors (no allocations in inner loop)
        def nbrs(x, y):
            out = []
            if x > 0:     out.append((x - 1, y))
            if x + 1 < H:   out.append((x + 1, y))
            if y > 0:     out.append((x, y - 1))
            if y + 1 < W:   out.append((x, y + 1))
            return out

        # local expected dirtiness with tiny overlay deltas (no deepcopy)
        def p_dirty_local(x, y, a_delta, b_delta, lv_delta, sim_time):
            a = self.belief.alpha[x, y] + a_delta.get((x, y), 0)
            b = self.belief.beta[x, y] + b_delta.get((x, y), 0)
            e = (a / (a + b)) if (a + b) > 0 else 0.5
            lv = lv_delta.get((x, y), self.belief.last_visit[x, y])
            dt = sim_time - lv
            if dt < 0: dt = 0
            w = 1.0 - np.exp(-dt / max(self.belief.decay_const, 1e-9))
            p = w * e
            # clamp
            if p < 0.0:
                p = 0.0
            elif p > 1.0:
                p = 1.0
            return p

        # one rollout from a first-step (nx, ny)
        def rollout_from(nx, ny):
            a_delta, b_delta, lv_delta = {}, {}, {}  # tiny per-rollout overlays
            sim_time = int(self.belief.current_step)
            x, y = nx, ny
            cum, gamma = 0.0, 1.0

            for _ in range(horizon):
                # observe at (x,y)
                p = p_dirty_local(x, y, a_delta, b_delta, lv_delta, sim_time)
                dirty = (rng.rand() < p)

                # reward
                step_r = (self.reward_scale if dirty else 0.0) - self.move_cost
                cum += gamma * step_r

                # local belief update overlays
                if dirty:
                    a_delta[(x, y)] = a_delta.get((x, y), 0) + 1
                else:
                    b_delta[(x, y)] = b_delta.get((x, y), 0) + 1
                lv_delta[(x, y)] = sim_time + 1

                # next move (probability matching on neighbors)
                nbr = nbrs(x, y)
                if not nbr: break
                probs = np.asarray([p_dirty_local(xx, yy, a_delta, b_delta, lv_delta, sim_time) for (xx, yy) in nbr],
                                   float)
                s = probs.sum()
                if s <= 1e-12:
                    k = rng.randint(len(nbr))
                else:
                    probs /= s
                    k = rng.choice(len(nbr), p=probs)
                x, y = nbr[k]

                sim_time += 1
                gamma *= discount

            return cum

        # evaluate each legal first move with equal rollouts; no UCB
        stats = []
        for dx, dy in actions:
            nx, ny = self.x + dx, self.y + dy
            if not (0 <= nx < H and 0 <= ny < W):
                continue

            total = 0.0
            for _ in range(n_sim):
                total += rollout_from(nx, ny)
            mean_return = total / n_sim if n_sim > 0 else -np.inf

            # citable count-based visit bonus
            v = self.visit_count[nx, ny]
            visit_bonus = alpha_visit / (1.0 + v)
            score = mean_return + visit_bonus

            stats.append((nx, ny, mean_return, score))

        if not stats:
            return (self.x, self.y), (self.x, self.y, 0.0, 0.0)

        # ε-tie band + same-direction tie-break
        best = max(stats, key=lambda t: t[3])
        eps = 0.02
        candidates = [a for a in stats if abs(a[3] - best[3]) < eps] or [best]

        chosen = candidates[0]
        for a in candidates:
            if (a[0] - self.x, a[1] - self.y) == self.last_move:
                chosen = a
                break

        best_action = (chosen[0], chosen[1])
        self.last_move = (best_action[0] - self.x, best_action[1] - self.y)
        return best_action, chosen

    def monte_carlo_action_ucb_adaptive(
            self,
            budget=200,  # total rollout budget (across all first-step actions)
            base_horizon=5,
            discount=0.95,
            c_ucb=1.0,
            eta_visit=0.0,  # visit shaping strength (potential-based)
            eps=0.0,  # ε-greedy inside rollout
            temp_minmax=(1.0, 1.0),  # softmax temperature bounds
    ):
        """
        Monte-Carlo action selection with *adaptive* UCB allocation and low-variance rollouts.
        Returns: ((nx, ny), (nx, ny, mean_return, ucb_score))
        """
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        n, m = self.n, self.m
        if not hasattr(self, "visit_count"):
            self.visit_count = np.zeros((n, m), dtype=int)
        if not hasattr(self, "last_move"):
            self.last_move = (0, 0)

        # enumerate legal first-step actions
        arms = []
        for dx, dy in actions:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < n and 0 <= ny < m:
                arms.append((nx, ny, dx, dy))
        if not arms:
            return (self.x, self.y), (self.x, self.y, 0.0, 0.0)

        # per-arm stats
        counts = {(nx, ny): 0 for (nx, ny, _, _) in arms}
        means = {(nx, ny): 0.0 for (nx, ny, _, _) in arms}

        # common random numbers: one RNG per arm for variance reduction
        rngs = {(nx, ny): np.random.RandomState(hash((nx, ny, self.x, self.y)) & 0xFFFFFFFF)
                for (nx, ny, _, _) in arms}

        # precompute neighbor lists (speed)
        def neighbors(x, y):
            H, W = self.belief.alpha.shape
            out = []
            if x > 0:     out.append((x - 1, y))
            if x + 1 < H: out.append((x + 1, y))
            if y > 0:     out.append((x, y - 1))
            if y + 1 < W: out.append((x, y + 1))
            return out

        # temperature from local uncertainty (Shannon entropy of Beta mean approx)
        def local_temp(x, y):
            p = float(self.belief.expected_dirtiness(x, y))
            ent = -(p * np.log(p + 1e-12) + (1 - p) * np.log(1 - p + 1e-12))
            tmin, tmax = temp_minmax
            return tmin + (tmax - tmin) * np.clip(ent / np.log(2.0), 0.0, 1.0)

        # potential-based shaping Φ(s) = η/(1+visit(x,y))
        def phi(x, y):
            return eta_visit / (1.0 + self.visit_count[x, y])

        # One rollout from first-step (nx,ny); rollbackable belief updates (delta stack)
        def simulate_from(nx, ny, rng):
            # adaptive horizon by local uncertainty
            horizon = int(round(base_horizon + 2 * local_temp(nx, ny)))
            x, y = nx, ny

            # small stacks of deltas to undo belief updates at the end
            delta_stack = []  # items: (i,j, d_alpha, d_beta, last_visit_old)
            cum = 0.0
            γ = 1.0
            last_dxdy = (nx - self.x, ny - self.y)

            # cache Φ at start to compute shaped reward γΦ(s')-Φ(s)
            phi_prev = phi(self.x, self.y)

            for t in range(horizon):
                # 1) observe dirty via Bernoulli(p) with CRN
                p_dirty = float(self.belief.expected_dirtiness(x, y))
                dirty_obs = (rng.rand() < p_dirty)

                # 2) step reward (base)
                step_r = (self.reward_scale if dirty_obs else 0.0) - self.move_cost

                # 3) belief update at (x,y) with rollback bookkeeping
                #    record old last_visit once; record Δα, Δβ
                last_visit_old = self.belief.last_visit[x, y]
                d_alpha = 1 if dirty_obs else 0
                d_beta = 0 if dirty_obs else 1
                self.belief.alpha[x, y] += d_alpha
                self.belief.beta[x, y] += d_beta
                self.belief.last_visit[x, y] = self.belief.current_step + t + 1
                delta_stack.append((x, y, d_alpha, d_beta, last_visit_old))

                # 4) shaped reward
                phi_curr = phi(x, y)
                shaped = step_r + discount * phi_curr - phi_prev
                phi_prev = phi_curr

                cum += γ * shaped

                # 5) next move: ε-greedy over softmax of shaped “desirability”
                nbrs = neighbors(x, y)
                if not nbrs:
                    break

                # desirability: expected dirtiness + small same-direction bias + inverse visit
                desir = []
                temp = local_temp(x, y)
                for (xx, yy) in nbrs:
                    ed = float(self.belief.expected_dirtiness(xx, yy))
                    sd = 0.05 if (xx - x, yy - y) == last_dxdy else 0.0
                    iv = 1.0 / (1.0 + self.visit_count[xx, yy])
                    desir.append(ed + 0.1 * iv + sd)
                desir = np.array(desir, dtype=float)
                desir = desir - desir.max()
                probs = np.exp(desir / max(temp, 1e-3))
                probs /= probs.sum()

                if rng.rand() < eps:
                    k = rng.randint(len(nbrs))
                else:
                    k = rng.choice(len(nbrs), p=probs)

                x2, y2 = nbrs[k]
                last_dxdy = (x2 - x, y2 - y)
                x, y = x2, y2
                γ *= discount

            # rollback belief
            for (i, j, da, db, lv_old) in reversed(delta_stack):
                self.belief.alpha[i, j] -= da
                self.belief.beta[i, j] -= db
                self.belief.last_visit[i, j] = lv_old

            return cum

        # Initialization: one pull per arm (ensures nonzero counts)
        for (nx, ny, _, _) in arms:
            r = simulate_from(nx, ny, rngs[(nx, ny)])
            counts[(nx, ny)] += 1
            means[(nx, ny)] = r

        # Adaptive UCB allocation until budget
        total = len(arms)  # already did one each
        while total < budget:
            # compute UCB scores
            logN = np.log(total + 1.0)
            scores = []
            for (nx, ny, _, _) in arms:
                n = counts[(nx, ny)]
                cb = c_ucb * np.sqrt(logN / n)
                # add a light *static* visit encouragement only at root
                # score = means[(nx, ny)] + cb + eta_visit / (1.0 + self.visit_count[nx, ny])
                score = means[(nx, ny)] + cb
                scores.append((score, nx, ny))
            _, ax, ay = max(scores, key=lambda t: t[0])

            # one more rollout for the chosen arm
            r = simulate_from(ax, ay, rngs[(ax, ay)])
            n = counts[(ax, ay)]
            means[(ax, ay)] = (means[(ax, ay)] * n + r) / (n + 1)
            counts[(ax, ay)] = n + 1
            total += 1

        # choose best arm, tie-break same-direction
        best_mean = max(((means[(nx, ny)], nx, ny) for (nx, ny, _, _) in arms), key=lambda t: t[0])
        best_score = best_mean[0]
        cands = [(nx, ny) for (nx, ny, _, _) in arms if abs(means[(nx, ny)] - best_score) < 1e-3] or [
            (best_mean[1], best_mean[2])]

        chosen = cands[0]
        for (nx, ny) in cands:
            if (nx - self.x, ny - self.y) == self.last_move:
                chosen = (nx, ny);
                break

        self.last_move = (chosen[0] - self.x, chosen[1] - self.y)
        ucb_dbg = max(score for (score, nx, ny) in scores if (nx, ny) == chosen)
        return chosen, (chosen[0], chosen[1], means[chosen], ucb_dbg)

    def move_and_observe_mc(self, horizon=5, n_sim=50, discount=0.95):
        # nx, ny, exp_return = *self.monte_carlo_action(horizon, n_sim, discount)[0], 0
        # (nx, ny), exp_return = self.monte_carlo_action_ucb_3(horizon, n_sim, discount)
        # (nx, ny), exp_return = self.monte_carlo_action_ucb_adaptive(horizon, n_sim, discount)
        # (nx, ny), dbg = self.monte_carlo_action_ucb_adaptive(
        #     budget=n_sim * 4,  # or whatever budget you prefer
        #     base_horizon=horizon,
        #     discount=discount
        # )
        (nx, ny), exp_return = self.monte_carlo_action_ucb_3_fast(horizon, n_sim, discount)
        self.x, self.y = nx, ny
        obs = self.env.observe(nx, ny)
        self.observe_and_clean(nx, ny)
        self.total_steps += 1
        return nx, ny, obs

    def observe_and_clean(self, x, y, obs=None):
        if obs is None:
            obs = self.env.observe(x, y)  # only read if caller didn't
        self.belief.update(x, y, obs)
        if obs == 1:
            self.env.clean_tile(x, y)
            immediate_reward = self.reward_scale
        else:
            immediate_reward = 0.0
        self.total_reward += immediate_reward - self.move_cost
        self.visit_count[x, y] += 1

