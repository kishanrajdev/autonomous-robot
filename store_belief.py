import json

def save_belief_to_json(belief, max_steps=1000, step = 0):
    # Append the new belief
    new_belief_dict = {
        "step": step,
        "belief": {
            f"({i},{j})": float(belief.expected_dirtiness(i, j))
            for i in range(belief.alpha.shape[0])
            for j in range(belief.alpha.shape[1])
        }
    }

    filename = 'belief.json'
    try:
        with open(filename, "r") as f:
            belief_array = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        belief_array = []  # start fresh if file doesn't exist or is empty

    belief_array.append(new_belief_dict)
    # Trim to last max_steps
    if len(belief_array) > max_steps:
        belief_array = belief_array[-max_steps:]

    with open(filename, "w") as f:
        json.dump(belief_array, f, indent=4)

    return belief_array


def save_episode_info_to_json(result):
    filename = 'episodes.json'
    try:
        with open(filename, "r") as f:
            episodes = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        episodes = []  # start fresh if file doesn't exist or is empty

    episodes.append(result)
    with open(filename, "w") as f:
        json.dump(episodes, f, indent=4)


def save_episode_info_to_jsonl(result, algo = "proposed"):
    with open(f"json/episodes_{algo}.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")


def save_belief_to_jsonl(belief, step = 0, episode = 0):
    # Append the new belief
    new_belief_dict = {
        "episode": episode,
        "step": step,
        "belief": {
            f"({i},{j})": float(belief.expected_dirtiness(i, j))
            for i in range(belief.alpha.shape[0])
            for j in range(belief.alpha.shape[1])
        }
    }

    with open("json/belief.jsonl", "a") as f:
        f.write(json.dumps(new_belief_dict) + "\n")
