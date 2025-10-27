import json
import numpy as np
from store_belief import save_episode_info_to_jsonl

N, M = 20, 20
EPISODES = 100

dirty_tiles_config = []

with open('dirt_seeds.json', 'r') as file:
    data = json.load(file)

def boustrophedon_path():
    path = []
    for i in range(N):
        row = list(range(M)) if i % 2 == 0 else list(reversed(range(M)))
        for j in row:
            path.append((i, j))
    return path


def run_boustrophedon_episode(episode=0):
    path = boustrophedon_path()
    visited = np.zeros((N, M), dtype=int)
    dirty_tile_steps = {str(tile): None for tile in dirty_tiles_config}

    steps = 0
    for pos in path:
        i, j = pos
        visited[i, j] += 1
        steps += 1

        # record first time we hit a dirty tile
        tile = (pos[0], pos[1])
        if str(tile) in dirty_tile_steps and dirty_tile_steps[str(tile)] is None:
            dirty_tile_steps[str(tile)] = steps

    return {
        "steps": steps,
        "path": path,
        "episode": episode,
        "visited": visited.tolist(),
        "dirty_tile_steps": dirty_tile_steps
    }


for ep in range(1, EPISODES + 1):
    dirty_tiles_config = [tuple(dirty_tile) for dirty_tile in data["episodes_regions"][ep - 1]]
    result = run_boustrophedon_episode(ep)
    save_episode_info_to_jsonl(result, "boustrophedon")
    print(f"Episode {ep} finished in {result['steps']} steps")