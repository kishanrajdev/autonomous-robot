import json
import numpy as np
from store_belief import save_episode_info_to_jsonl

N, M = 20, 20
EPISODES = 100
dirty_tiles_config = []


with open('dirt_seeds.json', 'r') as file:
    data = json.load(file)

# Spiral path generator
def spiral_path():
    path = []
    left, right, top, bottom = 0, M - 1, 0, N - 1
    while left <= right and top <= bottom:
        # Top row
        for j in range(left, right + 1):
            path.append((top, j))
        top += 1
        # Right col
        for i in range(top, bottom + 1):
            path.append((i, right))
        right -= 1
        if top <= bottom:
            # Bottom row
            for j in range(right, left - 1, -1):
                path.append((bottom, j))
            bottom -= 1
        if left <= right:
            # Left col
            for i in range(bottom, top - 1, -1):
                path.append((i, left))
            left += 1
    return path


def run_spiral_episode(episode=0):
    path = spiral_path()
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
    dirty_tiles_config = [tuple(dirty_tile)  for dirty_tile in data["episodes_regions"][ep - 1]]
    result = run_spiral_episode(ep)
    save_episode_info_to_jsonl(result, "spiral")
    print(f"Episode {ep} finished in {result['steps']} steps")
