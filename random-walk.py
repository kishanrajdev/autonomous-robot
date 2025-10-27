import random
import json
import numpy as np
from store_belief import save_episode_info_to_jsonl

N, M = 20, 20
EPISODES = 100
directions = [(-1,0), (0,1), (1,0), (0,-1)]
dirty_tiles_config = []

with open('dirt_seeds.json', 'r') as file:
    data = json.load(file)

def run_random_walk_episode(episode = 0):
    pos = (0, 0)  # Or use random start: (random.randint(0,N-1), random.randint(0,M-1))
    visited = np.zeros((N, M), dtype=int)
    path = [pos]
    visited[pos[0], pos[1]] = 1
    steps = 1
    dirty_tile_steps = {str(tile): None for tile in dirty_tiles_config}
    while not visited.all():
        x, y = pos
        candidates_unvisited = []
        candidates_all = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < M:
                candidates_all.append((nx, ny))
                if visited[nx, ny] == 0:
                    candidates_unvisited.append((nx, ny))
        if candidates_unvisited:
            pos = random.choice(candidates_unvisited)
        elif candidates_all:
            pos = random.choice(candidates_all)
        visited[pos[0],pos[1]] += 1

        # record first step when a dirty tile is reached
        tile = (pos[0],pos[1])
        if str(tile) in dirty_tile_steps and dirty_tile_steps[str(tile)] is None:
            dirty_tile_steps[str(tile)] = steps

        path.append(pos)
        steps += 1
    return {
        "steps": steps,
        "path": path,
        "episode": episode,
        "visited": visited.tolist(),
        "dirty_tile_steps": dirty_tile_steps
    }


for ep in range(1, EPISODES+1):
    dirty_tiles_config = [tuple(dirty_tile) for dirty_tile in data["episodes_regions"][ep - 1]]
    result = run_random_walk_episode(ep)
    save_episode_info_to_jsonl(result, "random_walk")
    print(f"Episode {ep} finished in {result['steps']} steps")
