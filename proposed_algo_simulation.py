import numpy as np
import json
from proposed_algo.environment import GridEnvironment
from proposed_algo.belief import BeliefGrid
from proposed_algo.robot import Robot
from store_belief import save_episode_info_to_jsonl, save_belief_to_jsonl

# --- Parameters ---
n, m = 20, 20
episodes = 100

with open('dirt_seeds.json', 'r') as file:
    data = json.load(file)

# --- Environment, Belief, Robot ---
env = GridEnvironment(n, m)
belief = BeliefGrid(n, m)
robot = Robot(env, belief, n, m)

# --- Run episodes ---
for episode in range(1, episodes + 1):
    step = 0
    visited = np.zeros((n, m), dtype=int)

    robot.total_reward = 0.0
    robot.visit_count = np.zeros((n, m), dtype=int)
    robot.total_steps = 0
    robot.x, robot.y = 0, 0  # reset start pos

    robot.observe_and_clean(0, 0)
    visited[0, 0] = 1


    # get dirty tiles_config from json data
    dirty_tiles_config = [tuple(dirty_tile) for dirty_tile in data["episodes_regions"][episode - 1]]

    env.set_dirty_tiles(dirty_tiles_config)

    # Track when dirty tiles are first reached
    dirty_tile_steps = {str(tile): None for tile in dirty_tiles_config}

    path = [(robot.x, robot.y)]   # record path

    while not visited.all():
        x, y, obs = robot.move_and_observe_mc(horizon=10, n_sim=100, discount=0.9)
        belief.evolve()
        save_belief_to_jsonl(belief, step, episode)
        visited[x, y] += 1
        path.append((x, y))
        step += 1

        # record first step when a dirty tile is reached
        tile = (x, y)
        if str(tile) in dirty_tile_steps and dirty_tile_steps[str(tile)] is None:
            dirty_tile_steps[str(tile)] = step

    # --- Save episode data ---
    episode_info = {
        "episode": episode,
        "steps": step,
        "reward": round(robot.total_reward, 2),
        "path": path,
        "visited": visited.tolist(),
        "dirty_tile_steps": dirty_tile_steps
    }
    save_episode_info_to_jsonl(episode_info)

    print(f"Episode {episode} finished in {step} steps. Reward = {robot.total_reward:.2f}.")
