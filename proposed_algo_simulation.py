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
    robot.x, robot.y = 0, 0  # reset start

    # 1) Apply episode's initial dirt
    dirty_tiles_config = [tuple(d) for d in data["episodes_regions"][episode - 1]]
    env.set_dirty_tiles(dirty_tiles_config)

    # 2) Initialize first-arrival tracker *before* any observation/clean
    dirty_tile_steps = {str(tile): None for tile in dirty_tiles_config}

    # 3) Observe start tile once
    start_obs = env.observe(0, 0)

    # If start tile is configured dirty, record first arrival as step 0 (your convention)
    if str((0, 0)) in dirty_tile_steps and start_obs == 1:
        dirty_tile_steps[str((0, 0))] = 0

    # 4) Update belief/env exactly once for the start tile
    robot.observe_and_clean(0, 0, start_obs)
    visited[0, 0] = 1

    path = [(robot.x, robot.y)]

    # Choose world model consistency
    dynamic_world = True

    while not visited.all():
        x, y, obs = robot.move_and_observe_mc(horizon=10, n_sim=100, discount=0.9)

        if dynamic_world:
            # env.evolve(dirt_stay=0.9, dirt_new=0.05)
            belief.evolve()

        # Log belief snapshot
        save_belief_to_jsonl(belief, step, episode)

        visited[x, y] += 1
        path.append((x, y))

        # If this tile was configured dirty and not yet recorded, set first-arrival time
        if str((x, y)) in dirty_tile_steps and dirty_tile_steps[str((x, y))] is None and obs == 1:
            # Use step+1 if your convention is "after the move"; otherwise keep step
            dirty_tile_steps[str((x, y))] = step + 1

        step += 1

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

