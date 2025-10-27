import json
import random

# --- Parameters ---
n, m = 20, 20
episodes = 100

# --- Predefined Dirty Regions (each region is a block of tiles) ---
regions = {
    "top_left": [(i, j) for i in range(0, 4) for j in range(0, 3)],          # 4x3 block
    "top_right": [(i, j) for i in range(0, 3) for j in range(16, 20)],       # 3x4 block
    "bottom_left": [(i, j) for i in range(15, 20) for j in range(0, 5)],     # 5x5 block
    "bottom_right": [(i, j) for i in range(17, 20) for j in range(15, 20)],  # 3x5 block
    "center": [(i, j) for i in range(8, 12) for j in range(8, 12)],          # 4x4 block
    "horizontal_strip": [(10, j) for j in range(20)],                        # row 10 dirty
    "vertical_strip": [(i, 10) for i in range(20)],                          # column 10 dirty
    "center_right": [(i, j) for i in range(8, 12) for j in range(17, 20)],   # 4x3 block
    "center_left": [(i, j) for i in range(8, 12) for j in range(0, 3)],   # 4x3 block
}

episodes_regions = []
episode_regions_names = []
for episode in range(1, episodes + 1):
    # --- Randomly pick dirty regions for this episode ---
    num_regions = random.randint(1, 4)  # choose 1 to 4 regions
    chosen_region_names = random.sample(list(regions.keys()), num_regions)
    dirty_tiles_config = []
    for r in chosen_region_names:
        dirty_tiles_config.extend(regions[r])

    episodes_regions.append(dirty_tiles_config)
    episode_regions_names.append(chosen_region_names)

print(episode_regions_names)
with open('dirt_seeds.json', "w") as f:
    json.dump({ "episodes_regions": episodes_regions, "episode_regions_names": episode_regions_names, "regions": regions }, f, indent=2)


