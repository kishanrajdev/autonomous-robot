import numpy as np

class GridEnvironment:
    def __init__(self, n, m, dirt_prob=0.5):
        self.grid = np.zeros((n, m), dtype=int)  # start clean

    def evolve(self, dirt_stay=0.9, dirt_new=0.05):
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i, j] == 1:
                    # Dirty stays dirty with dirt_stay, can become clean
                    self.grid[i, j] = 1 if np.random.rand() < dirt_stay else 0
                else:
                    # Clean becomes dirty with dirt_new, else stays clean
                    self.grid[i, j] = 1 if np.random.rand() < dirt_new else 0

    def observe(self, x, y):
        return self.grid[x, y]

    def clean_tile(self, x, y):
        self.grid[x, y] = 0

    def set_dirty_tiles(self, coords):
        for x, y in coords:
            self.grid[x, y] = 1