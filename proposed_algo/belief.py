import numpy as np

class BeliefGrid:
    def __init__(self, n, m, alpha_init=1.0, beta_init=1.0, decay_const = 10):
        self.rows = n
        self.cols = m
        self.alpha = np.full((n, m), alpha_init)
        self.beta = np.full((n, m), beta_init)
        self.current_belief = np.full((n, m), 0.0)
        self.cool_down = np.full((n, m), 0)
        self.last_visit = np.full((n, m), 0)
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.current_step = 0
        self.decay_const = decay_const

    def update(self, x, y, obs):
        if obs == 1:  # dirty
            self.alpha[x, y] += 1
        else:  # clean
            self.beta[x, y] += 1
        self.current_belief[x, y] = 0 #if tile is observed dirty it is cleaned, or it is already clean, in either case set belief to 0
        self.current_step +=  1
        self.last_visit[x, y] = self.current_step


    def evolve(self):
        for i in range(self.rows):
            for j in range(self.cols):
                expected_dirtiness = self.alpha[i, j] / (self.alpha[i, j] + self.beta[i, j])
                time_since = self.current_step - self.last_visit[i, j]
                weight = 1 - np.exp(-time_since / self.decay_const)
                self.current_belief[i, j] = weight * expected_dirtiness

    def expected_dirtiness(self, x, y):
        return self.current_belief[x, y]

    def visit_count(self, x, y):
        # Subtract prior for true observed count
        return (self.alpha[x, y] - self.alpha_init) + (self.beta[x, y] - self.beta_init)
