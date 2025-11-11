import numpy as np

class BeliefGrid:
    def __init__(self, n, m, alpha_init=1.0, beta_init=1.0, decay_const = 20):
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
        e = self.alpha / (self.alpha + self.beta)
        e[(self.alpha + self.beta) == 0] = 0.5
        time_since = self.current_step - self.last_visit
        time_since = np.maximum(time_since, 0)
        weight = 1 - np.exp(-time_since / self.decay_const)
        self.current_belief = weight * e

    def expected_dirtiness(self, x, y):
        # Compute decayed belief on the fly:
        a = self.alpha[x, y]
        b = self.beta[x, y]
        e = (a / (a + b)) if (a + b) > 0 else 0.5
        time_since = self.current_step - self.last_visit[x, y]
        if time_since < 0:
            time_since = 0
        weight = 1 - np.exp(-time_since / self.decay_const)
        return weight * e

    def visit_count(self, x, y):
        # Subtract prior for true observed count
        return (self.alpha[x, y] - self.alpha_init) + (self.beta[x, y] - self.beta_init)
