import numpy as np
import copy

from monte_carlo_simple_cleaning import observe


class Robot:
    def __init__(self, env, belief, n, m, start=(0, 0), move_cost=0.1, reward_scale=1.0):
        self.env = env
        self.belief = belief
        self.x, self.y = start
        self.n = n
        self.m = m
        self.move_cost = move_cost
        self.reward_scale = reward_scale
        self.total_reward = 0.0
        self.total_steps = 1
        self.last_move = (0, 0)

    def select_action_ucb(self):
        ucb_values = np.zeros((self.n, self.m))
        for i in range(self.n):
            for j in range(self.m):
                mean = self.belief.expected_dirtiness(i, j)
                count = self.belief.visit_count(i, j)
                # To avoid log(0) and division by zero
                count = max(count, 1e-5)
                ucb = mean + np.sqrt(2 * np.log(self.total_steps + 1) / count)
                ucb_values[i, j] = ucb
        idx = np.unravel_index(np.argmax(ucb_values), ucb_values.shape)
        return idx

    def select_best_tile_ucb(self):
        best_utility = -np.inf
        best_tile = (self.x, self.y)
        for i in range(self.n):
            for j in range(self.m):
                mean = self.belief.expected_dirtiness(i, j)
                count = self.belief.visit_count(i, j)
                count = max(count, 1e-5)  # avoid division by zero
                # UCB: mean + exploration bonus (sqrt term)
                ucb_score = mean + np.sqrt(2 * np.log(self.total_steps + 1) / count)
                move_distance = abs(self.x - i) + abs(self.y - j)
                utility = self.reward_scale * ucb_score - self.move_cost * move_distance
                if utility > best_utility:
                    best_utility = utility
                    best_tile = (i, j)
        return best_tile, best_utility

    def move_and_clean_best(self):
        (nx, ny), utility = self.select_best_tile_ucb()
        move_distance = abs(self.x - nx) + abs(self.y - ny)
        self.x, self.y = nx, ny
        obs = self.env.observe(nx, ny)
        self.belief.update(nx, ny, obs)
        if obs == 1:
            self.env.clean_tile(nx, ny)
            immediate_reward = self.reward_scale
        else:
            immediate_reward = 0
        move_penalty = self.move_cost * move_distance
        self.total_reward += immediate_reward - move_penalty
        self.total_steps += 1
        return nx, ny, obs, immediate_reward, move_penalty, utility

    def move_and_observe_ucb(self):
        x, y = self.select_action_ucb()
        self.x, self.y = x, y
        obs = self.env.observe(x, y)
        self.belief.update(x, y, obs)
        self.total_steps += 1
        if obs == 1:
            self.env.clean_tile(x, y)
        return x, y, obs

    def monte_carlo_action(self, horizon=5, n_sim=50, discount=0.95):
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        best_return = -float('inf')
        best_action = (self.x, self.y)
        n, m = self.n, self.m

        for dx, dy in actions:
            nx, ny = self.x + dx, self.y + dy
            if not (0 <= nx < n and 0 <= ny < m):
                continue  # skip invalid moves
            returns = []
            for _ in range(n_sim):
                # Start fresh for each simulation
                sim_x, sim_y = nx, ny
                sim_belief = copy.deepcopy(self.belief)
                cum_reward = 0.0
                curr_discount = 1.0

                for t in range(horizon):
                    # At each step, select a random move
                    valid_moves = []
                    for adx, ady in actions:
                        nnx, nny = sim_x + adx, sim_y + ady
                        if 0 <= nnx < n and 0 <= nny < m:
                            valid_moves.append((nnx, nny))
                    if not valid_moves:
                        break  # reached edge

                    # For the first step of simulation, use the chosen initial move
                    if t == 0:
                        nnx, nny = sim_x, sim_y  # already applied
                    else:
                        nnx, nny = valid_moves[np.random.choice(len(valid_moves))]

                    # Use belief for expected reward
                    expect_dirt = sim_belief.expected_dirtiness(nnx, nny)
                    reward = self.reward_scale * expect_dirt
                    move_penalty = self.move_cost

                    cum_reward += curr_discount * (reward - move_penalty)
                    # Simulate cleaning: pretend now we know it's clean
                    # (this is optional and can be omitted for pure open-loop planning)
                    # sim_belief.alpha[nnx, nny] += 0
                    # sim_belief.beta[nnx, nny] += 1  # bias: "saw clean" in sim
                    sim_x, sim_y = nnx, nny
                    curr_discount *= discount

                returns.append(cum_reward)
            mean_return = np.mean(returns)
            if mean_return > best_return:
                best_return = mean_return
                best_action = (nx, ny)
        return best_action, best_return

    def monte_carlo_action_ucb(self, horizon=5, n_sim=50, discount=0.95, c_ucb=1.5):
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        n, m = self.n, self.m
        action_stats = []  # [(nx, ny, mean_return, std_return, ucb_score)]

        total_actions = 0  # count sims performed across all actions

        for dx, dy in actions:
            nx, ny = self.x + dx, self.y + dy
            if not (0 <= nx < n and 0 <= ny < m):
                continue
            returns = []
            for _ in range(n_sim):
                sim_x, sim_y = nx, ny
                sim_belief = copy.deepcopy(self.belief)
                cum_reward = 0.0
                curr_discount = 1.0

                for t in range(horizon):
                    valid_moves = []
                    for adx, ady in actions:
                        nnx, nny = sim_x + adx, sim_y + ady
                        if 0 <= nnx < n and 0 <= nny < m:
                            valid_moves.append((nnx, nny))
                    if not valid_moves:
                        break
                    if t == 0:
                        nnx, nny = sim_x, sim_y
                    else:
                        nnx, nny = valid_moves[np.random.choice(len(valid_moves))]
                    expect_dirt = sim_belief.expected_dirtiness(nnx, nny)
                    reward = self.reward_scale * expect_dirt
                    move_penalty = self.move_cost
                    cum_reward += curr_discount * (reward - move_penalty)
                    sim_x, sim_y = nnx, nny
                    curr_discount *= discount
                returns.append(cum_reward)
                total_actions += 1
            mean_return = np.mean(returns)
            action_sims = len(returns)
            # (Optional) add standard deviation for reference
            std_return = np.std(returns)
            ucb = mean_return + c_ucb * np.sqrt(np.log(n_sim + 1) / (action_sims + 1e-5)) # used total_actions instead of n_sim earlier
            action_stats.append((nx, ny, mean_return, std_return, ucb))

        # Pick the action with the HIGHEST UCB score
        best = max(action_stats, key=lambda t: t[4])  # t is ucb
        best_action = (best[0], best[1])
        best_return = best
        return best_action, best_return

    def monte_carlo_action_ucb_2(self, horizon=5, n_sim=50, discount=0.95, c_ucb=1.5):
        """
        Monte Carlo action selection with UCB exploration for a grid-based cleaning robot.
        Ensures fair exploration of all valid moves, including corners and edges.
        """
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        n, m = self.n, self.m
        action_stats = []  # [(nx, ny, mean_return, std_return, ucb_score)]

        total_actions = 0  # Total rollouts performed

        for dx, dy in actions:
            nx, ny = self.x + dx, self.y + dy
            # Skip invalid moves
            if not (0 <= nx < n and 0 <= ny < m):
                continue

            returns = []

            for _ in range(n_sim):
                sim_x, sim_y = nx, ny
                sim_belief = copy.deepcopy(self.belief)
                cum_reward = 0.0
                curr_discount = 1.0

                for t in range(horizon):
                    # Compute valid moves from current position
                    valid_moves = []
                    for adx, ady in actions:
                        nnx, nny = sim_x + adx, sim_y + ady
                        if 0 <= nnx < n and 0 <= nny < m:
                            valid_moves.append((nnx, nny))
                    if not valid_moves:
                        break

                    # Weighted sampling: prioritize tiles with higher expected dirtiness
                    dirt_probs = np.array([sim_belief.expected_dirtiness(x, y) + 1e-5
                                           for x, y in valid_moves])
                    dirt_probs /= dirt_probs.sum()
                    sim_x, sim_y = valid_moves[np.random.choice(len(valid_moves), p=dirt_probs)]

                    # Compute expected reward
                    expect_dirt = sim_belief.expected_dirtiness(sim_x, sim_y)
                    reward = self.reward_scale * expect_dirt
                    cum_reward += curr_discount * (reward - self.move_cost)

                    # Optional: update simulated belief (assume tile cleaned)
                    # sim_belief.update_tile(sim_x, sim_y, observed=False)

                    # Apply discount
                    curr_discount *= discount

                returns.append(cum_reward)
                total_actions += 1

            mean_return = np.mean(returns)
            std_return = np.std(returns)
            n_a = len(returns)
            ucb_score = mean_return + c_ucb * np.sqrt(np.log(total_actions + 1) / (n_a + 1e-5))
            action_stats.append((nx, ny, mean_return, std_return, ucb_score))

        # Select the action with the highest UCB score
        best = max(action_stats, key=lambda t: t[4])
        best_action = (best[0], best[1])
        return best_action, best

    def monte_carlo_action_ucb_3(self, horizon=5, n_sim=50, discount=0.95, c_ucb=1.5, alpha_visit=1.0):
        """
        Returns the best next move (nx, ny) for coverage + expected reward
        """

        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        n, m = self.n, self.m

        action_stats = []  # [(nx, ny, mean_return, ucb_score)]
        total_actions = 0  # total simulations across all first-step actions

        # Initialize visit count if not done
        if not hasattr(self, "visit_count"):
            self.visit_count = np.zeros((n, m), dtype=int)

        # Count how many times each first-step action has been simulated
        n_a_dict = {}  # (nx, ny) -> number of simulations
        for dx, dy in actions:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < n and 0 <= ny < m:
                n_a_dict[(nx, ny)] = 0
                total_actions += n_sim  # each valid action will run n_sim rollouts

        for dx, dy in actions:
            nx, ny = self.x + dx, self.y + dy
            if not (0 <= nx < n and 0 <= ny < m):
                continue

            returns = []
            for _ in range(n_sim):
                # Simulate the rollout
                sim_x, sim_y = nx, ny
                cum_reward = 0.0
                curr_discount = 1.0

                # Optional: copy belief if needed for dirt reward
                sim_belief = copy.deepcopy(self.belief)

                for t in range(horizon):
                    valid_moves = [(sim_x + adx, sim_y + ady) for adx, ady in actions
                                   if 0 <= sim_x + adx < n and 0 <= sim_y + ady < m]
                    if not valid_moves:
                        break

                    # Random next step in rollout
                    sim_x, sim_y = valid_moves[np.random.choice(len(valid_moves))]

                    # Reward: for coverage-only, could be 1 per new tile
                    reward = self.reward_scale * sim_belief.expected_dirtiness(sim_x, sim_y)
                    move_penalty = self.move_cost
                    cum_reward += curr_discount * (reward - move_penalty)
                    curr_discount *= discount

                returns.append(cum_reward)
                n_a_dict[(nx, ny)] += 1  # increment simulations for this first-step action

            mean_return = np.mean(returns)

            # Visit bonus for coverage
            visit_bonus = alpha_visit / (1 + self.visit_count[nx, ny])

            # Proper UCB for this first-step action
            cb_term = c_ucb * np.sqrt(np.log(total_actions + 1) / (n_a_dict[(nx, ny)] + 1e-5))
            ucb = mean_return + cb_term + visit_bonus

            # add straight-line bias
            # if (dx, dy) == self.last_move:
            #     ucb += 0.05

            action_stats.append((nx, ny, mean_return, ucb))

        # Pick the action with highest UCB
        best = max(action_stats, key=lambda t: t[3])

        # Collect all actions within epsilon of best_score
        candidates = [a for a in action_stats if abs(a[3] - best[3]) < 0.02]
        # Tie-breaker: prefer action in same direction as last_move
        best = candidates[0]  # default if none matches
        for a in candidates:
            if (a[0] - self.x, a[1] - self.y) == self.last_move:
                best = a
                break

        best_action = (best[0], best[1])

        #for straight line bias
        self.last_move = (best_action[0] - self.x, best_action[1] - self.y)

        return best_action, best

    def move_and_observe_mc(self, horizon=5, n_sim=50, discount=0.95):
        # nx, ny, exp_return = *self.monte_carlo_action(horizon, n_sim, discount)[0], 0
        (nx, ny), exp_return = self.monte_carlo_action_ucb_3(horizon, n_sim, discount)

        self.x, self.y = nx, ny
        obs = self.env.observe(nx, ny)
        self.observe_and_clean(nx, ny)
        self.total_steps += 1
        return nx, ny, obs

    def observe_and_clean(self, x, y):
        obs = self.env.observe(x, y)
        self.belief.update(x, y, obs)
        if obs == 1:
            self.env.clean_tile(x, y)
            immediate_reward = self.reward_scale
        else:
            immediate_reward = 0
        self.total_reward += immediate_reward - self.move_cost
        # Update visit count for chosen tile
        self.visit_count[x, y] += 1
