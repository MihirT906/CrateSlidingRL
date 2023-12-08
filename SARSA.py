import random
import math
from EnvMap import EnvMap
from plot import plot_line

random.seed(42)

class SARSA:
    def __init__(self, num_grid_rows=3, num_grid_cols=3, num_crates=3, obstacles={}):
        self.num_grid_rows = num_grid_rows
        self.num_grid_cols = num_grid_cols
        self.num_crates = num_crates
        self.slidingMap = EnvMap(rows=self.num_grid_rows, cols=self.num_grid_cols)
        self.init_pitchfork_distribution = {'p1': {(0, 0): 0.2, (1, 0): 0.2, (2, 0): 0.2, (3, 0): 0 if self.num_grid_rows < 4 else 0.2}}
        self.alpha = 4e-1
        self.epsilon = 0.0
        self.discount_param = 1
        self.default_q_value = 0.0
        self.slidingMap.MOVE_REWARD = 0
        self.slidingMap.GOAL_REWARD = 10
        self.init_board = {
            'obstacles': obstacles,
            'goals': {'g1': (1, self.num_grid_cols-1)},
        }
        self.q_values = {}

        self.get_initial_state()
        self.initialise_q_values()

    def initialise_q_values(self):
        for state in self.slidingMap.states:
            for action in self.slidingMap.actions:
                self.q_values[(state, action)] = self.default_q_value

    def get_initial_state(self):
        pitchfork_state_map = {}
        for entity in self.init_pitchfork_distribution.keys():
            positions = list(self.init_pitchfork_distribution[entity].keys())
            probabilities = list(self.init_pitchfork_distribution[entity].values())
            chosen_position = random.choices(positions, probabilities)[0]
            pitchfork_state_map[entity] = chosen_position
        
        crate_state_map = {}
        allowed_spots = random.sample([(row, col) for row in range(0, self.num_grid_rows) for col in range(1, self.num_grid_cols) if (row, col) not in self.init_board['obstacles'].values()], self.num_crates)
        for i in range(0, self.num_crates):
            crate_id = 'c' + str(i)
            crate_state_map[crate_id] = allowed_spots[i]
            
        self.slidingMap.setup_board(self.init_board['obstacles'], crate_state_map, pitchfork_state_map, self.init_board['goals'])
        return self.slidingMap.getCurrentState()

    def get_epsilon_greedy_action_probabilities(self, current_state, epsilon=0.0):
        highest_q_value = -math.inf
        probs = []
        best_actions_idx = []
        base_prob = epsilon / len(self.slidingMap.actions)
        for idx, action in enumerate(self.slidingMap.actions):
            probs.append(base_prob)
            q_value = self.q_values[(current_state, action)]
            if q_value > highest_q_value:
                highest_q_value = q_value
                best_actions_idx = [idx]
            elif q_value == highest_q_value:
                best_actions_idx.append(idx)
        for best_action_idx in best_actions_idx:
            probs[best_action_idx] = (1 - epsilon)/len(best_actions_idx) + base_prob
        return probs

    def get_next_action(self, current_state, epsilon):
        return random.choices(self.slidingMap.actions, weights=self.get_epsilon_greedy_action_probabilities(current_state, epsilon))[0]
    
    def get_next_state(self, _current_state, current_action):
        return self.slidingMap.computeNextState(current_action)

    def get_reward(self, _current_state, _current_action, _next_state):
        return self.slidingMap.computeReward()

    def run_episode(self, max_episode_length=None):
        current_state = self.get_initial_state()
        episode_length = 0

        while True:
            if self.slidingMap.isTerminalState() or (max_episode_length is not None and episode_length == max_episode_length):
                break
            episode_length += 1

            current_action = self.get_next_action(current_state, self.epsilon)
            next_state = self.get_next_state(current_state, current_action)
            reward = self.get_reward(current_state, current_action, next_state)
            next_action = self.get_next_action(next_state, self.epsilon)

            current_q_value = self.q_values[(current_state, current_action)]
            next_q_value = self.q_values[(next_state, next_action)]

            self.q_values[(current_state, current_action)] = current_q_value + self.alpha*(reward + self.discount_param*next_q_value - current_q_value)
            current_state = next_state
            current_action = next_action
        return episode_length


def run_trial(qLearner, descriptor="", maxEpisodeCount=500, maxEpisodeLength=500, show_results=False):
    q_values_hist = []
    episode_length_hist = []
    trajectory_history = []

    print("[Episode 1] Running...")
    qLearner.epsilon = 1.0
    episode_length = qLearner.run_episode(maxEpisodeLength)
    episode_length_hist.append(episode_length)
    q_values_hist.append(qLearner.q_values.copy())
    trajectory_history.append(qLearner.slidingMap.trajectory)
    if show_results:
        qLearner.slidingMap.simulateTrajectory(100, 2500)

    for episode_idx in range(1, maxEpisodeCount):
        print(f"[Episode {episode_idx + 1}] Running...")
        qLearner.epsilon = 1/(episode_idx+1)
        episode_length = qLearner.run_episode(maxEpisodeLength)
        episode_length_hist.append(episode_length)
        q_values_hist.append(qLearner.q_values.copy())
        trajectory_history.append(qLearner.slidingMap.trajectory)

    cumulative_episode_lengths = [0]
    for episode_length in episode_length_hist:
        cumulative_episode_lengths.append(cumulative_episode_lengths[-1] + episode_length)

    plot_line(cumulative_episode_lengths, range(0, maxEpisodeCount+1), "Timesteps", "Episode Counts", f"SARSA - Timesteps vs Episode Counts ({descriptor})", show_results)
    if show_results:
        qLearner.slidingMap.simulateTrajectory(100, 2500)

# sarsaLearner = SARSA()
# run_trial(sarsaLearner, "basic", show_results=True)

# sarsaLearner = SARSA(4, 4, 2)
# run_trial(sarsaLearner, "basic - large and simple", maxEpisodeCount=800, show_results=True)

# sarsaLearner = SARSA(num_crates=4)
# # sarsaLearner.slidingMap.MOVE_REWARD = -1
# # sarsaLearner.slidingMap.GOAL_REWARD = 100
# run_trial(sarsaLearner, "basic - complex", show_results=True)

# sarsaLearner = SARSA(obstacles={'o1': (1, 1)})
# run_trial(sarsaLearner, "basic - obstacle", show_results=True)
