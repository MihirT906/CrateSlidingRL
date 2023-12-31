import random
import math
from EnvMap import EnvMap
from plot import plot_line, compare_lines_lat, mean_squared_error
# from Simulator import Simulator

random.seed(42)

class QLearningEligibilityTraces:
    def __init__(self, num_grid_rows=3, num_grid_cols=3, num_crates=3, obstacles={}, default_q_value=50.0):
        self.num_grid_rows = num_grid_rows
        self.num_grid_cols = num_grid_cols
        self.num_crates = num_crates
        self.slidingMap = EnvMap(rows=self.num_grid_rows, cols=self.num_grid_cols)
        self.init_pitchfork_distribution = {'p1': {(0, 0): 0.2, (1, 0): 0.2, (2, 0): 0.2, (3, 0): 0 if self.num_grid_rows < 4 else 0.2}}
        self.alpha = 0.9
        self.epsilon = 0.0
        self.discount_param = 1
        self.lam = 0.5
        self.default_q_value = default_q_value
        self.slidingMap.MOVE_REWARD = -1
        self.slidingMap.GOAL_REWARD = 10
        self.init_board = {
            'obstacles': obstacles,
            'goals': {'g1': (1, self.num_grid_cols-1)},
        }
        self.q_values = {}
        self.e_traces = {}

        self.get_initial_state()
        self.initialise_q_values()

    def initialise_q_values(self):
        for state in self.slidingMap.states:
            for action in self.slidingMap.actions:
                self.q_values[(state, action)] = self.default_q_value
                self.e_traces[(state, action)] = 0.0

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

    def get_next_action(self, current_state, epsilon, candidate_action=None):
        selected_action = random.choices(self.slidingMap.actions, weights=self.get_epsilon_greedy_action_probabilities(current_state, epsilon))[0]
        if candidate_action is not None:
            selected_action_q_val = self.q_values[(current_state, selected_action)]
            candidate_action_q_val = self.q_values[(current_state, candidate_action)]
            if candidate_action_q_val == selected_action_q_val:
                return candidate_action
        return selected_action
    
    def get_next_state(self, _current_state, current_action):
        return self.slidingMap.computeNextState(current_action)

    def get_reward(self, _current_state, _current_action, _next_state):
        return self.slidingMap.computeReward()
    
    def get_greedy_policy(self):
        policy = {}
        for state in self.slidingMap.states:
            policy[(state, self.get_next_action(state, 0.0))] = 1.0
        return policy

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
            best_next_action = self.get_next_action(next_state, 0.0,  candidate_action=next_action) # using a greedy policy here

            current_q_value = self.q_values[(current_state, current_action)]
            next_q_value = self.q_values[(next_state, best_next_action)]
            delta = reward + self.discount_param * next_q_value - current_q_value

            self.e_traces[(current_state, current_action)] = self.e_traces[(current_state, current_action)] + 1
            for state in self.slidingMap.states:
                for action in self.slidingMap.actions:
                    self.q_values[(state, action)] = self.q_values[(state, action)] + self.alpha * delta * self.e_traces[(state, action)]

                    if next_action == best_next_action:
                        self.e_traces[(state, action)] = self.discount_param * self.lam * self.e_traces[(state, action)]
                    else:
                        self.e_traces[(state, action)] = 0.0

            current_state = next_state
            current_action = next_action
        return episode_length


def run_trial(qLearner, descriptor="", maxEpisodeCount=150, maxEpisodeLength=500, show_results=False):
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
    if show_results:
        plot_line(cumulative_episode_lengths, range(0, maxEpisodeCount+1), "Timesteps", "Episode Counts", f"Q-Learning with ET - Timesteps vs Episode Counts ({descriptor})", show_results)
    
    deviation_over_episodes = []
    for idx in range(1, len(q_values_hist)):
        deviation_over_episodes.append(mean_squared_error(q_values_hist[idx], q_values_hist[idx-1]))

    if show_results:
        qLearner.slidingMap.simulateTrajectory(100, 2500)
    return cumulative_episode_lengths, deviation_over_episodes

# qLearner = QLearningEligibilityTraces()
# run_trial(qLearner, "basic", show_results=True)
# sim = Simulator(qLearner.slidingMap)
# sim.simulate(qLearner.get_greedy_policy())
# sim.visualize(500, 2000)

# qLearner = QLearningEligibilityTraces(4, 4, 2)
# run_trial(qLearner, "basic - large and simple", maxEpisodeCount=25, show_results=True)

# qLearner = QLearningEligibilityTraces(num_crates=4)
# run_trial(qLearner, "basic - complex", maxEpisodeCount=25, show_results=True)

# qLearner = QLearningEligibilityTraces(obstacles={'o1': (1, 1)})
# run_trial(qLearner, "basic - obstacle", show_results=True)

# ## Basic Obstacle Experiment - for Alpha ##
# MAX_EPISODES_COUNTS = 25
# alpha_candidates = [1, 9e-1, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4]
# MAX_TRIALS = len(alpha_candidates)
# MAX_SAMPLES = 10
# cumulative_episode_lengths_by_alpha = []
# for trial_idx in range(0, MAX_TRIALS):
#     cumulative_episode_lengths_by_sample = []
#     for sample_idx in range(0, MAX_SAMPLES):
#         print(f'[Sample {sample_idx}]')
#         qLearner = QLearningEligibilityTraces(obstacles={'o1': (1, 1)})
#         qLearner.alpha = alpha_candidates[trial_idx]
#         cumulative_episode_lengths, deviation_over_episodes = run_trial(qLearner, "basic - obstacle", maxEpisodeCount=MAX_EPISODES_COUNTS)
#         cumulative_episode_lengths_by_sample.append(cumulative_episode_lengths)
#     mean_cumulative_episode_lengths = []
#     for episode_idx in range(0, MAX_EPISODES_COUNTS+1):
#         mean_cumulative_episode_lengths.append(sum([cumulative_episode_lengths[episode_idx] for cumulative_episode_lengths in cumulative_episode_lengths_by_sample])/float(MAX_SAMPLES))
#     cumulative_episode_lengths_by_alpha.append(mean_cumulative_episode_lengths)
# compare_lines_lat(cumulative_episode_lengths_by_alpha, range(0, MAX_EPISODES_COUNTS+1), "Timesteps", "Episode Counts", "Q Learning With ET - Timesteps vs Episode Counts - Obstacle - Alpha Variations", alpha_candidates, False)


# # ## Basic Obstacle Experiment - for Default Q_Value ##
# MAX_EPISODES_COUNTS = 25
# default_q_vals_candidates = [0.0, 50, -50, 5e3, -5e3, 5e5, -5e5]
# MAX_TRIALS = len(default_q_vals_candidates)
# MAX_SAMPLES = 10
# cumulative_episode_lengths_by_q_vals = []
# for trial_idx in range(0, MAX_TRIALS):
#     cumulative_episode_lengths_by_sample = []
#     for sample_idx in range(0, MAX_SAMPLES):
#         print(f'[Sample {sample_idx}]')
#         qLearner = QLearningEligibilityTraces(obstacles={'o1': (1, 1)}, default_q_value=default_q_vals_candidates[trial_idx])
#         cumulative_episode_lengths, deviation_over_episodes = run_trial(qLearner, "basic - obstacle", maxEpisodeCount=MAX_EPISODES_COUNTS)
#         cumulative_episode_lengths_by_sample.append(cumulative_episode_lengths)
#     mean_cumulative_episode_lengths = []
#     for episode_idx in range(0, MAX_EPISODES_COUNTS+1):
#         mean_cumulative_episode_lengths.append(sum([cumulative_episode_lengths[episode_idx] for cumulative_episode_lengths in cumulative_episode_lengths_by_sample])/float(MAX_SAMPLES))
#     cumulative_episode_lengths_by_q_vals.append(mean_cumulative_episode_lengths)
# compare_lines_lat(cumulative_episode_lengths_by_q_vals, range(0, MAX_EPISODES_COUNTS+1), "Timesteps", "Episode Counts", "Q Learning With ET - Timesteps vs Episode Counts - Obstacle - Default Q Val Variations", default_q_vals_candidates, False)


# # ## Basic Obstacle Experiment - for Reward Functions ##
# MAX_EPISODES_COUNTS = 25
# reward_spec_candidates = [(-1, 0), (0, 10), (-1, 10), (-5, 10)]
# MAX_TRIALS = len(reward_spec_candidates)
# MAX_SAMPLES = 10
# cumulative_episode_lengths_by_reward_spec = []
# for trial_idx in range(0, MAX_TRIALS):
#     cumulative_episode_lengths_by_sample = []
#     for sample_idx in range(0, MAX_SAMPLES):
#         print(f'[Sample {sample_idx}]')
#         qLearner = QLearningEligibilityTraces(obstacles={'o1': (1, 1)})
#         qLearner.slidingMap.MOVE_REWARD = reward_spec_candidates[trial_idx][0]
#         qLearner.slidingMap.GOAL_REWARD = reward_spec_candidates[trial_idx][1]
#         cumulative_episode_lengths, deviation_over_episodes = run_trial(qLearner, "basic - obstacle", maxEpisodeCount=MAX_EPISODES_COUNTS)
#         cumulative_episode_lengths_by_sample.append(cumulative_episode_lengths)
#     mean_cumulative_episode_lengths = []
#     for episode_idx in range(0, MAX_EPISODES_COUNTS+1):
#         mean_cumulative_episode_lengths.append(sum([cumulative_episode_lengths[episode_idx] for cumulative_episode_lengths in cumulative_episode_lengths_by_sample])/float(MAX_SAMPLES))
#     cumulative_episode_lengths_by_reward_spec.append(mean_cumulative_episode_lengths)
# compare_lines_lat(cumulative_episode_lengths_by_reward_spec, range(0, MAX_EPISODES_COUNTS+1), "Timesteps", "Episode Counts", "Q Learning With ET - Timesteps vs Episode Counts - Obstacle - Reward Spec Variations", [f"{spec[0]}_{spec[1]}" for spec in reward_spec_candidates], False)


# # ## Basic Obstacle Experiment - for Discount Parameter Functions ##
# MAX_EPISODES_COUNTS = 25
# discount_param_candidates = [1, 0.9, 0.75, 0.5, 0.25, 0.0]
# MAX_TRIALS = len(discount_param_candidates)
# MAX_SAMPLES = 10
# cumulative_episode_lengths_by_discount_param = []
# for trial_idx in range(0, MAX_TRIALS):
#     cumulative_episode_lengths_by_sample = []
#     for sample_idx in range(0, MAX_SAMPLES):
#         print(f'[Sample {sample_idx}]')
#         qLearner = QLearningEligibilityTraces(obstacles={'o1': (1, 1)})
#         qLearner.discount_param = discount_param_candidates[trial_idx]
#         cumulative_episode_lengths, deviation_over_episodes = run_trial(qLearner, "basic - obstacle", maxEpisodeCount=MAX_EPISODES_COUNTS)
#         cumulative_episode_lengths_by_sample.append(cumulative_episode_lengths)
#     mean_cumulative_episode_lengths = []
#     for episode_idx in range(0, MAX_EPISODES_COUNTS+1):
#         mean_cumulative_episode_lengths.append(sum([cumulative_episode_lengths[episode_idx] for cumulative_episode_lengths in cumulative_episode_lengths_by_sample])/float(MAX_SAMPLES))
#     cumulative_episode_lengths_by_discount_param.append(mean_cumulative_episode_lengths)
# compare_lines_lat(cumulative_episode_lengths_by_discount_param, range(0, MAX_EPISODES_COUNTS+1), "Timesteps", "Episode Counts", "Q Learning With ET - Timesteps vs Episode Counts - Obstacle - Discount Param Variations", discount_param_candidates, False)


# # ## Basic Obstacle Experiment - for lam ##
# MAX_EPISODES_COUNTS = 25
# lam_candidates = [1, 0.9, 0.75, 0.5, 0.25, 0.0]
# MAX_TRIALS = len(lam_candidates)
# MAX_SAMPLES = 10
# cumulative_episode_lengths_by_lam = []
# for trial_idx in range(0, MAX_TRIALS):
#     cumulative_episode_lengths_by_sample = []
#     for sample_idx in range(0, MAX_SAMPLES):
#         print(f'[Sample {sample_idx}]')
#         qLearner = QLearningEligibilityTraces(obstacles={'o1': (1, 1)})
#         qLearner.lam = lam_candidates[trial_idx]
#         cumulative_episode_lengths, deviation_over_episodes = run_trial(qLearner, "basic - obstacle", maxEpisodeCount=MAX_EPISODES_COUNTS)
#         cumulative_episode_lengths_by_sample.append(cumulative_episode_lengths)
#     mean_cumulative_episode_lengths = []
#     for episode_idx in range(0, MAX_EPISODES_COUNTS+1):
#         mean_cumulative_episode_lengths.append(sum([cumulative_episode_lengths[episode_idx] for cumulative_episode_lengths in cumulative_episode_lengths_by_sample])/float(MAX_SAMPLES))
#     cumulative_episode_lengths_by_lam.append(mean_cumulative_episode_lengths)
# compare_lines_lat(cumulative_episode_lengths_by_lam, range(0, MAX_EPISODES_COUNTS+1), "Timesteps", "Episode Counts", "Q Learning With ET - Timesteps vs Episode Counts - Obstacle - Lamda Variations", lam_candidates, False)


# # Basic Obstacle Experiment ##
# MAX_SAMPLES = 5
# MAX_EPISODES_COUNTS = 175
# cumulative_episode_lengths_over_samples = []
# deviation_between_episodes_over_samples = []
# for sample_idx in range(0, MAX_SAMPLES):
#     print(f'[Sample {sample_idx}]')
#     qLearner = QLearningEligibilityTraces(obstacles={'o1': (1, 1)})
#     cumulative_episode_lengths, deviation_over_episodes = run_trial(qLearner, "basic - obstacle", maxEpisodeCount=MAX_EPISODES_COUNTS)
#     cumulative_episode_lengths_over_samples.append(cumulative_episode_lengths)
#     deviation_between_episodes_over_samples.append(deviation_over_episodes)
# mean_cumulative_episode_lengths = []
# mean_deviation_between_samples = []
# for episode_idx in range(0, MAX_EPISODES_COUNTS):
#     mean_cumulative_episode_lengths.append(sum([cumulative_episode_lengths[episode_idx] for cumulative_episode_lengths in cumulative_episode_lengths_over_samples])/float(MAX_SAMPLES))
# for episode_idx in range(0, MAX_EPISODES_COUNTS-1):
#     mean_deviation_between_samples.append(sum([deviation_over_episodes[episode_idx] for deviation_over_episodes in deviation_between_episodes_over_samples])/float(MAX_SAMPLES))
# plot_line(mean_cumulative_episode_lengths, range(0, MAX_EPISODES_COUNTS), "Timesteps", "Episode Counts", "Q Learning With ET - Cumulative timesteps vs Episode Counts - Obstacle")
# plot_line(range(1, MAX_EPISODES_COUNTS), mean_deviation_between_samples, "Episode Counts", "MSE Difference from previous epsiode q values", "Q Learning With ET - MSE difference from previous Q Values - Obstacle", True)
