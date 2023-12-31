from EnvMap import EnvMap
from Visualizer import Visualizer
import random

random.seed(42)

class Simulator:
    def __init__(self, envMap):
        self.envMap = EnvMap(rows=envMap.ROWS, cols=envMap.COLS)
        self.ui = Visualizer(self.envMap.ROWS, self.envMap.COLS)
        self.envMap.setup_board(envMap.initial_board['obstacle'], envMap.initial_board['crate'], envMap.initial_board['pitchfork'], envMap.initial_board['goal'])
        self.ui.setup_entities(self.envMap.initial_board['obstacle'], self.envMap.initial_board['crate'], self.envMap.initial_board['pitchfork'], self.envMap.initial_board['goal'])

    def simulate(self, policy={}, max_episode_length=100):
        episode_length = 0
        episode_return = 0
        print('[Simulator] Starting simulation')
        while True:
            episode_return += self.envMap.computeReward()
            if self.envMap.isTerminalState() or (max_episode_length is not None and episode_length >= max_episode_length):
                break
            current_state = self.envMap.getCurrentState()
            possible_actions, probs = zip(*[(key[1], policy[key]) for key in policy if key[0] == current_state])
            selected_action = random.choices(possible_actions, weights=probs)[0]
            self.envMap.computeNextState(selected_action)
            episode_length += 1
        print(f'[Simulator] Ran for {episode_length} timesteps')
        return episode_return

    def visualize(self, timestep=500, delay=0):
        self.ui.run(self.envMap.trajectory, timestep, delay)

    