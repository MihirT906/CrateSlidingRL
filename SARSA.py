import random
from EnvMap import EnvMap

class SARSA:
    def __init__(self):
        self.slidingMap = EnvMap()
        #self.curr_state = None
        self.init_distribution = {'p1': {(0,0): 0.25, (0,1): 0.5, (0,2): 0.25}}
        self.epsilon = 0.5
        self.num_of_actions = 8
        self.init_board = {
        'obstacles': {'o1': (1,1)},
        'crates': {'c1': (0,1)},
        'goals': {'g1': (1,2)},
        }
        
    def initialise_q_values(self):
        pass
    
    def get_initial_state(self):
        pitchfork_state_map = {}
        for entity in self.init_distribution.keys():
            positions = list(self.init_distribution[entity].keys())
            probabilities = list(self.init_distribution[entity].values())
            chosen_position = random.choices(positions, probabilities)[0]
            pitchfork_state_map[entity] = chosen_position
            
        self.slidingMap.setup_board(self.init_board['obstacles'], self.init_board['crates'], pitchfork_state_map, self.init_board['goals'])
        
        return self.slidingMap.curr_state
        
    def compute_epsilon_greedy(self, curr_state):
        pass
        
        
    def one_episode(self):
        curr_state = self.get_initial_state()
        print(curr_state)
    
    
sarsa = SARSA()
sarsa.one_episode()