import random
import math
import sys
from EnvMap import EnvMap
import itertools
import numpy as np
from Simulator import Simulator

GAMMA = 0.9

class ValueIteration:
    def __init__(self, num_grid_rows=3, num_grid_cols=3, num_crates=1, obstacles={}):
        self.num_grid_rows = num_grid_rows
        self.num_grid_cols = num_grid_cols
        
        self.num_crates = 3
        self.slidingMap = EnvMap(rows=self.num_grid_rows, cols=self.num_grid_cols)
        self.value_matrix = {}
        self.policy_matrix = {}
        self.default_initial_value = 0
        self.init_pitchfork_distribution = {'p1': {(0, 0): 0.2, (1, 0): 0.2, (2, 0): 0.2, (3, 0): 0 if self.num_grid_rows < 4 else 0.2}}
        self.num_pitchforks = len(self.init_pitchfork_distribution)
        self.pitchfork_keys = []
        self.crate_keys = []
        self.states = []
        self.sorted_values = []
        self.init_board = {
            'obstacles': obstacles,
            'goals': {'g1': (1, self.num_grid_cols-1)}
        }
        self.direction_list = ['U', 'R', 'D', 'L']
        
    
    def get_all_possible_states(self):
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
        self.states = self.slidingMap.states

    def initialise_value_matrix(self):
        #print(len(self.slidingMap.states)) 72 states
        self.get_all_possible_states()
        for state in self.states:
            self.value_matrix[state] = self.default_initial_value
        
        #self.show_value_matrix()
    
    def show_value_matrix(self):
        print(self.value_matrix)
        
    def get_next_state_list_for_one(self, s):
        self.slidingMap.updateStates(s)
        if(self.slidingMap.isTerminalState()):
            return []
        action_list = []
        next_state_list = []
        for idx, entity in enumerate(self.slidingMap.MOVABLE_ENTITIES):
            action_list.append((entity, 'R'))
            action_list.append((entity, 'L'))
            action_list.append((entity, 'U'))
            action_list.append((entity, 'D'))

               
        for action in action_list:
            next_state_list.append(self.slidingMap.computeNextState(action))
            self.slidingMap.updateStates(s)
            # print(action)
            # print(self.slidingMap.computeNextState(action))
        
        return next_state_list
    
    def compute_transition_prob(self,s,a,next_s):
        self.slidingMap.updateStates(s)
        if(self.slidingMap.computeNextState(a) == next_s):
            return 1
        return 0
    
    def compute_reward(self, s):
        self.slidingMap.updateStates(s)
        return self.slidingMap.computeReward()
    
    def is_terminal(self, s):
        self.slidingMap.updateStates(s)
        if(self.slidingMap.isTerminalState()):
            return True
        return False

    def iterate(self):
        entity_action_pairs = list(itertools.product(self.slidingMap.MOVABLE_ENTITIES, self.direction_list))
        for s in self.states:
            if(self.is_terminal(s)):
                self.policy_matrix[s] = None
                continue
            t_max = -sys.maxsize - 1
            for a in entity_action_pairs:
                t_sum = 0
                for next_s in self.get_next_state_list_for_one(s):
                    
                    p = self.compute_transition_prob(s,a,next_s)
                    R = self.compute_reward(next_s)
                    t_sum += p*(R + GAMMA*self.value_matrix[next_s])
                    
                if(t_sum > t_max):
                    self.policy_matrix[s] = a
                t_max = max(t_max, t_sum)
            self.value_matrix[s] = t_max
            
    def show_sorted_value_matrix(self):
        if self.value_matrix is None:
            return

        self.sorted_values = dict(sorted(self.value_matrix.items(), key=lambda item: item[1]))
        
        for key, val in self.sorted_values.items():
            print(f"State: {key}, Value: {val}")
            

    
    def find_optimal(self):
        count = 0
        while True:
            v_curr = self.value_matrix.copy()
            self.iterate()
            delta = max(abs(v_curr[key] - self.value_matrix[key]) for key in v_curr)
            #delta = np.max(absolute_diff)
            #print(absolute_diff)
            #print(delta)
            if(delta < 0.0001):
                break
            
            count+=1
            print(count)
                    
            if(count>300):
                break
    
    def show_policy(self):
        self.sorted_values = dict(sorted(self.value_matrix.items(), key=lambda item: item[1]))
        for key in self.sorted_values:
            object_value = self.policy_matrix[key]
            print(f"State: {key}, Action: {object_value}")
            
    def create_policy(self):
        optimal_deterministic_policy = {state_action_pair: 1 for state_action_pair in self.policy_matrix.items()}
        return optimal_deterministic_policy
    
    def get_optimal_policy(self):
        self.initialise_value_matrix()
        self.find_optimal()
        return self.create_policy()


vi = ValueIteration(obstacles={'o1': (1, 1)})
optimal_policy = vi.get_optimal_policy()

#vi.show_policy()


# print(vi.slidingMap.initial_board)
# print(vi.slidingMap.MOVABLE_ENTITIES)
# print([key[0] for key in optimal_policy if key[0] == ((1, 0), (0, 1), (1, 1), (0, 2))])
#print(optimal_policy[(((1, 0), (0, 1), (1, 1), (0, 2)), ('c1', 'U'))])
sim = Simulator(vi.slidingMap)
print(sim.envMap.getCurrentState())
sim.simulate(optimal_policy)
sim.visualize(500, 1000)


# vi.initialise_value_matrix()

# print(vi.get_next_state_list_for_one(((1, 2), (2, 1))))

# print(len(vi.states))

# print(vi.show_sorted_value_matrix())


# vi.find_optimal()
# print("new value matrix")

# vi.show_sorted_value_matrix()
# vi.show_policy()

# print(vi.create_policy())

# test_state = ((0,2), (0,0))
# print(vi.value_matrix[test_state])
# test_state = ((0,1), (0,0))
# print(vi.value_matrix[test_state])
# test_state = ((0,2), (1,2))
# print(vi.value_matrix[test_state])


