import numpy as np
import random
from Visualizer import EpisodeSimulator

class EnvMap:
    def __init__(self):
        self.ROWS = 3
        self.COLS = 3
        self.OBSTACLE_STATES = {}
        self.CRATE_STATES = {}
        self.PITCHFORK_STATES = {}
        self.GOAL_STATES = {}
        self.GOAL_REWARD = 0
        self.MOVE_REWARD = -1

        self.curr_state = (self.PITCHFORK_STATES, self.CRATE_STATES)
        self.curr_action = None
        
        self.initial_board = {
            'obstacle': self.OBSTACLE_STATES.copy(), 
            'crate': self.CRATE_STATES.copy(), 
            'pitchfork': self.PITCHFORK_STATES.copy(), 
            'goal': self.GOAL_STATES.copy()
        }
        self.trajectory = []

    def setup_board(self, obstacle_states, crate_states, pitchfork_states, goal_states):
        self.OBSTACLE_STATES = obstacle_states
        self.CRATE_STATES = crate_states
        self.PITCHFORK_STATES = pitchfork_states
        self.GOAL_STATES = goal_states
        self.curr_state = (self.PITCHFORK_STATES, self.CRATE_STATES)

        self.initial_board = {
            'obstacle': self.OBSTACLE_STATES.copy(), 
            'crate': self.CRATE_STATES.copy(), 
            'pitchfork': self.PITCHFORK_STATES.copy(), 
            'goal': self.GOAL_STATES.copy()
        }

    def isLegalMove(self, entity, new_pos):
        #If action results in entity going out of bounds, then the entity stays in the same position
        if(not ((0 <= new_pos[0]<= self.ROWS-1) and (0 <= new_pos[1]<= self.COLS-1))):
            return False

        #If action results in entity going into an obstacle, the entity stays in the same position
        if new_pos in self.OBSTACLE_STATES.values():
            return False

        #If action results in entity going into another entity, the entity stays in the same position
        if new_pos in self.CRATE_STATES.values():
            for crate_entity, crate_position in self.CRATE_STATES.items():
                if crate_entity != entity and crate_position == new_pos:
                    return False

        if new_pos in self.PITCHFORK_STATES.values():
            for pitchfork_entity, pitchfork_position in self.PITCHFORK_STATES.items():
                if pitchfork_entity != entity and pitchfork_position == new_pos:
                    return False
        return True


    def computeNextState(self, action=None):
        #Exit early if no action has been specified
        if action is None and self.curr_action is None:
            return None

        entity, direction = action if action is not None else self.curr_action

        #Determine state dict
        state_lookup = {}
        if entity in self.CRATE_STATES:
            state_lookup = self.CRATE_STATES
        elif entity in self.PITCHFORK_STATES:
            state_lookup = self.PITCHFORK_STATES

        current_position = state_lookup[entity]
        if direction == 'R':
            new_position = (current_position[0], current_position[1]+1)
        elif direction == 'L':
            new_position = (current_position[0], current_position[1]-1)
        elif direction == 'U':
            new_position = (current_position[0]-1, current_position[1])
        elif direction == 'D':
            new_position = (current_position[0]+1, current_position[1])

        state_lookup[entity] = new_position if self.isLegalMove(entity, new_position) else current_position
        self.trajectory.append((entity, direction, state_lookup[entity]))
        return state_lookup[entity]


    def computeReward(self):
        if self.curr_state in self.GOAL_STATES.values():
            return self.GOAL_REWARD
        return self.MOVE_REWARD

    def drawBoard(self):
        ui = EpisodeSimulator(self.ROWS, self.COLS)
        ui.setup_entities(self.OBSTACLE_STATES, self.CRATE_STATES, self.PITCHFORK_STATES, self.GOAL_STATES)
        ui.run([])
        
    def simulateTrajectory(self):
        ui = EpisodeSimulator(self.ROWS, self.COLS)
        ui.setup_entities(self.initial_board['obstacle'], self.initial_board['crate'], self.initial_board['pitchfork'], self.initial_board['goal'])
        ui.run(self.trajectory)


slidingMap = EnvMap()
slidingMap.setup_board(
    {'o1': (1,1)},
    {'c1': (0,1)},
    {'p1': (1,0)},
    {'g1': (1,2)},
)
slidingMap.drawBoard() # view board at any given instance

actions = [
    ('c1', 'D'),
    ('c1', 'R'),
    ('p1', 'R'),
    ('p1', 'U'),
    ('p1', 'R'),
    ('p1', 'D'),
    ('p1', 'R'),
    ('c1', 'D'),
    ('c1', 'D'),
    ('p1', 'R'),
    ('p1', 'D')
]
for action in actions:
    slidingMap.computeNextState(action)

slidingMap.simulateTrajectory() # replay trajectory