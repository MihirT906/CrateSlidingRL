import numpy as np
from itertools import permutations
from Visualizer import Visualizer

class EnvMap:
    def __init__(self, rows=3, cols=3):
        self.ROWS = rows
        self.COLS = cols
        self.OBSTACLE_STATES = {}
        self.CRATE_STATES = {}
        self.PITCHFORK_STATES = {}
        self.GOAL_STATES = {}
        self.GOAL_REWARD = 0
        self.MOVE_REWARD = -1
        self.MOVABLE_ENTITIES = []

        self.curr_action = None
        
        self.initial_board = {
            'obstacle': self.OBSTACLE_STATES.copy(), 
            'crate': self.CRATE_STATES.copy(), 
            'pitchfork': self.PITCHFORK_STATES.copy(), 
            'goal': self.GOAL_STATES.copy()
        }
        self.states = []
        self.actions = []
        self.trajectory = []

    def setup_board(self, obstacle_states, crate_states, pitchfork_states, goal_states):
        self.OBSTACLE_STATES = obstacle_states.copy()
        self.CRATE_STATES = crate_states.copy()
        self.PITCHFORK_STATES = pitchfork_states.copy()
        self.GOAL_STATES = goal_states.copy()
        self.MOVABLE_ENTITIES = list(self.PITCHFORK_STATES.keys()) + list(self.CRATE_STATES.keys())

        self.initial_board = {
            'obstacle': self.OBSTACLE_STATES.copy(), 
            'crate': self.CRATE_STATES.copy(), 
            'pitchfork': self.PITCHFORK_STATES.copy(), 
            'goal': self.GOAL_STATES.copy()
        }

        # get state permutations
        allowed_states = [(row, col) for row in range(0, self.ROWS) for col in range(0, self.COLS) if (row, col) not in self.OBSTACLE_STATES.values()]
        self.states = list(permutations(allowed_states, len(self.MOVABLE_ENTITIES)))

        # get actions
        self.actions = [(entity, action) for entity in self.MOVABLE_ENTITIES for action in ['U', 'R', 'D', 'L']]
        self.trajectory = []

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

    def getCurrentState(self):
        combined_map = {**self.PITCHFORK_STATES, **self.CRATE_STATES}
        return tuple(combined_map[entity] for entity in self.MOVABLE_ENTITIES)

    def updateStates(self, state_tuple):
        for idx, entity in enumerate(self.MOVABLE_ENTITIES):
            if entity in self.PITCHFORK_STATES:
                self.PITCHFORK_STATES[entity] = state_tuple[idx]
            if entity in self.CRATE_STATES:
                self.CRATE_STATES[entity] = state_tuple[idx]

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
        self.trajectory.append((entity, direction, state_lookup[entity], self.computeReward()))
        return self.getCurrentState()

    def isPitchforkOnAGoal(self):
        goals_pos = self.GOAL_STATES.values()
        return any([pitchfork_pos in goals_pos for pitchfork_pos in self.PITCHFORK_STATES.values()])

    def computeReward(self):
        if self.isPitchforkOnAGoal():
            return self.GOAL_REWARD
        return self.MOVE_REWARD

    def isTerminalState(self):
        return self.isPitchforkOnAGoal()

    def drawBoard(self):
        ui = Visualizer(self.ROWS, self.COLS)
        ui.setup_entities(self.OBSTACLE_STATES, self.CRATE_STATES, self.PITCHFORK_STATES, self.GOAL_STATES)
        ui.run([])
        
    def simulateTrajectory(self, timestep=500, delay=0):
        ui = Visualizer(self.ROWS, self.COLS)
        ui.setup_entities(self.initial_board['obstacle'], self.initial_board['crate'], self.initial_board['pitchfork'], self.initial_board['goal'])
        ui.run(self.trajectory, timestep, delay)

