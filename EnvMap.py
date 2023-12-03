import numpy as np
import random

class EnvMap:
    def __init__(self):
        self.ROWS = 3
        self.COLS = 3
        self.OBSTACLE_STATES = {'o1': (1,1)}
        self.CRATE_STATES = {'c1': (0,1)}
        self.PITCHFORK_STATES = {'p1': (1,0)}
        self.GOAL_STATES = {'g1': (1,2)}
        self.GOAL_REWARD = 0
        self.MOVE_REWARD = -1

        self.curr_state = (self.PITCHFORK_STATES, self.CRATE_STATES)
        self.curr_action = None


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


    def computeNextState(self):
        #Exit early if no action has been specified
        if self.curr_action is None:
            return None

        entity, direction = self.curr_action

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

        print(self.curr_state)
        return state_lookup[entity]


    def computeReward(self):
        if self.curr_state in self.GOAL_STATES.values():
            return self.GOAL_REWARD
        return self.MOVE_REWARD

    def drawBoard(self):
        pass

slidingMap = EnvMap()
slidingMap.curr_action = ('c1', 'D')
print(slidingMap.computeNextState())
slidingMap.curr_action = ('c1', 'R')
print(slidingMap.computeNextState())