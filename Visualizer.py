from tkinter import *

class Visualizer:
    def __init__(self, height, width):
        self.block_color = 'white'
        self.block_size = 75
        self.root = Tk()
        
        grid_padding = 1
        self.frm = Frame(
            self.root, 
            height=height * (self.block_size + 2*grid_padding),
            width=width * (self.block_size + 2*grid_padding), 
            background='lightgray'
        )

        self.frm.grid_propagate(0)
        self.frm.grid(column=0, row=0, padx=20, pady=20)
        for i in range(0, width):
            for j in range(0, height):
                cur_frame = Frame(self.frm, background=self.block_color, height=self.block_size, width=self.block_size)
                cur_frame.grid(column=i, row=j, padx=grid_padding, pady=grid_padding)

        self.ui_mapper = {}

    def create_square(self, side, canvasName, color):
        return canvasName.create_rectangle(0, 0, side, side, fill=color)

    def addEntity(self, x, y, color='black', padding=0, label=None):
        cv = Canvas(self.frm, height=self.block_size-padding, width=self.block_size-padding, background=self.block_color, highlightthickness=0)
        square_side = cv.winfo_reqheight()
        cv.create_rectangle(0, 0, square_side, square_side, fill=color)
        if label is not None:
            cv.create_text(square_side/2, square_side/2, text=label, fill='white')
        cv.grid(column=x, row=y, padx=0, pady=0)
        return cv

    def setup_entities(self, obstacle_states, crate_states, pitchfork_states, goal_states):
        self.ui_mapper = {}

        config = {}
        for goal_id in goal_states:
            config[goal_id] = (goal_states[goal_id], 'green', 0, '')
        for obstacle_id in obstacle_states:
            config[obstacle_id] = (obstacle_states[obstacle_id], 'black', 0, '')
        for crate_id in crate_states:
            config[crate_id] = (crate_states[crate_id], 'blue', 30, crate_id)
        for pitchfork_id in pitchfork_states:
            config[pitchfork_id] = (pitchfork_states[pitchfork_id], 'red', 30, pitchfork_id)

        for entity_id in config:
            entity_pos = config[entity_id][0]
            self.ui_mapper[entity_id] = self.addEntity(entity_pos[1], entity_pos[0], config[entity_id][1], config[entity_id][2], config[entity_id][3])


    def move_entity(self, entity, x, y, padx=0, pady=0):
        entity.grid(column=x, row=y, padx=padx, pady=pady)

    def run(self, actions, timestep=500, delay=0):
        for idx, action in enumerate(actions):
            self.root.after(timestep * (idx + 1) + delay, self.move_entity, self.ui_mapper[action[0]], action[2][1], action[2][0])
        self.root.mainloop()
