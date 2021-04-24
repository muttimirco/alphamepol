import sys, os
sys.path.append(os.getcwd() + '/gym-minigrid')

from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class CustomSimpleCrossingEnv(MiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, size=8):
        super().__init__(
            grid_size=size,
            max_steps=10*size*size
        )

    def _gen_grid(self, width, height):

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        if self.goal_position is not None:
            # Place a goal
            self.put_obj(Goal(), self.goal_position[0], self.goal_position[1])

        # Create a vertical splitting wall
        splitIdx = 2

        self.put_obj(Wall(), 11, 1)
        self.put_obj(Wall(), 12, 1)
        self.put_obj(Wall(), 13, 1)
        self.put_obj(Wall(), 11, 2)
        self.put_obj(Wall(), 11, 3)
        self.put_obj(Wall(), 4, 4)
        self.put_obj(Wall(), 5, 4)
        self.put_obj(Wall(), 6, 4)
        self.put_obj(Wall(), 4, 6)
        self.put_obj(Wall(), 5, 6)
        self.put_obj(Wall(), 5, 7)
        self.put_obj(Wall(), 5, 8)
        self.put_obj(Wall(), 6, 7)
        self.put_obj(Wall(), 5, 13)
        self.put_obj(Wall(), 8, 13)
        self.put_obj(Wall(), 12, 11)
        self.put_obj(Wall(), 13, 11)
        self.put_obj(Wall(), 14, 11)
        self.put_obj(Wall(), 11, 5)
        self.put_obj(Wall(), 11, 6)
        self.put_obj(Wall(), 12, 6)
        self.put_obj(Wall(), 13, 6)
        self.put_obj(Wall(), 15, 6)
        self.put_obj(Wall(), 16, 6)

        # Place the agent
        self.agent_pos = np.array([1, 16]) # last bottom square
        self.agent_dir = 0

        self.mission = "get to the goal"

class CustomSimpleCrossingEnv18x18(CustomSimpleCrossingEnv):
    def __init__(self, goal_position=None, config=0):
        self.goal_position = goal_position
        self.config = config
        self.num_features = 64
        super().__init__(size=18)

register(id='MiniGrid-CustomSimpleCrossing-18x18-v0', entry_point='envs.minigrid.custom_simplecrossing:CustomSimpleCrossingEnv18x18')