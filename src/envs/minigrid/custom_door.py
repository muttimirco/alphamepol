from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class CustomDoorEnv(MiniGridEnv):
    """
    Environment with a door, sparse reward
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
        self.grid.vert_wall(splitIdx, 0)

        # Place additional walls
        self.put_obj(Wall(), 7, 3)
        self.put_obj(Wall(), 8, 3)

        # Place the agent
        self.agent_pos = np.array([1, 8]) # last bottom square
        self.agent_dir = 0

        # Place a door in the wall
        doorIdx = 1
        self.put_obj(Door('yellow', is_locked=False), splitIdx, doorIdx)

        # Place a yellow key
        #self.put_obj(Key('yellow'), splitIdx-1, 1)

        self.mission = "use the key to open the door and then get to the goal"
    
    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction of forward movement.
        """
        assert self.agent_dir >= 0 and self.agent_dir < 4
        if (self.config == 1) and (self.agent_dir == 1): # pointing down
            return DIR_TO_VEC[3] # up
        elif (self.config == 1) and (self.agent_dir == 3): # pointing up
            return DIR_TO_VEC[1] # down
        elif (self.config == 1) and (self.agent_dir == 0): # pointing right
            return DIR_TO_VEC[2] # left
        elif (self.config == 1) and (self.agent_dir == 2): # pointing left
            return DIR_TO_VEC[0] # right
        else:
            return DIR_TO_VEC[self.agent_dir]

class CustomDoorEnv10x10(CustomDoorEnv):
    def __init__(self, goal_position=None, config=1):
        self.goal_position = goal_position
        self.config = config
        self.num_features = 64
        super().__init__(size=10)

register(id='MiniGrid-CustomDoor-10x10-v0', entry_point='envs.minigrid.custom_door:CustomDoorEnv10x10')