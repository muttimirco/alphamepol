import gym
import numpy as np
import scipy.stats as stats
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from envs.multi_grid_configs import walls_structures

class BoundingBox:
    """
    2d bounding box.
    """
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def check_if_inside(self, x, y):
        """
        Checks whether point (x,y) is inside the bounding box.
        """
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def rect(self, tx, ty, scale):
        """
        Returns a ready to render rect given translation and scale factors: (top left coordinates, width, height).
        """
        return (self.xmin*scale + tx, self.ymin*scale + ty, (self.xmax - self.xmin)*scale, (self.ymax - self.ymin)*scale)

def generate_gridworld(dim, walls_structure):
    # Compute the length of each square that will be part of a wall
    side_length = (dim*2) / len(walls_structure)
    # Resulting walls
    walls = []
    # Small distance to avoid blank lines in the pygame render
    delta = 1e-6
    # Build blocks
    for i in range(len(walls_structure)):
        for j in range(len(walls_structure)):
            # check if (i, j) is a wall
            if walls_structure[i][j] == 1:
                # map (i, j) to the real dimensions
                real_x = -dim + i * side_length
                real_y = -dim + j * side_length
                wall = BoundingBox(real_x-delta, real_x + side_length, real_y-delta, real_y + side_length)
                walls.append(wall)
    return walls

class MultiGrid(gym.Env):

    def __init__(self, dim=1, max_delta=0.2, wall_width=0.4, configuration=0):

        self.num_features = 2

        # The gridworld bottom left corner is at (-self.dim, -self.dim)
        # and the top right corner is at (self.dim, self.dim)
        self.dim = dim

        # The maximum change in position obtained through an action
        self.max_delta = max_delta

        # Maximum (dx,dy) action
        self.max_action = np.array([self.max_delta, self.max_delta], dtype=np.float32)
        self.action_space = gym.spaces.Box(-self.max_action, self.max_action, dtype=np.float32)

        # Maximum (x,y) position
        self.max_position = np.array([self.dim, self.dim], dtype=np.float32)
        self.observation_space = gym.spaces.Box(-self.max_position, self.max_position, dtype=np.float32)

        # Current state
        self.state = None

        # Initial state
        init_top_left = np.array([-self.dim, -self.dim], dtype=np.float32)
        init_bottom_right = np.array([-self.dim+0.5, -self.dim+0.5], dtype=np.float32)
        self.init_states = gym.spaces.Box(init_top_left, init_bottom_right, dtype=np.float32)

        self.wall_width = wall_width

        self.configuration = configuration

        # Generate the walls and hence the gridworld given a structure represented by a matrix of 1 and 0 (see multi_grid_configs.py)
        self.walls = generate_gridworld(self.dim, walls_structures[self.configuration])

        # Render stuff
        self.game_display = None
        self.DISPLAY_WIDTH = 800
        self.DISPLAY_HEIGHT = 600
        self.SCALE = 200
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.AGENT_RADIUS = 3
    
    def seed(self, seed):
        self.init_states.seed(seed)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

    def reset(self):
        # Start near the bottom left corner of the bottom left room
        self.state = self.init_states.sample()

        # Reset pygame
        self.game_display = None

        return self.state

    def render(self):
        if self.game_display is None:
            pygame.init()
            self.game_display = pygame.display.set_mode((self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
            pygame.display.set_caption('MultiGrid')

        # Draw background
        self.game_display.fill(self.WHITE)

        # Draw walls
        for bbox in self.walls:
            pygame.draw.rect(self.game_display, self.BLACK, bbox.rect(int(self.DISPLAY_WIDTH/2), int(self.DISPLAY_HEIGHT/2), self.SCALE))

        xmin = -self.dim*self.SCALE + self.DISPLAY_WIDTH/2
        xmax = self.dim*self.SCALE + self.DISPLAY_WIDTH/2
        ymin = -self.dim*self.SCALE + self.DISPLAY_HEIGHT/2
        ymax = self.dim*self.SCALE + self.DISPLAY_HEIGHT/2

        pygame.draw.line(self.game_display, self.BLACK, (xmin, ymin), (xmin, ymax))
        pygame.draw.line(self.game_display, self.BLACK, (xmin, ymax), (xmax, ymax))
        pygame.draw.line(self.game_display, self.BLACK, (xmax, ymin), (xmax, ymax))
        pygame.draw.line(self.game_display, self.BLACK, (xmin, ymin), (xmax, ymin))

        # Draw agent
        # Take agent (x,y), change y sign, scale and translate
        agent_x, agent_y = self.state * np.array([1, -1], dtype=np.int32) * self.SCALE + np.array([self.DISPLAY_WIDTH/2, self.DISPLAY_HEIGHT/2], dtype=np.float32)
        pygame.draw.circle(self.game_display, self.BLUE, (int(agent_x), int(agent_y)), self.AGENT_RADIUS)

        # Update screen
        pygame.display.update()

    def step(self, action):
        assert action.shape == self.action_space.shape

        x, y = self.state

        dx = action[0]
        dx = np.clip(dx, -self.max_delta, self.max_delta)

        dy = action[1]
        dy = np.clip(dy, -self.max_delta, self.max_delta)
        
        # Multiple configurations scenario
        if self.configuration == 0:
            lower, upper = 0, self.max_delta / 1.3
            mu, sigma = self.max_delta / 2.6, self.max_delta / 20
        else:
            lower, upper = 0, self.max_delta / 1.6
            mu, sigma = self.max_delta / 3.2, self.max_delta / 20
        normal_dist = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        slope = normal_dist.rvs(1)[0]
        if self.configuration == 0:
            # Adversarial configuration: north-facing slope over the upper-half environment
            if -self.dim <= x <= 0:
                dx -= np.abs(slope)
        elif (self.configuration == 2) or (self.configuration == 6):
            # south-facing slope over the whole environment
            dx += np.abs(slope)
        elif (self.configuration == 1) or (self.configuration == 5) or (self.configuration == 9):
            # east-facing slope over the whole environment
            dy += np.abs(slope)
        elif (self.configuration == 3):
            # south-east-facing slope over the whole environment
            dx += np.abs(slope)
            dy += np.abs(slope)
        elif (self.configuration == 4) or (self.configuration == 7) or (self.configuration == 8):
            # no slope
            pass
        
        new_x = x + dx
        new_y = y + dy
        
        if np.abs(new_x) >= self.dim or np.abs(new_y) >= self.dim:
            new_x, new_y = self.recompute_position_boundaries(x, y, dx, dy, new_x, new_y)
            #new_x, new_y = x, y

        # Check hit with a wall
        for bbox in self.walls:
            if bbox.check_if_inside(new_x, new_y):
                new_x, new_y = self.recompute_position(x, y, dx, dy)

        self.state = np.array([new_x, new_y], dtype=np.float32)
        done = False
        reward = 0

        return self.state, reward, done, {}

    def get_sample_points(self, size):
        xs = np.linspace(-self.dim, self.dim, size)
        ys = np.linspace(-self.dim, self.dim, size)
        return np.array([[x, y] for x in xs for y in ys])

    # This function is called when the agent hits a wall or a door.
    # Return a new position computed so that to avoid the aliasing problem
    def recompute_position(self, x, y, dx, dy):
        if dx < 0 and dy < 0:
            new_x = x + np.random.uniform(0, self.max_delta/2)
            new_y = y + np.random.uniform(0, self.max_delta/2)
        elif dx > 0 and dy > 0:
            new_x = x - np.random.uniform(0, self.max_delta/2)
            new_y = y - np.random.uniform(0, self.max_delta/2)
        elif dx > 0 and dy < 0:
            new_x = x - np.random.uniform(0, self.max_delta/2)
            new_y = y + np.random.uniform(0, self.max_delta/2)
        elif dx < 0 and dy > 0:
            new_x = x + np.random.uniform(0, self.max_delta/2)
            new_y = y - np.random.uniform(0, self.max_delta/2)
        
        # Additional check because it can happen that, in the angles created by the central walls,
        # moving the agent in the opposite direction will move it again inside the walls.
        for bbox in self.walls:
            if bbox.check_if_inside(new_x, new_y):
                new_x, new_y = x, y

        return new_x, new_y

    # This function is called when the agent hits the boundaries of the environment.
    # Return a new position computed so that to avoid the aliasing problem
    def recompute_position_boundaries(self, x, y, dx, dy, new_x, new_y, counter=0):
        # Special case: if the agent is very near to one angle, we have to understand if the intersection point is
        # with the horizontal or vertical wall.
        upper_wall = False
        if np.abs(new_x) >= self.dim and np.abs(new_y) >= self.dim:
            # Get the angle coordinates
            x_angle = np.sign(new_x)
            y_angle = np.sign(new_y)
            # Compute the angle between the line connecting the agent to the angle and dx
            alpha_ref = np.degrees(np.arccos((x_angle-x) / (np.sqrt((x_angle-x)**2 + (y_angle-y)**2))))
            # Compute the angle between the line corresponding to the taken direction and dx
            alpha = np.degrees(np.arccos((dx) / (np.sqrt((dx)**2 + (dy)**2))))
            if alpha >= alpha_ref:
                upper_wall = True
        
        if np.abs(new_x) >= self.dim and not upper_wall:
            # Compute the intersection
            x_inters = -self.dim if x < 0 else self.dim
            # If dx is very small (--> 0), y_inters can be a very large value
            if np.isnan(dy/dx) or np.isinf(dy/dx):
                y_inters = np.clip(y, -self.dim, self.dim)
            else:
                y_inters = (dy/dx)*(x_inters - x) + y
            # Still because of very small dx
            if (np.abs(y_inters) > self.dim):
                y_inters = np.clip(y, -self.dim, self.dim)
            # Compute the new increment
            if dy >= 0:
                if new_x < 0:
                    new_dx = y + dy - y_inters
                    new_dy = x_inters - (x + dx)
                else:
                    new_dx = y_inters - (y + dy)
                    new_dy = x + dx - x_inters
            else:
                if new_x < 0:
                    new_dx = y_inters - (y + dy)
                    new_dy = x + dx - x_inters
                else:
                    new_dx = y + dy - y_inters
                    new_dy = x_inters - (x + dx)
        elif np.abs(new_y) >= self.dim:
            # Compute the intersection
            y_inters = -self.dim if y < 0 else self.dim
            # If dy is very small (--> 0), x_inters can be a very large value
            if np.isnan(dx/dy) or np.isinf(dx/dy):
                x_inters = np.clip(x, -self.dim, self.dim)
            else:
                x_inters = (dx/dy)*(y_inters - y) + x
            # Still because of very small dy
            if (np.abs(x_inters) > self.dim):
                x_inters = np.clip(x, -self.dim, self.dim)
            # Compute the new increment
            if new_y < 0:
                if dx > 0:
                    new_dx = y_inters - (y + dy)
                    new_dy = x + dx - x_inters
                else:
                    new_dx = y + dy - y_inters
                    new_dy = x_inters - (x + dx)
            else:
                if dx >= 0:
                    new_dx = y + dy - y_inters
                    new_dy = x_inters - (x + dx)
                else:
                    new_dx = y_inters - (y + dy)
                    new_dy = x + dx - x_inters

        # Compute the new point
        new_x = x_inters + new_dx
        new_y = y_inters + new_dy
        # When x,y is near the borders it can happen that after the update the agent goes a bit off again.
        # Check recursively:
        if counter > 3: # Avoid infinite recursion
            return x, y
        if np.abs(new_x) >= self.dim or np.abs(new_y) >= self.dim:
            return self.recompute_position_boundaries(x_inters, y_inters, new_dx, new_dy, new_x, new_y, counter=counter+1)

        return new_x, new_y

if __name__ == '__main__':
    env = MultiGrid()
    s = env.reset()
    env.render()
    for _ in range(5000):
        env.render()
        s, r, done, _ = env.step(env.action_space.sample())