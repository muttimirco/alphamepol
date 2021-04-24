import gym

class CustomRewardEnv(gym.Wrapper):
    """
    Environment wrapper to provide a custom reward function
    and episode termination.
    """
    def __init__(self, env, get_reward):
        super().__init__(env)
        self.get_reward = get_reward

    def step(self, a, goal_position):
        s, r, d, i = super().step(a)
        r, d = self.get_reward(s[:2], r, d, i, goal_position)
        return s, r, d, i