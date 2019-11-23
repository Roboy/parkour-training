import numpy as np
import gym
from gym import spaces
from gym_parkour.envs.bullet_environment import BulletEnvironment


class ParkourGym(gym.Env):
    def __init__(self, *args, **kwargs):
        self.bullet = BulletEnvironment()
        action_low = np.array([-1, -1, -1])
        observation_low = np.array([-2, -2, -2])
        self.action_space = spaces.Box(action_low, -action_low)
        self.observation_space = spaces.Box(observation_low, -observation_low)

    def step(self, action):
        obs = 0
        pass

    def reset(self):
        return [0, 0, 0]
