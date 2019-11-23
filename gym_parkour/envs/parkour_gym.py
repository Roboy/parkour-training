import numpy as np
import gym
from gym import spaces
from gym_parkour.envs.bullet_environment import BulletEnvironment
from gym_parkour.envs.robot_bases import MJCFBasedRobot
import pybullet as p

class ParkourGym(gym.Env):
    def __init__(self, *args, **kwargs):
        physics_client = p.connect(p.GUI)
        robot_model_path = '/home/alex/parkour-training/gym_parkour/envs/assets/humanoid.xml'
        p.setGravity(0, 0, -10)
        self.robot = MJCFBasedRobot(robot_model_path, robot_name='', action_dim=21, obs_dim=21)
        self.robot.reset(p)
        action_low = np.array([-1, -1, -1])
        observation_low = np.array([-2, -2, -2])
        self.action_space = spaces.Box(action_low, -action_low)
        self.observation_space = spaces.Box(observation_low, -observation_low)

    def step(self, action):
        obs = 0
        pass

    def reset(self):
        return [0, 0, 0]
