import os
import pybullet
import numpy as np
from pybulletgym.envs.roboschool.robots.robot_bases import XmlBasedRobot
from abc import ABC, abstractmethod


class ParkourRobot(ABC):
    def __init__(self, target_position_xy):
        self.target_position_xy = target_position_xy
        power = 0.41
        self.power = power

    @abstractmethod
    def reset(self, bullet_client):
        pass

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        for n, j in enumerate(self.ordered_joints):
            j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))

    @abstractmethod
    def calc_state(self):
        # return the robot state. This is what is returned by gym.step as state
        pass

    @abstractmethod
    def calc_reward(self, action, ground_ids)-> (float, dict):
        pass

    @abstractmethod
    def is_alive(self):
        # return True if robot is still alive. Otherwise False
        # if False is returned the done flag will be set
        pass

    @abstractmethod
    def get_camera_pos(self):
        pass

    @abstractmethod
    def get_pos_xyz(self):
        pass

    def calc_potential(self):
        # necessary method for BulletBaseEnv
        return 0
