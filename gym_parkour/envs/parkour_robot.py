import numpy as np
from pybulletgym.envs import gym_utils as ObjectHelper
from pybulletgym.envs.roboschool.robots.locomotors.walker_base import WalkerBase


class ParkourRobot(WalkerBase):
    def __init__(self):
        WalkerBase.__init__(self, power=0.41)
        self.walk_target_x = 15
        self.walk_target_y = 0
        self.flag = None
        # self.target_index = 0
        # self.targets = [(0, 0), (18, 0.0), (27, -2), (37, 3), (45, -3)]

    def robot_specific_reset(self, bullet_client):
        # self.flag = ObjectHelper.get_sphere(self._p, self.walk_target_x, self.walk_target_y, 0.7) # can cause reset error
        WalkerBase.robot_specific_reset(self, bullet_client)

    def calc_state(self):
        state = WalkerBase.calc_state(self)
        if self.walk_target_dist < 1:
            state = WalkerBase.calc_state(self)  # calculate state again, against new flag pos
            self.potential = self.calc_potential()  # avoid reward jump
        return state
