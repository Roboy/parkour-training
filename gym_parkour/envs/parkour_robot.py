import numpy as np
from pybulletgym.envs import gym_utils as ObjectHelper
from pybulletgym.envs.roboschool.robots.locomotors.walker_base import WalkerBase


class ParkourRobot(WalkerBase):
    def __init__(self):
        WalkerBase.__init__(self, power=0.41)
        self.flag = None
        self.target_index = 0
        self.targets = [(0, 0), (18, 0.0), (27, -2), (37, 3), (45, -3)]

    def robot_specific_reset(self, bullet_client):
        WalkerBase.robot_specific_reset(self,bullet_client)
        self.target_index = 0
        self.flag_reposition()

    def flag_reposition(self):
        if self.flag:
            #for b in self.flag.bodies:
            #	print("remove body uid",b)
            #	p.removeBody(b)
            x = 0
            # self._p.resetBasePositionAndOrientation(self.flag.bodies[0],[self.walk_target_x, self.walk_target_y, 0.7], [0,0,0,1])
        else:
            self.flag = ObjectHelper.get_sphere(self._p, self.walk_target_x, self.walk_target_y, 0.7)
        self.flag_timeout = 6000/self.scene.frame_skip  # match Roboschool

    def calc_state(self):
        self.flag_timeout -= 1
        state = WalkerBase.calc_state(self)
        if self.walk_target_dist < 1 or self.flag_timeout <= 0:
            self.flag_reposition()
            state = WalkerBase.calc_state(self)  # calculate state again, against new flag pos
            self.potential = self.calc_potential()	   # avoid reward jump
        return state
