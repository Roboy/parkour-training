import os
import pybullet
import numpy as np
from pybulletgym.envs.roboschool.robots.robot_bases import XmlBasedRobot
from abc import ABC, abstractmethod


class ParkourRobot(XmlBasedRobot, ABC):
    def __init__(self, target_position_xy):
        self.target_position_xy = target_position_xy
        power = 0.41
        self.power = power

    @abstractmethod
    def robot_specific_reset(self, bullet_client):
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
    def calc_reward(self, action, ground_ids ):
        pass

    @abstractmethod
    def is_alive(self):
        # return True if robot is still alive. Otherwise False
        # if False is returned the done flag will be set
        pass

    def calc_potential(self):
        # necessary method for BulletBaseEnv
        return 0

    def reset(self, bullet_client):
        full_path = os.path.join(os.path.dirname(__file__), "assets", self.model_xml)

        self._p = bullet_client
        # print("Created bullet_client with id=", self._p._client)
        if self.doneLoading == 0:
            self.ordered_joints = []
            self.doneLoading = 1
            if self.self_collision:
                self.objects = self._p.loadMJCF(full_path,
                                                flags=pybullet.URDF_USE_SELF_COLLISION | pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self._p, self.objects)
            else:
                self.objects = self._p.loadMJCF(full_path)
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self._p, self.objects)
        self.robot_specific_reset(self._p)
