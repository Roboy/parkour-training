import os
import pybullet
from pybulletgym.envs.roboschool.robots.locomotors.walker_base import WalkerBase
from pybulletgym.envs.roboschool.robots.robot_bases import URDFBasedRobot
from gym_parkour.envs.parkour_robot import ParkourRobot
import numpy as np


class Biped(ParkourRobot, URDFBasedRobot):
    self_collision = False
    foot_list = ["foot_right", "foot_left"]  # "left_hand", "right_hand"

    def __init__(self, random_yaw=False, random_lean=False, **kwargs):
        ParkourRobot.__init__(self, **kwargs)
        URDFBasedRobot.__init__(self, 'biped.urdf', 'base', action_dim=17, obs_dim=44)
        self.random_yaw = random_yaw
        self.random_lean = random_lean

    # overwrite ParkourRobot
    def apply_action(self, a):
        assert (np.isfinite(a).all())
        force_gain = 1
        for i, m, power in zip(range(17), self.motors, self.motor_power):
            m.set_motor_torque(float(force_gain * power * self.power * np.clip(a[i], -1, +1)))

    def alive_bonus(self, z, pitch):
        return +2 if z > 0.78 else -1  # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying

    # overwrite ParkourRobot
    def calc_state(self, target_position_xy):
        j = np.array([j.current_relative_position() for j in self.ordered_joints], dtype=np.float32).flatten()
        # even elements [0::2] position, scaled to -1..+1 between limits
        # odd elements  [1::2] angular speed, scaled to show -1..+1
        self.joint_speeds = j[1::2]
        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

        body_pose = self.robot_body.pose()
        parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
        self.body_xyz = (
            parts_xyz[0::3].mean(), parts_xyz[1::3].mean(),
            body_pose.xyz()[2])  # torso z is more informative than mean z
        self.body_rpy = body_pose.rpy()
        z = self.body_xyz[2]
        if self.initial_z is None:
            self.initial_z = z
        r, p, yaw = self.body_rpy
        self.walk_target_theta = np.arctan2(target_position_xy[1] - self.body_xyz[1],
                                            target_position_xy[0] - self.body_xyz[0])
        angle_to_target = self.walk_target_theta - yaw

        rot_speed = np.array(
            [[np.cos(-yaw), -np.sin(-yaw), 0],
             [np.sin(-yaw), np.cos(-yaw), 0],
             [0, 0, 1]]
        )
        vx, vy, vz = np.dot(rot_speed, self.robot_body.speed())  # rotate speed back to body point of view

        more = np.array([z - self.initial_z,
                         np.sin(angle_to_target), np.cos(angle_to_target),
                         0.3 * vx, 0.3 * vy, 0.3 * vz,
                         # 0.3 is just scaling typical speed into -1..+1, no physical sense here
                         r, p], dtype=np.float32)
        return np.clip(np.concatenate([more] + [j] + [self.feet_contact]), -5, +5)

    def calc_reward(self, action, ground_ids):
        # 2 here because 17 joints produce a lot of electricity cost just from policy noise
        # living must be better than dying
        alive = +2 if self.body_xyz[2] > 0.78 else -1

        feet_collision_cost = 0.0
        for i, f in enumerate(
                self.feet):  # TODO: Maybe calculating feet contacts could be done within the robot code
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            # print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if ground_ids & contact_ids:
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                # feet_collision_cost += self.foot_collision_cost
                self.feet_contact[i] = 1.0
            else:
                self.feet_contact[i] = 0.0

        electricity_cost = self.electricity_cost * float(np.abs(
            action * self.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(action).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.joints_at_limit)

        rewards = [
            alive,
            electricity_cost,
            joints_at_limit_cost,
            feet_collision_cost
        ]
        return sum(rewards)

    def is_alive(self):
        body_pitch = self.body_rpy[1]   # not a good predictor
        body_height = self.body_xyz[2]
        if body_height < 0.5:
            return False
        else:
            return True

    def reset(self, bullet_client):
        self._p = bullet_client
        self.ordered_joints = []

        full_path = os.path.join(os.path.dirname(__file__), "assets", self.model_urdf)
        print(full_path)

        if self.self_collision:
            self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self._p,
                self._p.loadURDF(full_path,
                basePosition=self.basePosition,
                baseOrientation=self.baseOrientation,
                useFixedBase=self.fixed_base,
                flags=pybullet.URDF_USE_SELF_COLLISION))
        else:
            self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self._p,
                self._p.loadURDF(full_path,
                basePosition=self.basePosition,
                baseOrientation=self.baseOrientation,
                useFixedBase=self.fixed_base))

        for j in self.ordered_joints:
            j.reset_current_position(0,0)
        self.feet = [self.parts[f] for f in self.foot_list]
        self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)
        try:
            self.scene.actor_introduce(self)
        except AttributeError:
            pass
        self.initial_z = None

        