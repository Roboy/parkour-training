import os
import pybullet
import math
from pybulletgym.envs.roboschool.robots.locomotors.walker_base import WalkerBase
import copy
from pybulletgym.envs.roboschool.robots.robot_bases import URDFBasedRobot
from gym_parkour.envs.parkour_robot import ParkourRobot
import numpy as np


class Laikago(ParkourRobot, URDFBasedRobot):
    self_collision = True
    foot_list = ["toeRL", "toeRR", 'toeFL', 'toeFR']  # "left_hand", "right_hand"
    electricity_cost = -2.0
    stall_torque_cost = -0.1 * 4.25  # cost for running electric current through a motor even at zero rotational speed, small
    joints_at_limit_cost = -0.1  # discourage stuck joints

    def __init__(self, random_yaw=False, random_lean=False, **kwargs):
        # quadruped = p.loadURDF("laikago/laikago_toes.urdf", [0, 0, .5], [0, 0.5, 0.5, 0],
        #                        flags=urdfFlags,
        #                        useFixedBase=False)
        ParkourRobot.__init__(self, **kwargs)
        URDFBasedRobot.__init__(self, 'laikago/laikago_toes.urdf', 'base', action_dim=16, obs_dim=60)
        self.last_action = (0,) * 16
        self.action_difference = 0  # l2 norm between successive actions -> important for position control
        self.positions = (0,) * 16
        self.torques = (0,) * 16
        self.velocities = (0, ) * 16

    # overwrite ParkourRobot
    def apply_action(self, a):
        a = np.array(a)
        assert (np.isfinite(a).all())
        self._p.setJointMotorControlArray(self.robot_body.bodies[self.robot_body.bodyIndex],
                                          jointIndices=range(16),
                                          controlMode=self._p.POSITION_CONTROL,
                                          targetPositions=a,
                                          forces=(5,) * 16)
        self.action_difference = np.linalg.norm(a - self.last_action)
        self.last_action = a
        self.read_joint_states()
        # force_gain = 1
        # for i, m, motor_range in zip(range(12), self.motors, self.motor_ranges):
        #     m.set_motor_torque(float(force_gain * power * self.power * np.clip(a[i], -1, +1)))
        # m.set_position(float(motor_range * np.clip(a[i], -1, +1)))

    # overwrite ParkourRobot
    def calc_state(self, target_position_xy, ground_ids):
        # j = np.array([j.current_relative_position() for j in self.ordered_joints], dtype=np.float32).flatten()
        # print(j.shape)
        # print('j relative: ' + str(j))
        # print('positions: ' + str(self.positions))
        j = self.positions + self.velocities
        # print(len(j))
        # even elements [0::2] position, scaled to -1..+1 between limits
        # odd elements  [1::2] angular speed, scaled to show -1..+1
        self.joint_speeds = j[1::2]
        self.joints_at_limit = np.count_nonzero(np.abs(self.positions) > 0.99)

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
        for i, f in enumerate(self.feet):
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            if ground_ids & contact_ids:
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                self.feet_contact[i] = 1.0
            else:
                self.feet_contact[i] = 0.0
        # print(self.feet_contact)
        return np.clip(np.concatenate([more] + [j] + [self.feet_contact] + [np.array(self.last_action)]), -5, +5)

    def read_joint_states(self):
        joint_states = self._p.getJointStates(
            bodyUniqueId=self.robot_body.bodies[self.robot_body.bodyIndex],
            jointIndices=range(16))
        self.positions = [j_state[0] for j_state in joint_states]
        self.velocities = [j_state[1] for j_state in joint_states]
        self.torques = [j_state[3] for j_state in joint_states]

    def calc_reward(self, action, ground_ids):
        # living must be better than dying
        # alive = +1.5 if self.body_xyz[2] > 0.3 and self.body_rpy[0] > 1 else -10
        alive = 0
        # electricity_cost = self.electricity_cost * float(np.abs(action).mean())
        # action * self.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        # electricity_cost += self.stall_torque_cost * float(np.square(self.joint_speeds).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.joints_at_limit)
        action_difference_cost = -.1 * self.action_difference
        torque_cost = -0.01 * np.sum(np.abs(np.array(self.torques)))

        rewards = [
            alive,
            # electricity_cost,
            joints_at_limit_cost,
            action_difference_cost,
            torque_cost,
            # feet_collision_cost
        ]
        # print('alive: ' + str(alive))
        # print('electricity_cost: ' + str(electricity_cost))
        # print('joints at limit cost: '+ str(joints_at_limit_cost))
        # print('action difference cost: ' + str(action_difference_cost))
        # print('torque cost: ' + str(torque_cost))
        info = dict(
            alive_bonus=alive,
            # electricity_cost=electricity_cost,
            joints_at_limit_cost=joints_at_limit_cost,
            torque_cost=torque_cost,
            action_difference_cost=action_difference_cost
        )
        return sum(rewards), info

    def is_alive(self):
        body_height = self.body_xyz[2]
        roll, pitch, yaw = self.body_rpy
        if body_height < 0.3 or roll < 1.0:  # prevent flipping and falling down
            return False
        else:
            return True

    def reset(self, bullet_client):
        self._p = bullet_client
        self.ordered_joints = []

        full_path = os.path.join(os.path.dirname(__file__), "assets", self.model_urdf)
        print(full_path)
        self.basePosition = (0, 0, 0.5)
        self.baseOrientation = self._p.getQuaternionFromEuler((math.pi / 2, 0, math.pi / 2))
        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self._p,
                                                                                       self._p.loadURDF(full_path,
                                                                                                        basePosition=self.basePosition,
                                                                                                        baseOrientation=self.baseOrientation,
                                                                                                        useFixedBase=self.fixed_base,
                                                                                                        flags=pybullet.URDF_USE_SELF_COLLISION))
        ignored_joints = ['jtoeFL', 'jtoeFR', 'jtoeRR', 'jtoeRL']
        for ignored_joint in ignored_joints:
            self.jdict.pop(ignored_joint)

        for j in self.ordered_joints:
            if j.joint_name in ignored_joints:
                self.ordered_joints.remove(j)

        for j in self.ordered_joints:
            j.reset_current_position(0, 0)
        self.feet = [self.parts[f] for f in self.foot_list]
        self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)
        try:
            self.scene.actor_introduce(self)
        except AttributeError:
            pass
        self.initial_z = None
        self.motor_names = ["FR_hip_motor_2_chassis_joint", "FL_hip_motor_2_chassis_joint",
                            "RL_hip_motor_2_chassis_joint", "RR_hip_motor_2_chassis_joint"]
        self.motor_ranges = [0.3, 0.3, 0.3, 0.3]
        self.motor_names += ["FR_upper_leg_2_hip_motor_joint", "FL_upper_leg_2_hip_motor_joint",
                             "RL_upper_leg_2_hip_motor_joint", "RR_upper_leg_2_hip_motor_joint"]
        self.motor_ranges += [2, 2, 2, 2]
        self.motor_names += ["FR_lower_leg_2_upper_leg_joint", "FL_lower_leg_2_upper_leg_joint",
                             "RL_lower_leg_2_upper_leg_joint", "RR_lower_leg_2_upper_leg_joint"]
        self.motor_ranges += [2, 2, 2, 2]
        # self.motor_names += ["jtoeRR", "jtoeRL",
        #                      "jtoeFL", "jtoeFR"]
        # self.motor_power += [10, 10, 10, 10]
        self.motors = [self.jdict[n] for n in self.motor_names]

    def get_camera_pos(self):
        pass

    def get_pos_xyz(self):
        parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
        body_pose = self.robot_body.pose()
        return (
            parts_xyz[0::3].mean(), parts_xyz[1::3].mean(),
            body_pose.xyz()[2])  # torso z is more informative than mean z
