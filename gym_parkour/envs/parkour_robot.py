import os
import pybullet
from pybulletgym.envs.roboschool.robots.robot_bases import XmlBasedRobot
import numpy as np


class ParkourRobot(XmlBasedRobot):
    def __init__(self):
        power = 0.41
        self.power = power
        self.camera_x = 0
        self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
        self.walk_target_x = 1e3  # kilometer away
        self.walk_target_y = 0
        self.body_xyz = [0, 0, 0]

    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client
        # self.flag = ObjectHelper.get_sphere(self._p, self.walk_target_x, self.walk_target_y, 0.7) # can cause reset error
        for j in self.ordered_joints:
            j.reset_current_position(self.np_random.uniform(low=-0.1, high=0.1), 0)

        self.feet = [self.parts[f] for f in self.foot_list]
        self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)
        try:
            self.scene.actor_introduce(self)
        except AttributeError:
            pass
        self.initial_z = None

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        for n, j in enumerate(self.ordered_joints):
            j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))

    def calc_state(self):
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
        self.walk_target_theta = np.arctan2(self.walk_target_y - self.body_xyz[1],
                                            self.walk_target_x - self.body_xyz[0])
        self.walk_target_dist = np.linalg.norm(
            [self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0]])
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

        s = self.calc_state()  # optimization: calc_state() can calculate something in self.* for calc_potential() to use

        return s
