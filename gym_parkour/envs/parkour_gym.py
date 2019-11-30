from pybulletgym.envs.roboschool.envs.locomotion.walker_base_env import WalkerBaseBulletEnv
from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.roboschool.scenes import StadiumScene
from gym_parkour.envs.humanoid import Humanoid
from gym_parkour.envs.track_scene import TrackScene
import pybullet
import numpy as np


class ParkourGym(BaseBulletEnv):
    def __init__(self, render=False):
        self.robot = Humanoid()
        # self.electricity_cost = 4.25 * WalkerBaseBulletEnv.electricity_cost
        # self.stall_torque_cost = 4.25 * WalkerBaseBulletEnv.stall_torque_cost
        # print("WalkerBase::__init__")
        BaseBulletEnv.__init__(self, self.robot, render)
        self.camera_x = 0
        self.walk_target_x = 5  # kilometer away
        self.walk_target_y = 0
        self.stateId = -1

    def create_single_player_scene(self, bullet_client):
        self.scene = TrackScene(bullet_client, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        self.scene.zero_at_running_strip_start_line = False
        return self.scene

    def reset(self):
        print('RESETTING')
        if self.stateId >= 0:
            # print("restoreState self.stateId:",self.stateId)
            self._p.restoreState(self.stateId)

        r = BaseBulletEnv._reset(self)
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(self._p,
                                                                                             self.scene.ground_plane_mjcf)
        self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex], self.parts[f].bodyPartIndex) for f in
                               self.foot_ground_object_names])
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
        if self.stateId < 0:
            self.stateId = self._p.saveState()
        # print("saving state self.stateId:",self.stateId)
        self.steps = 0
        return r

    def move_robot(self, init_x, init_y, init_z):
        "Used by multiplayer stadium to move sideways, to another running lane."
        self.cpp_robot.query_position()
        pose = self.cpp_robot.root_part.pose()
        pose.move_xyz(init_x, init_y,
                      init_z)  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
        self.cpp_robot.set_pose(pose)

    electricity_cost = -2.0 * 4.25  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
    stall_torque_cost = -0.1 * 4.25  # cost for running electric current through a motor even at zero rotational speed, small
    foot_collision_cost = -1.0  # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
    foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
    joints_at_limit_cost = -0.1  # discourage stuck joints

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit

        alive = float(self.robot.alive_bonus(state[0] + self.robot.initial_z, self.robot.body_rpy[
            1]))  # state[0] is body height above ground, body_rpy[1] is pitch
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True
        distance_to_target = np.linalg.norm([self.robot.body_xyz[0] - self.walk_target_x, self.robot.body_xyz[1] - self.walk_target_y])
        if distance_to_target < 1:
            print('target reached!!!')
            done = True
        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i, f in enumerate(
                self.robot.feet):  # TODO: Maybe calculating feet contacts could be done within the robot code
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            # print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if self.ground_ids & contact_ids:
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                # feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0

        electricity_cost = self.electricity_cost * float(np.abs(
            a * self.robot.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)

        rewards = [
            alive,
            progress,
            electricity_cost,
            joints_at_limit_cost,
            feet_collision_cost
        ]
        # self.HUD(state, a, done)
        # self.reward += sum(self.rewards)

        return state, sum(rewards), bool(done), {}

    def camera_adjust(self):
        x, y, z = self.body_xyz
        self.camera_x = 0.98 * self.camera_x + (1 - 0.98) * x
        self.camera.move_and_look_at(self.camera_x, y - 2.0, 1.4, x, y, 1.0)
