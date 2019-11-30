import pybullet
import numpy as np
from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from gym_parkour.envs.humanoid import Humanoid
from gym_parkour.envs.track_scene import TrackScene


class ParkourGym(BaseBulletEnv):
    def __init__(self, render=False):
        self.target_position_xy = (15, 0)
        self.robot = Humanoid(target_position_xy=self.target_position_xy)
        BaseBulletEnv.__init__(self, self.robot, render)
        self.camera_x = 0
        self.saved_state_id = None

    # Overwrite BaseBulletEnv
    def create_single_player_scene(self, bullet_client):
        self.scene = TrackScene(bullet_client, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        self.scene.zero_at_running_strip_start_line = False
        return self.scene

    def reset(self):
        print('RESETTING')
        if self.saved_state_id is not None:
            # print("restoreState self.stateId:",self.stateId)
            self._p.restoreState(self.saved_state_id)

        r = BaseBulletEnv._reset(self)
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(self._p,
                                                                                             self.scene.ground_plane_mjcf)
        self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex], self.parts[f].bodyPartIndex) for f in
                               self.foot_ground_object_names])
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
        if self.saved_state_id is None:
            self.saved_state_id = self._p.saveState()
        print("saving state self.saved_state_id:", self.saved_state_id)
        return self.robot.calc_state(self.target_position_xy)

    electricity_cost = -2.0 * 4.25  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
    stall_torque_cost = -0.1 * 4.25  # cost for running electric current through a motor even at zero rotational speed, small
    foot_collision_cost = -1.0  # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
    foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
    joints_at_limit_cost = -0.1  # discourage stuck joints

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()

        distance_to_target = np.linalg.norm(np.array(self.robot.body_xyz[0:2]) - np.array(self.target_position_xy))
        done = False
        if distance_to_target < 1 or not self.robot.is_alive():
            done = True

        state = self.robot.calc_state(self.target_position_xy)  # also calculates self.joints_at_limit
        reward = self.robot.calc_reward(a, self.ground_ids)
        return state, reward, bool(done), {}

    def camera_adjust(self):
        # useless?
        pass
        # x, y, z = self.body_xyz
        # self.camera_x = 0.98 * self.camera_x + (1 - 0.98) * x
        # self.camera.move_and_look_at(self.camera_x, y - 2.0, 1.4, x, y, 1.0)
