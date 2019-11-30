import pybullet
import numpy as np
from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from gym_parkour.envs.humanoid import Humanoid
from gym_parkour.envs.track_scene import TrackScene


class ParkourGym(BaseBulletEnv):
    foot_ground_object_names = {"floor"}  # to distinguish ground and other objects

    def __init__(self, render=False):
        self.target_position_xy = (15, 0)
        self.robot = Humanoid(target_position_xy=self.target_position_xy)
        BaseBulletEnv.__init__(self, self.robot, render)
        self.saved_state_id = None

    # Overwrite BaseBulletEnv
    def create_single_player_scene(self, bullet_client):
        self.scene = TrackScene(bullet_client, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        self.scene.zero_at_running_strip_start_line = False
        return self.scene

    def reset(self):
        if self.saved_state_id is not None:
            # restore state of pybullet with saved state from first reset
            self._p.restoreState(self.saved_state_id)
        else:
            r = BaseBulletEnv._reset(self)
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

            self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(self._p,
                                                                                                 self.scene.ground_plane_mjcf)
            self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex], self.parts[f].bodyPartIndex) for f in
                                   self.foot_ground_object_names])
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
            self.saved_state_id = self._p.saveState()

        return self.robot.calc_state(self.target_position_xy)

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()

        distance_to_target = np.linalg.norm(np.array(self.robot.body_xyz[0:2]) - np.array(self.target_position_xy))
        done = False
        if distance_to_target < 1 or not self.robot.is_alive():
            done = True

        state = self.robot.calc_state(self.target_position_xy)
        reward = self.robot.calc_reward(a, self.ground_ids)
        return state, reward, bool(done), {}
