import pybullet
import numpy as np
from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from gym_parkour.envs.humanoid import Humanoid
from gym_parkour.envs.track_scene import TrackScene
from gym.spaces.dict import Dict
import gym.spaces as spaces

class ParkourGym(BaseBulletEnv):
    foot_ground_object_names = {"starting_ground"}  # to distinguish ground and other objects

    def __init__(self, render=False):
        self.target_position_xy = (15, 0)
        self.robot = Humanoid(target_position_xy=self.target_position_xy)
        BaseBulletEnv.__init__(self, self.robot, render)
        self.saved_state_id = None
        self.action_space = Dict()
        self.observation_space = spaces.Dict({
            'sensors': self.robot.observation_space,
            'camera': spaces.Box(low=0, high=255, shape=(80, 80, 3)),
        })
        self.action_space = self.robot.action_space
        self.observation_space = self.robot.observation_space

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
            BaseBulletEnv._reset(self)
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

            self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(self._p,
                                                                                                 self.scene.ground_plane_mjcf)
            self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex], self.parts[f].bodyPartIndex) for f in
                                   self.foot_ground_object_names])
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
            self.saved_state_id = self._p.saveState()

        state = self.robot.calc_state(self.target_position_xy)
        self.last_position = self.robot.body_xyz
        return state

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()

        distance_to_target = np.linalg.norm(np.array(self.robot.body_xyz[0:2]) - np.array(self.target_position_xy))
        done = False
        if distance_to_target < 1 or not self.robot.is_alive():
            done = True

        state = self.robot.calc_state(self.target_position_xy)
        robot_specific_reward = self.robot.calc_reward(a, self.ground_ids)
        velocity = np.linalg.norm(np.array(self.target_position_xy) - np.array(self.last_position[:2])) \
                    - np.linalg.norm(np.array(self.target_position_xy) - np.array(self.robot.body_xyz[:2]))
        self.last_position = self.robot.body_xyz
        reward = robot_specific_reward + 3e2 * velocity
        print('robot reward: ' + str(robot_specific_reward) + ' velocity: ' + str(velocity) + str(' reward: ' + str(reward)))

        # base_pos = [0, 0, 0]
        # if hasattr(self, 'robot'):
        #     if hasattr(self.robot, 'body_xyz'):
        #         base_pos = self.robot.body_xyz
        # base_pos = self.robot.body_xyz
        # view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
        #     cameraTargetPosition=base_pos,
        #     distance=self._cam_dist,
        #     yaw=self._cam_yaw,
        #     pitch=self._cam_pitch,
        #     roll=0,
        #     upAxisIndex=2)
        # view_matrix = self._p.computeViewMatrix(
        #     cameraEyePosition=self.robot.body_xyz,
        #     cameraTargetPosition=(10, 0, 0),
        #     cameraUpVector=(1, 0, 1)
        # )
        # proj_matrix = self._p.computeProjectionMatrixFOV(
        #     fov=60, aspect=1.0, # float(self._render_width) / self._render_height,
        #     nearVal=0.1, farVal=100.0)
        # (_, _, px, _, _) = self._p.getCameraImage(
        #     width=80, height=80, viewMatrix=view_matrix,
        #     projectionMatrix=proj_matrix,
        #     renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
        # )
        # rgb_array = np.array(px)
        # rgb_array = rgb_array[:, :, :3]
        # # import matplotlib.pyplot as plt
        # # plt.imshow(rgb_array)
        # observation = {
        #     'sensors': state,
        #     'camera': rgb_array
        # }
        # follow robot with camera
        if self.isRender:
            robot_position = self.robot.body_xyz
            camInfo = self._p.getDebugVisualizerCamera()
            curTargetPos = camInfo[11]
            distance = camInfo[10]
            yaw = camInfo[8]
            pitch = camInfo[9]
            targetPos = [0.95 * curTargetPos[0] + 0.05 * robot_position[0], 0.95 * curTargetPos[1] + 0.05 * robot_position[1],
                         curTargetPos[2]]
            self._p.resetDebugVisualizerCamera(distance, yaw, pitch, targetPos)
        return state, reward, bool(done), {}
