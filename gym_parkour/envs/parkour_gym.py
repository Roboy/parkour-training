import pybullet
import numpy as np
from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from gym_parkour.envs.humanoid import Humanoid
from gym_parkour.envs.laikago import Laikago
from gym_parkour.envs.stadium_scene import StadiumScene
from gym_parkour.envs.track_scene import TrackScene
from gym_parkour.envs.wall_scene import WallScene
from gym.spaces.dict import Dict
import gym.spaces as spaces
import random
import copy
import time


class ParkourGym(BaseBulletEnv):
    foot_ground_object_names = {"floor"}  # to distinguish ground and other objects

    def __init__(self, render=False, vision=False):
        # self.robot = Humanoid(target_position_xy=self.target_position_xy)
        self.robot = Laikago()
        BaseBulletEnv.__init__(self, self.robot, render)
        self.saved_state_id = None
        self.vision = vision
        # self.action_space = Dict()
        if vision:
            self.observation_space = spaces.Dict({
                'robot_state': self.robot.observation_space,
                'camera': spaces.Box(low=0, high=255, shape=(30, 30, 1)),
            })
        else:
            self.observation_space = self.robot.observation_space
        self.action_space = self.robot.action_space

    # Overwrite BaseBulletEnv
    def create_single_player_scene(self, bullet_client):
        self.scene = StadiumScene(bullet_client, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
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
            self.target_marker_id = self._p.loadURDF('/home/alex/parkour-training/gym_parkour/envs/assets/target_marker.urdf')
            self.saved_state_id = self._p.saveState()

        self.set_target()
        self.last_distance_to_target = np.linalg.norm(
            np.array(self.robot.get_pos_xyz()[0:2]) - np.array(self.target_position_xy))
        return self.get_obs()

    def step(self, a):
        start = time.time()
        self.robot.apply_action(a)
        self.scene.global_step()
        # print('step: ' + str(time.time() - start))

        robot_specific_reward, env_info = self.robot.calc_reward(a, self.ground_ids)
        distance_to_target = self.get_distance_to_target()
        done = False
        if distance_to_target < 1 or not self.robot.is_alive():
            done = True
        velocity = self.last_distance_to_target - distance_to_target
        velocity_reward = 1e2 * velocity
        env_info['velocity_reward'] = velocity_reward
        reward = robot_specific_reward + velocity_reward
        # print('velocity reward: ' + str(velocity_reward))
        self.last_distance_to_target = copy.copy(distance_to_target)
        # print('time after reward: ' + str(time.time() - start))
        # follow robot with camera
        if self.isRender:
            robot_position = self.robot.body_xyz
            camInfo = self._p.getDebugVisualizerCamera()
            curTargetPos = camInfo[11]
            distance = camInfo[10]
            yaw = camInfo[8]
            pitch = camInfo[9]
            targetPos = [0.95 * curTargetPos[0] + 0.05 * robot_position[0],
                         0.95 * curTargetPos[1] + 0.05 * robot_position[1],
                         curTargetPos[2]]
            self._p.resetDebugVisualizerCamera(distance, yaw, pitch, targetPos)
        observation = self.get_obs()
        # print('obs: ' + str(time.time() - start))
        if distance_to_target < 0.5:
            self.set_target()
        # print('reward: ' + str(reward))
        return observation, reward, bool(done), env_info

    def get_distance_to_target(self):
        return np.linalg.norm(np.array(self.robot.get_pos_xyz()[0:2]) - np.array(self.target_position_xy))


    def set_target(self):
        self.target_position_xy = (random.randint(-5, 5), random.randint(-5, 5))
        self.last_distance_to_target = self.get_distance_to_target()
        self._p.resetBasePositionAndOrientation(self.target_marker_id, posObj=list(self.target_position_xy) + [1],
                                                ornObj=(1, 1, 1, 0))

    def get_obs(self):
        start = time.time()
        robot_state = self.robot.calc_state(self.target_position_xy, self.ground_ids)
        # print('robot state: ' + str(time.time() - start))
        if self.vision:
            base_pos = list(self.robot.get_pos_xyz())
            base_pos[2] += 0.7
            view_matrix = self._p.computeViewMatrix(
                cameraEyePosition=base_pos,  # self.robot.body_xyz + [0, 0, 1],
                cameraTargetPosition=self.target_position_xy + (1,),
                cameraUpVector=(1, 0, 1)
            )
            proj_matrix = self._p.computeProjectionMatrixFOV(
                fov=60, aspect=1.0,  # float(self._render_width) / self._render_height,
                nearVal=0.1, farVal=100.0)
            (_, _, px, _, _) = self._p.getCameraImage(
                width=20, height=20, viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
            )
            rgb_array = np.array(px)
            rgb_array = rgb_array[:, :, :3]
            gray_img = np.mean(rgb_array, axis=2)
            # for testing
            if random.randint(0, 100) %20 == 0:
                import matplotlib.pyplot as plt
                # plt.imshow(gray_img, cmap='gray')
                plt.imshow(rgb_array)
                plt.show()
            observation = {
                'robot_state': robot_state,
                'camera': gray_img
            }
        else:
            observation = robot_state

        return observation
