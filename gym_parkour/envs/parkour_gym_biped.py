import pybullet
import numpy as np
from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from gym_parkour.envs.biped import Biped
from gym_parkour.envs.track_scene import TrackScene


class ParkourGymBiped(BaseBulletEnv):
    def __init__(self, render=False):
        self.camera_x = 0
        self.target_position_xy = (3, 0)
        self.saved_state_id = None
        self.robot = Biped(target_position_xy=self.target_position_xy)
        BaseBulletEnv.__init__(self, self.robot, render)
        self.action_space = self.robot.action_space
        self.observation_space = self.robot.observation_space

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

    electricity_cost = -2.0 * 4.25  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
    stall_torque_cost = -0.1 * 4.25  # cost for running electric current through a motor even at zero rotational speed, small
    foot_collision_cost = -1.0  # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
    foot_ground_object_names = set(["starting_ground"])  # to distinguish ground and other objects
    joints_at_limit_cost = -0.1  # discourage stuck joints

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()
        state = self.robot.calc_state(self.target_position_xy)  # also calculates self.joints_at_limit
        done = False
        if not np.isfinite(state).all():  # check state
            print("~INF~", state)
            done = True

        alive = float(self.robot.alive_bonus(state[0] + self.robot.initial_z, self.robot.body_rpy[
            1]))  # state[0] is body height above ground, body_rpy[1] is pitch
        if alive < 0:
            done = True

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
            electricity_cost,
            joints_at_limit_cost,
            feet_collision_cost
        ]
        # self.reward += sum(self.rewards)
        distance_to_target = np.linalg.norm(np.array(self.robot.body_xyz[0:2]) - np.array(self.target_position_xy))
        if distance_to_target < 1:
            print('target reached!!!')
            done = True

        return state, sum(rewards), bool(done), {}

    def camera_adjust(self):
        x, y, z = self.body_xyz
        self.camera_x = 0.98 * self.camera_x + (1 - 0.98) * x
        self.camera.move_and_look_at(self.camera_x, y - 2.0, 1.4, x, y, 1.0)
