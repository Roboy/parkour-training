from pybulletgym.envs.roboschool.envs.locomotion.walker_base_env import WalkerBaseBulletEnv
from pybulletgym.envs.roboschool.robots.locomotors import Humanoid
# from pybulletgym.envs.roboschool.robots.locomotors import HumanoidFlagrun, HumanoidFlagrunHarder
from gym_parkour.envs.stadium import StadiumScene
from gym_parkour.envs.humanoid_flagrun import HumanoidFlagrun


class ParkourGym(WalkerBaseBulletEnv):
    def __init__(self, render=False):
        self.robot = HumanoidFlagrun()
        WalkerBaseBulletEnv.__init__(self, self.robot, render=render)
        WalkerBaseBulletEnv.walk_target_x = 100
        WalkerBaseBulletEnv.walk_target_y = 0
        self.electricity_cost = 4.25 * WalkerBaseBulletEnv.electricity_cost
        self.stall_torque_cost = 4.25 * WalkerBaseBulletEnv.stall_torque_cost

    def create_single_player_scene(self, bullet_client):   # useless ?
        scene = StadiumScene(bullet_client, gravity=9.8, timestep=0.0165/4, frame_skip=4)
        WalkerBaseBulletEnv.stadium_scene = scene
        # s = WalkerBaseBulletEnv.create_single_player_scene(self, bullet_client)
        scene.zero_at_running_strip_start_line = False
        return scene
