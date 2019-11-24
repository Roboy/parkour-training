from pybulletgym.envs.roboschool.envs.locomotion.walker_base_env import WalkerBaseBulletEnv
from pybulletgym.envs.roboschool.robots.locomotors import Humanoid
from pybulletgym.envs.roboschool.robots.locomotors import HumanoidFlagrun, HumanoidFlagrunHarder


class ParkourGym(WalkerBaseBulletEnv):
    def __init__(self, render=False):
        self.robot = HumanoidFlagrun()
        WalkerBaseBulletEnv.__init__(self, self.robot, render=render)
        WalkerBaseBulletEnv.walk_target_x = 100
        WalkerBaseBulletEnv.walk_target_y = 0
        self.electricity_cost = 4.25 * WalkerBaseBulletEnv.electricity_cost
        self.stall_torque_cost = 4.25 * WalkerBaseBulletEnv.stall_torque_cost

    # def create_single_player_scene(self, bullet_client):   # useless ?
    #     s = WalkerBaseBulletEnv.create_single_player_scene(self, bullet_client)
    #     s.zero_at_running_strip_start_line = False
    #     return s
