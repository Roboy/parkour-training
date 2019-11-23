import pybullet as p


class BulletEnvironment():
    def __init__(self):
        physics_client = p.connect(p.GUI)
        robot_model_path = '/home/alex/parkour-training/gym_parkour/envs/assets/humanoid.xml'
        self.robot_id = p.loadMJCF(robot_model_path)
        p.setGravity(0, 0, -10)

    def apply_torques(self):
        pass

    def step_simulation(self):
        pass

    def get_observation(self):
        pass