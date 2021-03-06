import math
import os
from gym_parkour.envs.stadium_scene import StadiumScene


class WallScene(StadiumScene):
    walls_loaded = False

    def episode_restart(self, bullet_client):
        # let parent object load plane
        StadiumScene.episode_restart(self, bullet_client)

        if not self.walls_loaded:
            wall_filename = os.path.join(os.path.dirname(__file__), "assets", "wall.urdf")
            self._p.loadURDF(wall_filename, useFixedBase=1, basePosition=(14, 2, 0),
                             baseOrientation=self._p.getQuaternionFromEuler([0, 0, 0.07 * math.pi]))
            self._p.loadURDF(wall_filename, useFixedBase=1, basePosition=(10, -2, 0),
                             baseOrientation=self._p.getQuaternionFromEuler([0, 0, 0.05 * math.pi]))
            self._p.loadURDF(wall_filename, useFixedBase=1, basePosition=(19, -2, 0),
                             baseOrientation=self._p.getQuaternionFromEuler([0, 0, 0.05 * math.pi]))
            self.walls_loaded = True
