
import os, inspect
import pybullet
import math
from pybulletgym.envs.roboschool.scenes.scene_bases import Scene

class TrackScene(Scene):
    track_loaded = False
    starting_ground_loaded = False
    multiplayer = False
    stadiumLoaded = 0
    
    def episode_restart(self, bullet_client):
        self._p = bullet_client
        # Scene.episode_restart(self, bullet_client)
        if not self.starting_ground_loaded:
            track_filename = os.path.join(os.path.dirname(__file__), "assets", "starting_ground.urdf")
            self.ground_plane_mjcf = self._p.loadURDF(track_filename, useFixedBase=1, basePosition=(0, 0, -2), baseOrientation=(-1, 0, 0, -1))
            self.starting_ground_loaded = True

        if not self.track_loaded:
            track_filename = os.path.join(os.path.dirname(__file__), "assets", "track.urdf")
            self._p.loadURDF(track_filename, useFixedBase=1, basePosition=(1.5, 0, -2), baseOrientation=(-1, 0, 0, -1))
            self.track_loaded = True