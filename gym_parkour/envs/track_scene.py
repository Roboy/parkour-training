import os
from gym_parkour.envs.stadium_scene import StadiumScene


class TrackScene(StadiumScene):
    track_loaded = False

    def episode_restart(self, bullet_client):
        # let parent object load plane
        StadiumScene.episode_restart(self, bullet_client)

        if not self.track_loaded:
            track_filename = os.path.join(os.path.dirname(__file__), "assets", "track.urdf")
            self._p.loadURDF(track_filename, useFixedBase=1, basePosition=(7, 0, 0), baseOrientation=(-1, 0, 0, -1))
            self.track_loaded = True
