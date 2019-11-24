import math
import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0,parentdir)
from pybulletgym.envs.roboschool.scenes.scene_bases import Scene

import pybullet


class StadiumScene(Scene):
	multiplayer = False
	zero_at_running_strip_start_line = True   # if False, center of coordinates (0,0,0) will be at the middle of the assets
	stadium_halflen   = 105*0.25	# FOOBALL_FIELD_HALFLEN
	stadium_halfwidth = 50*0.25	 # FOOBALL_FIELD_HALFWID
	stadiumLoaded = 0

	def episode_restart(self, bullet_client):
		self._p = bullet_client
		# Scene.episode_restart(self, bullet_client)
		if self.stadiumLoaded == 0:
			self.stadiumLoaded = 1

			# stadium_pose = cpp_household.Pose()
			# if self.zero_at_running_strip_start_line:
			#	 stadium_pose.set_xyz(27, 21, 0)  # see RUN_STARTLINE, RUN_RAD constants

			filename = os.path.join(os.path.dirname(__file__), "assets", "plane_stadium.sdf")
			# block_filename = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "scenes", "block.urdf")
			# pc_track_filename = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "scenes", "RoboyParkourChallengeTrack-Simplistic-2019-09-19.sdf")
			# track_filename = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "scenes", "track.urdf")
			# wall_filename = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "scenes", "wall.urdf")
			# # self.ground_plane_mjcf=self._p.loadSDF(filename)
			# print(block_filename)
			# # self._p.loadURDF(track_filename, useFixedBase=1, basePosition=(7, 0, -.6), baseOrientation=(-1, 0, 0, -1))
			# self._p.loadURDF(block_filename, useFixedBase=1, basePosition=(5, 0, -0.2), baseOrientation=(1, 0, 0, 0))
			# self._p.loadURDF(block_filename, useFixedBase=1, basePosition=(6, 0, -0.2), baseOrientation=(1, 0, 0, 0))
			# self._p.loadURDF(wall_filename, useFixedBase=1, basePosition=(14, 2, 0), baseOrientation=self._p.getQuaternionFromEuler([0, 0, 0.07 * math.pi]))
			# self._p.loadURDF(wall_filename, useFixedBase=1, basePosition=(10, -2, 0), baseOrientation=self._p.getQuaternionFromEuler([0, 0, 0.05 * math.pi]))
			# self._p.loadURDF(wall_filename, useFixedBase=1, basePosition=(19, -2, 0), baseOrientation=self._p.getQuaternionFromEuler([0, 0, 0.05 * math.pi]))
			# self._p.loadURDF(wall_filename, useFixedBase=1, basePosition=(23, 2, 0), baseOrientation=self._p.getQuaternionFromEuler([0, 0, 0.05 * math.pi]))
			# self.ground_plane_mjcf=self._p.loadSDF(pc_track_filename)
			# #filename = os.path.join(pybullet_data.getDataPath(),"stadium_no_collision.sdf")
			self.ground_plane_mjcf = self._p.loadSDF(filename)
			#
			for i in self.ground_plane_mjcf:
				self._p.changeDynamics(i,-1,lateralFriction=0.8, restitution=0.5)
				# self._p.changeVisualShape(i,-1,rgbaColor=[1,1,1,0.8])
				# self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION,1)
			x = 2
		#	for j in range(pybullet.getNumJoints(i)):
		#		self._p.changeDynamics(i,j,lateralFriction=0)
		#despite the name (stadium_no_collision), it DID have collision, so don't add duplicate ground