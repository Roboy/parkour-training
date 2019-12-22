"""
MIT License
Copyright (c) 2018 Benjamin Ellenberger
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
# add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import gym_parkour
import gym
import numpy as np
import pybullet as p
import pybulletgym.envs
import os.path
import time


def relu(x):
    return np.maximum(x, 0)



class ReactivePolicy:
    "Simple multi-layer perceptron policy, no internal state"
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def act(self, observations):
        action = np.zeros((self.action_space.shape[0]))
        ###############################################################
        ## Implement your action policy here


        ###############################################################
        return action
def main():
    print("create env")
    # env = gym.make("HumanoidFlagrunHarderPyBulletEnv-v0")
    env = gym.make("ParkourBiped-v0")
    env.render(mode="human")
    print (env.observation_space)
    print (type(env.action_space))
    pi = ReactivePolicy(env.observation_space, env.action_space)

    while 1:
        frame = 0
        score = 0
        restart_delay = 0
        obs = env.reset()
        torsoId = -1
        for i in range(p.getNumBodies()):
            print(p.getBodyInfo(i))
            if p.getBodyInfo(i)[0].decode() == "base":
                torsoId = i
                print("found humanoid torso")

        while 1:
            time.sleep(0.02)
            a = pi.act(obs)
            obs, r, done, _ = env.step(a)
            score += r
            frame += 1
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            camInfo = p.getDebugVisualizerCamera()
            curTargetPos = camInfo[11]
            distance = camInfo[10]
            yaw = camInfo[8]
            pitch = camInfo[9]
            targetPos = [0.95 * curTargetPos[0] + 0.05 * humanPos[0], 0.95 * curTargetPos[1] + 0.05 * humanPos[1],
                         curTargetPos[2]]
            p.resetDebugVisualizerCamera(distance, yaw, pitch, targetPos)

            still_open = env.render("human")
            if still_open is None:
                return
            if not done:
                continue
            if restart_delay == 0:
                print("score=%0.2f in %i frames" % (score, frame))
                restart_delay = 60 * 2  # 2 sec at 60 fps
            else:
                restart_delay -= 1
                if restart_delay == 0:
                    break


if __name__ == "__main__":
    main()
