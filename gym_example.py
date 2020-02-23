import gym_parkour
from gym_parkour.envs import biped, humanoid, laikago
import gym
import time

env = gym.make('ParkourChallenge-v0', robot=humanoid.Humanoid(), render=True)
while True:
    done = False
    obs = env.reset()
    print('reset')
    step = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render(mode='human')
        time.sleep(0.04)
        step += 1
