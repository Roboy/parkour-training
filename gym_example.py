import gym_parkour
from gym_parkour.envs.biped import Biped
import gym
import time

env = gym.make('ParkourChallenge-v0', robot=Biped(), render=True)
while True:
    done = False
    obs = env.reset()
    step = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render(mode='human')
        time.sleep(0.04)
        step += 1
