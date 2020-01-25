import gym_parkour
import gym
import time

env = gym.make('ParkourChallenge-v0', render=True)
while True:
    done = False
    obs = env.reset()
    step = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render(mode='human')
        time.sleep(0.08)
        step += 1
