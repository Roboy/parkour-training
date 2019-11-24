import gym_parkour
import gym
import time

env = gym.make('ParkourChallenge-v0', render=True)
while True:
    done = False
    obs = env.reset()

    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(obs)
        env.render(mode='human')
        time.sleep(0.02)
