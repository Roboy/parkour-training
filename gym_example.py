import gym_parkour
import gym
import time
import pybulletgym

env = gym.make('ParkourChallenge-v0', render=True)
# env = gym.make('HopperPyBulletEnv-v0')
while True:
    done = False
    obs = env.reset()
    step = 0
    while not done:
        # action = env.action_space.sample()
        action = (0, ) * 16
        if step < 17:
            action = [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            action = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        start = time.time()
        obs, reward, done, info = env.step(action)
        print('time: ' + str(time.time() - start))
        # print(obs)
        env.render(mode='human')
        time.sleep(0.08)
        step += 1
