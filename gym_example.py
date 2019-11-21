import gym_parkour
import gym

env = gym.make('ParkourChallenge-v0')
obs = env.reset()
print(obs)