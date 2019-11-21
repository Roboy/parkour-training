from gym.envs.registration import register


print('registering env')
register(
    id='ParkourChallenge-v0',
    entry_point='gym_parkour.envs.parkour_gym:ParkourGym',
)

