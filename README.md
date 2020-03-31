# parkour-training

This is one of two main repositories for the parkour project at Roboy. 

Our  environment, initial training experiments and simple walking demo can be found here. 

For the advanced experiments with learning motion-capture based and compositional policies, refer to [parkour_learning](https://github.com/Roboy/parkour_learning)

For the simple deployable environment in pybullet-gym, refer to [parkour-training-starting-kit](https://github.com/Roboy/parkour-training-starting-kit)


![alt text](HumanoidWithTrack.png "Humanoid in front of challenge track")


# Dependencies and installation
We use OpenAI Gym, PyBullet, PyBullet Gymperium and BAIR's deep learning library lpyyt for the training and the environment.


### Installing pybullet 
```
pip install pybullet

git clone https://github.com/openai/gym.git
cd gym
pip install -e .

git clone https://github.com/benelot/pybullet-gym.git
cd pybullet-gym
pip install -e .

git clone https://github.com/astooke/rlpyt.git
cd rlpyt
pip install -e .

```

## Installing gym_parkour
```bash
git clone https://github.com/Roboy/parkour-training
cd parkour-training
pip install -e .
```

## Examples
Watch pretrained humanoid failing on the parkour track
```bash
cd parkour-training
python3 enjoy_humanoid
```

## Environments
ParkourChallenge-v0


## Bugs
pbullet migth throw an error if several environments run in parallel:
pybullet.error: Not connected to physics server.

for fix see here: https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=12722
