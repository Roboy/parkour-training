# parkour-training

This is one of two main repositories for the parkour project at Roboy. Our initial goal was to provide an environment and initial solution for the [parkour challenge](https://roboy.org/win).

After some time and direction changes, we conctentrated on trying to pass through the bouncy-catle track untill the ramp and after using a simple policy, to experiment with training a motion-capture mimicking solution.

## Repositories and code stucture

Our  environment, initial training experiments and simple walking demo can be found here.

For the advanced experiments with learning motion-capture based and compositional policies, refer to [parkour_learning](https://github.com/Roboy/parkour_learning)

For the simple deployable environment in pybullet-gym, refer to [parkour-training-starting-kit](https://github.com/Roboy/parkour-training-starting-kit)


![alt text](HumanoidWithTrack.png "Humanoid in front of challenge track")


# Dependencies and installation
We use OpenAI Gym, PyBullet, PyBullet Gymperium and BAIR's deep learning library rlpyt for the training and the environment.


### Installing pybullet 
```bash
pip install pybullet
```

### Installing OpenAI Gym
```bash
git clone https://github.com/openai/gym.git
cd gym
pip install -e .
```

### Installing PyBullet Gymperium

```bash
git clone https://github.com/benelot/pybullet-gym.git
cd pybullet-gym
pip install -e .
```

### Installing rlpyt
```bash
git clone https://github.com/astooke/rlpyt.git
cd rlpyt
pip install -e .

```

## Installing gym_parkour

And finally, to install the content of this repository you need:

```bash
git clone https://github.com/Roboy/parkour-training
cd parkour-training
pip install -e .
```

# Repository structure 

This repository mainly consist of our primary environment ParkourChallenge-v0, which is implemented, following a hierarchical structure with easily swappable robots and scenes. 

All our robots are based on the ParkourRobot abstract class that you can find in ```gym_parkour/envs/parkour_robot.py```, which demands the implementation of state calculation, actions, updates, reward calculation, etc.  

The canonical robot used for the demo is a basic pybullet humanoid with all the methods and specifications implemented in the ```gym_parkour/envs/humanoid.py```. 

The main environment is implemented in ```gym_parkour/envs/parkour_gym.py``` and follows the BaseBulletEnv specifications from pybullet-gym.


## Examples

For a pre-trained model that used the proximity to the goal as the main reward, we've created a demo, that changes the goal positions to follow the track progression.

You can watch pretrained humanoid running through the parkour track like this:
```bash
cd parkour-training
python3 enjoy_humanoid
```

## Bugs
pbullet migth throw an error if several environments run in parallel:
pybullet.error: Not connected to physics server.

for fix see here: https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=12722
