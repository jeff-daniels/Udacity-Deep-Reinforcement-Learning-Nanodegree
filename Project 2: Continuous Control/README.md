# Udacity Deep Reinforcement Learning Nanodegree: Continuous Control Project  

The goal of this project is to develop a deep reinforcement learning model that interacts with the Unity ML-Agents [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) Environment.  In this environment, a double-jointed arm with a hand at the end of it moves to target locations that are constantly in motion.  For each time step that the hand in a target location, a reward of +0.1 is provided by the environment.  The objective of this project is maximize the rewards by having the hand located in the target position for as long as possible.  

The observation space is 33 variables corresponding the position, rotation, velocity, and angular velocity of the two arm Rigidbodies.  The action space is 4 variables corresponding to the torque applied to the two joints.  The action space is continuous with a range between -1 and 1.  

There were two options for the environments, one with a single arm, and one with 20 arms.  The benchmark score that must be achieved is a score of +30 over 100 consecutive episodes.  For the 20 arm environment, the average score of all the arms is must be +30 over 100 consecutive episodes.  For this project, the benchmark was achieved using the 20 arm environment.

## Setup
Follow the instructions in the [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) which will install PyTorch, the ML-Agents toolkit and other Python packages required for the project.

The pre-built environment can be downloaded from one of the links below.  This evironment includes Unity.  
**Version 1: One (1) Agent**
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

**Version 2: Twenty (20) Agents** 
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)  

Then, place the file in the `p2_continuous-control/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.
Finally, if you followed the instructions effectively, open `Continuous_Control.ipynb` (located in the `p2_continuous-control/` folder in the DRLND GitHub repository) and follow the instructions to learn how to use the Python API to control the agent.  This notebook was primary prototyping platform for developing the model.  The model was trained in this notebook and after the benchmark score was achieved, the model weights were saved as `checkpoint_actor.pth` and `checkpoint_critic.pth`.  

## Solution:  
The model used were part of a Deep Deterministic Policy Gradient (DDPG) algorithm.  More information about using Actor-Critic methods can be found in this [paper](https://arxiv.org/abs/1509.02971).  Details about my implementation can be found in [`Report.ipynb`]()

## Description of Files:  
* [`Continuous_Control.ipynb`](https://github.com/jeff-daniels/Udacity-DRLND/blob/master/Project%202:%20Continuous%20Control/Continuous_Control.ipynb)  The notebook where the model was trained.  
* [`checkpoint_06-without_memory.tar`](https://github.com/jeff-daniels/Udacity-DRLND/blob/master/Project%202:%20Continuous%20Control/checkpoint_06-without_memory.tar)  Backup file of the model to resume training.  Contains all the model weights, optimizers, a dictionary of hyperparameters used, and an array of scores for all the episodes.  Usually contains a Replay Buffer memory that is too large to be uploaded.  
* [`checkpoint_actor.pth`](https://github.com/jeff-daniels/Udacity-DRLND/blob/master/Project%202:%20Continuous%20Control/checkpoint_actor.pth)  Actor model weights.  
* [`checkpoint_critic.pth`](https://github.com/jeff-daniels/Udacity-DRLND/blob/master/Project%202:%20Continuous%20Control/checkpoint_critic.pth)  Critic model weights.  
* [`ddpg_agent.py`](https://github.com/jeff-daniels/Udacity-DRLND/blob/master/Project%202:%20Continuous%20Control/ddpg_agent.py)  Module for creating an **Agent** that implements DDPG.  
* [`helper.py`](https://github.com/jeff-daniels/Udacity-DRLND/blob/master/Project%202:%20Continuous%20Control/helper.py)  Module of various helper functions.  
* [`model.py`](https://github.com/jeff-daniels/Udacity-DRLND/blob/master/Project%202:%20Continuous%20Control/model.py)  Module for building the **Actor** and **Critic** neural networks.  
* [`Report.ipynb`]()  Report describing learning algorithm. Describes details of the implementation, along with ideas for future work.

