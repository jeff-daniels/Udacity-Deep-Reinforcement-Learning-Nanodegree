a README that describes how someone not familiar with this project should use your repository. The README should be designed for a general audience that may not be familiar with the Nanodegree program; you should describe the environment that you solved, along with how to install the requirements before running the code in your repository.
the code that you use for training the agent, along with the trained model weights.
a report describing your learning algorithm. This is where you will describe the details of your implementation, along with ideas for future work

# Udacity Deep Reinforcement Learning Nanodegree: Collaboration and Competition Project  

The goal of this project is to develop a deep reinforcement learning model that interacts with an environment similar to the Unity ML-Agents [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.  In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,  
* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores. This yields a single score for each episode. 
* The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Setup  
### Step 1: Activate the Environment
Follow the instructions in the [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) which will install PyTorch, the ML-Agents toolkit and other Python packages required for the project.

(*For Windows users*) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

### Step 2: Download the Unity Environment
The pre-built environment can be downloaded from one of the links below.  This evironment includes Unity.  
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Then, place the file in the `p3_collab-compet/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.

### Step 3: Explort the Environment
Finally, if you followed the instructions effectively, open `Tennis.ipynb` (located in the `p3_collab-compet/` folder in the DRLND GitHub repository) and follow the instructions to learn how to use the Python API to control the agent.  This notebook was primary prototyping platform for developing the model.  The model was trained in this notebook and after the benchmark score was achieved, the model weights were saved as `checkpoint_actor.pth` and `checkpoint_critic.pth`.  

## Solution:  
The model used were part of a Deep Deterministic Policy Gradient (DDPG) algorithm.  More information about using Actor-Critic methods can be found in this [paper](https://arxiv.org/abs/1509.02971).  Details about my implementation can be found in [`Report.ipynb`]()

## Description of Files:  
* [`Tennis.ipynb`]()  The notebook where the model was trained.  
* [`Project 3: Collaboration and Competition Parameter.ods`]()  Spreadsheet used for keeping track of hyperparameters during tuning.  
* [`Report.md`]()  Report describing learning algorithm. Describes details of the implementation, along with ideas for future work. 
* [`checkpoint_actor.pth`]()  Actor model weights.  
* [`checkpoint_critic.pth`]()  Critic model weights.  
* [`ddpg_agent.py`]()  Module for creating an **Agent** that implements DDPG.  
* [`helper.py`]()  Module of various helper functions.  
* [`model.py`]()  Module for building the **Actor** and **Critic** neural networks.  
* [`workspace_utils.py`]()  Module for that disallows workspace slumbering.  
