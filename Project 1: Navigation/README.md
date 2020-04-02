# Project 1: Navigation

## Background  
In this project, an agent is trained to navigate (and collect bananas!) in a large, square world.  The goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

## Environment  
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.   

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

0 - move forward.  
1 - move backward.  
2 - turn left.  
3 - turn right.  

## Agent
A [`Unity Machine Learning agent`](https://github.com/Unity-Technologies/ml-agents) was used.  And trained on a Deep Q-Network.

## This repository uses code from:
The [`Udacity Deep Reinforcement Learning Nanodegree`](https://github.com/udacity/deep-reinforcement-learning.git) repository.

The outline for the report can be found in the [`p1_navigation folder`](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation)  
Code that was modified to implement a Deep Q-Network is found in the [`dqn/solution folder`](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn)

## List of Files:
- [`Report.ipynb`]() A report describing the learning algorithm, the details for implementation, and ideas for future work.
- [`dqn_network.py`]() Code for a Deep Q-Network containing the Agent class that is used in the learning algorithm.
- [`model.py`]() Code for the QNetwork class which used to construct a pyTorch deep neural network.
- [`checkpoint.pth`]() Trained model weights of the final and best implementation of the learning algorithm.
- [`Navigation_Hyperparameters.ods`]() A spreadsheet tracking the hyperparameter tuning process.

