# Udacity Deep Reinforcement Learning Nanodegree  
## Project 2: Continuous Control  

This report outlines how I achieved a benchmark score for the Reacher 20 arm environment using a Deep Deterministic Policy Gradients (DDPG) algorithm.  

### Learning Algorithm  
The code for the [`ddpg-pendulum`](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) algorithm in the [Udacity Deep Reinforcement Learning Nanodegree program repository](https://github.com/udacity/deep-reinforcement-learning) was adapted for the Unity ML-Agents Reacher environment with twenty arms.  Much of the modifications were done on the [`ddpg_agent.py`](https://github.com/jeff-daniels/Udacity-DRLND/blob/master/Project%202:%20Continuous%20Control/ddpg_agent.py), mostly to adapt it to reading states and performing actions for 20 arms.  Although the arms acted independently, they shared the same networks and learning buffer.  You can think of the algorithm as having 20 agents or just one agent controlling 20 arms, but I'll outline what is actually going on in the training function `ddpg` in the `Continuous_Control.ipynb` notebook:  
1. **Twenty** states are observed  
1. **Twenty** actions are taken
1. **Twenty** next_states are observed
1. **Twenty** rewards are recieved
1. **Twenty** dones are observed
1. **Twenty** sets of state, action, reward, next_state, and done are added to **one** replay buffer.  
1. **One** batch-sized sample from the replay buffer is used to train or update **one** set of networks including:
   * actor_local
   * actor_target
   * critic_local
   * critic_target

The report clearly describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures for any neural networks.

### Plot of rewards  
A plot of rewards per episode is included to illustrate that either:

[version 1] the agent receives an average reward (over 100 episodes) of at least +30, or
[version 2] the agent is able to receive an average reward (over 100 episodes, and over all 20 agents) of at least +30.
The submission reports the number of episodes needed to solve the environment.  

### Ideas for Future Work
The submission has concrete future ideas for improving the agent's performance.
