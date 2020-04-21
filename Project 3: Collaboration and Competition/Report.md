# Udacity Deep Reinforcement Learning Nanodegree  
## Project 3: Collaboration and Competition

This report outlines how I achieved a benchmark score for the Tennis environment using a Deep Deterministic Policy Gradients (DDPG) algorithm.  

### Learning Algorithm  
The code for the [`ddpg-pendulum`](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) algorithm in the [Udacity Deep Reinforcement Learning Nanodegree program repository](https://github.com/udacity/deep-reinforcement-learning) was adapted for the Unity ML-Agents Tennis environment.  Much of the modifications were done on the [`ddpg_agent.py`](https://github.com/jeff-daniels/Udacity-DRLND/blob/master/Project%203:%20Collaboration%20and%20Competition/ddpg_agent.py), mostly to adapt it to reading states and performing actions for 2 tennis players.  Most of this work was done in the previous project [Continuous Control](https://github.com/jeff-daniels/Udacity-DRLND/tree/master/Project%202:%20Continuous%20Control) where the DDPG agent was used to control 20 reacher arms.  Although the tennis players acted independently, they shared the same networks and learning buffer.  You can think of the algorithm as having 2 agents or just one agent controlling 2 players, but I'll outline what is actually going on in the training function `ddpg` in the `Tennis.ipynb` notebook:  
1. **Two** states are observed  
1. **Two** actions are taken
1. **Two** next_states are observed
1. **Two** rewards are recieved
1. **Two** dones are observed
1. **Two** sets of state, action, reward, next_state, and done are added to **one** replay buffer.  
1. **One** batch-sized sample from the replay buffer is used to train or update **one** set of networks including:
   * actor_local
   * actor_target
   * critic_local
   * critic_target

The hyperparameters in the `ddpg_agent` module include:  

BUFFER_SIZE = int(1e5)  # replay buffer size  
BATCH_SIZE = 128        # minibatch size  
GAMMA = 0.99            # discount factor  
TAU = 1e-3              # for soft update of target parameters  
LR_ACTOR = 3e-4         # learning rate of the actor   
LR_CRITIC = 1e-3        # learning rate of the critic  
WEIGHT_DECAY = 0.0000   # L2 weight decay  
UPDATE_EVERY = 20        # how often to update the networks in time steps  
N_UPDATES = 20           # how many updates to perform per UPDATE_EVERY   

These `model` hyperparameters are:   

FC1_UNITS_ACTOR = 512   # number of nodes in first hidden layer for Actor    
FC2_UNITS_ACTOR = 256   # number of nodes in second hidden layer for Actor       
FCS1_UNITS_CRITIC = 512 # number of nodes in first hidden layor for Critic  
FC2_UNITS_CRITIC = 256  # number of nodes in second hidden layor for Critic  

The actor network hidden layers are fully connected and use relu activations and a tanh activation for the action.  Input dimensions are the size of the observation space and output dimensions are the size of the action space.  

The critic network layers are also fully connected and use relu activations but don't use any activation for the action.  Input dimensions are the size of the state plus the action size.  

This model differs from the `ddpg-pendulum` by using a slightly different network layer sizes because I felt like using model architecture that I was used to using and hopefully it handles batch processing better.  It has an UPDATE_EVERY parameter to stabilize training, but every time step interval it does train, it trains multiple times, in this case the same number of training iterations occur as if you trained every time step.  That seemed to work well.

### Plot of rewards
The environment reached a benchmark average reward of +0.5 over 100 episodes after 7120 episodes. To be clear, the score for each episode is the maximum score of the two agents. A plot is shown below along with a sample of the output taken in the middle of training.  
![Plot of rewards](https://github.com/jeff-daniels/Udacity-DRLND/blob/master/Project%203:%20Collaboration%20and%20Competition/scores.png)
```                                   
Episode: 6000	Average Max Score: 0.344	Average steps: 138.3	Size of memory: 100000                                   
Episode: 6100	Average Max Score: 0.373	Average steps: 147.3	Size of memory: 100000                                   
Episode: 6200	Average Max Score: 0.435	Average steps: 170.1	Size of memory: 100000                                   
Episode: 6300	Average Max Score: 0.471	Average steps: 185.6	Size of memory: 100000                                   
Episode: 6400	Average Max Score: 0.384	Average steps: 148.3	Size of memory: 100000                                   
Episode: 6500	Average Max Score: 0.171	Average steps: 74.5	Size of memory: 100000                                   
Episode: 6508	Scores: 0.500 0.490	Steps: 204	Time Elapsed: 47.0 min 
```

### Ideas for Future Work
There could have been more hyperparameter tuning, but I found the training to be slow and frustrating so I was happy to get just one model that achieved the bench mark. I think it took about 2.5 hours to train my only successful model.  It seems like a lot of steps compared to my classmates.  Running some randomized hyperparameter search could help me find better hyperparameters but I'd be interested mostly in making the hidden layers larger, increasing the replay memory size, and maybe playing around with how frequently the models update.

In the last project I expressed a lot of impatience using the online workspace because it fell asleep or I lost my internet connection.  Since then, I started using Udacity's workplace_utils module to keep my session active.  I have also gotten deft at saving my models frequently, writing functions to do this for me in my helper file.

In my last project I expressed interest in running Proximal Policy Optimization (PPO) and I decided early on in the project that this would be the algorithm I would use.  I thought it would be more educational to start all over with a new algorithm rather than just reuse the model from the last project.  I ended up adapting Shangtong Zhang's [code](https://github.com/ShangtongZhang/DeepRL) to the tennis environment.  Although I successfully implemented it, the models did not perform well and were quite unstable.  The work related to this futile effort can be found in the setup-DeepRL and the single-optimizer branchs of this repository.  The experience working with Shangtong's repository taught me a lot about modularized code and I would have liked to get PPO to run well, but time contraints forced me to settle on using my old DDPG code.

Because I have invested time in learning to use to Shangtong's code, I would feel confident implementing any one of the other algorithms he has provided examples for such as: Synchronous Advantage Actor Critic (A2C) or the Twined Delayed DDPG (TD3) algorithms.  

It might also be interesting to try my unsuccessful PPO algorithm on the previous project's Reacher 20 environment.  I suspect PPO wasn't as successful in the Tennis environment because it has such a cold start.  It doesn't have enough experience to know that it should hit the ball and I suspect its initial actions are not exploratory enough.  I also found that default rollout length of 2048 made running the algorithm very slow.  This doesn't seem like enough trajectories to be useful, especially when so many of the earlier trajectories sampled have little rewards.  DDPG, in contrast, could draw upon its replay buffer of 100,000 experiences quite quickly.  The only slow aspect of DDPG was how long it takes to save the model because I saved the replay buffer as well.  In the past, I found that saving a replay buffer of size 1e6 takes about two minutes.  It's not completely necessary to save the replay buffer, you can just resume training with saved model weights and an empty replay buffer, but it seemed a good practice for consistent results in experimentation.  Of course there could be faster ways of saving the model, using Amazon's E2C for example.

The environment could also be altered to give different rewards.  I think initially, getting a higher score just means not being the agent who is served the ball.  I think the structure of the rewards is such that doing nothing might be optimized.
