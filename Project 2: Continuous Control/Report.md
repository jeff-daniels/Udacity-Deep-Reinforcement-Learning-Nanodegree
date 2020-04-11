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

The hyperparameters in the `ddpg_agent` module include:  

  BUFFER_SIZE = int(1e6)  # replay buffer size  
   BATCH_SIZE = 128        # minibatch size  
   GAMMA = 0.99            # discount factor  
   TAU = 1e-3              # for soft update of target parameters  
   LR_ACTOR = 3e-4         # learning rate of the actor   
   LR_CRITIC = 1e-3        # learning rate of the critic  
   WEIGHT_DECAY = 0.0000   # L2 weight decay  
   UPDATE_EVERY = 20       # how often to update the networks in time steps  
   N_UPDATES = 20          # how many updates to perform per UPDATE_EVERY  

These `model` hyperparameters are:   

   FC1_UNITS_ACTOR = 256   # number of nodes in first hidden layer for Actor  
   FC2_UNITS_ACTOR = 128   # number of nodes in second hidden layer for Actor     
   FCS1_UNITS_CRITIC = 256 # number of nodes in first hidden layor for Critic  
   FC2_UNITS_CRITIC = 128  # number of nodes in second hidden layor for Critic  

The actor network hidden layers are fully connected and use relu activations and a tanh activation for the action.  Input dimensions are the size of the observation space and output dimensions are the size of the action space.  

The critic network layers are also fully connected and use relu activations but don't use any activation for the action.  Input dimensions are the size of the state plus the action size.  

This model differs from the `ddpg-pendulum` by using a larger buffer size to keep track of twenty agents.  The network is smaller because I felt like using more conventional model architecture.  It has an UPDATE_EVERY parameter to stabilize training, but every time step interval it does train, it trains multiple times, in this case the same number of training iterations occur as if you trained every time step.  That seemed to work well.

### Plot of rewards  
The environment reached a benchmark average reward of 30 over 100 episodes after 153 episodes.  To be clear, the scores are an average score for the 20 agents.  A plot as well as the output are shown below.  
![Plot of rewards](https://github.com/jeff-daniels/Udacity-DRLND/blob/master/Project%202:%20Continuous%20Control/scores.png)
```
Episode 110	Average Score: 13.78                                         
Episode 120	Average Score: 15.07                                    
Episode 130	Average Score: 18.71                                         
Episode 140	Average Score: 21.32                                    
Episode 150	Average Score: 21.13                                    
Episode 160	Average Score: 22.09                                         
Episode 170	Average Score: 22.85                                    
Episode 180	Average Score: 24.80                                    
Episode 190	Average Score: 28.81                                    
Episode 200	Average Score: 29.38                                    
Episode 210	Average Score: 31.69                                    
Episode 220	Average Score: 34.53                                    
Episode 230	Average Score: 33.69                                    
Episode 240	Average Score: 35.31                                    
Episode 250	Average Score: 33.75                                    
Episode 253	Score: 34.71	Time Step 1001	Size of memory: 1000000     
Environment solved in 153 episodes!	Average Score: 30.03 
```

### Ideas for Future Work  
There could have been more hyperparameter tuning, but I found the training to be slow and frustrating so I was happy to get just one model that achieved the bench mark.  To reach this benchmark took about 90 minutes of training which is quite a long time when working on an unstable Workspace that falls asleep without warning or loses its connection for whatever reason.  Other frustrations included local internet and power outages.  I tried to run it locally on my very unqualified laptop but I couldn't even get the environment set up because my computer refused to install TensorFlow and claimed to be low on disk space.  Obviously a better computer would solve some of my problems.  More likely I will investigate how to use cloud computing, ie. AWS Sagemaker or Google CoLab.  

Obviously DDPG is not the only algrorithm for Continuous Control Reinforcement Learning and I would have liked to have had the time to explore some of the algorithms described in the suggested reading `Benchmarking Deep Reinforcement Learning for Continuous Control`(https://arxiv.org/abs/1604.06778).  Proximal Policy Optimization (PPO) is a method that gets mentioned a lot but I have no idea how it works.   OpenAI seems to be enamored by PPO so I should probably learn how it works.  Distributed Distributional Deep Deterministic Policy Gradient (D4PG) also seems to be an improvement upon DDPG.  I wonder how much the simple Reacher environment would benefit from these new algorithms.
The submission has concrete future ideas for improving the agent's performance.
