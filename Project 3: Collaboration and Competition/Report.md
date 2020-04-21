# Udacity Deep Reinforcement Learning Nanodegree  
## Project 3: Collaboration and Competition

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
