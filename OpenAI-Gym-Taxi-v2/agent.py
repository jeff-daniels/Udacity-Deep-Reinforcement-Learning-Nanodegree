import numpy as np
import random
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, alpha=0.3, gamma=0.95, eps_start=1.0, eps_decay=0.9, eps_min=0.0001):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.alpha = alpha
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_min = eps_min

    def update_Q_sarsamax(self, alpha, gamma, Q, state, action, reward, next_state=None):
        """Returns updated Q-value for the most recent experience."""
        current = Q[state][action]  # estimate in Q-table (for current state, action pair)
        Qsa_next = np.max(Q[next_state]) if next_state is not None else 0  # value of next state 
        target = reward + (gamma * Qsa_next)               # construct TD target
        new_value = current + (alpha * (target - current)) # get updated value 
        return new_value
    

    def epsilon_greedy(self, Q, state, nA, eps):
        """Selects epsilon-greedy action for supplied state.

        Params
        ======
            Q (dictionary): action-value function
            state (int): current state
            nA (int): number actions in the environment
            eps (float): epsilon
        """
        if random.random() > eps: # select greedy action with probability epsilon
            return np.argmax(Q[state])
        else:                     # otherwise, select an action randomly
            return random.choice(np.arange(nA))

    def select_action(self, state, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        eps = max(self.eps_start*(self.eps_decay**i_episode), self.eps_min)
        action = self.epsilon_greedy(self.Q, state, self.nA, eps)
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q[state][action] = self.update_Q_sarsamax(self.alpha, self.gamma, self.Q, state, action, reward, next_state)
