##############################################################################
# helper functions for use on Udacity Deep Reinforcement Learning Nanodegree #
# Project 2: Continuous Control                                              #
##############################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def print_hyperparameters(agent=None):
    ''' Prints out the agents Hyperparameters'''
    if agent is not None:
        print('BUFFER_SIZE = {}'.format(agent.BUFFER_SIZE))
        print('BATCH_SIZE = {}'.format(agent.BATCH_SIZE))
        print('GAMMA = {}'.format(agent.GAMMA))
        print('TAU = {}'.format(agent.TAU))
        print('LR_ACTOR = {}'.format(agent.LR_ACTOR))
        print('LR_CRITIC = {}'.format(agent.LR_CRITIC))
        print('WEIGHT_DECAY = {}'.format(agent.WEIGHT_DECAY))
        print('UPDATE_EVERY = {}'.format(agent.UPDATE_EVERY))
        print('N_UPDATES = {}'.format(agent.N_UPDATES))          
        print('FC_UNITS_ACTOR = {}'.format(agent.FC_UNITS_ACTOR))
        print('FCS1_UNITS_CRITIC = {}'.format(agent.FCS1_UNITS_CRITIC))
        print('FC2_UNITS_CRITIC = {}'.format(agent.FC2_UNITS_CRITIC))
        print('FC3_UNITS_CRITIC = {}\n'.format(agent.FC3_UNITS_CRITIC))
    return None

def plot_scores(scores, rolling_window=10, title = 'Scores'):
    """ Plots the average score across agents for each episode and a rolling average"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), np.mean(scores, 1))
    rolling_mean = pd.Series(np.mean(scores,1)).rolling(rolling_window).mean()
    plt.plot(rolling_mean)
    plt.title(title)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    return None