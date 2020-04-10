##############################################################################
# helper functions for use on Udacity Deep Reinforcement Learning Nanodegree #
# Project 2: Continuous Control                                              #
##############################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import importlib
import torch
import time
import ddpg_agent

def get_hyperparameters(agent):
    hyper_dict = {
        'BUFFER_SIZE': agent.BUFFER_SIZE,
        'BATCH_SIZE': agent.BATCH_SIZE,
        'GAMMA': agent.GAMMA,
        'TAU': agent.TAU,
        'LR_ACTOR': agent.LR_ACTOR,
        'LR_CRITIC': agent.LR_CRITIC,
        'WEIGHT_DECAY': agent.WEIGHT_DECAY,
        'UPDATE_EVERY': agent.UPDATE_EVERY,
        'N_UPDATES': agent.N_UPDATES,
        'FC1_UNITS_ACTOR': agent.FC1_UNITS_ACTOR,
        'FC2_UNITS_ACTOR': agent.FC2_UNITS_ACTOR,
        'FCS1_UNITS_CRITIC': agent.FCS1_UNITS_CRITIC,
        'FC2_UNITS_CRITIC': agent.FC2_UNITS_CRITIC
    }
    return hyper_dict

def print_hyperparameters(agent=None):
    ''' Prints out the agents Hyperparameters'''
    if agent is not None:
        hyper_parameters = get_hyperparameters(agent)
    for key, value in hyper_parameters.items():
        print(key, '=', value)
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
