"""
Utility functions
Udacity Deep Reinforcement Learning Nanodegree
Project 3: Collaboration and Competition
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from collections import deque
import matplotlib.pyplot as plt
import time 


def play_episode(agent):
    """ 
    Plays a single episode of the environment
    returns the mean score of all the agents, and the combined steps of the agents
    """
    config = agent.config
    env = config.environment
    brain_name = config.brain_name
    env_info = env.reset(train_mode=True)[brain_name]    
    states = env_info.vector_observations                 
    scores = np.zeros(config.num_agents)
    steps = 0
    while True:
        prediction = agent.network(states)
        env_info = env.step(prediction['a'].cpu().detach().numpy())[brain_name]
        next_states = env_info.vector_observations         
        rewards = env_info.rewards                         
        dones = env_info.local_done                     
        scores += rewards                      
        states = next_states      
        steps += config.num_agents
        if np.any(dones):                                  
            break
    
    return scores, steps

def save_checkpoint(agent, scores_per_episode, steps_per_episode, path='checkpoint.tar'):
    """ saves the model weights and optimizers, the scores_per_episode and the ReplayBuffer memory"""
    torch.save({'scores_per_episode': scores_per_episode, 
                'steps_per_episode': steps_per_episode,
                'network_dict': agent.network.state_dict(),
                'actor_opt_dict': agent.actor_opt.state_dict(),
                'critic_opt_dict': agent.critic_opt.state_dict()
              }, path)

def load_checkpoint(agents, scores_per_episode, steps_per_episode, path):
    """ loads a checkpoint to pick up training from the last saved checkpoint"""
    checkpoint = torch.load(path)
    scores_per_episode = checkpoint['scores_per_episode']
    steps_per_episode = checkpoint['steps_per_episode']
    agents.network.load_state_dict(checkpoint['network_dict'])
    agents.actor_opt.load_state_dict(checkpoint['actor_opt_dict'])
    agents.critic_opt.load_state_dict(checkpoint['critic_opt_dict'])
    
    return agents, scores_per_episode, steps_per_episode


def run_steps(agent):
    config = agent.config
    scores_per_episode = []
    steps_per_episode = []
    max_scores_window = deque(maxlen=config.scores_window_len)
    steps_window = deque(maxlen=config.scores_window_len)
    toc = time.time()
    
    if config.load_checkpoint:
        agent, scores_per_episode, steps_per_episode = \
        load_checkpoint(agent, scores_per_episode,
                        steps_per_episode, config.load_checkpoint_path)
        for score in scores_per_episode:
            max_scores_window.append(score.max())
        for step in steps_per_episode:
            steps_window.append(step)
    
    for episode in range(len(scores_per_episode)+1, config.max_episodes+1):
        agent.step()
        score, steps = play_episode(agent)
        scores_per_episode.append(score)
        max_scores_window.append(score.max())
        steps_per_episode.append(steps)
        steps_window.append(steps)
        
        # Print out scores after each episode and every print interval
        tic = time.time()
        s = '\rEpisode: {}\tScores: {:.3f} {:.3f}\tSteps: {}\tTime Elapsed: {:.1f} min'
        print(s.format(episode, score[0], score[1], steps, (tic-toc)/60), end=" ")
        
        if episode % config.print_interval == 0:                
            s = '\rEpisode: {}\tAverage Max Score: {:.3f}\tAverage steps: {:.1f}'
            print(s.format(episode, np.mean(max_scores_window), 
                           np.mean(steps_per_episode)), end=" "*35+"\n")
        
        if episode % config.save_interval == 0:
            path = f'checkpoint-{episode}.tar'
            save_checkpoint(agent, scores_per_episode, steps_per_episode, path)
        
        if (np.mean(max_scores_window) >= config.target_score and 
            episode >= config.scores_window_len):
            print('\nEnvironment solved in {} episodes!!!'.format(episode))
            print('Average score over {} episodes was: {:.3f}'
                  .format(config.scores_window_len, np.mean(max_scores_window)))
            print('Time Elapsed: {:.1f} min'.format((tic-toc)/60))
            path = f'checkpoint-{episode}.tar'
            save_checkpoint(agent, scores_per_episode, steps_per_episode, path)
            break
        
    print('Time Elapsed: {:.1f} min'.format((tic-toc)/60))
    
    return (agent, scores_per_episode, steps_per_episode)

        
      

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