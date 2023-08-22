from math import log
import gymnasium as gym
import numpy as np
from td3_torch import Agent
from utils import plot_learning_curve
import datetime
import os
def create_unique_filename(base_filename):
    if not os.path.exists(base_filename):
        return base_filename

    filename, ext = os.path.splitext(base_filename)
    serial_number = 1
    while True:
        new_filename = f"{filename}_{serial_number}{ext}"
        if not os.path.exists(new_filename):
            return new_filename
        serial_number += 1


if __name__ == '__main__':
    env = gym.make('BipedalWalker-v3')
    #env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(alpha=0.001, beta=0.001, 
                input_dims=env.observation_space.shape, tau=0.005,
                env=env, batch_size=100, layer1_size=400, layer2_size=300,
                n_actions=env.action_space.shape[0])
    n_games = 1500
    filename = 'Walker2d_' + str(n_games) + '_2.png'
    figure_file = 'plots/' + filename
    
    logfile = 'Walker2d_' + str(n_games) + '_2.log'
    log_file = 'logs/' + logfile
    figure_file = create_unique_filename(figure_file)
    log_file = create_unique_filename(log_file)


    best_score = env.reward_range[0]
    score_history = []
    avg_score_history = []
    #agent.load_models()

    for i in range(n_games):
        observation,info = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_score_history.append(avg_score)
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.2f' % score,
                'trailing 100 games avg %.3f' % avg_score)
        open(log_file, 'a').write(str(datetime.datetime.now())+","+str(i)+","+str(score)+","+str(avg_score)+"\n")

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
