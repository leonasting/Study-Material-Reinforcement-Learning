import gymnasium as gym
import numpy as np
from actor_critic_torch import Agent
from utils import plot_learning_curve
import torch
if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, lr=5e-6, input_dims=[8], n_actions=4,
                  fc1_dims=2048, fc2_dims=1536)
    n_games = 3000

    fname = 'ACTOR_CRITIC_' + 'lunar_lander_' + str(agent.fc1_dims) + \
            '_fc1_dims_' + str(agent.fc2_dims) + '_fc2_dims_lr' + str(agent.lr) +\
            '_' + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    scores = []
    for i in range(n_games):
        done = False
        observation, info = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            agent.learn(observation, reward, observation_, done)
            observation = observation_
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)

    x = [i+1 for i in range(n_games)]
    torch.save(agent.actor_critic.state_dict(), 'lunar_agent_actor_critic.pth')
    plot_learning_curve(x, scores, figure_file)
    
