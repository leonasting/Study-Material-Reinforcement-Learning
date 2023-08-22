import numpy as np
import gymnasium as gym
# This script is used to demonstrate TD(0) prediction of value dunction for CartPole-v1


def simple_policy(state):
    """
    Consumes discretized state and returns action
    """
    action = 0 if state < 5 else 1
    return action

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    alpha = 0.1
    gamma = 0.99

    states = np.linspace(-0.2094, 0.2094, 10)# Dicretizing the state space into 10 states
    V = {}
    for state in range(len(states)+1):
        V[state] = 0

    for i in range(5000):
        observation, info = env.reset()
        done = False
        while not done:
            state = int(np.digitize(observation[2], states))
            # Filter over current State space to get into discrete state space
            action = simple_policy(state)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state_ = int(np.digitize(observation_[2], states))
            V[state] = V[state] + alpha*(reward + gamma*V[state_] - V[state])
            observation = observation_

    for state in V:
        print(state, '%.3f' % V[state])