import gym

import random
import torch as T
import numpy as np

import torch.nn as nn # Base module
import torch.nn.functional as F #All the activation functions
from torch import optim# optimizers class

#Source:https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/actor_critic/actor_critic_continuous.py

### Creating a generic Network based on RL with Phil for mountain car

# All pytorch classes derive from nn.Module (base class for pytorch)
class GenericNetwork(nn.Module):

    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        """
        lr=  Learning Rate of Actor (Actor learns to take action based on state at t with  Parameter theta:Sutton)
        input_dims = Observation provided from the enironment
        fc1 = Dimension for the first fully connected layer of hidden deep neural network
        fc2 = Dimension for second fully connected layer of hidden deep neural network

        """
        super(GenericNetwork, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        """ 
        below are th definition of deep neural network using linear transformations with random initialisation for
        weight matrix for indivifual linear transformation matrix 
        """
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        # *self.. is used to unpack the input dims so that the same code would be able to handle multi dimensition
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # seld.parameters comes from self.nn

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        # Sends entire network to device

    def forward(self, observation):
        """
        Actors feed forward function

        """
        state = T.tensor(observation, dtype=T.float).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Final layer output will not have activations
        # Range of values permissible for state value function and actions are totally different.
        # We will define the activation function later

        return x

    # Agent Class

class Agent(object):
    def __init__(self, alpha, beta, input_dims, gamma=0.99, n_actions=2,
                 layer1_size=64, layer2_size=64, n_outputs=1):
        """
        alpha = learning reward from actor
        beta = learning reward from critics
        gamma = discount for future reward
        layer_1 =
        """

        self.gamma=gamma
        self.log_probs =None
        # Log probs will be used in calculation of loss function the actor network
        """
        log of the probability of selecting an action
        Actor -is deep neural network estimates of the probability
        of choosing the action given you are at some state or the policy
        When we dealing with probability and when we do the gradient of that we end with log term 
        """
        self.n_outputs = n_outputs
        self.actor = GenericNetwork(alpha, input_dims, layer1_size,
                                    layer2_size, n_actions=n_actions)
        self.critic = GenericNetwork(beta, input_dims, layer1_size,
                                    layer2_size, n_actions=1)# Real valued scalar number for critiv
        """
        Both actor and critic can have different hidden layers
        For continuous problem we usually have only 1 action .
        We can choose the action by probablity distribution
        Normal distribution - defined for Mean and standard distribution
        thats why 2 actions for actor are mean and standard distribution
        
        Critic Network is tasked with calculating value for the particular state.
        """

    def choose_actions(self,observation):
        mu, sigma = self.actor.forward(observation)
        """
        sigma can not negative
        """
        sigma = T.exp(sigma)
        action_probs = T.distributions.Normal(mu, sigma)
        # Agent want to lean mu and sigma  to maximize reqard over time
        probs = action_probs.sample(sample_shape=T.Size([self.n_outputs]))# n_outputs=1 without bound
        self.log_probs = action_probs.log_prob(probs).to(self.actor.device)
        action = T.tanh(probs)# Bounding the limit to -1 to 1

        return action.item()
        # item gets values as open ai gym does not take tensor


    """
    Learning Function 
    It is in Temporal Style
    Similar to Q - Learning
    It is in contrast to policy gradient.
    In policy gradient method the style is Monta Carlo Style:
    Where we accumulate the memories over the course of episode 
    Replay the memories to training the network.
    
    Replay Memories samples randomly
    """

    def learn(self, state, reward, new_state, done):
        """
        done flag: Terminal state has been reached.
        """
        self.actor.optimizer.zero_grad()
        # We dont want to impact the gradient of current sample from previous gradient
        self.critic.optimizer.zero_grad()

        critic_value_ = self.critic.forward(new_state)
        critic_value = self.critic.forward(state)

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)

        delta = reward + self.gamma*critic_value*(1-int(done)) - critic_value_
        # int()done ensures we dont accumulate extra reward

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2
        """
        - Previous video
        critic loss - 
        delta- difference between next state - previous state
        USe that to guide the evolution of the probabilities in the policy space over time 
        with actor loss and critic loss just seek to minimize the quantity
        Delta **2 it is strictly positive
        
        
        """
        (actor_loss + critic_loss).backward()# we can have only one back probagation
        self.actor.optimizer.step()
        self.critic.optimizer.step()
        # without above step it wont learn


