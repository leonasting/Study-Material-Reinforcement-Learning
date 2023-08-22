import numpy as np

# Q learning agent which works with dicretized state space and action space only.

class Agent():
    def __init__(self, lr, gamma, n_actions, state_space, eps_start, eps_end,
                 eps_dec):
        """
        lr: learning rate for Q learning which is alpha
        gamma: discount factor for future rewards
        n_actions: number of actions
        state_space: state space of the environment which is 4 tuple dicretized
                        in case of CartPole-v1 to make it dicrete space problem
        eps_start: starting epsilon value for epsilon-greedy policy
        eps_end: minimum epsilon value for epsilon-greedy policy
        eps_dec: epsilon decrement value for epsilon-greedy policy

        Note: linear epsilon decay is used
        
        """
        self.lr = lr
        self.gamma = gamma
        self.actions = [i for i in range(n_actions)]
        self.states = state_space
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec

        self.Q = {}

        self.init_Q()

    def init_Q(self):
        for state in self.states:
            for action in self.actions:
                self.Q[(state, action)] = 0.0

    def max_action(self, state):
        actions = np.array([self.Q[(state, a)] for a in self.actions])
        action = np.argmax(actions)

        return action

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            action = self.max_action(state)

        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                if self.epsilon>self.eps_min else self.eps_min

    def learn(self, state, action, reward, state_):
        a_max = self.max_action(state_)

        self.Q[(state, action)] = self.Q[(state, action)] + self.lr*(reward +
                                        self.gamma*self.Q[(state_, a_max)] -
                                        self.Q[(state, action)])