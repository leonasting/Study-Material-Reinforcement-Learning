import numpy as np

class Agent():
    def __init__(self, gamma=0.99):
        """
        gamma: discount factor for future rewards
        """
        self.V = {} # Agents estimated value of each state/ State value function
        self.sum_space = [i for i in range(4, 22)]# Possible sums of player's cards
        self.dealer_show_card_space = [i+1 for i in range(10)] # Possible values of dealer's showing card
        self.ace_space = [False, True]# Players usable ace or not
        self.action_space = [0, 1] # stick or hit

        self.state_space = []# 3 tuple of player's sum, dealer's showing card and usable ace or not permutation
        self.returns = {}# Returns that followed visit to each state - Key is 3 value tuple 
        self.states_visited = {} # first visit or not for each state - Key is 3 value tuple 
        self.memory = [] # States encountered and rewards received
        self.gamma = gamma

        self.init_vals()

    def init_vals(self):
        """
        Initializes default values for each state
        for state_space, returns, states_visited and V
        """
        for total in self.sum_space:
            for card in self.dealer_show_card_space:
                for ace in self.ace_space:
                    self.V[(total, card, ace)] = 0 # Only state matters is terminal state rest is arbitrary
                    self.returns[(total, card, ace)] = []
                    self.states_visited[(total, card, ace)] = 0
                    self.state_space.append((total, card, ace))

    def policy(self, state):
        """
        Current state is the input
        """
        total, _, _ = state
        action = 0 if total >= 20 else 1
        return action


    def update_V(self):
        for idt, (state, _) in enumerate(self.memory):
            G = 0
            if self.states_visited[state] == 0:
                self.states_visited[state] += 1
                discount = 1
                for t, (_, reward) in enumerate(self.memory[idt:]):
                    G += reward * discount
                    discount *= self.gamma
                    self.returns[state].append(G)

        for state, _ in self.memory:
            self.V[state] = np.mean(self.returns[state])

        for state in self.state_space:
            self.states_visited[state] = 0

        self.memory = []