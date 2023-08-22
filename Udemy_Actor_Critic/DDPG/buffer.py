import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        """
        max_size - maximum size of the buffer
        input_shape - shape of the input observation
        n_actions - number of actions
        """
        self.mem_size = max_size
        self.mem_cntr = 0 # keep track of how many transitions we've stored
        self.state_memory = np.zeros((self.mem_size, *input_shape))# *input_shape is the same as input_shape[0], input_shape[1] - Multi dimensional array
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))# actions - continous numbers, dimensions are dependent on the count of components
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype='bool')

    def store_transition(self, state, action, reward, state_, done):
        """
        state - current state
        state_ - next state
        """
        index = self.mem_cntr % self.mem_size# TO assign position in the memory ,% as wrapper
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)# batch sampling from the memory indexes

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


