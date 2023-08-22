import os
import numpy as np
import torch as T # pytorch
import torch.nn as nn # neural network
import torch.nn.functional as F # activation functions
import torch.optim as optim # optimizer - Adam 

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, \
                 chkpt_dir='tmp/ddpg'):
        """
        beta - learning rate
        name - Checkpoint file name
        """
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ddpg')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # In the paper they have used batch normalization to deal with multple input scales
        """
        Layer Norm used as alternative to Batch Norm
        It normalizes the different scales of the inputs.

        So it won't hurt the extensibility, but it has the added bonus that it is independent of the batch
        size, whereas the batch norm depends on the batch size.

        Another technical detail is that when you go to copy parameters with the model, when you're doing the
        transfer of parameters from the regular to the target networks, the batch norm doesn't seem to keep
        track of the running mean and running variants when you do that.
        So you get an error and so you have to use a flag in the load state dict function, which is strict
        equals false.
        So it doesn't seem like it's working 100% of the way it is intended.
        So the layer norm deals with that problem, has the exact same properties we want and is a little bit
        faster to boot, so I'm going to use that.
        """
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        #self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        #self.bn2 = nn.BatchNorm1d(self.fc2_dims)


        """
        Next, we need a layer to handle the input of our action values.
        And this is a totally separate input layer.
        And this deals with the verbiage in the paper where they say that the actions are not included until
        the second hidden layer.
        And finally, we're going to need a layer to join everything together to get the actual critic values.
        """
        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        
        self.q = nn.Linear(self.fc2_dims, 1)

        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)

        f4 = 1./np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)

        self.optimizer = optim.Adam(self.parameters(), lr=beta,
                                    weight_decay=0.01)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        """
        Functions to translate state and action in q values

        Now there is a bit of debate here, so the debate centers around the question of should you activate
        the function before or after the batch normalization?
        I don't have a definitive answer to this.
        I haven't played around with it a whole lot.
        I believe it works both ways.
        However, I've opted to opted to do it this way for the reason that the Relu introduces a non-linearity
        and lops off everything in the negative domain.
        While what if those values in the negative domain are actually useful?
        You know, maybe we're dealing with velocities or something of that nature where they could be negative,
        right, if they're in the opposite direction.
        So to me, lopping off all of the negative values before normalizing probably loses some important information
        about the environment.
        And so I opt to activate it after the batch normalization.
        So then we proceed with passing the values through the second fully connected layer.

        I'm not going to activate it yet.
        What I'm going to do first is go ahead and introduce the action value at this time.
        And then I'm going to go ahead and add the state values and action values and activate those.
        Now, there is a bit of a debate here as well.
        So if you look up solutions on the Internet, some people will concatenate these instead of adding them.
        I do not believe that to be correct.
        And the reason is that if you think of the table.
        The table should have shape number of rows where the number of rows corresponds to the number of states
        you're dealing with, and the number of columns corresponds to the value of each action for that particular
        state.
        And so there's going to be some sort of baseline value to that for each state.
        And then the value of the action over that tells you what you're going to gain by taking that action.
        Put it another way.
        If you were to concatenate them, you're going to end up with something with N actions plus one in shape,
        right?
        Because you're going to have state values added on to that.
        And of course, that is going to give you something a little bit wonky and dimensionality wise.
        Doesn't quite make sense.
        """


        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        #state_value = F.relu(state_value) # To avoid loosing negative values
        #action_value = F.relu(self.action_value(action))
        action_value = self.action_value(action)
        state_action_value = F.relu(T.add(state_value, action_value)) # to have consistent shape as the state
        #state_action_value = T.add(state_value, action_value)
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        T.save(self.state_dict(), checkpoint_file)

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ddpg')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        #self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        #self.bn2 = nn.BatchNorm1d(self.fc2_dims)

        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f3 = 0.003
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device =T.device('cuda' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = T.tanh(self.mu(x)) # Tanhh used to bound the actions between -1 and 1

        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        T.save(self.state_dict(), checkpoint_file)
