# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 2
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 26th October 2020, by alessior@kth.se
#

# Load packages
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple



class Agent(object):
    ''' Base agent class

        Args:
            n_actions (int): actions dimensionality

        Attributes:
            n_actions (int): where we store the dimensionality of an action
    '''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        pass

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> np.ndarray:
        ''' Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent.
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)

class ExperienceReplayBuffer(object):
    """ Class used to store a buffer containing experiences of the RL agent.
    """
    def __init__(self, maximum_length):
        # Create buffer of maximum length
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        # Append experience to the buffer
        self.buffer.append(experience)

    def __len__(self):
        # overload len operator
        return len(self.buffer)

    def sample_batch(self, n):
        """ Function used to sample experiences from the buffer.
            returns 5 lists, each of size n. Returns a list of state, actions,
            rewards, next states and done variables.
        """
        # If we try to sample more elements that what are available from the
        # buffer we raise an error
        if n > len(self.buffer):
            raise IndexError('Tried to sample too many elements from the buffer!')

        # Sample without replacement the indices of the experiences
        # np.random.choice takes 3 parameters: number of elements of the buffer,
        # number of elements to sample and replacement.
        indices = np.random.choice(
            len(self.buffer),
            size=n,
            replace=False
        )

        # Using the indices that we just sampled build a list of chosen experiences
        batch = [self.buffer[i] for i in indices]

        # batch is a list of size n, where each element is an Experience tuple
        # of 5 elements. To convert a list of tuples into
        # a tuple of list we do zip(*batch). In this case this will return a
        # tuple of 5 elements where each element is a list of n elements.
        #return batch
        return zip(*batch)


class ActorNetwork(nn.Module): #Actor is policy
    """ Actor feedforward neural network """
    def __init__(self, dev, input_size, output_size=2):
        super().__init__()

        #Define number of neurons: given by the exercise
        num_neurons_l1 = 400
        num_neurons_l2 = 200

        # INPUT: state
        self.input_layer = nn.Linear(input_size, num_neurons_l1, device=dev)
        self.input_layer_activation = nn.ReLU()

        self.hidden_layer = nn.Linear(num_neurons_l1, num_neurons_l2, device=dev)
        self.hidden_layer_activation = nn.ReLU()

        # OUTPUT: action (2 dimensions, between -1 and 1)
        self.output_layer = nn.Linear(num_neurons_l2, output_size, device=dev)
        self.output_layer_activation = nn.Tanh()

    def forward(self, x):
        # Computation policy(s)

        # Compute first layer
        l1 = self.input_layer(x)
        l1 = self.input_layer_activation(l1)

        l2 = self.hidden_layer(l1)
        l2 = self.hidden_layer_activation(l2)

        # Compute output layer
        out = self.output_layer(l2)
        out = self.output_layer_activation(out)

        return out



class CriticNetwork(nn.Module): #Critic is Q
    """ Critic feedforward neural network """
    def __init__(self, dev, input_size, action_size, output_size=1):
        super().__init__()
        # Define number of neurons: given by the exercise
        num_neurons_l1 = 400
        num_neurons_l2 = 200

        # INPUT: state
        self.input_layer = nn.Linear(input_size, num_neurons_l1, device=dev)
        self.input_layer_activation = nn.ReLU()

        # Concatenate action
        self.hidden_layer = nn.Linear(num_neurons_l1+action_size, num_neurons_l2, device=dev)
        self.hidden_layer_activation = nn.ReLU()

        # OUTPUT: Q(s,a) (1 dimension)
        self.output_layer = nn.Linear(num_neurons_l2, output_size, device=dev)

    def forward(self, x, a):
        # Computation of Q(s,a)

        # Compute first layer
        l1 = self.input_layer(x)
        l1 = self.input_layer_activation(l1)

        # Concatenate output first layer + action
        concat = torch.cat([l1, a], dim=1)

        l2 = self.hidden_layer(concat)
        l2 = self.hidden_layer_activation(l2)

        # Compute output layer
        out = self.output_layer(l2)

        return out



class DDPGAgent(object):
    ''' Base agent class'''

    def __init__(self, batch_size, discount_factor, lr_actor, lr_critic, action_size, dim_state, mu, sigma, dev):
        #Parameters
        self.dev = dev

        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.dim_state = dim_state
        self.action_size = action_size
        self.prev_noise = np.array([0,0])
        self.mu = mu
        self.sigma = sigma

        #Critic networks (Q(s,a))
        self.main_critic = CriticNetwork(self.dev, dim_state, action_size)
        self.target_critic = CriticNetwork(self.dev, dim_state, action_size)

        #Actor networks (policy(s))
        self.main_actor = ActorNetwork(self.dev, dim_state)  #main ANN: ANN to update in every batch size
        self.target_actor = ActorNetwork(self.dev, dim_state) #final ANN

        #OPITMIZER
        self.optimizer_critic = optim.Adam(self.main_critic.parameters(), lr=self.lr_critic) #I put the optimizer on the main_ann 'cause is the one we are gonna train
        self.optimizer_actor = optim.Adam(self.main_actor.parameters(), lr=self.lr_actor)

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation with added noise '''
        w = np.random.normal(0, self.sigma, size=2)
        n = -self.mu*self.prev_noise + w
        a = self.main_actor.forward(torch.tensor(state, device=self.dev)).detach().cpu().numpy() + n
        self.prev_noise = n

        a = np.clip(a,-1,1).reshape(-1) # It can happen that the noise makes it go outside this range
        return a

    def backward_critic(self, buffer):
        ''' Performs a backward pass on the critic network '''
        # Sample a batch of batch_size elements
        states, actions, rewards, next_states, dones = buffer.sample_batch( n=self.batch_size)

        # Training process, set gradients to 0
        self.optimizer_critic.zero_grad()

        # Compute target Q values
        with torch.no_grad():
            target_next_actions = self.target_actor.forward(torch.tensor(next_states, device=self.dev))
            target_Qs = self.target_critic.forward(torch.tensor(next_states, device=self.dev),target_next_actions)
            y_values = (torch.tensor(rewards, device=self.dev, dtype=torch.float32)) + (self.discount_factor*target_Qs.squeeze())*(torch.tensor(dones, device=self.dev)==False)

        # Compute main Q values used for the gradient
        states = torch.tensor(states, device=self.dev,requires_grad=True, dtype=torch.float32)
        actions = torch.tensor(actions, device=self.dev,requires_grad=True, dtype=torch.float32)
        q_values = self.main_critic.forward(states, actions).squeeze()

        # Compute loss function
        loss = nn.functional.mse_loss(q_values, y_values)

        # Compute gradient
        loss.backward()

        # Clip gradient norm to 1
        nn.utils.clip_grad_norm_(self.main_critic.parameters(), max_norm=1.)

        # Perform backward pass (backpropagation)
        self.optimizer_critic.step()
        return


    def backward_actor(self, buffer):
        ''' Performs a backward pass on the critic network '''
        # Sample a batch of batch_size elements
        states, actions, rewards, next_states, dones = buffer.sample_batch(n=self.batch_size)

        # Training process, set gradients to 0
        self.optimizer_actor.zero_grad()

        # Compute main Q values used for the gradient
        states = torch.tensor(states, device=self.dev, requires_grad=True, dtype=torch.float32)
        policy_actions = self.main_actor.forward(states)
        q_values = self.main_critic.forward(states,policy_actions).squeeze()

        # Compute loss function
        loss = -torch.mean(q_values)

        # Compute gradient
        loss.backward()

        # Clip gradient norm to 1
        nn.utils.clip_grad_norm_(self.main_actor.parameters(), max_norm=1.)

        # Perform backward pass (backpropagation)
        self.optimizer_actor.step()


    def save_ann(self, main_nn, target_nn, filename_main='neural-network-main-1.pth',filename_target='neural-network-target-1.pth'):
        '''Save network in working directory'''
        torch.save(main_nn, filename_main)
        print(f'Saved main_network as {filename_main}')
        torch.save(target_nn, filename_target)
        print(f'Saved main_network as {filename_target}')
        return


