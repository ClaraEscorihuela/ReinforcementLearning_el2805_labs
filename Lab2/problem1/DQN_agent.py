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
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th October 2020, by alessior@kth.se
#

# Load packages
import gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

class Agent(object):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''
    def __init__(self, network):
        self.network = torch.load(network) #The netwok will be the trained network

    def choose_action(self, state: np.ndarray):
        ''' Performs a forward computation '''
        q_values = self.network(torch.tensor([state])) #With the netwok we have trained we look fo the best action
        action = q_values.max(1)[1].item()
        return action




class RandomAgent():
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def choose_action(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action

### Neural Network ###
class MyNetwork(nn.Module): #--> TA Github
    """ Create a feedforward neural network """
    def __init__(self, input_size, output_size):
        super().__init__()

        #Define number of neurons: we should try different number of neurons (he ficat 16 i32 pq diuen entre 8 i 128)
        num_neurons_l1 = 64
        num_neurons_l2 = 64

        # Create input layer with ReLU activation
        self.input_layer = nn.Linear(input_size, num_neurons_l1) #inside: (num_inputs = num_outputs_before,num_output)
        self.input_layer_activation = nn.ReLU()

        self.hidden_layer = nn.Linear(num_neurons_l1, num_neurons_l2)  # inside: (num_inputs = num_outputs_before,num_output)
        self.hidden_layer_activation = nn.ReLU()

        # Create output layer
        self.output_layer = nn.Linear(num_neurons_l2, output_size)

    def forward(self, x):
        # Function used to compute the forward pass

        # Compute first layer
        l1 = self.input_layer(x)
        l1 = self.input_layer_activation(l1)

        l2 = self.hidden_layer(l1)
        l2 = self.hidden_layer_activation(l2)

        # Compute output layer
        out = self.output_layer(l2)
        return out

### Experience class ###
Experience = namedtuple('Experience',['state', 'action', 'reward', 'next_state', 'done'])

class ExperienceReplayBuffer(object): #-> TA GITHUBG
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
        #return batch #I have changed it for the starting point of the training
        return zip(*batch)


class DQNAgent(): #class Agent
    def __init__(self, batch_size, discount_factor, lr, num_actions, dim_state):
        #Parameters
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.learning_rate = lr
        self.num_actions = num_actions #not really sure how to do this part
        self.dim_state = dim_state  # not really sure how to do this part

        #ANN
        self.main_ann = MyNetwork(dim_state, num_actions)  #main ANN: ANN to update in every batch size
        self.target_ann = MyNetwork(dim_state, num_actions) #final ANN

        #OPITMIZER
        self.optimizer = optim.Adam(self.main_ann.parameters(), lr=self.learning_rate) #I put the optimizer on the main_ann 'cause is the one we are gonna train

    def choose_action(self, state, epsilon):
        """Function to choose action once the agent is already trained"""
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(self.num_actions)
        else:
            q_values = self.main_ann(torch.tensor([state]))
            action = q_values.max(1)[1].item()

        return action


    def train(self, buffer):
        ### TRAINING ###
        # Perform training only if we have more than batch_szie elements in the buffer
        # Sample a batch of batch_size elements
        states, actions, rewards, next_states, dones = buffer.sample_batch(
            n=self.batch_size)

        # Training process, set gradients to 0
        self.optimizer.zero_grad()

        # Compute output of the target network given the states batch
        qvalues_targ_ann = self.target_ann(torch.tensor(next_states,
                                      requires_grad=True,
                                      dtype=torch.float32)) #NEXT STATES! DIF FROM TA

        #Compute the maximum action of the target network
        acmax_targ_ann = qvalues_targ_ann.max(1)[0]

        #Compute y values: IMPORTANT: TRANSFORM TUPLE TO TENSOR!
        y_values = (torch.tensor(rewards,dtype=torch.float32)) + (self.discount_factor*acmax_targ_ann)*(torch.tensor(dones)==False)

        # Compute output
        q_values = self.main_ann.forward(torch.tensor(states,
                                                          requires_grad=True,
                                                          dtype=torch.float32)
                                             ).gather(1, torch.tensor(actions).unsqueeze(1))
        q_values = q_values.reshape(-1)  # collapse one dim

        # Compute loss function
        loss = nn.functional.mse_loss(q_values,y_values) #(input, target: el profe fica com a input els valors q surten de q(s,a) ara mateix)

        # Compute gradient
        loss.backward()

        # Clip gradient norm to 1
        nn.utils.clip_grad_norm_(self.main_ann.parameters(), max_norm=1.) #Recomndation and fone by the profesor, we clip the main network, which is the one used for training

        # Perform backward pass (backpropagation)
        self.optimizer.step()

    def update_ann(self):
        self.target_ann.load_state_dict(self.main_ann.state_dict()) #https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c
        print('I am updating Q')

    def save_ann(self, filename_main='neural-network-main-1.pth',filename_target='neural-network-target-1.pth'):
        '''Save network in working directory'''
        torch.save(self.main_ann, filename_main)
        print(f'Saved main_network as {filename_main}')
        torch.save(self.main_ann, filename_target)
        print(f'Saved main_network as {filename_target}')
        return





