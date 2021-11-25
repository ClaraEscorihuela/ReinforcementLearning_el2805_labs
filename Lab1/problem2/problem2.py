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
# Course: EL2805 - Reinforcement Learning - Lab 1 Problem 4
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
import pickle

# Functions used during training
def running_average(x, N):
    ''' Function used to compute the running mean
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def scale_state_variables(s, low, high):
    ''' Rescaling of s to the box [0,1]^2 '''
    x = (s - low) / (high - low)
    return x

def basis_function(eta, state):

    "Function that will transform the states to the new basis"
    basis = np.zeros((np.shape(eta)[0]))
    for i in range (np.shape(eta)[0]):
        #base_i = np.cos(np.pi*np.dot(np.transpose(eta[:,i]),state))
        base_i = np.cos(np.pi * np.matmul(np.transpose(eta[:, i]), state))
        basis[i] = base_i
    return basis.transpose()

def Q_calculation(w, basis, action):
    """Function to calculate Q new"""
    Q_new = np.dot(np.transpose(w[:,action]),basis)
    return Q_new

def choose_action(epsilon, w, basis):
    """Function to choose action"""
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.choice(np.arange(k))
    else:
        action = np.argmax([Q_calculation(w,basis,action) for action in range(k)])
    return action

def scale_basis(eta, alpha):
    """Function to scale the Fourier Basis"""
    norm_eta = np.sqrt()
    norm_eta[norm_eta == 0] = 1  # if ||eta_i||=0 then alpha_i=alpha
    alpha = np.divide(alpha, norm_eta)

    return alpha

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
state = env.reset()
k = env.action_space.n      # tells you the number of actions (3)
low, high = env.observation_space.low, env.observation_space.high #tells us the low, high limits for the state (state == pos,vel)

# Parameters
N_episodes = 500        # Number of episodes to run for training
discount_factor = 1.    # Value of gamma
n = state.size #== 2
gamma = 1 #discount factor 
lamda = 0.01 #eligibility_parameter
alpha = 0.1

# Reward
episode_reward_list = []  # Used to save episodes reward

# Training process
eta = np.array([[1,0],[0,2],[0,1],[2,0],[1,1],[2,2]])
eta_transpose = np.array([[0,0],[1,0],[0,2],[0,1],[2,0],[1,1],[2,2]]).transpose()
w = np.zeros((np.shape(eta_transpose)[0],k))
for i in range(N_episodes):
    # Reset enviroment data
    done = False
    scaling_bases = False
    state = scale_state_variables(env.reset(), low, high)
    eta_transpose = np.array([[0,0],[1,0],[0,2],[0,1],[2,0],[1,1],[2,2]]).transpose()
    eta = np.array([[0,0],[1,0],[0,2],[0,1],[2,0],[1,1],[2,2]])
    total_episode_reward = 0.
    epsilon = 0.01
    z = np.zeros((np.shape(eta_transpose)[0],k))

    gamma = 1
    lamda = 0.9

    if not scaling_bases:
        alpha = 0.5
    else:
        alpha = scale_basis(eta, alpha)

    while not done:
        # Take an action
        # env.action_space.n tells you the number of actions
        # available
        basis = basis_function(eta_transpose, state)
        action = choose_action(epsilon, w, basis)
            
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action)
        next_state = scale_state_variables(next_state, env.observation_space.low,env.observation_space.high)

        # Choose the next action
        next_basis = basis_function(eta, next_state)
        next_action = choose_action(epsilon, w, next_basis)

        #Update z
        z = gamma*lamda* z
        z[:,action] += basis
        z = np.clip(z, -5, 5)

        #Update w
        delta = reward + gamma*Q_calculation(w, next_basis, next_action)-Q_calculation(w,basis,action)
        w = w + alpha*delta*z


        # Update episode reward
        total_episode_reward += reward
        # Update state for next iteration
        state = next_state


    # Append episode reward
    episode_reward_list.append(total_episode_reward)

    # Close environment
    env.close()

with open('weights.pkl', 'wb') as f:
    pickle.dump(w,f)

# Plot Rewards
plt.plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
plt.plot([i for i in range(1, N_episodes+1)], running_average(episode_reward_list, 10), label='Average episode reward')
plt.xlabel('Episodes')
plt.ylabel('Total reward')
plt.title('Total Reward vs Episodes')
plt.legend()
plt.grid(alpha=0.3)
plt.show()