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
# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()
k = env.action_space.n      # tells you the number of actions
low, high = env.observation_space.low, env.observation_space.high #tells us the low, high limits for the state (state == pos,vel)

# Parameters
N_episodes = 100        # Number of episodes to run for training
discount_factor = 1.    # Value of gamma


# Reward
episode_reward_list = []  # Used to save episodes reward

#----------------------------------------------------------------------------------------------------------------------
# CREATE FOURIER BASIS
#-----------------------------------------------------------------------------------------------------------------------

def basis_function(eta, state):

    "Function that will transform the states to the new basis"
    basis = np.zeros((np.shape(eta)[0]))
    for i in range (np.shape(eta)[0]):
        base_i = np.cos(np.pi*np.dot(np.transpose(eta[:,i]),state))
        basis[i] = base_i
    return basis.transpose()

def Q_calculation(w, basis, action):
    Q_new = np.dot(np.transpose(w[:,action]),basis)
    return Q_new




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

def scale_state_variables(s, low=env.observation_space.low, high=env.observation_space.high):
    ''' Rescaling of s to the box [0,1]^2 '''
    x = (s - low) / (high - low)
    return x

def choose_action(epsilon, w, basis):
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.choice(np.arange(k))
    else:
        action = np.argmax([Q_calculation(w,basis,action) for action in range(k)])
    return action

# Training process
for i in range(N_episodes):
    # Reset enviroment data
    done = False
    state = scale_state_variables(env.reset())
    eta = np.array([[0,0],[1,0],[1,0],[1,1]]).transpose()
    total_episode_reward = 0.
    epsilon = 0.01
    w = np.zeros((np.shape(eta)[0],k))
    z = np.zeros((np.shape(eta)[0],k))
    gamma = 1
    lamda = 0.9
    alpha = 0.01

    while not done:
        # Take an action
        # env.action_space.n tells you the number of actions
        # available
        env.render()
        basis = basis_function(eta, state)
        action = choose_action(epsilon, w, basis)
            
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action)
        next_state = scale_state_variables(next_state)

        # Choose the next action
        next_basis = basis_function(eta, next_state)
        next_action = choose_action(epsilon, w, next_basis)

        #Update z
        z[:,action] = gamma*lamda* z[:,action]+basis

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