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
# Last update: 20th November 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DDPG_agent import *
from DDPG_soft_updates import soft_updates

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(dev)

#Parameters to save the network
legend_main_critic = 'main_critic.pth'
legend_target_critic = 'target_critic.pth'
legend_main_actor = 'main_actor.pth'
legend_target_actor = 'target_actor.pth'



def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

# Import and initialize Mountain Car Environment
env = gym.make('LunarLanderContinuous-v2')
env.reset()

# Parameters
N_episodes = 300                # Number of episodes to run for training
discount_factor = 0.99          # Value of gamma
n_ep_running_average = 50       # Running average of 50 episodes
m = len(env.action_space.high)  # dimensionality of the action
dim_state = len(env.observation_space.high)  # State dimensionality
lr_actor = 5e-5                 # For actor model
lr_critic = 5e-4                # For actor model
batch_size = 64                 # N
buffer_size = 30000             # L
sync_target_frames = buffer_size / batch_size  # C
tau = 0.001                     #soft update parameter
d = 2

#Noise parameters
mu = 0.15
sigma = 0.2


# Reward
episode_reward_list = []  # this list contains the total reward per episode
episode_number_of_steps = []  # this list contains the number of steps per episode

# Experience buffer initialization
agent = RandomAgent(m)
buffer = ExperienceReplayBuffer(maximum_length=buffer_size)
while len(buffer) < buffer_size:
    # Reset enviroment data
    done = False
    state = env.reset()
    while not done:
        # Take an action
        action = agent.forward(state)
        # Get next state, reward, done and append to buffer
        next_state, reward, done, _ = env.step(action)
        buffer.append((state, action, reward, next_state, done))
        # Update state for next iteration
        state = next_state



# DQN agent initialization
agent = DDPGAgent(batch_size, discount_factor, lr_actor, lr_critic, m, dim_state, mu, sigma, dev)

# Training process
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

for i in EPISODES:
    # Reset environment data
    done = False
    state = env.reset()
    total_episode_reward = 0.
    t = 0


    while not done:
        # Take a random action
        action = agent.forward(state)

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action)
        buffer.append((state, action, reward, next_state, done))

        # Train in case the buffer is bigger than the batch size
        if len(buffer) > batch_size:
            agent.backward_critic(buffer)

            if t % d == 0:
                agent.backward_actor(buffer)

                #Soft updates
                agent.target_actor = soft_updates(agent.main_actor, agent.target_actor, tau)
                agent.target_critic = soft_updates(agent.main_critic, agent.target_critic, tau)

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t+= 1

    # Append episode reward
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)
    # Close environment
    env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, n_ep_running_average)[-1],
        running_average(episode_number_of_steps, n_ep_running_average)[-1]))

agent.save_ann(agent.main_actor,agent.target_actor,filename_main=legend_main_actor,filename_target=legend_target_actor)
agent.save_ann(agent.main_critic,agent.target_critic,filename_main=legend_main_critic,filename_target=legend_target_critic)

# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.savefig("plot.png")