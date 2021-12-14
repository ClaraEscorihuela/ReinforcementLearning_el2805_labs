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
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import *

#Parameters for epsilon measurments
min_epsilon = 0.05
max_epsilon = 0.99
Z = 0.925

#Parameters to save the network
legend_main = 'main_network_150ep_disc0.98.pth'
legend_target = 'target_network_150ep_disc0.98.pth'
comparative = True

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

def epsilon_decay_exponencial(k, N_episodes):
    epsilon = max(min_epsilon, max_epsilon * (min_epsilon / max_epsilon) ** ((k - 1) / (N_episodes * Z - 1)))
    return epsilon


def main():

    #dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("Using", dev)

    # Import and initialize the discrete Lunar Laner Environment
    env = gym.make('LunarLander-v2')
    env.reset()

    # Parameters
    N_episodes = 300                             # Number of episodes
    discount_factor = 0.98                       # Value of the discount factor
    n_ep_running_average = 50                    # Running average of 50 episodes
    n_actions = env.action_space.n               # Number of available actions
    dim_state = len(env.observation_space.high)  # State dimensionality

    lr = 0.0005
    batch_size = 50                              # N
    buffer_size = 10000                          # L
    sync_target_frames = buffer_size/batch_size  # C

    episode_reward_list = []       # this list contains the total reward per episode
    episode_number_of_steps = []   # this list contains the number of steps per episode

    ### Training process
    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

    # Experience buffer initialization
    buffer = ExperienceReplayBuffer(maximum_length=buffer_size)

    # Random agent initialization
    # agent = RandomAgent(n_actions)

    # DQN agent initialization
    agent = DQNAgent(batch_size, discount_factor, lr, n_actions, dim_state)

    for i in EPISODES:

        # Reset enviroment data and initialize variables
        done = False
        state = env.reset()
        total_episode_reward = 0.
        t = 0
        frame_idx = 1
        epsilon = epsilon_decay_exponencial(i + 1, N_episodes)

        while not done:
            # Take a random action
            action = agent.choose_action(state, epsilon)

            # Get next state, reward and done. Append into a buffer
            next_state, reward, done, _ = env.step(action)
            buffer.append((state, action, reward, next_state, done)) #apend to buffer

            # Train in case the buffer is bigger than the batch size
            if len(buffer)>batch_size:
                agent.train(buffer)  # train on buffer [the function will take a sample of the buffer]

            #Update Q
            if frame_idx % sync_target_frames == 0:
                print('frame_idx' + str(frame_idx), 'sync_target_frames' + str(sync_target_frames))
                agent.update_ann()

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            t+= 1
            frame_idx += 1

        # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        #HERE WE SHOULD INCLUDE A CHECK POINT OF THE REWARD, BUT I AM NOT SURE OF HOW MUCH SHOULD I PUT!

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


    agent.save_ann(filename_main=legend_main,filename_target=legend_target)
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
    plt.show()


if __name__ == "__main__":
    comparison_to_random()