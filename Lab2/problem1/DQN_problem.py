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
from DQN_agent import *
from tqdm import trange
import matplotlib as mpl
import matplotlib.pyplot as plt


#Parameters for epsilon measurments
min_epsilon = 0.05
max_epsilon = 0.99
Z = 0.925


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
    """ Exponential epsilon decay calculation
    """

    epsilon = max(min_epsilon, max_epsilon * (min_epsilon / max_epsilon) ** ((k - 1) / (N_episodes * Z - 1)))
    return epsilon

def test_agent(env, agent, N):
    """ Let the agent behave with the policy it follows
    """

    env.reset()
    episode_reward_list = []
    episode_number_of_steps = []

    EPISODES = trange(N, desc='Episodes', leave=True)
    for i in EPISODES:
        done = False
        state = env.reset()
        total_episode_reward = 0
        t = 0

        while not done:
            action = agent.choose_action(state)
            # Get next state, reward and done. Append into a buffer
            next_state, reward, done, _ = env.step(action)
            # Update episode reward
            total_episode_reward += reward
            # Update state for next iteration
            state = next_state
            t += 1

        # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        # Close environment
        env.close()

    return N, episode_reward_list, episode_number_of_steps

def compare_to_random(n_episodes, network_filename = ''):
    """ Compare random with trained agent
    """

    env = gym.make('LunarLander-v2')

    # Random agent initialization
    random_agent=RandomAgent(env.action_space.n)
    num_episodes_random,random_rewards,_ = test_agent(env, random_agent, n_episodes)

    # DQN agent initialization
    dqn_agent = Agent(network_filename)
    num_episodes_dqn,dqn_rewards,_ = test_agent(env, dqn_agent, n_episodes)

    # Plot rewads
    fig = plt.figure(figsize=(9, 9))

    xr = [i for i in range(1, num_episodes_random+1)]
    xdqn = [i for i in range(1, num_episodes_dqn+1)]
    plt.plot(xr, random_rewards, label='Random Agent')
    plt.plot(xdqn, dqn_rewards, label='Trained DQN Agent')
    plt.ylim(-400, 400)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Reward vs Episodes')
    plt.legend()
    plt.show()

def train(N_episodes, discount_factor, n_ep_running_average, lr, batch_size, buffer_size, legend_main, legend_target):
    """ Train agent
    """

    #dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("Using", dev)

    # Import and initialize the discrete Lunar Laner Environment
    env = gym.make('LunarLander-v2')
    env.reset()

    # Parameters
    n_actions = env.action_space.n               # Number of available actions
    dim_state = len(env.observation_space.high)  # State dimensionality
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

    draw_plots(N_episodes, episode_reward_list, n_ep_running_average, episode_number_of_steps)
    agent.save_ann(filename_main=legend_main,filename_target=legend_target)


def draw_plots(N_episodes, episode_reward_list, n_ep_running_average, episode_number_of_steps):
    """ Draw total reward and number of steps plot
    """
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

def optimal_policy_plot(network= 'network_main_1.phd'):
    """ 3D plot for optimal policy
    """

    n_y = 100
    n_om = 100
    ys = np.linspace(0, 1.5, n_y)
    ws = np.linspace(-np.pi, np.pi, n_om)

    Ys, Ws = np.meshgrid(ys, ws)

    Q_network = torch.load(network)

    Q = np.zeros((len(ys), len(ws)))
    action = np.zeros((len(ys), len(ws)))
    for y_idx, y in enumerate(ys):
        for w_idx, w in enumerate(ws):
            state = torch.tensor((0, y, 0, 0, w, 0, 0, 0), dtype=torch.float32)
            Q[w_idx, y_idx] = Q_network(state).max(0)[0].item()  # Max
            action[w_idx,y_idx] = torch.argmax(Q_network(state)).item()

    #3d plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(Ws, Ys, Q, cmap=mpl.cm.coolwarm)
    ax.set_ylabel('height (y)')
    ax.set_xlabel('angle (ω)')
    ax.set_zlabel('V(s(y,ω))')
    plt.show()

    # 3d plot
    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')
    surf2 = ax2.plot_surface(Ws, Ys, action)
    ax2.set_ylabel('height (y)')
    ax2.set_xlabel('angle (ω)')
    ax2.set_zlabel('Best Action')
    plt.show()

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    im = ax3.pcolormesh(Ys, Ws, action)
    cbar = fig3.colorbar(im, ticks=[0, 1, 2, 3])
    ax3.set_ylabel('angle (w)')
    ax3.set_xlabel('height (y)')
    #cbar.ax3.set_yticklabels(['nothing (0)', 'left (1)', 'main (2)', 'right (3)'])
    ax3.set_title('Best Action')
    plt.show()



if __name__ == "__main__":
    # Parameters to define
    legend_main = 'neural-network-1.pth'
    legend_target = 'neural-network-1-target.pth'
    n_episodes = 300  # Number of episodes
    discount_factor = 0.98  # Value of the discount factor
    n_ep_running_average = 50  # Running average of 50 episodes

    lr = 0.0005
    batch_size = 50  # N
    buffer_size = 10000  # L

    train_ex = True
    comparative = False
    network_filename = 'neural-network-1.pth'
    plot_3d = False

    if train_ex == True:
        train(n_episodes, discount_factor, n_ep_running_average, lr, batch_size, buffer_size,legend_main, legend_target)

    if comparative == True:
        compare_to_random(n_episodes, network_filename=network_filename)

    if plot_3d == True:
        optimal_policy_plot(network = network_filename)