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
import gym
from tqdm import trange
from DDPG_agent import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from DDPG_soft_updates import soft_updates



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


def test_agent(env, agent, N):
    """ Let the agent behave with the policy it follows"""
    env.reset()
    episode_reward_list = []
    episode_number_of_steps = []

    # Training process
    EPISODES = trange(N, desc='Episode: ', leave=True)

    for i in EPISODES:
        # Reset enviroment data
        done = False
        state = env.reset()
        total_episode_reward = 0.
        t = 0
        while not done:
            # Take a random action
            action = agent.forward(state)
            # Get next state and reward.
            next_state, reward, done, _ = env.step(action)
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

    return N, episode_reward_list, episode_number_of_steps

def compare_to_random(n_episodes, network_filename = ''):

    env = gym.make('LunarLanderContinuous-v2')

    # Random agent initialization
    random_agent=RandomAgent(len(env.action_space.high))
    num_episodes_random,random_rewards,_ = test_agent(env, random_agent, n_episodes)

    # DQN agent initialization
    ddpg_agent = Agent(network_filename)
    num_episodes_ddpg,ddpg_rewards,_ = test_agent(env, ddpg_agent, n_episodes)

    # Plot rewads
    fig = plt.figure(figsize=(9, 9))

    xr = [i for i in range(1, num_episodes_random+1)]
    xddpg = [i for i in range(1, num_episodes_ddpg+1)]
    plt.plot(xr, random_rewards, label='Random Agent')
    plt.plot(xddpg, ddpg_rewards, label='Trained DDPG Agent')
    plt.ylim(-400, 400)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Reward vs Episodes')
    plt.legend()
    plt.show()


def train(N_episodes, discount_factor, n_ep_running_average, lr_actor, lr_critic, batch_size, buffer_size,tau, d, mu, sigma):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(dev)

    # Import and initialize Mountain Car Environment
    env = gym.make('LunarLanderContinuous-v2')
    env.reset()
    m = len(env.action_space.high)  # dimensionality of the action
    dim_state = len(env.observation_space.high)  # State dimensionality


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

    draw_plots(N_episodes, episode_reward_list, n_ep_running_average, episode_number_of_steps)

    # Save the models
    legend_main_critic = 'main_critic.pth'
    legend_target_critic = 'target_critic.pth'
    legend_main_actor = 'main_actor.pth'
    legend_target_actor = 'target_actor.pth'

    agent.save_ann(agent.main_actor,agent.target_actor,filename_main=legend_main_actor,filename_target=legend_target_actor)
    agent.save_ann(agent.main_critic,agent.target_critic,filename_main=legend_main_critic,filename_target=legend_target_critic)



def draw_plots(N_episodes, episode_reward_list, n_ep_running_average, episode_number_of_steps):
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



def optimal_policy_plot(actor_network= 'neural-network-2-actor.phd', critic_network= 'neural-network-2-critic.phd'):
    n_y = 100
    n_om = 100
    ys = np.linspace(0, 1.5, n_y)
    ws = np.linspace(-np.pi, np.pi, n_om)

    Ys, Ws = np.meshgrid(ys, ws)

    policy_network = torch.load(actor_network)
    Q_network = torch.load(critic_network)

    Q = np.zeros((len(ys), len(ws)))
    action = np.zeros((len(ys), len(ws)))
    for y_idx, y in enumerate(ys):
        for w_idx, w in enumerate(ws):
            state = torch.tensor((0, y, 0, 0, w, 0, 0, 0), dtype=torch.float32)
            a = policy_network(state)
            action[w_idx, y_idx] = a[1].item()
            Q[w_idx, y_idx] = Q_network(torch.reshape(state, (1,-1)),torch.reshape(a, (1,-1))).item()  # Max

    #3d plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(Ws, Ys, Q, cmap=mpl.cm.coolwarm)
    ax.set_ylabel('height (y)')
    ax.set_xlabel('angle (ω)')
    ax.set_zlabel('Q(s(y,ω), π(s))')
    plt.show()

    # 3d plot
    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')
    surf2 = ax2.plot_surface(Ws, Ys, action)
    ax2.set_ylabel('height (y)')
    ax2.set_xlabel('angle (ω)')
    ax2.set_zlabel('Best Action - Engine direction')
    plt.show()

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    im = ax3.pcolormesh(Ys, Ws, action)
    cbar = fig3.colorbar(im, ticks=[0, 1, 2, 3])
    ax3.set_ylabel('angle (w)')
    ax3.set_xlabel('height (y)')
    #cbar.ax3.set_yticklabels(['nothing (0)', 'left (1)', 'main (2)', 'right (3)'])
    ax3.set_title('Best Action - Engine direction')
    plt.show()




if __name__ == "__main__":

    n_episodes = 50  # Number of episodes - 50 for comparative
    discount_factor = 0.99  # Value of gamma
    n_ep_running_average = 50  # Running average of 50 episodes
    lr_actor = 5e-5  # For actor model
    lr_critic = 5e-4  # For actor model
    batch_size = 64  # N
    buffer_size = 30000  # L
    tau = 0.001  # soft update parameter
    d = 2
    # Noise parameters
    mu = 0.15
    sigma = 0.2


    train_ex = True
    comparative = False
    actor_filename = 'neural-network-2-actor.pth'
    critic_filename = 'neural-network-2-critic.pth'
    plot_3d = False

    if train_ex == True:
        train(n_episodes, discount_factor, n_ep_running_average, lr_actor, lr_critic, batch_size, buffer_size,tau, d, mu, sigma)

    if comparative == True:
        compare_to_random(n_episodes, network_filename=actor_filename)

    if plot_3d == True:
        optimal_policy_plot(actor_network = actor_filename, critic_network = critic_filename)

