import numpy as np
import maze2 as mz
import matplotlib.pyplot as plt
# Description of the maze as a numpy array
maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 5],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]
])

env = mz.Maze(maze)
# Finite horizon
horizon = 20
"""
# Solve the MDP problem with dynamic programming
V, policy= mz.dynamic_programming(env,horizon)
# Simulate the shortest path starting from position A
method = 'DynProg'
start  = (0,0, 6,5,0)
path, win = env.simulate(start, policy, method)
"""



Q, policy, reward_list, value_list = mz.q_learning(env, gamma=0.95, n_episodes=50000, T=30, epsilon= 0.1, player_state=(0,0,6,5,0), alpha_exponent=0.8)


print("Mean reward per thousand episodes")
for i in range(50):
    print((i+1)*1000,": mean espiode reward: ", np.mean(reward_list[1000*i:1000*(i+1)]))

plt.plot(reward_list)
plt.show()

plt.plot(value_list)
plt.show()