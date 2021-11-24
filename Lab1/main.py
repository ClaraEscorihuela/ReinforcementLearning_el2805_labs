import numpy as np
import maze as mz
import matplotlib.pyplot as plt
# Description of the maze as a numpy array
maze1 = np.array([
    [0, 0, 1, 0, 0, 0, 0, 5],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]
])

env = mz.Maze(maze1)
# Finite horizon
horizon = 30
# Solve the MDP problem with dynamic programming
V, policy= mz.dynamic_programming(env,horizon)
# Simulate the shortest path starting from position A
method = 'DynProg';
start  = (0,0, 4,5,0)
path, win = env.simulate(start, policy, method);



# Show the shortest path
mz.animate_solution(maze, path)
Q, policy, reward_list, value_list = mz.sarsa(env, gamma=0.95, n_episodes=50000, T=30, player_state=(0,0,4,5), epsilon_decay=True, alpha_exponent=0.8)


print("Mean reward per thousand episodes")
for i in range(50):
    print((i+1)*1000,": mean espiode reward: ", np.mean(reward_list[1000*i:1000*(i+1)]))

plt.plot(reward_list)
plt.show()

plt.plot(value_list)
plt.show()