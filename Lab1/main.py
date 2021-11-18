import numpy as np
import maze as mz

# Description of the maze as a numpy array
maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
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
# Solve the MDP problem with dynamic programming
V, policy= mz.dynamic_programming(env,horizon)

method = 'DynProg'
start  = (0,0, 4,5)
path, win = env.simulate(start, policy, method)

mz.animate_solution(maze, path)