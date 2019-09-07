#!/usr/bin/env python3
import os
import sys
import numpy as np
from matplotlib import pyplot as plt

sys.path.append('../')
from path_planning.AStar import AStar

def main():
    grid = np.zeros((100, 60))

    obstacles = []
    for i in range(80):
        obstacles.append((i, 20))
        grid[i, 20] = 1.0
    for i in range(80):
        obstacles.append((99-i, 40))
        grid[99-i, 40] = 1.0

    start = (50, 10)
    goal = (50, 50)

    path_planner = AStar(grid, start, goal)
    path = path_planner.retrieve_path()

    closed_list = path_planner.closed_list

    # Plot obstacles
    plt.figure()
    plt.axis('equal')
    for obs in obstacles:
        plt.plot(obs[0], obs[1], '.k')

    # Plot boundaries
    plt.plot([-1]*(grid.shape[1]+2), range(-1, grid.shape[1]+1, 1), '.k')
    plt.plot(range(-1, grid.shape[0]+1, 1), [-1]*(grid.shape[0]+2), '.k')
    plt.plot([grid.shape[0]]*(grid.shape[1]+2), range(-1, grid.shape[1]+1, 1), '.k')
    plt.plot(range(-1, grid.shape[0]+1, 1), [grid.shape[1]+1]*(grid.shape[0]+2), '.k')

    for i, node in enumerate(closed_list):
        plt.plot(node[0], node[1], 'xc')
        plt.plot(start[0], start[1], 'xr')
        plt.plot(goal[0], goal[1], 'xg')
        # Make plotting faster
        if i%20 == 0:
            plt.pause(0.01)

    for p in path:
        plt.plot(p[0], p[1], '.r')
    plt.show()

if __name__=="__main__":
    main()
