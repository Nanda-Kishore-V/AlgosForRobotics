#!/usr/bin/env python3
import os
import sys
import numpy as np
from matplotlib import pyplot as plt

sys.path.append('../')
from path_planning.LPAStar import LPAStar

def main():
    grid = np.zeros((80, 100))
    start = (2, 17)
    goal = (62, 75)
    path_planner = LPAStar(grid.copy(), start, goal)
    path = path_planner.retrieve_path()

    obstacle_groups = []
    obstacles = []
    for i in range(5, 20):
        for j in range(10, 30):
            obstacles.append((i, j))
    obstacle_groups.append(obstacles)

    obstacles = []
    for i in range(40, 60):
        for j in range(10, 80):
            obstacles.append((i, j))
    obstacle_groups.append(obstacles)

    obstacles = []
    for i in range(30, 35):
        for j in range(70, 90):
            obstacles.append((i, j))
    obstacle_groups.append(obstacles)

    obstacles = []
    for i in range(35, 40):
        for j in range(75, 80):
            obstacles.append((i, j))
    obstacle_groups.append(obstacles)

    fig = plt.figure(1)
    # Plot boundaries
    plt.plot([-1]*(grid.shape[1]+2), range(-1, grid.shape[1]+1, 1), '.k')
    plt.plot(range(-1, grid.shape[0]+1, 1), [-1]*(grid.shape[0]+2), '.k')
    plt.plot([grid.shape[0]]*(grid.shape[1]+2), range(-1, grid.shape[1]+1, 1), '.k')
    plt.plot(range(-1, grid.shape[0]+1, 1), [grid.shape[1]+1]*(grid.shape[0]+2), '.k')

    for p in path:
        plt.plot(p[0], p[1], '.r')
    plt.plot(start[0], start[1], '.b')
    plt.plot(goal[0], goal[1], '.g')

    plt.axis('equal')
    plt.title("Dynamic Replanning using Lifelong Planning A*")

    plt.pause(1)

    for idx, obstacle in enumerate(obstacle_groups):
        for obs in obstacle:
            grid[obs] = 1.0

        path_planner.generate_path(grid.copy())
        path = path_planner.retrieve_path()

        plt.cla()

        # Plot boundaries
        plt.plot([-1]*(grid.shape[1]+2), range(-1, grid.shape[1]+1, 1), '.k')
        plt.plot(range(-1, grid.shape[0]+1, 1), [-1]*(grid.shape[0]+2), '.k')
        plt.plot([grid.shape[0]]*(grid.shape[1]+2), range(-1, grid.shape[1]+1, 1), '.k')
        plt.plot(range(-1, grid.shape[0]+1, 1), [grid.shape[1]+1]*(grid.shape[0]+2), '.k')

        # Plot obstacles
        for i in range(idx+1):
            for obs in obstacle_groups[i]:
                plt.plot(obs[0], obs[1], '.k')

        for p in path:
            plt.plot(p[0], p[1], '.r')
        plt.plot(start[0], start[1], '.b')
        plt.plot(goal[0], goal[1], '.g')

        plt.axis('equal')
        plt.title("Dynamic Replanning using Lifelong Planning A*")

        plt.pause(1)

    plt.show()

def main2():
    grid = np.zeros((5, 5))
    grid[1, 2] = 1
    grid[1, 3] = 1
    grid[2, 1] = 1

    start = (0, 4)
    goal = (3, 1)

    path_planner = LPAStar(grid, start, goal)

    data_g = np.zeros((5, 5))
    data_rhs = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
           data_g[i, j] = path_planner.g.get((i, j), np.inf)
           data_rhs[i, j] = path_planner.rhs.get((i, j), np.inf)

    print("G:")
    print(data_g)
    print("RHS:")
    print(data_rhs)

    grid_copy = grid.copy()
    grid_copy[1, 4] = 1
    path_planner.generate_path(grid_copy)

    data_g = np.zeros((5, 5))
    data_rhs = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
           data_g[i, j] = path_planner.g.get((i, j), np.inf)
           data_rhs[i, j] = path_planner.rhs.get((i, j), np.inf)

    print("G:")
    print(data_g)
    print("RHS:")
    print(data_rhs)

if __name__=="__main__":
    main()
    # main2()
