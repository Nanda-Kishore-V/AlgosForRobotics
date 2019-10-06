#!/usr/bin/env python3
import os
import sys
import numpy as np
from matplotlib import pyplot as plt

sys.path.append('../')
from path_planning.DStarLite import DStarLite

from tempfile import mkstemp, mkdtemp
from imageio import imread, mimsave

def main():
    true_grid = np.zeros((80, 100))
    obstacles = []
    for i in range(5, 20):
        for j in range(10, 30):
            true_grid[i, j] = 1.0
            obstacles.append((i, j))

    for i in range(40, 60):
        for j in range(10, 80):
            true_grid[i, j] = 1.0
            obstacles.append((i, j))

    for i in range(30, 35):
        for j in range(70, 90):
            true_grid[i, j] = 1.0
            obstacles.append((i, j))

    for i in range(35, 40):
        for j in range(75, 80):
            true_grid[i, j] = 1.0
            obstacles.append((i, j))

    grid_belief = np.zeros((80, 100))
    start = (2, 17)
    goal = (62, 75)

    robot_mask = np.zeros_like(grid_belief)
    robot_mask[max(0, start[0]-5):min(grid_belief.shape[0], start[0]+5), \
               max(0, start[1]-5):min(grid_belief.shape[1], start[1]+5)] = 1.0

    grid_belief = true_grid * robot_mask

    path_planner = DStarLite(grid_belief.copy(), start, goal)
    path = path_planner.retrieve_path()

    fig = plt.figure(1)
    # Plot boundaries
    plt.plot([-1]*(true_grid.shape[1]+2), range(-1, true_grid.shape[1]+1, 1), '.k')
    plt.plot(range(-1, true_grid.shape[0]+1, 1), [-1]*(true_grid.shape[0]+2), '.k')
    plt.plot([true_grid.shape[0]]*(true_grid.shape[1]+2), range(-1, true_grid.shape[1]+1, 1), '.k')
    plt.plot(range(-1, true_grid.shape[0]+1, 1), [true_grid.shape[1]+1]*(true_grid.shape[0]+2), '.k')

    I, J = np.nonzero(grid_belief)
    for i, j in zip(I, J):
        plt.plot(i, j, '.k')

    for p in path:
        plt.plot(p[0], p[1], '.r')
    plt.plot(start[0], start[1], '.b')
    plt.plot(goal[0], goal[1], '.g')

    plt.axis('equal')
    plt.title("Dynamic Replanning using D* Lite")

    tempdir = mkdtemp()
    image_name = []

    _, filename = mkstemp(dir=tempdir)
    filename += '.png'
    fig.savefig(filename)
    image_name.append(filename)

    plt.pause(1)

    actual_path = [start]

    while start != goal:
        start = path[1]
        actual_path.append(start)

        robot_mask[max(0, start[0]-5):min(grid_belief.shape[0], start[0]+5), \
                   max(0, start[1]-5):min(grid_belief.shape[1], start[1]+5)] = 1.0
        grid_belief = true_grid * robot_mask

        path_planner.generate_path(start, grid_belief.copy())
        path = path_planner.retrieve_path()

        plt.cla()
        # Plot boundaries
        plt.plot([-1]*(true_grid.shape[1]+2), range(-1, true_grid.shape[1]+1, 1), '.k')
        plt.plot(range(-1, true_grid.shape[0]+1, 1), [-1]*(true_grid.shape[0]+2), '.k')
        plt.plot([true_grid.shape[0]]*(true_grid.shape[1]+2), range(-1, true_grid.shape[1]+1, 1), '.k')
        plt.plot(range(-1, true_grid.shape[0]+1, 1), [true_grid.shape[1]+1]*(true_grid.shape[0]+2), '.k')

        I, J = np.nonzero(grid_belief)
        for i, j in zip(I, J):
            plt.plot(i, j, '.k')

        for p in path:
            plt.plot(p[0], p[1], '.r')
        plt.plot(start[0], start[1], '.b')
        plt.plot(goal[0], goal[1], '.g')

        plt.axis('equal')
        plt.title("Dynamic Replanning using D* Lite")

        fig.savefig(filename)
        _, filename = mkstemp(dir=tempdir)
        filename += '.png'
        fig.savefig(filename)
        image_name.append(filename)

        plt.pause(0.01)

    actual_path.append(goal)

    for p in actual_path:
        plt.plot(p[0], p[1], '.r')

    plt.plot(actual_path[0][0], actual_path[0][1], '.b')
    plt.plot(actual_path[0][0], actual_path[0][1], '.g')

    for obs in obstacles:
        plt.plot(obs[0], obs[1], '.k')

    fig.savefig(filename)
    _, filename = mkstemp(dir=tempdir)
    filename += '.png'
    fig.savefig(filename)
    image_name.append(filename)
    plt.show()

    images = []
    for png in image_name:
        img = imread(png)
        images.append(img)
    for i in range(10):
        images.append(img)
    mimsave('DStarLite.gif', images)

def main2():
    grid = np.zeros((5, 5))
    grid[1, 2] = 1
    grid[1, 3] = 1
    grid[2, 1] = 1

    start = (0, 4)
    goal = (3, 1)

    path_planner = DStarLite(grid, start, goal)

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
    grid_copy[3, 2] = 1
    path_planner.generate_path((3, 3), grid_copy)

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
