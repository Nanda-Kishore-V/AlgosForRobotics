#!/usr/bin/env python3
import sys
import numpy as np
from matplotlib import pyplot as plt

sys.path.append("../")
from path_planning.RRTStar import RRTStar

def main():
    start = (1, 1)
    goal = (2, 1)
    limits = [(0, 3), (0, 3)]
    obstacles = [(1.5, 0.1, 0.2),
                 (1.5, 0.4, 0.2),
                 (1.5, 0.7, 0.2),
                 (1.5, 1.0, 0.2),
                 (1.5, 1.3, 0.2),
                 (1.5, 1.9, 0.2),
                 (1.5, 2.2, 0.2),
                 (1.5, 2.5, 0.2),
                 (1.5, 2.8, 0.2)]
    path_planner = RRTStar(start, goal, limits, obstacles)
    path_planner.generate_path()
    path = path_planner.retrieve_path()

    plt.figure()
    plt.title("RRT* path planning")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(path_planner.root.location[0], path_planner.root.location[1], 'xr')
    plt.plot(path_planner.goal[0], path_planner.goal[1], 'xg')

    fig = plt.gcf()
    ax = fig.gca()
    for obs in path_planner.obstacles:
        circle = plt.Circle((obs[0], obs[1]), obs[-1], color='k')
        ax.add_artist(circle)

    plt.axis('equal')
    plt.xlim([path_planner.limits[0][0], path_planner.limits[0][1]])
    plt.ylim([path_planner.limits[1][0], path_planner.limits[1][1]])

    for p in path_planner.node_list:
        plt.plot(p.location[0], p.location[1], 'xc')
        if p.parent is not None:
            plt.plot([p.location[0], p.parent.location[0]], [p.location[1], p.parent.location[1]], '-k')

        plt.plot(path_planner.root.location[0], path_planner.root.location[1], 'xr')
        plt.plot(path_planner.goal[0], path_planner.goal[1], 'xg')

        plt.pause(0.001)

    for i in range(len(path)-1):
        plt.plot(path[i][0], path[i][1], '.r')
        plt.plot([path[i][0], path[i+1][0]], [path[i][1], path[i+1][1]], '-r')
    plt.show()

if __name__=="__main__":
    main()
