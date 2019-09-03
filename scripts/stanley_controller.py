#!/usr/bin/env python
import os
import sys

import numpy as np
from matplotlib import pyplot as plt

sys.path.append('..')
from models.car_rear_axle import Car
from controllers.StanleyController import StanleyController
from controllers.LongitudinalPID import LongitudinalPID
from trajectory_generation.CubicSpline import Spline2D

WAYPOINT_RADIUS = 5.0
GOAL_EPS = 0.5

def distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def find_closest_waypoints(car, cx, cy, cyaw):
    wps = np.empty((3, 0))
    last_idx = None
    idx = -1
    for wp_x, wp_y, wp_yaw in zip(cx, cy, cyaw):
        idx += 1
        if distance((wp_x, wp_y), (car.x, car.y)) < WAYPOINT_RADIUS:
            wps = np.hstack((wps, np.array([[wp_x], [wp_y], [wp_yaw]])))
            last_idx = idx

    return last_idx, wps

def main():
    car = Car(x=0, y=1, yaw=0)
    lateral_controller = StanleyController(x=car.x, y=car.y, yaw=car.yaw,
                                              v=car.v, delta=car.delta, K=2)
    longitudinal_controller = LongitudinalPID(v=car.v)

    # Target track
    ax = [0.0, 100.0, 100.0, 50.0, 60.0]
    ay = [0.0, 0.0, -30.0, -20.0, 0.0]
    goal_x = ax[-1]
    goal_y = ay[-1]

    spline = Spline2D(ax, ay)
    s = np.linspace(0, 1, num=500)
    cx, cy, cyaw, ck = spline.get_discrete_points(s)

    target_speed = 30.0/3.6

    dt = 0.01
    x_hist = []
    y_hist = []
    for _ in range(5000):
        last_idx, closest_waypoints = find_closest_waypoints(car, cx, cy, cyaw)
        if last_idx is None:
            break

        throttle = longitudinal_controller.get_throttle_input(car.v, dt,
                                                              target_speed)
        steer = lateral_controller.get_steer_input(car.x, car.y, car.yaw, car.v,
                                                   closest_waypoints)

        car.step(throttle, steer, dt)

        print("Car speed {}".format(car.v))
        x_hist.append(car.x)
        y_hist.append(car.y)

        plt.cla()
        plt.plot(cx, cy, 'r')
        plt.arrow(car.x, car.y, 0.5 * np.cos(car.yaw), 0.5 * np.sin(car.yaw),
                 fc='b', ec='k', head_width=1, head_length=1)
        plt.plot(x_hist, y_hist, '-b')
        plt.title('Stanley Control for steering')
        plt.xlabel('x (in m)')
        plt.ylabel('y (in m)')
        plt.pause(dt)

        if distance((car.x, car.y), (goal_x, goal_y)) < GOAL_EPS:
            print("Reached goal!")
            break

    plt.show()

if __name__=="__main__":
    main()
