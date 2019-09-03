#!/usr/bin/env python
"""
Based on the paper "A Tutorial On Autonomous Vehicle Steering
Controller Design, Simulation and Implementation"
Link: https://arxiv.org/pdf/1803.03758.pdf
"""
import os
import sys

import numpy as np
from matplotlib import pyplot as plt

sys.path.append('..')
from models.car_rear_axle import Car
from controllers.LQR import LQR
from controllers.LongitudinalPID import LongitudinalPID
from trajectory_generation.CubicSpline import Spline2D

GOAL_EPS = 0.1

def distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def main():
    Q = np.eye(4)
    R = 0.1*np.eye(1)
    car = Car(x=0, y=-0.5, yaw=0, L=0.5)
    lateral_controller = LQR(x=car.x, y=car.y, yaw=car.yaw,
                             v=car.v, delta=car.delta, L = 0.5, Q=Q,
                             R=R, K=1)
    longitudinal_controller = LongitudinalPID(v=car.v, L=0.5)

    # Target track
    ax = [0.0, 6.0, 12.5, 10.0, 7.5, 3.0, -1.0]
    ay = [0.0, -3.0, -5.0, 6.5, 3.0, 5.0, -2.0]
    goal_x= ax[-1]
    goal_y = ay[-1]

    spline = Spline2D(ax, ay)
    s = np.linspace(0, 1, num=500)
    cx, cy, cyaw, ck = spline.get_discrete_points(s)

    target_speed = 10.0/3.6

    dt = 0.01
    x_hist = []
    y_hist = []
    for _ in range(5000):
        throttle = longitudinal_controller.get_throttle_input(car.v, dt,
                                                              target_speed)
        steer = lateral_controller.get_steer_input(car.x, car.y, car.yaw, car.v,
                                                   np.stack((cx, cy, cyaw, ck)))

        car.step(throttle, steer, dt)

        print("Car speed {}".format(car.v))
        x_hist.append(car.x)
        y_hist.append(car.y)

        plt.cla()
        plt.plot(cx, cy, 'r')
        plt.arrow(car.x, car.y, 0.1 * np.cos(car.yaw), 0.1 * np.sin(car.yaw),
                 fc='b', ec='k', head_width=0.1, head_length=0.1)
        plt.plot(x_hist, y_hist, '-b')
        plt.title('LQR controller for steering')
        plt.xlabel('x (in m)')
        plt.ylabel('y (in m)')
        plt.pause(dt)

        if distance((car.x, car.y), (goal_x, goal_y)) < GOAL_EPS:
            print("Reached goal!")
            break

    plt.show()

if __name__=="__main__":
    main()
