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

def distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def main():
    car = Car(x=0, y=0, yaw=0)
    lateral_controller = LQR(x=car.x, y=car.y, yaw=car.yaw,
                             v=car.v, delta=car.delta, Q=10*np.eye(4),
                             R=1, K=1)
    longitudinal_controller = LongitudinalPID(v=car.v)

    # Target track
    ax = [0.0, 100.0, 100.0, 50.0, 60.0]
    ay = [0.0, 0.0, -30.0, -20.0, 0.0]

    spline = Spline2D(ax, ay)
    s = np.linspace(0, 1, num=500)
    cx, cy, cyaw, ck = spline.get_discrete_points(s)

    target_speed = 30.0/3.6

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
        plt.arrow(car.x, car.y, 0.5 * np.cos(car.yaw), 0.5 * np.sin(car.yaw),
                 fc='b', ec='k', head_width=1, head_length=1)
        # plt.plot(car.x, car.y, 'xb')
        plt.plot(x_hist, y_hist, '-b')
        plt.title('Stanley Control for steering')
        plt.xlabel('x (in m)')
        plt.ylabel('y (in m)')
        plt.pause(dt)

    plt.show()

if __name__=="__main__":
    main()
