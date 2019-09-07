#!/usr/bin/env python3
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

# Vehicle parameters
WIDTH = 1.0  # m
WHEEL_LEN = 0.3  # m
WHEEL_WIDTH = 0.2  # m
TREAD = 0.7  # m
L = 3.0  # m

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

def plot_car(x, y, yaw, delta=0.0, cabcolor="-r", truckcolor="-k"):
    x_f = x + np.cos(yaw) * L
    y_f = y + np.sin(yaw) * L

    plt.plot([x, x_f], [y, y_f], 'k')

    rear_axle_x1 = x + WIDTH * np.cos(yaw - 1.57) / 2
    rear_axle_y1 = y + WIDTH * np.sin(yaw - 1.57) / 2
    rear_axle_x2 = x + WIDTH * np.cos(yaw + 1.57) / 2
    rear_axle_y2 = y + WIDTH * np.sin(yaw + 1.57) / 2

    plt.plot([rear_axle_x1, rear_axle_x2], [rear_axle_y1, rear_axle_y2], 'k')

    front_axle_x1 = x_f + WIDTH * np.cos(yaw - 1.57) / 2
    front_axle_y1 = y_f + WIDTH * np.sin(yaw - 1.57) / 2
    front_axle_x2 = x_f + WIDTH * np.cos(yaw + 1.57) / 2
    front_axle_y2 = y_f + WIDTH * np.sin(yaw + 1.57) / 2

    plt.plot([front_axle_x1, front_axle_x2], [front_axle_y1, front_axle_y2], 'k')

    right_rear_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN,
                                  WHEEL_LEN],
                                 [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD,
                                  WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD,
                                  -WHEEL_WIDTH - TREAD]])
    right_front_wheel = np.copy(right_rear_wheel)

    left_rear_wheel = np.copy(right_rear_wheel)
    left_rear_wheel[1, :] *= -1

    left_front_wheel = np.copy(right_front_wheel)
    left_front_wheel[1, :] *= -1


    R_yaw = np.array([[np.cos(yaw), np.sin(yaw)],
                     [-np.sin(yaw), np.cos(yaw)]])
    R_delta = np.array([[np.cos(delta), np.sin(delta)],
                     [-np.sin(delta), np.cos(delta)]])

    right_rear_wheel = R_yaw.T @ right_rear_wheel
    left_rear_wheel = R_yaw.T @ left_rear_wheel

    right_front_wheel = R_delta.T @ right_front_wheel
    left_front_wheel = R_delta.T @ left_front_wheel
    right_front_wheel[0, :] += L
    left_front_wheel[0, :] += L
    right_front_wheel = R_yaw.T @ right_front_wheel
    left_front_wheel = R_yaw.T @ left_front_wheel

    right_rear_wheel[0, :] += x
    right_rear_wheel[1, :] += y
    left_rear_wheel[0, :] += x
    left_rear_wheel[1, :] += y
    right_front_wheel[0, :] += x
    right_front_wheel[1, :] += y
    left_front_wheel[0, :] += x
    left_front_wheel[1, :] += y

    plt.plot(np.array(right_rear_wheel[0, :]).flatten(),
             np.array(right_rear_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(left_rear_wheel[0, :]).flatten(),
             np.array(left_rear_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(right_front_wheel[0, :]).flatten(),
             np.array(right_front_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(left_front_wheel[0, :]).flatten(),
             np.array(left_front_wheel[1, :]).flatten(), truckcolor)

    plt.plot(x, y, '*g')

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
        plot_car(car.x, car.y, car.yaw, car.delta)
        plt.plot(x_hist, y_hist, '-b')
        plt.title('Stanley Control for steering')
        plt.xlabel('x (in m)')
        plt.ylabel('y (in m)')
        plt.axis('equal')
        plt.pause(0.001)

        if distance((car.x, car.y), (goal_x, goal_y)) < GOAL_EPS:
            print("Reached goal!")
            break

    plt.show()

if __name__=="__main__":
    main()
