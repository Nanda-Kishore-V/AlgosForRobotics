#!/usr/bin/env python
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import matplotlib as mpl

sys.path.append('..')
from models.car_rear_axle import Car
from controllers.ModelPredictiveController import MPC
from trajectory_generation.CubicSpline import Spline2D

GOAL_EPS = 0.1

# Vehicle parameters
WIDTH = 1.0  # m
WHEEL_LEN = 0.3  # m
WHEEL_WIDTH = 0.2  # m
TREAD = 0.7  # m
L = 3.0  # m

def distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def find_closest_waypoint(car, cx, cy):
    distances = np.sum(( np.array([[car.x], [car.y]]) -
                         np.stack((cx, cy)) )**2, axis=0)
    idx = np.argmin(distances)

    return idx, cx[idx], cy[idx]

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

def smooth_yaw(yaw):
    for i in range(len(yaw) - 1):
        dyaw = yaw[i+1] - yaw[i]

        while dyaw >= np.pi/2.0:
            yaw[i+1] -= 2.0 * np.pi
            dyaw = yaw[i+1] - yaw[i]

        while dyaw <= -np.pi/2.0:
            yaw[i+1] += 2.0 * np.pi
            dyaw = yaw[i+1] - yaw[i]

    return yaw

def generate_speed_profile(car, cx, cy, cyaw, target_speed):
    direction = 1.0
    speed_profile = [target_speed] * len(cx)

    for i in range(len(cx) - 1):
        dx = cx[i+1] - cx[i]
        dy = cy[i+1] - cy[i]

        forward_direction = np.arctan2(dy, dx)

        if dx != 0 and dy != 0:
            d_yaw = abs(MPC.bound_angles(forward_direction - cyaw[i]))
            if d_yaw >= car.delta_max:
                direction = -1.0
            else:
                direction = 1.0
        if direction != -1.0:
            speed_profile[i] = target_speed
        else:
            speed_profile[i] = -target_speed

    return speed_profile

def main():
    # Target track
    ax = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
    ay = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]

    spline = Spline2D(ax, ay)
    s = np.linspace(0, 1, num=500)
    cx, cy, cyaw, ck = spline.get_discrete_points(s)

    goal_x= ax[-1]
    goal_y = ay[-1]

    cyaw = smooth_yaw(cyaw)

    Q = np.eye(4)
    Qf = np.eye(4)
    R = 0.01*np.eye(2)
    Rd = 0.01*np.eye(2)

    car = Car(x=cx[0], y=cy[0], yaw=cyaw[0], L=3)
    controller = MPC(x=car.x, y=car.y, yaw=car.yaw,
                     v=car.v, delta=car.delta, L=3, Q=Q,
                     R=R, Qf=Qf, Rd = Rd)

    target_speed = 30.0/3.6
    speed_profile = generate_speed_profile(car, cx, cy, cyaw, target_speed)

    dt = 0.1
    x_hist = []
    y_hist = []
    for _ in range(5000):
        throttle, steer, xs, ys, vs, yaws = \
            controller.get_inputs(car.x, car.y, car.yaw,
                                  car.v, np.stack((cx, cy, cyaw, ck)),
                                  speed_profile, dt)

        car.step(throttle, steer, dt)

        print("Car speed {}".format(car.v))
        x_hist.append(car.x)
        y_hist.append(car.y)

        plt.cla()
        plt.plot(cx, cy, 'r')
        plot_car(car.x, car.y, car.yaw, car.delta)
        plt.plot(x_hist, y_hist, '-b')
        plt.title('MPC controller for speed and steering')
        plt.xlabel('x (in m)')
        plt.ylabel('y (in m)')
        plt.axis("equal")
        plt.pause(0.0001)

        if distance((car.x, car.y), (goal_x, goal_y)) < GOAL_EPS:
            print("Reached goal!")
            break

    plt.show()

if __name__=="__main__":
    main()
