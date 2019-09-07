#!/usr/bin/env python3
import os
import sys

import numpy as np
from matplotlib import pyplot as plt

sys.path.append('..')
from models.car_rear_axle import Car
from controllers.PurePursuit import PurePursuit
from controllers.LongitudinalPID import LongitudinalPID

LOOKAHEAD_FACTOR = 0.2 # s
CONST_LOOKAHEAD_DIST = 3 # m
GOAL_EPS = 0.1 # m

# Vehicle parameters
WIDTH = 1.0  # m
WHEEL_LEN = 0.3  # m
WHEEL_WIDTH = 0.2  # m
TREAD = 0.7  # m
L = 3.0  # m

def distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def lookahead(car, cx, cy):
    distances = np.sum(( np.array([[car.x], [car.y]]) - np.stack((cx, cy)) )**2,
                       axis=0)
    idx = np.argmin(distances)

    found = False
    while idx < len(cx):
        if distances[idx] > LOOKAHEAD_FACTOR * car.v + CONST_LOOKAHEAD_DIST:
            found = True
            break
        idx += 1

    if found:
        return idx, cx[idx], cy[idx]
    else:
        return None, cx[-1], cy[-1]

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
    car = Car(x=0, y=-3, yaw=0.1, v=0)
    lateral_controller = PurePursuit(x=car.x, y=car.y, yaw=car.yaw, v=car.v,
                                     delta=car.delta, K=0.5)
    longitudinal_controller = LongitudinalPID(v=car.v)

    # Target track
    cx = np.arange(0, 50, 0.1)
    cy = [np.sin(ix / 5.0) * ix / 2.0 for ix in cx]
    goal_x = cx[-1]
    goal_y = cy[-1]

    target_speed = 30.0/3.6

    speed_profile = []
    x_hist = []
    y_hist = []
    dt = 0.01
    while True:
        idx, target_x, target_y = lookahead(car, cx, cy)
        target_waypoint = [target_x, target_y]
        throttle = longitudinal_controller.get_throttle_input(car.v, dt,
                                                               target_speed)
        steer = lateral_controller.get_steer_input(car.x, car.y, car.yaw, car.v,
                                                   target_waypoint)

        car.step(throttle, steer, dt)

        # print("Car speed {}".format(car.v))
        speed_profile.append(car.v)
        x_hist.append(car.x)
        y_hist.append(car.y)

        plt.cla()
        plt.plot(cx, cy, 'r')
        plot_car(car.x, car.y, car.yaw, car.delta)
        plt.plot(target_x, target_y, 'og')
        plt.plot(x_hist, y_hist, '-b')
        plt.title('Pure Pursuit Steering Control')
        plt.xlabel('x (in m)')
        plt.ylabel('y (in m)')
        plt.axis('equal')
        plt.pause(dt)

        if distance((car.x, car.y), (goal_x, goal_y)) < GOAL_EPS:
            print("Reached goal!")
            break
    plt.show()

if __name__=="__main__":
    main()
