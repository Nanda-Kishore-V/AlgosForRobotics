#!/usr/bin/env python
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

        print("Car speed {}".format(car.v))
        speed_profile.append(car.v)
        x_hist.append(car.x)
        y_hist.append(car.y)

        plt.cla()
        plt.plot(cx, cy, '.r')
        plt.arrow(car.x, car.y, 0.5 * np.cos(car.yaw), 0.5 * np.sin(car.yaw),
                 fc='b', ec='k', head_width=1, head_length=1)
        # plt.plot(car.x, car.y, 'xb')
        plt.plot(target_x, target_y, 'og')
        plt.plot(x_hist, y_hist, '-b')
        plt.title('Pure Pursuit Steering Control')
        plt.xlabel('x (in m)')
        plt.ylabel('y (in m)')
        plt.pause(dt)

        if distance((car.x, car.y), (goal_x, goal_y)) < GOAL_EPS:
            print("Reached goal!")
            break
    plt.show()

if __name__=="__main__":
    main()
