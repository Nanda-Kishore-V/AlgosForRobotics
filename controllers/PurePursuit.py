#!/usr/bin/env python
import numpy as np

class PurePursuit:
    """
    Pure Pursuit controller for lateral control
    Assumes real axle coordinates
    """
    def __init__(self, x=0, y=0, yaw=0, v=0, delta=0, max_steering_angle=1.22, L=3, K=1):
        # States
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

        # Steering angle
        self.delta = delta
        self.max_steering_angle = max_steering_angle

        # Wheel base
        self.L = L

        # Control gain
        self.K = K

    def update_position(self, x, y):
        self.x = x
        self.y = y

    def update_speed(self, v):
        self.v = v

    def update_yaw(self, yaw):
        self.yaw = yaw

    def update_steering_angle(self, steer):
        self.delta = steer * self.max_steer_angle

    def get_steer_input(self, x, y, yaw, v, target):
        self.update_position(x, y)
        self.update_yaw(yaw)
        self.update_speed(v)

        goal_x = target[0]
        goal_y = target[1]

        alpha = np.arctan2(goal_y - self.y, goal_x - self.x) - self.yaw
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

        numerator = 2*self.L*np.sin(alpha)
        denominator = self.K * self.v
        steer = np.arctan2(numerator, 0.001 + denominator)

        self.delta = steer

        return steer
