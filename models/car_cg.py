#!/usr/bin/env python
import numpy as np

class Car:
    """
    Kinematic model of a car-like robot with ref point on the center of gravity
    States: x, y, yaw, v
    Inputs: a, delta
    """
    def __init__(self, x=0, y=0, yaw=0, beta=0, v=0):
        self.x = 0
        self.y = 0
        self.yaw = 0
        self.beta = 0
        self.v = 0
        self.a = 0
        self.delta = 0

        self.L = 3 # m
        self.l_r = self.L/2
        self.a_max = 1 # m/s^2
        self.delta_max = 1.22 # rad

    def _update_controls(self, v, delta):
        self.a = np.fmin(np.fmax(v, self.a_min), self.a_max)
        self.delta = np.fmin(np.fmax(delta, -self.delta_max), self.delta_max)

    def model(self, v, delta):
        self._update_controls(v, delta)

        state_dot = np.array([[0], [0], [0], [0]])
        state_dot[0, 0] = self.v * np.cos(self.yaw + self.beta)
        state_dot[1, 0] = self.v * np.sin(self.yaw + self.beta)
        state_dot[2, 0] = self.v * np.tan(self.delta) * np.cos(self.beta)/ self.L
        state_dot[3, 0] = self.a
        self.beta = np.arctan2(self.l_r * np.tan(self.delta), self.L)

        return state_dot
