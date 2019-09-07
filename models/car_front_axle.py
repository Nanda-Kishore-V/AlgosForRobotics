#!/usr/bin/env python3
import numpy as np

class Car:
    """
    Kinematic model of a car-like robot with ref point on front axle
    States: x, y, yaw, v
    Inputs: a, delta
    """
    def __init__(self, x=0, y=0, yaw=0, v=0):
        self.x = 0
        self.y = 0
        self.yaw = 0
        self.v = 0
        self.a = 0
        self.delta = 0

        self.L = 3 # m
        self.a_max = 1 # m/s^2
        self.delta_max = 1.22 # rad

    def _update_controls(self, v, delta):
        self.a = np.fmin(np.fmax(v, -self.a_max), self.a_max)
        self.delta = np.fmin(np.fmax(delta, -self.delta_max), self.delta_max)

    def model(self, v, delta):
        self._update_controls(v, delta)

        state_dot = np.array([0., 0., 0., 0.])
        state_dot[0] = self.v * np.cos(self.yaw + self.delta)
        state_dot[1] = self.v * np.sin(self.yaw + self.delta)
        state_dot[2] = self.v * np.sin(self.delta) / self.L
        state_dot[3] = self.a

        return state_dot

    def step(self, a, delta, dt):
        state_dot = self.model(a, delta)
        state = self.get_state()
        self.set_state(state + state_dot * dt)

    def get_state(self):
        state = np.array([0., 0., 0., 0.])
        state[0] = self.x
        state[1] = self.y
        state[2] = self.yaw
        state[3] = self.v

        return state

    def set_state(self, state):
        self.x = state[0]
        self.y = state[1]
        self.yaw = state[2]
        self.v = state[3]
