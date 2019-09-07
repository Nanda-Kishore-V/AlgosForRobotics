#!/usr/bin/env python3
import numpy as np

class LongitudinalPID:
    """
    PID controller for longitudinal control
    """
    def __init__(self, v=0, L=3, Kp=1, Kd=0.01, Ki=0.01):
        # States
        self.v = v
        self.prev_v = v
        self.sum_error = 0

        # Wheel base
        self.L = L

        # Control gain
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def update_speed(self, v):
        self.v = v

    def get_throttle_input(self, v, dt, target_speed):
        self.update_speed(v)

        error = target_speed - self.v
        self.sum_error += error
        throttle = self.Kp * error + \
            self.Ki * self.sum_error * dt + \
            self.Kd * (self.v - self.prev_v) / dt
        self.prev_v = self.v

        return throttle
