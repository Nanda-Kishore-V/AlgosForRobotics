#!/usr/bin/env python
import numpy as np
from numpy import linalg

class LQR:
    """
    LQR controller for lateral control
    """
    def __init__(self, x=0, y=0, yaw=0, v=0, delta=0,
                 max_steering_angle=1.22, L=3, Q=np.eye(4), R=1, K=1):
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
        self.Q = Q
        self.R = R
        self.K = K

        self.prev_y_e = 0
        self.prev_yaw_e = 0

    def update_position(self, x, y):
        self.x = x
        self.y = y

    def update_speed(self, v):
        self.v = v

    def update_yaw(self, yaw):
        self.yaw = yaw

    @staticmethod
    def solve_discrete_riccati(A, B, Q, R):
        P = Q
        for _ in range(1000):
            Pn = A.T @ P @ A - (A.T @ P @ B) @ linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A) + Q
            if linalg.norm(Pn - P, ord='fro') < 0.01:
                P = Pn
                break
            P = Pn

        return P

    @staticmethod
    def get_controller_gain(A, B, Q, R, P):
        K = linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        return K

    @staticmethod
    def bound_angles(theta):
        return (theta + np.pi) % (2*np.pi) - np.pi


    def create_A_and_B(self, dt=0.01):
        A = np.zeros((4, 4))
        A[0, 0] = 1
        A[0, 1] = dt
        A[1, 2] = self.v
        A[2, 2] = 1
        A[2, 3] = dt

        B = np.zeros((4, 1))
        B[3, 0] = self.v / self.L

        return A, B

    def find_closest_waypoint(self, cx, cy):
        distances = np.sum(( np.array([[self.x], [self.y]]) - np.stack((cx, cy)) )**2,
                           axis=0)
        idx = np.argmin(distances)

        return idx, cx[idx], cy[idx]

    def get_steer_input(self, x, y, yaw, v, waypoints, dt=0.01):
        self.update_position(x, y)
        self.update_yaw(yaw)
        self.update_speed(v)

        cx = waypoints[0]
        cy = waypoints[1]
        cyaw = waypoints[2]
        ck = waypoints[3]
        idx, target_x, target_y = self.find_closest_waypoint(cx, cy)

        curvature = ck[idx]

        y_e = np.sqrt((self.x - cx[idx])**2 + (self.y - cy[idx])**2)
        angle = self.bound_angles(cyaw[idx] - np.arctan2(cy[idx] - self.y,
                                                         cx[idx] - self.x))
        if angle < 0:
            y_e = -y_e

        yaw_e = self.bound_angles(self.yaw - cyaw[idx])

        A, B = self.create_A_and_B(dt)
        P = self.solve_discrete_riccati(A, B, self.Q, self.R)
        K = self.get_controller_gain(A, B, self.Q, self.R, P)

        error_vec = np.zeros((4, 1))
        error_vec[0, 0] = y_e
        error_vec[1, 0] = (y_e - self.prev_y_e)/dt
        error_vec[2, 0] = yaw_e
        error_vec[3, 0] = (yaw_e - self.prev_yaw_e)/dt

        self.prev_y_e = y_e
        self.prev_yaw_e = yaw_e

        steer_fb = self.bound_angles((- K @ error_vec)[0, 0])
        steer_ff = np.arctan2(self.L * curvature, 1)

        steer = steer_fb + steer_ff

        return steer

if __name__=="__main__":
    A = np.array([[1, 0.01, 0, 0],
                  [0, 0, 3, 0],
                  [0, 0, 1, 0.01],
                  [0, 0, 0, 0]])
    B = np.array([[0],
                  [0],
                  [0],
                  [1]])
    Q = np.eye(4)
    R = np.eye(1)
    P = LQR.solve_discrete_riccati(A, B, Q, R)
    print("P:")
    print(P)

    K = LQR.get_controller_gain(A, B, Q, R, P)
    print("K:")
    print(K)
