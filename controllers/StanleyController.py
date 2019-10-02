#!/usr/bin/env python3
import numpy as np

class StanleyController:
    """
    Stanley controller for lateral control
    Assumes rear axle model of the car
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

    def get_steer_input(self, x, y, yaw, v, waypoints):
        self.update_position(x, y)
        self.update_yaw(yaw)
        self.update_speed(v)

        x_f = self.x + self.L * np.cos(self.yaw)
        y_f = self.y + self.L * np.sin(self.yaw)

        cx = waypoints[0]
        cy = waypoints[1]
        # cyaw = waypoints[2]

        distances = np.sum(( np.array([[x_f], [y_f]]) - np.stack((cx, cy)) )**2, axis=0)
        idx = np.argmin(distances)
        cte = distances[idx]

        if idx != len(waypoints[0]):
            desired_heading = np.arctan2(cy[idx+1] - cy[idx], cx[idx+1] - cx[idx])
        else:
            desired_heading = np.arctan2(cy[idx] - cy[idx-1], cx[idx] - cx[idx-1])
        # desired_heading = cyaw[idx]
        heading_error = desired_heading - self.yaw
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        target_x, target_y = cx[idx], cy[idx]

        ##### Method 1 #############
        dx = x_f - target_x
        dy = y_f - target_y
        front_vec = [-np.cos(self.yaw + np.pi / 2),
                      - np.sin(self.yaw + np.pi / 2)]
        cte = np.dot([dx, dy], front_vec)

        ##### Method 2 #############
        # yaw_ct2vehicle = np.arctan2(y_f - target_y, x_f - target_x)
        # yaw_ct2heading = desired_heading - yaw_ct2vehicle
        # yaw_ct2heading = np.arctan2(np.sin(yaw_ct2heading), np.cos(yaw_ct2heading))
        # cte *= np.sign(yaw_ct2heading)

        steer = heading_error + np.arctan2(self.K * cte, self.v)
        self.delta = steer

        return steer
