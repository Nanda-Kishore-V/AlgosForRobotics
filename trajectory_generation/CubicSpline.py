#!/usr/bin/env python3
import numpy as np
from scipy.interpolate import CubicSpline

class Spline2D:
    def __init__(self, x, y):
        x = np.array(x)
        y = np.array(y)
        assert(x.shape == y.shape)
        self.x = x
        self.y = y
        self.s = np.linspace(0, 1, num=self.x.size)
        self.x_spline = None
        self.y_spline = None
        self.setup()

    def setup(self):
        self.x_spline = CubicSpline(self.s, self.x, bc_type='natural')
        self.y_spline = CubicSpline(self.s, self.y, bc_type='natural')

    def get_yaw(self, s):
        dx_i = self.x_spline(s, 1)
        dy_i = self.y_spline(s, 1)

        return np.arctan2(dy_i, dx_i)

    def get_curvature(self, s):
        dx_i = self.x_spline(s, 1)
        dy_i = self.y_spline(s, 1)
        ddx_i = self.x_spline(s, 2)
        ddy_i = self.y_spline(s, 2)

        return (dx_i * ddy_i - dy_i * ddx_i)/(dx_i**2 + dy_i**2)**1.5

    def get_discrete_points(self, s):
        xs = self.x_spline(s)
        ys = self.y_spline(s)
        yaws = self.get_yaw(s)
        ks = self.get_curvature(s)

        return xs, ys, yaws, ks

def main():
    x = np.arange(10)
    y = np.sin(x)
    spline = Spline2D(x, y)

    s = np.linspace(0, 1, num=100)
    xs, ys, yaws, ks = spline.get_discrete_points(s)

    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(x, y, 'o', label='data')
    plt.plot(xs, ys, '-', label='interp')
    plt.legend()
    plt.figure()
    plt.plot(s, yaws)
    plt.figure()
    plt.plot(s, ks)
    plt.show()

if __name__=="__main__":
    main()
