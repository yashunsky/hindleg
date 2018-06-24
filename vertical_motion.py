#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np

from kinematic_model import optimal
from work_area_model import draw_structure

from matplotlib import pyplot as plt

# -42.48105234 -52.59465547
X = -42.48105234
Y_MIN = -200
Y_MAX = -0
STEP = 0.5
MAX_SHIN_ANGLE = np.pi / 3


def get_limited_y_range(model, x, y_range, min_shin_angle, max_shin_angle):
    points = np.array((x * np.ones(y_range.shape), y)).T
    angles = np.array([get_angles(model, point, max_shin_angle)
                       for point in points])
    mask = np.logical_not(np.isnan(angles[:, 0]))
    return y_range[mask], points[mask], angles[mask]


def get_angles(model, point, max_shin_angle):
    state = model.inverse_kinematics(point)
    if state is None or state.shin_angle > max_shin_angle:
        return np.array((np.nan, np.nan, np.nan))
    else:
        return np.degrees(state.get_angles().values())


if __name__ == '__main__':
    model = optimal()

    y = np.arange(Y_MIN, Y_MAX, STEP)

    y, points, angles = get_limited_y_range(model, X, y, 0, MAX_SHIN_ANGLE)

    plt.plot(y, angles)
    plt.show()

    for point in points[:-1:20]:
        state = model.inverse_kinematics(point)
        if state is not None:
            draw_structure(state.get_joints().values())

    plt.axis('equal')
    plt.show()
