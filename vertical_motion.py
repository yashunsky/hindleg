#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np

from kinematic_model import optimal
from work_area_model import draw_structure

from matplotlib import pyplot as plt

# -42.48105234 -52.59465547
X = -42.48105234


def get_angles(model, point):
    state = model.inverse_kinematics(point)
    if state is None:
        return np.array((np.nan, np.nan, np.nan))
    else:
        return np.degrees(state.get_angles().values())


if __name__ == '__main__':
    model = optimal()

    y = np.arange(-20, -200, -1)

    points = np.array((X * np.ones(y.shape), y)).T

    angles = np.array([get_angles(model, point) for point in points])

    plt.plot(y, angles)
    plt.show()

    for point in points[:-1:20]:
        state = model.inverse_kinematics(point)
        if state is not None:
            draw_structure(state.get_joints().values())

    plt.axis('equal')
    plt.show()
