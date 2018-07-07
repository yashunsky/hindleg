#!/usr/bin python
# -*- coding: utf8 -*-

import numpy as np
from scipy.optimize import minimize

from kinematic_model import HindLeg
from vertical_motion import get_max_speeds


def model_factory(min_shin_angle, base, resolution):
    def get_model(x):
        (base_x_offset, knee_rod,
         knee_connection_rod, knee_offset, hip, shin) = x
        hing_leg = HindLeg(min_shin_angle, base, base_x_offset,
                           knee_rod, knee_connection_rod, knee_offset,
                           hip, shin)
        return -max(get_max_speeds(hing_leg, resolution)[:, 1])  # hack to find "min" value
    return get_model


if __name__ == '__main__':

    MIN_SHIN_ANGLE = np.pi / 12

    ANGULAR_RESOLUTION = 1.0
    SPATIAL_RESOLUTION = 5.0

    BASE = 20.0
    INITIAL_BASE_X_OFFSET = 0
    INITIAL_KNEE_ROD = 20.0
    INITIAL_KNEE_CONNECTION_ROD = 70.0
    INITIAL_KNEE_OFFSET = 20.0
    INITIAL_HIP = 70.0
    INITIAL_SHIN = 150.0

    vertical_speed = model_factory(MIN_SHIN_ANGLE, BASE,
                                   SPATIAL_RESOLUTION)

    x0 = np.array((INITIAL_BASE_X_OFFSET, INITIAL_KNEE_ROD,
                   INITIAL_KNEE_CONNECTION_ROD,
                   INITIAL_KNEE_OFFSET, INITIAL_HIP, INITIAL_SHIN))

    res = minimize(vertical_speed, x0, method='Nelder-Mead')
    print res
