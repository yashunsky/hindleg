#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
from scipy.optimize import minimize

from work_area_model import HindLeg, vec, angles_generator


def model_factory(min_shin_angle, base,
                  angles, resolution, mask=vec(True, True)):
    def get_model(x):
        (base_x_offset, knee_rod,
         knee_connection_rod, knee_offset, hip, shin) = x
        max_length = hip + shin - knee_offset
        hing_leg = HindLeg(min_shin_angle, base, base_x_offset,
                           knee_rod, knee_connection_rod, knee_offset,
                           hip, shin)
        result = hing_leg.get_max_cross_sections(angles, resolution)[mask]
        result = 1 - (result / max_length)
        print result
        return result[0] if len(result) == 1 else result
    return get_model


if __name__ == '__main__':

    ANGULAR_RESOLUTION = 5.0
    SPATIAL_RESOLUTION = 10.0

    BASE = 20.0
    INITIAL_KNEE_ROD = 20.0
    INITIAL_KNEE_CONNECTION_ROD = 70.0
    INITIAL_KNEE_OFFSET = 20.0
    INITIAL_HIP = 70.0
    INITIAL_SHIN = 150.0

    MASK = vec(False, True)

    ANGLES = list(angles_generator(hip_step=ANGULAR_RESOLUTION,
                                   knee_step=ANGULAR_RESOLUTION))

    model = model_factory(BASE, ANGLES, SPATIAL_RESOLUTION, mask=MASK)

    vertical_usage = model(INITIAL_KNEE_ROD,
                           INITIAL_KNEE_CONNECTION_ROD,
                           INITIAL_KNEE_OFFSET,
                           INITIAL_HIP,
                           INITIAL_SHIN)

    print vertical_usage
