#!/usr/bin/env python
# -*- coding: utf8 -*-

from kinematic_model import optimal
from utils import vec

from consts import G
from consts import EPSILON, STEP
from consts import X, Y_MIN, Y_MAX
from consts import M, MAX_TORQUE

import numpy as np
from matplotlib import pyplot as plt


def get_sholders_and_torque_coeff(model, point):
    state = model.inverse_kinematics(point)
    if state is None:
        return None

    hip_angle = state.hip_angle
    knee_angle = state.knee_angle
    shin_angle = state.shin_angle

    knee_angle_delta = EPSILON

    delta_state = model.forward_kinematics(hip_angle,
                                           knee_angle + knee_angle_delta)

    shin_angle_delta = delta_state.shin_angle - shin_angle

    # knee_drive_torque * knee_angle_delta == knee_torque * shin_angle_delta

    torque_coeff = knee_angle_delta / shin_angle_delta

    hip_sholder = (state.foot - state.hip_s) / 1000  # mm -> m
    knee_sholder = (state.foot - state.hip_e) / 1000  # mm -> m

    return state, hip_sholder, knee_sholder, torque_coeff


def get_force(model, point, hip_drive_torque, knee_drive_torque):

    params = get_sholders_and_torque_coeff(model, point)

    if params is None:
        return None, np.nan

    state, hip_sholder, knee_sholder, torque_coeff = params

    hip_torque = hip_drive_torque
    knee_torque = knee_drive_torque * torque_coeff

    # F = M / d

    return state, hip_torque / hip_sholder + knee_torque / knee_sholder


def get_torques(model, point, shin_force):
    params = get_sholders_and_torque_coeff(model, point)

    if params is None:
        return None, np.nan, np.nan

    state, hip_sholder, knee_sholder, torque_coeff = params

    # shin_force = hip_torque / hip_sholder + knee_torque / knee_sholder
    #
    # hip_torque = hip_drive_torque
    #
    # knee_torque = knee_drive_torque * knee_angle_delta / shin_angle_delta
    # torque_coeff = knee_angle_delta / shin_angle_delta
    # knee_torque = knee_drive_torque * torque_coeff

    # A * X = B
    # X = invA * B

    A = np.vstack([1.0 / hip_sholder, torque_coeff / knee_sholder]).T
    B = shin_force

    hip_drive_torque, knee_drive_torque = (np.matrix(A).I * np.matrix(B).T).A1

    return state, hip_drive_torque, knee_drive_torque

if __name__ == '__main__':
    model = optimal()

    ys = np.arange(Y_MIN, Y_MAX, STEP)

    torques = np.array([get_torques(model, vec(X, y), vec(0.0, 2 * G * M))[1:]
                        for y in ys])

    torques[np.abs(torques) > MAX_TORQUE] = np.nan

    plt.plot(ys, torques)

    plt.show()
