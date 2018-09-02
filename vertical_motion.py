#!/usr/bin python
# -*- coding: utf8 -*-

import numpy as np

from kinematic_model import optimal
from dynamic_model import get_torques
from utils import vec

from matplotlib import pyplot as plt

from consts import G
from consts import X, Y_MIN_UNBOUND as Y_MIN, Y_MAX_UNBOUND as Y_MAX
from consts import STEP
from consts import M, MAX_W, MAX_TORQUE


def get_limited_y_range(model, x, y_range, max_torque, force):
    points = np.array((x * np.ones(y_range.shape), y_range)).T
    angles = np.array([get_angles(model, point, max_torque, force)
                       for point in points])
    mask = np.logical_not(np.isnan(angles[:, 0]))
    return y_range[mask], points[mask], angles[mask]


def get_angles(model, point, max_torque, force):
    state, hip_torque, knee_torque = get_torques(model, point, force)

    if state is None or hip_torque > max_torque or knee_torque > max_torque:
        return np.ones(3) * np.nan
    else:
        return np.degrees(state.get_angles().values())


def stack_to_matrix(vector, size):
    return vector * np.ones((size, 1))


def deceleration_matrix(deceleration, width):
    size = deceleration.size
    result = np.zeros((size, width))
    for i in xrange(width):
        limit = i - width + size
        limit = limit if limit > 0 else 0
        result[0:limit, i] = np.nan
        result[limit:, i] = deceleration[:size - limit]
    return result


def get_max_deceleratable_speed(model, deceleration,
                                robot_mass,
                                max_angular_speed,
                                max_torque,
                                x, y_min, y_max, y_step,
                                plot=False, each_nth_curve=1):

    y = np.arange(y_min, y_max, y_step)

    force = robot_mass * vec(0.0, deceleration)

    y, points, angles = get_limited_y_range(model, x, y,
                                            max_torque, force)

    heights = (y - y[0]) / 1000  # m

    das = np.diff(angles[:, :2], axis=0)

    min_times = np.max(np.abs(das), axis=1) / max_angular_speed

    dhs = np.diff(heights)

    max_v_speed = dhs / min_times  # m/s

    start_v_speed = np.hstack((np.arange(0, max_v_speed[0], max_v_speed[1] - max_v_speed[0]), max_v_speed))

    dvs = -2 * deceleration * np.linspace(0, heights[-1], dhs.size)

    start_speed_matrix = stack_to_matrix(start_v_speed, dhs.size)

    speed_matrix = start_speed_matrix ** 2 + deceleration_matrix(dvs, start_v_speed.size)

    speed_matrix = np.sqrt(speed_matrix * (speed_matrix > 0))

    max_speed_matrix = stack_to_matrix(max_v_speed, start_v_speed.size).T

    speed_matrix[speed_matrix > max_speed_matrix] = np.nan

    mask = speed_matrix[-1, :] == 0

    possible_curves = speed_matrix[:, mask]
    best_curve = np.max(possible_curves, axis=1)
    height_mask = -np.isnan(best_curve)

    possible_heights = heights[:-1][height_mask]

    if plot:
        plt.plot(heights[:-1], max_v_speed)
        plt.plot(heights[:-1], speed_matrix[:, ::each_nth_curve])

        plt.plot(heights[:-1], best_curve, linewidth=2)

        plt.show()

    max_possible_speed = best_curve[-np.isnan(best_curve)][0]
    start_height = min(possible_heights) * 1000 + y[0]
    stop_height = max(possible_heights) * 1000 + y[0]

    return max_possible_speed, start_height, stop_height


def get_max_speeds(model, resolution):
    x_max = model.hip
    x_min = -model.hip
    y_min = -(model.hip + model.shin - model.knee_offset)
    y_max = 0

    return np.array([[x, get_max_deceleratable_speed(model, G, MAX_W,
                                                     x, y_min, y_max,
                                                     resolution)[0]]
                     for x in np.arange(x_min, x_max, resolution)])


if __name__ == '__main__':
    model = optimal()

    # print max(get_max_speeds(model, STEP), key=lambda x: x[1])
    # [-59.3         1.7997635]

    speed, h_min, h_max = get_max_deceleratable_speed(model, G,
                                                      M, MAX_W, MAX_TORQUE,
                                                      X, Y_MIN, Y_MAX, STEP,
                                                      True, 10)
    fall_height = (speed ** 2) / (2 * G)

    print 'terminal speed:', speed, 'm/s'
    print 'fall height:', fall_height, 'm'
    print 'start height:', h_min, 'mm'
    print 'stop height:', h_max, 'mm'
