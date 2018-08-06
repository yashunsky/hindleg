#!/usr/bin python
# -*- coding: utf8 -*-

import numpy as np

from kinematic_model import optimal

from matplotlib import pyplot as plt

from consts import G
from consts import X, Y_MIN_UNBOUND as Y_MIN, Y_MAX_UNBOUND as Y_MAX
from consts import STEP
from consts import MAX_W


def get_limited_y_range(model, x, y_range):
    points = np.array((x * np.ones(y_range.shape), y_range)).T
    angles = np.array([get_angles(model, point) for point in points])
    mask = np.logical_not(np.isnan(angles[:, 0]))
    return y_range[mask], points[mask], angles[mask]


def get_angles(model, point):
    state = model.inverse_kinematics(point)
    if state is None:
        return np.array((np.nan, np.nan, np.nan))
    else:
        return np.degrees(state.get_angles().values())


def stack_to_matrix(vector):
    return vector * np.ones((vector.size, 1))


def deceleration_matrix(deceleration):
    size = deceleration.size
    result = np.zeros((size, size))
    for i in xrange(size):
        result[0:i, i] = np.nan
        result[i:, i] = deceleration[:size - i]
    return result


def get_max_deceleratable_speed(model, deceleration, max_angular_speed,
                                x, y_min, y_max, y_step,
                                plot=False, each_nth_curve=1):

    y = np.arange(y_min, y_max, y_step)

    y, points, angles = get_limited_y_range(model, x, y)

    heights = (y - y[0]) / 1000  # m

    das = np.diff(angles[:, :2], axis=0)

    min_times = np.max(np.abs(das), axis=1) / max_angular_speed

    dhs = np.diff(heights)

    max_v_speed = dhs / min_times  # m/s

    dvs = -2 * deceleration * np.linspace(0, heights[-1], max_v_speed.size)

    max_speed_matrix = stack_to_matrix(max_v_speed)

    speed_matrix = max_speed_matrix ** 2 + deceleration_matrix(dvs)

    speed_matrix = np.sqrt(speed_matrix * (speed_matrix > 0))

    speed_matrix[speed_matrix > max_speed_matrix.T] = np.nan

    mask = speed_matrix[-1, :] == 0

    possible_curves = speed_matrix[:, mask]
    best_curve = np.max(possible_curves, axis=1)
    height_mask = -np.isnan(best_curve)

    possible_speeds = max_v_speed[mask]

    possible_heights = heights[:-1][height_mask]

    if plot:
        plt.plot(heights[:-1], max_v_speed)
        plt.plot(heights[:-1], speed_matrix[:, ::each_nth_curve])

        plt.plot(heights[:-1], best_curve, linewidth=2)

        plt.show()

    max_possible_speed = (max(possible_speeds)
                          if possible_speeds.size > 0 else 0)
    start_height = min(possible_heights)
    stop_height = max(possible_heights)

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

    speed, h_min, h_max = get_max_deceleratable_speed(model, G, MAX_W,
                                                      X, Y_MIN, Y_MAX, STEP,
                                                      True, 5)
    fall_height = (speed ** 2) / (2 * G)

    print 'terminal speed:', speed, 'm/s'
    print 'fall height:', fall_height, 'm'
    print 'start height:', h_min * 1000 + Y_MIN, 'mm'
    print 'stop height:', h_max * 1000 + Y_MIN, 'mm'
