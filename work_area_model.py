#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np

from itertools import chain

from matplotlib import pyplot as plt


def draw_joint(*args):
    start, end = args[0] if len(args) == 1 else args
    m = np.array((start, end))
    plt.plot(m[:, 0], m[:, 1], linewidth=4)


def draw_structure(joints):
    for joint in joints:
        draw_joint(joint)


def vec(*args):
    return np.array(args)


def get_max_sequence_length(sequence):
    max_length = 0
    length = 0

    for element in chain(sequence, [0]):
        if element:
            length += 1
        else:
            if length > max_length:
                max_length = length
            length = 0
    return max_length


def get_max_cloud_cross_sections(points, resolution, draw=False):
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])

    N = int((x_max - x_min) / resolution) + 1
    M = int((y_max - y_min) / resolution) + 1

    matrix = np.zeros((M, N), dtype=np.int8)

    for point in points:
        x, y = point
        col = int((x - x_min) / resolution)
        row = M - int((y - y_min) / resolution) - 1
        matrix[row, col] = 1

    horizontal = max((get_max_sequence_length(row) for row in matrix))
    vertical = max((get_max_sequence_length(col) for col in matrix.T))

    if horizontal > 0:
        horizontal -= 1
    if vertical > 0:
        vertical -= 1

    if draw:
        corner = vec(x_min, y_min) + (resolution / 2)

        matrix_points = []

        for x in xrange(N):
            for y in xrange(M):
                if matrix[M - y - 1, x]:
                    matrix_points.append(corner + vec(x, y) * resolution)
        points = np.array(matrix_points)
        plt.plot(points[:, 0], points[:, 1], '.', markersize=8)

    return vec(horizontal, vertical) * resolution


def angles_generator(hip_min=0, hip_max=180, hip_step=5,
                     knee_min=0, knee_max=180, knee_step=5):
    for hip_angle in np.arange(hip_min, hip_max, hip_step):
        for knee_angle in np.arange(knee_min, knee_max, knee_step):
            yield np.radians(vec(hip_angle, knee_angle))


class HindLeg(object):
    def __init__(self, min_shin_angle,
                 base, base_x_offset,
                 knee_rod, knee_connection_rod, knee_offset,
                 hip, shin):
        super(HindLeg, self).__init__()
        self.min_shin_angle = min_shin_angle
        self.base = base
        self.base_x_offset = base_x_offset
        self.knee_rod = knee_rod
        self.knee_connection_rod = knee_connection_rod
        self.knee_offset = knee_offset
        self.hip = hip
        self.shin = shin

    def forward_kinematics(self, hip_angle, knee_angle):
        hip_angle = hip_angle - np.pi
        knee_angle = knee_angle - np.pi / 2

        knee_rod_s = vec(0, 0)
        knee_rod_e = knee_rod_s + self.knee_rod * vec(np.cos(knee_angle),
                                                      np.sin(knee_angle))

        hip_s = knee_rod_s + vec(-self.base_x_offset, -self.base)
        hip_e = hip_s + self.hip * vec(np.cos(hip_angle),
                                       np.sin(hip_angle))

        knee, shin_angle = self.find_knee(knee_rod_e, hip_e)

        foot = knee - self.shin * vec(np.cos(shin_angle),
                                      np.sin(shin_angle))

        joints = {'hip': (hip_s, hip_e),
                  'shin': (knee, foot),
                  'knee_rod': (knee_rod_s, knee_rod_e),
                  'knee_rod_connection': (knee_rod_e, knee)}
        structure = {'foot': foot,
                     'shin_angle': shin_angle,
                     'joints': joints}

        result = structure

        if np.isnan(np.min(knee)) or not self.is_shin_angle_ok(shin_angle):
            result = None

        return result

    def is_shin_angle_ok(self, shin_angle):
        return (shin_angle > self.min_shin_angle and
                shin_angle < np.pi - self.min_shin_angle)

    def find_knee(self, knee_rod_e, hip_e):
        x, y = (hip_e - knee_rod_e) * vec(1, -1)

        l1 = self.knee_connection_rod
        l2 = self.knee_offset

        c = np.sqrt(x ** 2 + y ** 2)

        d = (c ** 2 + l2 ** 2 - l1 ** 2) / (2 * c)

        shin_angle = np.pi - (np.arctan2(y, x) + np.arccos(d / l2))

        knee = hip_e + l2 * vec(np.cos(shin_angle),
                                np.sin(shin_angle))

        return knee, shin_angle

    def get_cloud(self, angles, draw=False):
        points = []
        for hip_angle, knee_angle in angles:
            structure = self.forward_kinematics(hip_angle, knee_angle)
            if structure is not None:
                points.append(structure['foot'])
        cloud = np.array(points)

        if draw:
            plt.plot(cloud[:, 0], cloud[:, 1], '.', markersize=2)

        return cloud

    def get_max_cross_sections(self, angles, resolution,
                               draw_cloud=False, draw_matrix=False):
        points = self.get_cloud(angles, draw_cloud)
        return get_max_cloud_cross_sections(points, resolution, draw_matrix)


def draw_model(min_shin_angle, base, x, angles, resolution):
    base_x_offset, knee_rod, knee_connection_rod, knee_offset, hip, shin = x
    hing_leg = HindLeg(min_shin_angle, base, base_x_offset,
                       knee_rod, knee_connection_rod, knee_offset, hip, shin)

    hing_leg.get_max_cross_sections(angles, resolution, True, False)

    structure = hing_leg.forward_kinematics(np.radians(140), np.radians(95))
    if structure is not None:
        draw_structure(structure['joints'].values())


if __name__ == '__main__':
    MIN_SHIN_ANGLE = np.pi / 12

    base = 20.0
    base_x_offset = 0.0
    knee_rod = 20.0
    knee_connection_rod = 70.0
    knee_offset = 20.0
    hip = 70.0
    shin = 150.0

    x0 = base_x_offset, knee_rod, knee_connection_rod, knee_offset, hip, shin

    # 0.000200925892, 21.0661968, 73.515422, 19.7268278,
    # 68.0328677, 134.233191
    x1 = (0.0, 21.1, 73.5, 19.7, 68.0, 134.2)

    angles = list(angles_generator(hip_step=2, knee_step=2))

    draw_model(MIN_SHIN_ANGLE, base, x0, angles, 5)
    draw_model(MIN_SHIN_ANGLE, base, x1, angles, 5)

    plt.axis('equal')
    plt.show()
