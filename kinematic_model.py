#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np

from utils import vec, get_angle, delta


def find_left_triangle(a, b, c_s, c_e, get_b_global_angle=False):
    c_vec = (c_e - c_s) * vec(1, -1)

    c = np.linalg.norm(c_vec)

    d = (c ** 2 + b ** 2 - a ** 2) / (2 * c)

    b_global_angle = np.pi - (get_angle(c_vec) + np.arccos(d / b))

    vertex = c_e + b * vec(np.cos(b_global_angle),
                           np.sin(b_global_angle))

    return (vertex, b_global_angle) if get_b_global_angle else vertex


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

        self.knee_base = vec(0, 0)
        self.hip_base = self.knee_base + vec(-self.base_x_offset, -self.base)

    def forward_kinematics(self, hip_angle, knee_angle):
        hip_angle = hip_angle - np.pi
        knee_angle = knee_angle - np.pi / 2

        knee_rod_s = self.knee_base
        knee_rod_e = knee_rod_s + self.knee_rod * vec(np.cos(knee_angle),
                                                      np.sin(knee_angle))

        hip_s = self.hip_base
        hip_e = hip_s + self.hip * vec(np.cos(hip_angle),
                                       np.sin(hip_angle))

        knee, shin_angle = find_left_triangle(self.knee_connection_rod,
                                              self.knee_offset,
                                              knee_rod_e, hip_e,
                                              get_b_global_angle=True)

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

    def inverse_kinematics(self, foot):

        free_shin = self.shin - self.knee_offset

        hip_e = find_left_triangle(self.hip,
                                   free_shin,
                                   self.hip_base, foot)

        knee = foot + (hip_e - foot) * self.shin / free_shin

        knee_rod_e = find_left_triangle(self.knee_rod,
                                        self.knee_connection_rod,
                                        self.knee_base, knee)

        knee_rod = (self.knee_base, knee_rod_e)
        hip = (self.hip_base, hip_e)
        shin = (foot, knee)
        knee_rod_connection = (knee_rod_e, knee)

        knee_angle = get_angle(delta(knee_rod)) + np.pi / 2
        hip_angle = get_angle(delta(hip)) + np.pi

        return hip_angle, knee_angle
