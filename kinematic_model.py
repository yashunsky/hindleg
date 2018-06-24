#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np

from utils import vec, get_angle, delta, rotated, with_nan


def find_left_triangle(a, b, c_s, c_e, get_b_global_angle=False):
    c_vec = (c_e - c_s) * vec(1, -1)

    c = np.linalg.norm(c_vec)

    d = (c ** 2 + b ** 2 - a ** 2) / (2 * c)

    b_global_angle = np.pi - (get_angle(c_vec) + np.arccos(d / b))

    vertex = c_e + b * rotated(b_global_angle)

    return (vertex, b_global_angle) if get_b_global_angle else vertex


class HindLegState(object):
    """docstring for HindLegState"""
    def __init__(self, hip_s, hip_e, knee_rod_s, knee_rod_e, foot, knee,
                 hip_angle, knee_angle, shin_angle):
        super(HindLegState, self).__init__()
        self.hip_s = hip_s
        self.hip_e = hip_e
        self.knee_rod_s = knee_rod_s
        self.knee_rod_e = knee_rod_e
        self.foot = foot
        self.knee = knee
        self.hip_angle = hip_angle
        self.knee_angle = knee_angle
        self.shin_angle = shin_angle

    def get_joints(self):
        return {'hip': (self.hip_s, self.hip_e),
                'shin': (self.knee, self.foot),
                'knee_rod': (self.knee_rod_s, self.knee_rod_e),
                'knee_rod_connection': (self.knee_rod_e, self.knee)}

    def get_angles(self):
        return {'hip': self.hip_angle,
                'knee': self.knee_angle,
                'shin': self.shin_angle}

    def __str__(self):
        return str({'joints': self.get_joints(),
                    'angles': self.get_angles()})


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
        hip_global_angle = hip_angle - np.pi
        knee_global_angle = knee_angle - np.pi / 2

        knee_rod_s = self.knee_base
        knee_rod_e = knee_rod_s + self.knee_rod * rotated(knee_global_angle)

        hip_s = self.hip_base
        hip_e = hip_s + self.hip * rotated(hip_global_angle)

        knee, shin_angle = find_left_triangle(self.knee_connection_rod,
                                              self.knee_offset,
                                              knee_rod_e, hip_e,
                                              get_b_global_angle=True)

        if self.invalide_state(shin_angle, knee):
            return None

        foot = knee - self.shin * rotated(shin_angle)

        return HindLegState(hip_s=hip_s, hip_e=hip_e,
                            knee_rod_s=knee_rod_s, knee_rod_e=knee_rod_e,
                            foot=foot, knee=knee,
                            hip_angle=hip_angle, knee_angle=knee_angle,
                            shin_angle=shin_angle)

    def invalide_state(self, shin_angle, *points):
        return with_nan(points) or not self.is_shin_angle_ok(shin_angle)

    def is_shin_angle_ok(self, shin_angle):
        return (shin_angle > self.min_shin_angle and
                shin_angle < np.pi - self.min_shin_angle)

    def inverse_kinematics(self, foot):
        hip_s = self.hip_base
        knee_rod_s = self.knee_base
        free_shin = self.shin - self.knee_offset

        hip_e, shin_angle = find_left_triangle(self.hip, free_shin,
                                               self.hip_base, foot,
                                               get_b_global_angle=True)

        knee = foot + (hip_e - foot) * self.shin / free_shin

        knee_rod_e = find_left_triangle(self.knee_rod,
                                        self.knee_connection_rod,
                                        self.knee_base, knee)

        if self.invalide_state(shin_angle, knee, knee_rod_e):
            return None

        knee_rod = (self.knee_base, knee_rod_e)
        hip = (self.hip_base, hip_e)

        knee_angle = get_angle(delta(knee_rod)) + np.pi / 2
        hip_angle = get_angle(delta(hip)) + np.pi

        return HindLegState(hip_s=hip_s, hip_e=hip_e,
                            knee_rod_s=knee_rod_s, knee_rod_e=knee_rod_e,
                            foot=foot, knee=knee,
                            hip_angle=hip_angle, knee_angle=knee_angle,
                            shin_angle=shin_angle)
