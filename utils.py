#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np


def vec(*args):
    return np.array(args)


def get_angle(vector):
    x, y = vector
    return np.arctan2(y, x)


def delta(vecs):
    v1, v2 = vecs
    return v2 - v1


def rotated(angle):
    return vec(np.cos(angle), np.sin(angle))


def with_nan(*vectors):
    return np.isnan(vectors).any()
