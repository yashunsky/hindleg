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
