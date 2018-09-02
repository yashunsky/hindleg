#!/usr/bin/env python
# -*- coding: utf8 -*-

# phisics
G = 9.81  # m/s^2

# drive DSHV0507nТ — 6.4kg*cm  0.05s/60 degree
MAX_W = 60 / 0.05  # degree / s
MAX_TORQUE = 0.064 * G  # N*m

# entire future robot

# servos — 62g * 4
# battery — 300g
# control board — 20g
# misc — 32g
# sum = 600g

M = 0.3  # kg per leg if land on two

# best decceleration
X = -59.3  # mm
Y_MIN_UNBOUND = -250  # mm
Y_MAX_UNBOUND = -0  # mm
STEP = 0.5  # mm

Y_MIN = -222.0  # mm
Y_MAX = -56.0  # mm

# model
EPSILON = 0.00001
