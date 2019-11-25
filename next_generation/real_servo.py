#!/usr/bin python
# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
import numpy as np

G = 9.81


class Servo(object):
    def __init__(self, name, time_for_60_degrees, stall_torque_kfg_cm):
        super(Servo, self).__init__()
        self.name = name
        self.stall_torque_kfg_cm = stall_torque_kfg_cm
        self.max_w = np.radians(60) / time_for_60_degrees
        self.max_e = 2 * np.radians(60) / (time_for_60_degrees ** 2)
        self.time_for_45_degrees = time_for_60_degrees * 45 / 60


class MaxonServo(object):
    def __init__(self, name, t_acc, max_w, stall_torque):
        super(MaxonServo, self).__init__()
        self.name = name
        self.max_w = max_w
        self.mean_acc = max_w / (t_acc)
        self.stall_torque = stall_torque
        self.stall_torque_kfg_cm = stall_torque * 1000 / G

if __name__ == '__main__':
    # servo = MaxonServo('RX-28', 0.04315, 67 * 2 * np.pi / 60, 2.5)
    servo = MaxonServo('RX-24F', 0.04315, 126 * 2 * np.pi / 60, 2.6)

    joint_length = 0.15

    touchdown_speed = 2 * np.sin(np.pi / 4) * joint_length * servo.max_w

    supported_mass = 2 * servo.stall_torque_kfg_cm / (np.sqrt(2) * joint_length * 100 * G)

    # servo = Servo('DYNAMIXEL AX-12', 0.196, 16.5)
    # servo = Servo('DYNAMIXEL RX-28', 0.126, 37.7)


    # touchdown_speed = np.sqrt(2) * joint_length * servo.max_w

    # supported_mass = 2 * servo.stall_torque_kfg_cm / (np.sqrt(2) * joint_length * 100)

    print('touchdown speed: %f' % touchdown_speed)
    # print('aceleration time: %f' % servo.time_for_45_degrees)
    print('supported mass: %f' % supported_mass)
