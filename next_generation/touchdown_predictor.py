#!/usr/bin python
# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
import numpy as np

G = -9.81


def predict_touchdown(initial_height,
                      dampfer_initial_length, dampfer_remaining_length,
                      decceleration):
    # x_m(t) - center of mass position
    # x_i    - center of mass initial position
    # x_f(t) - feet position
    # d(t)   — dempfer lendth
    # d_i    - dempfer initial length
    # d_r    - dempfer remaining length
    # a      — max dempfer acceleration
    # v_td   — touchdown speed
    # t0     - decceleration start time
    # t1     - decceleration end time
    # t_td   - touchdoun time

    # x_m(t) = x_i + G * t^2 / 2
    # x_f(t) = x_m(t) - d(t)
    # d(t) = d_i + a * (t1 - t0) ^ 2 / 2 + (t - t1) * v_td
    #
    # v_td = a * (t1 - t0)
    # v_td = G * t_td
    # t1 - t0 = G * t_td / a = t_a

    # touchdown equation:
    # x_f(t_td) = 0
    # x_i + G * t_td^2 / 2 - d_i - a * t_a ^ 2 / 2 - a * (t_td - t1) * t_a = 0

    # adding conditions to reduce the number of variables
    # x_m(t_td) = d_r
    # d_r = x_i + G * t_td^2 / 2
    # t_td = sqrt(2 * (d_r - x_i) / G)
    #
    # a * (t_td - t1) * t_a = x_i + G * t_td^2 / 2 - d_i - a * t_a ^ 2 / 2
    # t_td - t1 = (d_r - d_i - a * t_a ^ 2 / 2) / (a * t_a)
    # t1 = t_td - (d_r - d_i - a * t_a ^ 2 / 2) / (a * t_a)
    # t0 = t1 - t_a

    x_i = initial_height
    d_i = dampfer_initial_length
    d_r = dampfer_remaining_length
    a = decceleration

    t_td = np.sqrt(2 * (d_r - x_i) / G)
    t_a = G * t_td / a
    t1 = t_td - (d_r - d_i - a * t_a ** 2 / 2) / (a * t_a)
    t0 = t1 - t_a

    #

    a = G * (x_i - d_r) / (d_i - d_r)

    t1 = np.sqrt((x_i - d_i) / (G ** 2 / (2 * a) - G / 2))
    t_a = G * t1 / a
    t0 = t1 - t_a
    t_td = t1

    v_td = G * t_td

    return {'t0': t0,
            't1': t1,
            'a': a,
            'touch_down_time': t_td,
            'touch_down_speed': v_td}

if __name__ == '__main__':
    initial_height = 0.5
    dampfer_initial_length = 0.3
    dampfer_remaining_length = dampfer_initial_length * 0.5
    decceleration = -20.0

    time_step = 0.0001

    data = predict_touchdown(initial_height=initial_height,
                             dampfer_initial_length=dampfer_initial_length,
                             dampfer_remaining_length=dampfer_remaining_length,
                             decceleration=decceleration)

    print(data)

    t0 = data['t0']
    t1 = data['t1']
    touch_down_time = data['touch_down_time']
    decceleration = data['a']

    ts = np.arange(0, touch_down_time, time_step)

    dempfer_model_acc = decceleration * np.ones(ts.shape)
    dempfer_model_acc[ts < t0] = 0
    dempfer_model_acc[ts > t1] = G
    dempfer_model_speed = np.cumsum(dempfer_model_acc) * time_step
    dempfer_model_length = np.cumsum(dempfer_model_speed) * time_step + dampfer_initial_length

    mass_position = initial_height + G * ts ** 2 / 2

    dempfer_length = np.ones(ts.shape) * dampfer_initial_length

    psedo_ts = np.copy(ts)
    psedo_ts[ts < t0] = t0
    psedo_ts[ts > t1] = t1

    dempfer_length += (decceleration * (psedo_ts - t0) ** 2 / 2)

    psedo_ts = np.copy(ts)
    psedo_ts[ts < t1] = t1

    final_speed = (t1 - t0) * decceleration

    dempfer_length += final_speed * (psedo_ts - t1)

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(5, 1)

    axes = [None] * 3

    axes[0] = fig.add_subplot(gs[:3, :])
    axes[1] = fig.add_subplot(gs[3, :])
    axes[2] = fig.add_subplot(gs[4, :])

    foot_position = mass_position - dempfer_length
    foot_model_position = mass_position - dempfer_model_length

    # axes[0].set_title('Touchdown prediction')
    axes[0].plot(ts, mass_position * 1000, label='Center of mass')
    axes[0].plot(ts, foot_model_position * 1000, label='Foot')
    axes[0].set_ylabel('Height (mm)')
    axes[0].grid(True)
    axes[0].legend(loc='lower left')
    axes[0].set_yticks(np.arange(0, 501, 100))
    axes[0].set_xticks(ts[::400])

    mass_speed = np.diff(mass_position) / time_step
    dempfer_speed = np.diff(dempfer_length) / time_step

    axes[1].plot(ts[1:], mass_speed, label='Center of mass')
    axes[1].plot(ts, dempfer_model_speed, label='Dampfer compression')
    axes[1].set_ylabel('Speed (m/s)')
    axes[1].grid(True)
    axes[1].legend(loc='lower left')
    axes[1].set_yticks(np.arange(-3.0, 0.01, 0.5))
    axes[1].set_xticks(ts[::400])

    mass_acc = np.diff(mass_speed) / time_step
    dempfer_acc = np.diff(dempfer_speed) / time_step

    axes[2].plot(ts[2:], mass_acc, label='Center of mass')
    axes[2].plot(ts, dempfer_model_acc, label='Dampfer compression')
    axes[2].set_ylabel('Acceleration (m/s²)')
    axes[2].grid(True)
    axes[2].legend(loc='lower left')
    axes[2].set_yticks(np.arange(-25, 0.01, 5))
    axes[2].set_xlabel('time (s)')
    axes[2].set_xticks(ts[::400])

    plt.show()
