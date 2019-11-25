#!/usr/bin python
# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
import numpy as np

from touchdown_predictor import predict_touchdown

G = -9.81


class Properties(object):
    def __init__(self, mass, stiffness_coeff, damping_ratio,
                 elastic_deformation_range, active_damping_range):
        super(Properties, self).__init__()
        self.mass = mass
        self.stiffness_coeff = stiffness_coeff
        self.damping_ratio = damping_ratio
        self.elastic_deformation_range = elastic_deformation_range
        self.active_damping_range = active_damping_range


class State(object):
    def __init__(self,
                 t, position, speed, acc,
                 dampfer_height, elastic_height,
                 previous_deformation):
        super(State, self).__init__()
        self.t = t
        self.position = position
        self.speed = speed
        self.acc = acc
        self.dampfer_height = dampfer_height
        self.elastic_height = elastic_height
        self.previous_deformation = previous_deformation

    def get_vector(self):
        return np.array([self.t, self.position, self.speed, self.acc,
                         self.elastic_height, self.dampfer_height])


class DampController(object):
    def __init__(self):
        super(DampController, self).__init__()

    def get_height(self, state, t):
        return None


class NoDampController(DampController):
    def __init__(self, constant_height):
        super(NoDampController, self).__init__()
        self.constant_height = constant_height

    def get_height(self, state, t):
        return self.constant_height


class ContactlessController(DampController):
    def __init__(self, active_damping_range):
        super(ContactlessController, self).__init__()
        self.dampfer_initial_length = active_damping_range
        self.dampfer_remaining_length = active_damping_range * 0.5
        self.damper_acc = -40

        self.height = active_damping_range
        self.speed = 0
        self.acc = 0
        self.touch_down_time = None
        self.t0 = None
        self.t1 = None
        self.t2 = None
        self.t3 = None
        self.post_touch_acc = None

    def get_height(self, state, t):
        if state is not None and self.touch_down_time is None:
            data = predict_touchdown(initial_height=state.position - state.elastic_height,
                                     dampfer_initial_length=self.dampfer_initial_length,
                                     dampfer_remaining_length=self.dampfer_remaining_length,
                                     decceleration=self.damper_acc)

            self.t0 = data['t0']
            self.t1 = data['t1']
            self.touch_down_time = data['touch_down_time']
            self.t2 = self.touch_down_time
            self.damper_acc = data['a']

            print(self.t0, self.t1 - self.t0, data['touch_down_speed'])

            v_td = data['touch_down_speed']
            d_r = self.dampfer_remaining_length
            t_f = - 2 * d_r / v_td
            self.t3 = self.t2 + t_f
            self.post_touch_acc = - v_td / t_f

        if self.touch_down_time is not None:
            dt = t - state.t

            if t >= self.t0 and t < self.t1:
                self.acc = self.damper_acc
            elif t < self.t2:
                self.acc = 0
            elif t < self.t3:
                self.acc = self.post_touch_acc
            else:
                self.acc = 0

            self.speed += self.acc * dt
            self.height += self.speed * dt

        return self.height


class PsedoLeg(object):
    def __init__(self, properties):
        super(PsedoLeg, self).__init__()
        self.mass = properties.mass
        self.stiffness_coeff = properties.stiffness_coeff
        self.damping_ratio = properties.damping_ratio
        self.elastic_deformation_range = properties.elastic_deformation_range
        self.active_damping_range = properties.active_damping_range
        self.damp_controller = DampController()
        self.name = None
        self.color = None

    def identify(self, name, color):
        self.name = name
        self.color = color

    def set_damp_controller(self, controller):
        self.damp_controller = controller

    def get_stiffness_coeff(self, deformation):
        return self.stiffness_coeff

    def with_damp_controller(self, controller):
        return PsedoLeg(self.mass, self.stiffness_coeff, self.damping_ratio,
                        self.elastic_deformation_range,
                        self.active_damping_range,
                        controller)

    def get_next_state(self, prev, t):
        new_deformation = prev.elastic_height - self.elastic_deformation_range

        dt = t - prev.t

        m = self.mass
        g = G
        k = self.get_stiffness_coeff(new_deformation)
        x = new_deformation
        c = self.damping_ratio
        v = (new_deformation - prev.previous_deformation) / dt

        f = m * g - k * x - c * v

        new_acc = f / m
        new_speed = prev.speed + new_acc * dt
        new_position = prev.position + new_speed * dt
        new_dampfer_height = self.damp_controller.get_height(prev, t)
        new_elastic_height = min(new_position - new_dampfer_height,
                                 self.elastic_deformation_range)

        if new_elastic_height < 0:
            print('robot is broken at %f' % t)
            new_elastic_height = np.NAN

        return State(t=t, position=new_position, speed=new_speed, acc=new_acc,
                     dampfer_height=new_dampfer_height,
                     elastic_height=new_elastic_height,
                     previous_deformation=new_deformation)

    def states_generator(self, drop_height, model_time, time_step):
        ts = np.arange(0, model_time, time_step)

        state = State(t=0.0, position=drop_height, speed=0.0, acc=G,
                      dampfer_height=self.damp_controller.get_height(None, 0),
                      elastic_height=self.elastic_deformation_range,
                      previous_deformation=0.0)

        yield state.get_vector()

        for t in ts[1:]:
            state = self.get_next_state(state, t)
            yield state.get_vector()

    def model(self, drop_height, model_time, time_step):
        return np.array(list(self.states_generator(drop_height,
                                                   model_time,
                                                   time_step)))


class NoDampPsedoLeg(PsedoLeg):
    def __init__(self, properties):
        super(NoDampPsedoLeg, self).__init__(properties=properties)
        controller = NoDampController(properties.active_damping_range / 2)
        self.set_damp_controller(controller)
        self.identify(name='no damp', color='r')


class LinearPsedoLeg(PsedoLeg):
    def __init__(self, properties):
        super(LinearPsedoLeg, self).__init__(properties=properties)
        controller = ContactlessController(properties.active_damping_range)
        self.set_damp_controller(controller)
        self.identify(name='active damp', color='g')


class MockPlot(object):
    def __init__(self):
        super(MockPlot, self).__init__()

    def plot(self, *args, **kwargs):
        pass

    def set_xlabel(self, *args, **kwargs):
        pass

    def set_ylabel(self, *args, **kwargs):
        pass

    def grid(self, *args, **kwargs):
        pass

    def legend(self, *args, **kwargs):
        pass

    def set_xticks(self, *args, **kwargs):
        pass

    def set_yticks(self, *args, **kwargs):
        pass


def compare(legs, drop_height, model_time, time_step):

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(5, 1)

    axes = [MockPlot()] * 6

    # axes[0] = fig.add_subplot(gs[:3, :])
    # axes[1] = fig.add_subplot(gs[3, :])
    # axes[2] = fig.add_subplot(gs[4, :])

    axes[3] = fig.add_subplot(gs[2, :])
    axes[4] = fig.add_subplot(gs[3, :])
    axes[5] = fig.add_subplot(gs[4, :])

    for leg in legs:
        name = leg.name
        color = leg.color

        data = leg.model(drop_height, model_time, time_step)
        ts = data[:, 0]

        foot = data[:, 1] - data[:, 4] - data[:, 5]

        axes[0].plot(ts, data[:, 1] * 1000, color, label=name + ': center of mass')
        axes[0].plot(ts, foot * 1000, '-.%s' % color,label=name + ': foot', linewidth=1)

        axes[1].plot(ts, (data[:, 4] - leg.elastic_deformation_range) * 1000, color)
        axes[2].plot(ts, data[:, 5] * 1000, color)

        axes[3].plot(ts, data[:, 2], color)

        damp_speed = np.diff(data[:, 5]) / time_step

        # axes[3].plot(ts[1:], damp_speed, '-.%s' % color, linewidth=1)

        damp_acc = np.diff(damp_speed) / time_step
        axes[4].plot(ts[2:], damp_acc, '-.%s' % color, linewidth=1)

        axes[5].plot(ts, data[:, 3], color)
        print(np.max(data[:, 3]))

    axes[0].set_ylabel('Height (mm)', fontsize='small')
    axes[0].grid(True)
    axes[0].legend(loc='upper right')
    axes[0].set_yticks(np.arange(0, 501, 100))

    axes[1].set_ylabel('Deformation (mm)', fontsize='small')
    axes[1].grid(True)
    axes[1].set_yticks(np.arange(-10, 0, 2))

    axes[2].set_ylabel('Dampfer height (mm)', fontsize='small')
    axes[2].grid(True)
    axes[2].set_yticks(np.arange(0, 301, 50))

    axes[3].set_ylabel('Speed (m/s)', fontsize='small')
    axes[3].grid(True)
    axes[3].set_yticks(np.arange(-3, 2, 1))

    axes[4].set_ylabel('Dampfer compression\nacceleration (m/s²)', fontsize='small')
    axes[4].grid(True)
    axes[4].set_yticks(np.arange(-50, 40, 20))

    axes[5].set_ylabel('Acceleration (m/s²)', fontsize='small')
    axes[5].grid(True)
    axes[5].set_yticks(np.arange(-100, 500, 100))

    axes[5].set_xlabel('time (s)')

    plt.show()


if __name__ == '__main__':
    properties = Properties(mass=2.5, stiffness_coeff=100000.0,
                            damping_ratio=250.0,
                            elastic_deformation_range=0.01,
                            active_damping_range=0.3)

    no_damping = NoDampPsedoLeg(properties)
    linear = LinearPsedoLeg(properties)

    legs = [no_damping, linear]
    compare(legs, drop_height=0.5, model_time=0.8, time_step=0.0001)
