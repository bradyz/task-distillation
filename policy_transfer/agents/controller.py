from collections import deque

import numpy as np
import carla


PID_PARAMS = [
        dict(K_P=0.5,  K_I=4.0, K_D=0.1,  fps=20, n=40),
        dict(K_P=1.5,  K_I=3.0, K_D=0.1,  fps=20, n=40),
        dict(K_P=0.75,  K_I=2.0, K_D=0.2, fps=20, n=30),
        dict(K_P=1.0,  K_I=3.0, K_D=0.05, fps=20, n=80),
        dict(K_P=1.25,  K_I=3.0, K_D=0.05,  fps=20, n=80),
        ]


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, fps=10, n=20, **kwargs):
        print(K_P, K_I, K_D, fps, n, **kwargs)

        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._dt = 1.0 / fps
        self._n = n
        self._window = deque(maxlen=self._n)

    def step(self, error):
        self._window.append(error)

        if len(self._window) >= 2:
            integral = np.mean(self._window) * self._dt
            derivative = (self._window[-1] - self._window[-2]) / self._dt
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


class Controller(object):
    def __init__(self, pid):
        self.brake_threshold = 0.5

        self.turn_control = PIDController(**PID_PARAMS[pid])
        self.speed_control = PIDController(K_P=1.0, K_I=1.0, K_D=0.5, fps=20, n=100)

    def postprocess(self, steer, throttle, brake):
        control = carla.VehicleControl()
        control.steer = np.clip(steer, -1.0, 1.0)
        control.throttle = np.clip(throttle, 0.0, 1.0)
        control.brake = np.clip(brake, 0.0, 1.0)
        control.manual_gear_shift = False

        return control

    def run_step(self, target, target_speed, speed, viz=None):
        target_speed = np.clip(target_speed, 0.0, 7.5)
        delta = np.clip(target_speed - speed, 0.0, 1)
        theta = np.radians(90 - np.degrees(np.arctan2(target[1], target[0])))

        steer = self.turn_control.step(theta)
        throttle = self.speed_control.step(delta)
        brake = 0.0

        self.speed = speed
        self.target_speed = target_speed

        # Slow or stop.
        if target_speed < self.brake_threshold:
            steer = 0.0
            throttle = 0.0
            brake = 0.5

        control = self.postprocess(steer, throttle, brake)

        if viz:
            viz.controller_draw(speed, target_speed, control)

        return control
