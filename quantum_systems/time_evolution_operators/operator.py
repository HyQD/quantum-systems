import numpy as np


class TimeEvolutionOperator:
    def set_system(self, system):
        self._system = system

    def h_t(self, current_time):
        return self._system.h

    def u_t(self, current_time):
        return self._system.u


class LaserField(TimeEvolutionOperator):
    def __init__(self, laser_pulse):
        if not callable(laser_pulse):
            laser_pulse = lambda t: laser_pulse

        self._laser_pulse = laser_pulse

    def h_t(self, current_time):
        return self._system.h + self._laser_pulse(current_time) * np.tensordot(
            self._system.polarization_vector,
            self._system.dipole_moment,
            axes=(0, 0),
        )
