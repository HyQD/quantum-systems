class TimeEvolutionOperator:
    def set_system(self, system):
        self._system = system

    def h_t(self, current_time):
        return self._system.h

    def u_t(self, current_time):
        return self._system.u


class LaserField(TimeEvolutionOperator):
    def __init__(self, laser_pulse, polarization_vector=None):
        self._laser_pulse = laser_pulse
        self._polarization = polarization_vector

    def h_t(self, current_time):
        np = self._system.np

        if not callable(self._laser_pulse):
            tmp = self._laser_pulse
            self._laser_pulse = lambda t: tmp

        if self._polarization is None:
            # Set default polarization along x-axis
            self._polarization = np.zeros(self._system.dipole_moment.shape[0])
            self._polarization[0] = 1

        if not callable(self._polarization):
            tmp = self._polarization
            self._polarization = lambda t: tmp

        return self._system.h + self._laser_pulse(current_time) * np.tensordot(
            self._polarization(current_time),
            self._system.dipole_moment,
            axes=(0, 0),
        )
