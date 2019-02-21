class TimeEvolutionOperator:
    def set_system(self, system):
        self._system = system

    def h_t(self, current_time):
        return self._system.h

    def u_t(self, current_time):
        return self._system.u


class LaserField(TimeEvolutionOperator):
    def __init__(self, laser_pulse, polarization_vector=None):
        if not callable(laser_pulse):
            laser_pulse = lambda t: laser_pulse

        self._laser_pulse = laser_pulse
        self._polarization_vector = polarization_vector

    def h_t(self, current_time):
        np = self._system.np

        if self._polarization_vector is None:
            # Set default polarization fully along x-axis
            self._polarization_vector = np.zeros(
                self._system.dipole_moment.shape[0]
            )
            self._polarization_vector[0] = 1

        return self._system.h + self._laser_pulse(current_time) * np.tensordot(
            self._system.polarization_vector,
            self._system.dipole_moment,
            axes=(0, 0),
        )
