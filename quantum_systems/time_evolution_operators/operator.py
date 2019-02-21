class TimeEvolutionOperator:
    def set_system(self, system):
        self._system = system

    def h_t(self, current_time):
        return self._system.h

    def u_t(self, current_time):
        return self._system.u


class LaserField(TimeEvolutionOperator):
    def __init__(self, laser_pulse, polarization_vector=None):
        np = self._system.np

        self._laser_pulse = (
            laser_pulse if callable(laser_pulse) else lambda t: laser_pulse
        )

        if polarization_vector is None:
            # Set default polarization fully along x-axis
            polarization_vector = np.zeros(self._system.dipole_moment.shape[0])
            polarization_vector[0] = 1

        self._polarization = (
            polarization_vector
            if callable(polarization_vector)
            else lambda t: polarization_vector
        )

    def h_t(self, current_time):
        return self._system.h + self._laser_pulse(current_time) * np.tensordot(
            self._polarization(current_time),
            self._system.dipole_moment,
            axes=(0, 0),
        )
