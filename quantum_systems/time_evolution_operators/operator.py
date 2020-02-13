import abc


class TimeEvolutionOperator(metaclass=abc.ABCMeta):
    @property
    def is_one_body_operator(self):
        """Property used to determine if the time-evolution operator only
        applies to the one-body part of the Hamilonian.

        Returns
        -------
        out : bool
        """

        return False

    @property
    def is_two_body_operator(self):
        """Property used to determine if the time-evolution operator only
        applies to the one-body part of the Hamilonian.

        Returns
        -------
        out : bool
        """

        return False

    def set_system(self, system):
        """Internal function used to set callback system. This is done in the
        QuantumSystem-class and allows the user to specify the time-evolution
        operator parameters when setting the operator.

        Parameters
        ----------
        system : QuantumSystem
            A QuantumSystem instance to apply the time-evolution operator to.
        """

        self._system = system

    def h_t(self, current_time):
        """Function computing the one-body part of the Hamiltonian for a
        specified time.

        Parameters
        ----------
        current_time : float
            The time-point to evaluate the one-body part of the Hamiltonian.

        Returns
        -------
        out : ndarray
            The one-body part of the Hamiltonian evaluated at the specified
            time-point.
        """

        return self._system.h

    def u_t(self, current_time):
        """Function computing the two-body part of the Hamiltonian for a
        specified time.

        Parameters
        ----------
        current_time : float
            The time-point to evaluate the two-body part of the Hamiltonian.

        Returns
        -------
        out : ndarray
            The two-body part of the Hamiltonian evaluated at the specified
            time-point.
        """

        return self._system.u


class LaserField(TimeEvolutionOperator):
    def __init__(self, laser_pulse, polarization_vector=None):
        self._laser_pulse = laser_pulse
        self._polarization = polarization_vector

    @property
    def is_one_body_operator(self):
        return True

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


class AdiabaticSwitching(TimeEvolutionOperator):
    def __init__(self, switching_function):
        self._switching_function = switching_function

        if not callable(switching_function):
            self._switching_function = lambda t: switching_function

    @property
    def is_two_body_operator(self):
        return True

    def u_t(self, current_time):
        np = self._system.np

        return self._switching_function(current_time) * self._system.u
