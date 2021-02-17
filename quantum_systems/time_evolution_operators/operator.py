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


class DipoleFieldInteraction(TimeEvolutionOperator):
    r"""Semi-classical time-dependent interaction between particles and an
    electric field in the dipole approximation. The contribution to the
    Hamiltonian is on the form:

    .. math:: \hat{h}_I(t) = -\hat{\mathbf{d}} \cdot \mathbf{E}(t),

    where :math:`\hat{\mathbf{d}}` is the dipole moment operator, and
    :math:`\mathbf{E}(t)` is the time-dependent electric field. Note that this
    is a time-dependent one-body operator as the dipole moment is a one-body
    operator. The electric field is given by

    .. math:: \mathbf{E}(t) = \boldsymbol{\epsilon}(t)E(t),

    where :math:`\boldsymbol{\epsilon}(t)` is the polarization of the electric
    field, and :math:`E(t)` is the electric field strength. This function
    includes potential envelope functions.

    Parameters
    ----------
    electric_field_strength : callable, float
        Function describing the electric field strength at a given specific time
        ``current_time``. This function should accept a single parameter as the
        current time. A constant electric field is supported as well.
    polarization_vector : np.array
        Vector specifying the polarization axis of the electric field. This can
        also be provided as a time-dependent function returning a polarization
        axis as a function of ``current_time``. Default is ``None`` which gives
        a constant polarization along the first axis of the dipole moment
        integrals.
    """

    def __init__(self, electric_field_strength, polarization_vector=None):
        self._electric_field_strength = electric_field_strength
        self._polarization = polarization_vector

    @property
    def is_one_body_operator(self):
        return True

    def h_t(self, current_time):
        np = self._system.np

        if not callable(self._electric_field_strength):
            tmp = self._electric_field_strength
            self._electric_field_strength = lambda t: tmp

        if self._polarization is None:
            # Set default polarization along x-axis
            self._polarization = np.zeros(self._system.dipole_moment.shape[0])
            self._polarization[0] = 1

        if not callable(self._polarization):
            tmp = self._polarization
            self._polarization = lambda t: tmp

        return self._system.h - self._electric_field_strength(
            current_time
        ) * np.tensordot(
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


class CustomOneBodyOperator(TimeEvolutionOperator):
    def __init__(self, weight, operator):
        self._weight = weight
        self._operator = operator

    @property
    def is_one_body_operator(self):
        return True

    def h_t(self, current_time):
        np = self._system.np

        if not callable(self._weight):
            tmp = self._weight
            self._weight = lambda t: tmp

        return self._system.h + self._weight(current_time) * self._operator
