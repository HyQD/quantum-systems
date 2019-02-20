from quantum_systems.system_helper import (
    transform_one_body_elements,
    transform_two_body_elements,
)


class QuantumSystem:
    """Base class defining some of the common methods used by all the different
    quantum systems.
    """

    def __init__(self, n, l, np=None):
        assert n <= l

        if np is None:
            import numpy as np

        self.np = np

        self.n = n
        self.l = l
        self.m = self.l - self.n

        self.o = slice(0, self.n)
        self.v = slice(self.n, self.l)

        self._h = None
        self._f = None
        self._u = None
        self._s = None
        self._dipole_moment = None
        self._polarization_vector = None

        self._time_evolution_operator = None
        self._envelope = lambda t: 0

        self._spf = None

    def setup_system(self):
        pass

    def construct_fock_matrix(self, h, u, f=None):
        """Function setting up the Fock matrix"""
        np = self.np
        o, v = (self.o, self.v)

        if f is None:
            f = np.zeros_like(h)

        f.fill(0)
        f += np.einsum("piqi -> pq", u[:, o, :, o])
        f += h

        return f

    def cast_to_complex(self):
        np = self.np

        self._h = self._h.astype(np.complex128)
        self._u = self._u.astype(np.complex128)

        if self._f is not None:
            self._f = self._f.astype(np.complex128)

    def change_basis(self, c):
        self._h = transform_one_body_elements(self._h, c, np=self.np)

        for i in range(self._dipole_moment.shape[0]):
            self._dipole_moment[i] = transform_one_body_elements(
                self._dipole_moment[i], c, np=self.np
            )

        self._u = transform_two_body_elements(self._u, c, np=self.np)
        self._f = self.construct_fock_matrix(self._h, self._u)

    @property
    def h(self):
        """Getter returning one-body matrix"""
        return self._h

    @property
    def f(self):
        """Getter returning one-body Fock matrix"""
        return self._f

    @f.setter
    def f(self, f):
        self._f = f

    @property
    def u(self):
        """Getter returning the antisymmetric two-body matrix"""
        return self._u

    @property
    def s(self):
        """Getter returning the overlap matrix of the atomic orbitals"""
        np = self.np

        if self._s is None:
            self._s = np.eye(*self._h.shape)

        return self._s

    @property
    def dipole_moment(self):
        return self._dipole_moment

    @property
    def polarization_vector(self):
        return self._polarization_vector

    @property
    def spf(self):
        """Getter returning the single particle functions, i.e, the eigenstates
        of the non-interacting Hamiltonian"""
        return self._spf

    def get_transformed_h(self, c):
        return transform_one_body_elements(self._h, c, np=self.np)

    def get_transformed_u(self, c):
        return transform_two_body_elements(self._u, c, np=self.np)

    def set_time_evolution_operator(self, time_evolution_operator):
        self._time_evolution_operator = time_evolution_operator
        self._time_evolution_operator.set_system(self)

    def set_polarization_vector(self, polarization_vector):
        self._polarization_vector = polarization_vector

    def h_t(self, current_time):
        if self._time_evolution_operator is None:
            return self._h

        return self._time_evolution_operator.h_t(current_time)

    def u_t(self, current_time):
        if self._time_evolution_operator is None:
            return self._u

        return self._time_evolution_operator.u_t(current_time)
