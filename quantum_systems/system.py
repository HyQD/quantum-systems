import numpy as np

from quantum_systems.system_helper import (
    transform_one_body_elements,
    transform_two_body_elements,
)


class QuantumSystem:
    """Base class defining some of the common methods used by all the different
    quantum systems.
    """

    def __init__(self, n, l):
        assert n <= l

        self.n = n
        self.l = l
        self.m = self.l - self.n

        self.o = slice(0, self.n)
        self.v = slice(self.n, self.l)

        self._h = None
        self._f = None
        self._off_diag_f = None
        self._u = None
        self._s = None
        self._dipole_moment = None
        self._polarization_vector = None

        self._propagator = None
        self._envelope = lambda t: 0

        self._spf = None

    def setup_system(self):
        pass

    def construct_fock_matrix(self):
        """Function setting up the Fock matrix"""
        o, v = (self.o, self.v)

        if self._f is None:
            self._f = np.zeros(self._h.shape, dtype=np.complex128)

        self._f.fill(0)
        self._f += np.einsum("piqi -> pq", self._u[:, o, :, o])
        self._f += self._h

        if self._off_diag_f is None:
            self._off_diag_f = np.zeros(self._h.shape, dtype=np.complex128)

        self._off_diag_f.fill(0)
        self._off_diag_f += self._f
        np.fill_diagonal(self._off_diag_f, 0)

    def cast_to_complex(self):
        self._h = self._h.astype(np.complex128)
        self._f = self._f.astype(np.complex128)
        self._off_diag_f = self._off_diag_f.astype(np.complex128)
        self._u = self._u.astype(np.complex128)

    def change_basis(self, c):
        self._h = transform_one_body_elements(self._h, c)
        self._u = transform_two_body_elements(self._u, c)

    @property
    def h(self):
        """Getter returning one-body matrix"""
        return self._h

    @property
    def f(self):
        """Getter returning one-body Fock matrix"""
        return self._f

    @property
    def off_diag_f(self):
        """Getter returning the one-body Fock matrix without the diagonal
        elements"""
        return self._off_diag_f

    @property
    def u(self):
        """Getter returning the antisymmetric two-body matrix"""
        return self._u

    @property
    def s(self):
        """Getter returning the overlap matrix of the atomic orbitals"""
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
        return transform_one_body_elements(self._h, c)

    def get_transformed_u(self, c):
        return transform_two_body_elements(self._u, c)

    def set_time_propagator(self, propagator):
        self._propagator = propagator
        self._propagator.set_system(self)

    def h_t(self, current_time):
        if self._propagator is None:
            return self._h

        return self._propagator.h_t(current_time)

    def u_t(self, current_time):
        if self._propagator is None:
            return self._u

        return self._propagator.u_t(current_time)
