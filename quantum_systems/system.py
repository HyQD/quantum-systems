import abc

import numpy as np


class QuantumSystem(metaclass=abc.ABCMeta):
    """Abstract base class defining some of the common methods used by all
    the different quantum systems.
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

        self._spf = None

    @abc.abstractmethod
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
    def spf(self):
        """Getter returning the single particle functions, i.e, the eigenstates
        of the non-interacting Hamiltonian"""
        return self._spf

    def h_t(self, current_time):
        return self._h

    def u_t(self, current_time):
        return self._u
