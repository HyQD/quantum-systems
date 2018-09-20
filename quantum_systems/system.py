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
        self._u = None

    @abc.abstractmethod
    def setup_system(self):
        raise NotImplementedError("Use a specific quantum system")

    def construct_fock_matrix(self):
        o, v = (self.o, self.v)

        self._f = np.einsum("piqi -> pq", self.u[:, o, :, o])
        self._f += self.h

    @property
    def h(self):
        return self._h

    @property
    def f(self):
        return self._f

    @property
    def u(self):
        return self._u
