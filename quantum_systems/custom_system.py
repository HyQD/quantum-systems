import numpy as np

from quantum_systems.system import QuantumSystem
from quantum_systems.system_helper import (
    add_spin_one_body,
    add_spin_two_body,
    anti_symmetrize_u,
)


class CustomSystem(QuantumSystem):
    """Custom quantum system where a user can pass in matrix elements from
    other sources. The purpose of this class is to allow usage of quantum
    solvers made by the author and collaborators using other sources of matrix
    elements.
    """

    def __init__(self, n, l):
        super().__init__(n, l)

    def set_h(self, h, add_spin=False):
        if add_spin:
            h = add_spin_one_body(h)

        self._h = h

    def set_u(self, u, add_spin=False, anti_symmetrize=False):
        if add_spin:
            u = add_spin_two_body(u)

        if anti_symmetrize:
            u = anti_symmetrize_u(u)

        self._u = u

    def set_s(self, s, add_spin=False):
        if add_spin:
            s = add_spin_one_body(s)

        self._s = s

    def set_dipole_moment(self, dipole_moment, add_spin=False):
        if len(dipole_moment.shape) < 3:
            dipole_moment = np.array([dipole_moment])

        if not add_spin:
            self._dipole_moment = dipole_moment
            return

        new_shape = tuple(map(lambda x: x * 2, dipole_moment.shape))
        self._dipole_moment = np.zeros(new_shape)
        for i in range(len(dipole_moment)):
            self._dipole_moment[i] = add_spin_one_body(dipole_moment[i])
