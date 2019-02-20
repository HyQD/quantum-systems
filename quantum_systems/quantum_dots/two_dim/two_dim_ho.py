import numpy as np
from quantum_systems import QuantumSystem
from quantum_systems.quantum_dots.two_dim.two_dim_helper import (
    get_coulomb_elements,
    get_one_body_elements,
    get_indices_nm,
    spf_state,
)
from quantum_systems.system_helper import (
    add_spin_one_body,
    add_spin_two_body,
    anti_symmetrize_u,
)


class TwoDimensionalHarmonicOscillator(QuantumSystem):
    def __init__(self, n, l, radius_length, num_grid_points, omega=1, mass=1):
        super().__init__(n, l)

        self.omega = omega
        self.mass = mass

        self.radius_length = radius_length
        self.num_grid_points = num_grid_points

        self.radius = np.linspace(0, self.radius_length, self.num_grid_points)
        self.theta = np.linspace(0, 2 * np.pi, self.num_grid_points)

        self._spf = np.zeros(
            (self.l // 2, self.num_grid_points, self.num_grid_points),
            dtype=np.complex128,
        )

    def setup_system(self):
        self.__h = self.omega * get_one_body_elements(self.l // 2).astype(
            np.complex128
        )
        self.__u = np.sqrt(self.omega) * get_coulomb_elements(
            self.l // 2
        ).astype(np.complex128)

        self._h = add_spin_one_body(self.__h, np=np)
        self._u = anti_symmetrize_u(add_spin_two_body(self.__u, np=np))
        self._f = self.construct_fock_matrix(self._h, self._u)

        self.cast_to_complex()

        self.setup_spf()

        if np is not self.np:
            self._h = self.np.asarray(self._h)
            self._u = self.np.asarray(self._u)
            self._f = self.np.asarray(self._f)
            self._spf = self.np.asarray(self._spf)

    def setup_spf(self):
        self.R, self.T = np.meshgrid(self.radius, self.theta)

        for p in range(self.l // 2):
            n, m = get_indices_nm(p)
            self._spf[p, :] += spf_state(
                self.R, self.T, n, m, self.mass, self.omega
            )


