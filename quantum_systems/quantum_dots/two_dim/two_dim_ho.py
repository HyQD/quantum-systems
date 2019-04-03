import numpy as np
from quantum_systems import QuantumSystem
from quantum_systems.quantum_dots.two_dim.two_dim_helper import (
    get_coulomb_elements,
    get_one_body_elements,
    get_indices_nm,
    spf_state,
    spf_norm,
    spf_radial_function,
    radial_integral,
    theta_1_integral,
    theta_2_integral,
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
            (self.l, self.num_grid_points, self.num_grid_points),
            dtype=np.complex128,
        )

    def setup_system(self):
        self.__h = self.omega * get_one_body_elements(
            self.l // 2, dtype=np.complex128
        )
        self.__u = np.sqrt(self.omega) * get_coulomb_elements(
            self.l // 2, dtype=np.complex128
        )

        self._h = add_spin_one_body(self.__h, np=np)
        self._u = anti_symmetrize_u(add_spin_two_body(self.__u, np=np))
        self._f = self.construct_fock_matrix(self._h, self._u)

        self.cast_to_complex()

        self.setup_spf()
        self.construct_dipole_moment()

        if np is not self.np:
            self._h = self.np.asarray(self._h)
            self._u = self.np.asarray(self._u)
            self._f = self.np.asarray(self._f)
            self._spf = self.np.asarray(self._spf)
            self._dipole_moment = self.np.asarray(self._dipole_moment)

    def setup_spf(self):
        self.R, self.T = np.meshgrid(self.radius, self.theta)

        for p in range(self.l // 2):
            self._spf[2 * p, :] += spf_state(
                self.R, self.T, p, self.mass, self.omega
            )
            self._spf[2 * p + 1, :] += self._spf[2 * p, :]

    def construct_dipole_moment(self):
        dipole_moment = np.zeros(
            (2, self.l // 2, self.l // 2), dtype=self._spf.dtype
        )

        for p in range(self.l // 2):
            n_p, m_p = get_indices_nm(p)

            norm_p = spf_norm(n_p, m_p, self.mass, self.omega)
            r_p = spf_radial_function(n_p, m_p, self.mass, self.omega)

            for q in range(self.l // 2):
                n_q, m_q = get_indices_nm(q)

                norm_q = spf_norm(n_q, m_q, self.mass, self.omega)
                r_q = spf_radial_function(n_q, m_q, self.mass, self.omega)

                norm = norm_p.conjugate() * norm_q
                I_r = radial_integral(r_p, r_q)
                I_theta_1 = theta_1_integral(m_p, m_q)
                I_theta_2 = theta_2_integral(m_p, m_q)

                # x-direction
                dipole_moment[0, p, q] = norm * I_r * I_theta_1
                # y-direction
                dipole_moment[1, p, q] = norm * I_r * I_theta_2

        self._dipole_moment = np.array(
            [add_spin_one_body(dipole_moment[i], np=np) for i in range(2)]
        )
