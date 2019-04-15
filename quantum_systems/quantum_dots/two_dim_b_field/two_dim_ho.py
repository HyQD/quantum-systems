import numpy as np
from quantum_systems import QuantumSystem
from quantum_systems.quantum_dots.two_dim_b_field.two_dim_helper import (
    get_coulomb_elements,
    get_one_body_elements,
    construct_dataframe,
)
from quantum_systems.quantum_dots.two_dim.two_dim_helper import (
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


class TwoDimHarmonicOscB(QuantumSystem):
    def __init__(
        self, n, l, radius_length, num_grid_points, omega_0=1, mass=1, omega_c=0
    ):
        super().__init__(n, l)

        self.omega_c = omega_c
        self.omega = np.sqrt(omega_0 * omega_0 + omega_c * omega_c / 4)
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

        num_orbitals = self.l // 2
        n_array = np.arange(num_orbitals)
        m_array = np.arange(-num_orbitals - 5, num_orbitals + 6)
        self.df = construct_dataframe(
            n_array, m_array, omega_c=self.omega_c, omega=self.omega
        )

        self.__h = self.omega * get_one_body_elements(
            num_orbitals, df=self.df
        ).astype(np.complex128)
        self.__u = np.sqrt(self.omega) * get_coulomb_elements(
            num_orbitals, df=self.df
        ).astype(np.complex128)

        self._h = add_spin_one_body(self.__h, np=np)
        self._u = anti_symmetrize_u(add_spin_two_body(self.__u, np=np))
        self._f = self.construct_fock_matrix(self.__h, self.__u)

        self.cast_to_complex()

        self.setup_spf()
        self.construct_dipole_moment()

        # Some numpy-not-numpy stuff
        if np is not self.np:
            self._h = self.np.asarray(self._h)
            self._u = self.np.asarray(self._u)
            self._f = self.np.asarray(self._f)
            self._spf = self.np.asarray(self._spf)

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
            # It is important that these are not floats
            # assoc_laguerre is picky
            n_p, m_p = self.df.loc[p, ["n", "m"]].values
            n_p = int(n_p)
            m_p = int(m_p)

            norm_p = spf_norm(n_p, m_p, self.mass, self.omega)
            r_p = spf_radial_function(n_p, m_p, self.mass, self.omega)

            for q in range(self.l // 2):
                n_q, m_q = self.df.loc[q, ["n", "m"]].values
                n_q = int(n_q)
                m_q = int(m_q)

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
