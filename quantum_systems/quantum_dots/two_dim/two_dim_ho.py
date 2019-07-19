import numpy as np
import math

from quantum_systems import QuantumSystem
from quantum_systems.quantum_dots.two_dim.two_dim_helper import (
    get_coulomb_elements,
    get_one_body_elements,
    get_double_well_one_body_elements,
    get_smooth_double_well_one_body_elements,
    get_indices_nm,
    spf_state,
    spf_norm,
    spf_radial_function,
    radial_integral,
    theta_1_integral,
    theta_2_integral,
    construct_dataframe,
    get_one_body_elements_B,
    get_coulomb_elements_B,
)
from quantum_systems.system_helper import (
    add_spin_one_body,
    add_spin_two_body,
    anti_symmetrize_u,
)


class TwoDimensionalHarmonicOscillator(QuantumSystem):
    """Create 2D harmonic oscillator, using 
    polar coordinates

    Parameters
    ----------
    n : int
        Number of electrons
    l : int
        Number of spinorbitals
    radius_length : int or float
        Radius of space over which to model wavefunction
    num_grid_points : int or float
        Defines resolution of numerical representation
        of wavefunction
    omega : float, default 1 
        Frequency of harmonic oscillator potential.
    mass : int of float, default 1
        Mass of electrons

    Attributes
    ----------
    h : np.array
        One-body matrix
    f : np.array
        Fock matrix
    u : np.array
        Two-body matrix

    Methods
    -------
    setup_system()
        Must be called to set up quantum system.
    setup_spf()
        Constructs single-particle functions. This method is called
        by setup_system().
    construct_dipole_moment()
        Constructs dipole moment. This method is called by
        setup_system().
    """

    def __init__(self, n, l, radius_length, num_grid_points, omega=1, mass=1):
        super().__init__(n, l)

        self.omega = omega
        self.mass = mass

        self.radius_length = radius_length
        self.num_grid_points = num_grid_points

        self.radius = np.linspace(0, self.radius_length, self.num_grid_points)
        self.theta = np.linspace(0, 2 * np.pi, self.num_grid_points)

    def setup_system(self, add_spin=True, anti_symmetrize=True):
        self._h = self.omega * get_one_body_elements(self.l // 2)
        self._u = np.sqrt(self.omega) * get_coulomb_elements(self.l // 2)
        self._s = np.eye(self.l // 2)

        self.setup_spf()
        self.construct_dipole_moment()
        self.cast_to_complex()
        self.change_module()

        if add_spin:
            self.change_to_spin_orbital_basis(anti_symmetrize=anti_symmetrize)

    def setup_spf(self):
        self._spf = np.zeros(
            (self.l // 2, self.num_grid_points, self.num_grid_points),
            dtype=np.complex128,
        )

        self.R, self.T = np.meshgrid(self.radius, self.theta)

        for p in range(self.l // 2):
            self._spf[p] += spf_state(self.R, self.T, p, self.mass, self.omega)

    def get_indices_nm(self, p):
        return get_indices_nm(p)

    def construct_dipole_moment(self):
        self._dipole_moment = np.zeros(
            (2, self.l // 2, self.l // 2), dtype=self._spf.dtype
        )

        for p in range(self.l // 2):
            n_p, m_p = self.get_indices_nm(p)

            norm_p = spf_norm(n_p, m_p, self.mass, self.omega)
            r_p = spf_radial_function(n_p, m_p, self.mass, self.omega)

            for q in range(self.l // 2):
                n_q, m_q = self.get_indices_nm(q)

                if abs(m_p - m_q) != 1:
                    continue

                norm_q = spf_norm(n_q, m_q, self.mass, self.omega)
                r_q = spf_radial_function(n_q, m_q, self.mass, self.omega)

                norm = norm_p.conjugate() * norm_q
                I_r = radial_integral(r_p, r_q)
                I_theta_1 = theta_1_integral(m_p, m_q)
                I_theta_2 = theta_2_integral(m_p, m_q)

                # x-direction
                self._dipole_moment[0, p, q] = norm * I_r * I_theta_1
                # y-direction
                self._dipole_moment[1, p, q] = norm * I_r * I_theta_2


class TwoDimensionalDoubleWell(TwoDimensionalHarmonicOscillator):
    """System constructing two-dimensional quantum dots with a double well
    potential barrier.

    Parameters
    ----------
    n : int
        Number of occupied spin-orbitals.
    l : int
        Number of spin-orbitals.
    radius : float
        Length of radial coordinate.
    num_grid_points : int
        Discretization of spatial coordinates. This includes both the radial and
        the angular part of the room.
    barrier_strength: float
        Barrier strength in the double well potential.
    l_ho_factor : float
        Factor of harmonic oscillator basis functions compared to the number of
        double-well functions. Note that l_ho_factor >= 1, as we need more
        harmonic oscillator basis functions than double-well functions in
        order to get a good representation of the basis elements.
    omega: float
        Frequency of the oscillator potential.
    mass: float
        Mass of the particles.

    """

    def __init__(
        self,
        n,
        l,
        radius,
        num_grid_points,
        barrier_strength=1,
        l_ho_factor=1.0,
        omega=1,
        mass=1,
    ):
        assert l_ho_factor >= 1, (
            "Number of harmonic oscillator functions must be higher than the"
            + " number of double-well basis functions"
        )

        l_ho = math.floor(l * l_ho_factor)
        super().__init__(
            n, l_ho, radius, num_grid_points, omega=omega, mass=mass
        )

        self.l_dw = l
        self.barrier_strength = barrier_strength

    def setup_system(self, axis=0, add_spin=True, anti_symmetrize=True):
        """Function setting up the one- and two-body elements, the
        single-particle functions, dipole moments and other quantities used by
        second quantization methods.
        
        Parameters
        ----------
        axis : int
            The axis argument specifies which
            axis the well barrier is aligned to. (0, 1) = (x, y).
        """

        super().setup_system(add_spin=add_spin, anti_symmetrize=anti_symmetrize)

        h_dw = get_double_well_one_body_elements(
            self.l // 2,
            self.omega,
            self.mass,
            self.barrier_strength,
            dtype=np.complex128,
            axis=axis,
        )

        self.epsilon, C = np.linalg.eigh(h_dw)
        self._h = np.diagflat(self.epsilon[: self.l_dw // 2])
        self._s = np.eye(self.l_dw // 2)
        C_dw = C[:, : self.l_dw // 2]

        if add_spin:
            self._h = add_spin_one_body(self._h, np=np)
            self._s = add_spin_one_body(self._s, np=np)
            C_dw = add_spin_one_body(C_dw, np=np)

        self.change_basis_two_body_elements(C_dw)
        self.change_basis_dipole_moment(C_dw)
        self.change_basis_spf(C_dw)

        self.set_system_size(self.n, self.l_dw)

        self.cast_to_complex()
        self.change_module()


class TwoDimSmoothDoubleWell(TwoDimensionalHarmonicOscillator):
    def __init__(
        self,
        n,
        l,
        radius,
        num_grid_points,
        a=2,
        b=2,
        l_ho_factor=1,
        omega=1,
        mass=1,
    ):

        assert l_ho_factor >= 1, (
            "Ensure number of harmonic oscillator functions are higher than"
            + " the number of double-well basis functions"
        )

        l_ho = math.floor(l * l_ho_factor)
        super().__init__(
            n, l_ho, radius, num_grid_points, omega=omega, mass=mass
        )

        self.l_dw = l
        self.a = a
        self.b = b

    def setup_system(self, add_spin=True, anti_symmetrize=True):
        """The wells are divided by x-axis by default.
        """

        super().setup_system(add_spin=add_spin, anti_symmetrize=anti_symmetrize)

        h_dw = get_smooth_double_well_one_body_elements(
            self.l // 2,
            self.omega,
            self.mass,
            a=self.a,
            b=self.b,
            dtype=np.complex128,
        )

        self.epsilon, C = np.linalg.eigh(h_dw)
        self._h = np.diagflat(self.epsilon[: self.l_dw // 2])
        self._s = np.eye(self.l_dw // 2)
        C_dw = C[:, : self.l_dw // 2]

        if add_spin:
            self._h = add_spin_one_body(self._h, np=np)
            self._s = add_spin_one_body(self._s, np=np)
            C_dw = add_spin_one_body(C_dw, np=np)

        self.change_basis_two_body_elements(C_dw)
        self.change_basis_dipole_moment(C_dw)
        self.change_basis_spf(C_dw)

        self.set_system_size(self.n, self.l_dw)

        self.cast_to_complex()
        self.change_module()


class TwoDimHarmonicOscB(TwoDimensionalHarmonicOscillator):
    """Create 2D harmonic oscillator under homogenous
    magnetic field, using polar coordinates

    Parameters
    ----------
    n : int
        Number of electrons
    l : int
        Number of spinorbitals
    radius_length : int or float
        Radius of space over which to model wavefunction
    num_grid_points : int or float
        Defines resolution of numerical representation
        of wavefunction
    omega_0 : float, default 1 
        Frequency of harmonic oscillator potential.
    mass : int or float, default 1
        Mass of electrons.
    omega_c : float, default 0
        Frequency corresponding to strength of magnetic field.

    Attributes
    ----------
    h : np.array
        One-body matrix
    f : np.array
        Fock matrix
    u : np.array
        Two-body matrix

    Methods
    -------
    setup_system()
        Must be called to set up quantum system.
    construct_dipole_moment()
        Constructs dipole moment. This method is called by
        setup_system().
    """

    def __init__(
        self, n, l, radius_length, num_grid_points, omega_0=1, mass=1, omega_c=0
    ):
        super().__init__(
            n, l, radius_length, num_grid_points, omega=omega_0, mass=mass
        )

        self.omega_c = omega_c
        self.omega = np.sqrt(omega_0 * omega_0 + omega_c * omega_c / 4)

    def setup_system(self, add_spin=True, anti_symmetrize=True):
        num_orbitals = self.l // 2
        n_array = np.arange(num_orbitals)
        m_array = np.arange(-num_orbitals - 5, num_orbitals + 6)
        self.df = construct_dataframe(
            n_array, m_array, omega_c=self.omega_c, omega=self.omega
        )

        self._h = get_one_body_elements_B(num_orbitals, df=self.df)
        self._s = np.eye(num_orbitals)
        self._u = np.sqrt(self.omega) * get_coulomb_elements_B(
            num_orbitals, df=self.df
        )

        self.setup_spf()
        self.construct_dipole_moment()
        self.cast_to_complex()
        self.change_module()

        if add_spin:
            self.change_to_spin_orbital_basis(anti_symmetrize=anti_symmetrize)

    def get_indices_nm(self, p):
        n, m = self.df.loc[p, ["n", "m"]].values

        return int(n), int(m)
