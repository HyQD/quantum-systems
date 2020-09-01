import numpy as np
import math

from quantum_systems import BasisSet
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


class TwoDimensionalHarmonicOscillator(BasisSet):
    """Create 2D harmonic oscillator, using polar coordinates

    Parameters
    ----------
    l : int
        Number of basis functions
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
    setup_basis()
        Must be called to set up quantum system.
    setup_spf()
        Constructs single-particle functions. This method is called
        by setup_basis().
    construct_position_integrals()
        Constructs position integrals. This method is called by
        setup_basis().
    """

    def __init__(
        self,
        l,
        radius_length,
        num_grid_points,
        omega=1,
        mass=1,
        verbose=False,
        **kwargs
    ):
        super().__init__(l, dim=2, **kwargs)

        self.omega = omega
        self.mass = mass
        self.verbose = verbose

        self.radius_length = radius_length
        self.num_grid_points = num_grid_points

        self.radius = np.linspace(0, self.radius_length, self.num_grid_points)
        self.theta = np.linspace(0, 2 * np.pi, self.num_grid_points)
        self.setup_basis()

    def setup_basis(self):
        self._h = self.omega * get_one_body_elements(self.l)
        self._u = np.sqrt(self.omega) * get_coulomb_elements(
            self.l, verbose=self.verbose
        )
        self._s = np.eye(self.l)

        self.setup_spf()
        self.construct_position_integrals()

        if self.np is not np:
            self.change_module(self.np)

    def setup_spf(self):
        self._spf = np.zeros(
            (self.l, self.num_grid_points, self.num_grid_points),
            dtype=np.complex128,
        )

        self.R, self.T = np.meshgrid(self.radius, self.theta)

        for p in range(self.l):
            self._spf[p] += spf_state(
                self.R, self.T, p, self.mass, self.omega, self.get_indices_nm
            )

    def get_indices_nm(self, p):
        return get_indices_nm(p)

    def construct_position_integrals(self):
        self._position = np.zeros((2, self.l, self.l), dtype=self._spf.dtype)

        for p in range(self.l):
            n_p, m_p = self.get_indices_nm(p)

            norm_p = spf_norm(n_p, m_p, self.mass, self.omega)
            r_p = spf_radial_function(n_p, m_p, self.mass, self.omega)

            for q in range(self.l):
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
                self._position[0, p, q] = norm * I_r * I_theta_1
                # y-direction
                self._position[1, p, q] = norm * I_r * I_theta_2


class TwoDimensionalDoubleWell(TwoDimensionalHarmonicOscillator):
    """System constructing two-dimensional quantum dots with a double well
    potential barrier.

    Parameters
    ----------
    l : int
        Number of spin-orbitals.
    radius : float
        Length of radial coordinate.
    num_grid_points : int
        Discretization of spatial coordinates. This includes both the radial and
        the angular part of the room.
    barrier_strength: float
        Barrier strength in the double well potential.
    omega: float
        Frequency of the oscillator potential.
    mass: float
        Mass of the particles.
    axis : int
        The axis argument specifies which
        axis the well barrier is aligned to. (0, 1) = (x, y).
    """

    def __init__(self, *args, barrier_strength=1, axis=0, **kwargs):
        self.barrier_strength = barrier_strength
        self.axis = axis

        super().__init__(*args, **kwargs)

    def setup_basis(self):
        """Function setting up the one- and two-body elements, the
        single-particle functions, position integrals and other quantities used
        by second quantization methods.
        """

        super().setup_basis()

        self._h = get_double_well_one_body_elements(
            self.l,
            self.omega,
            self.mass,
            self.barrier_strength,
            dtype=np.complex128,
            axis=self.axis,
        )

        self.change_module(self.np)


class TwoDimSmoothDoubleWell(TwoDimensionalHarmonicOscillator):
    def __init__(self, *args, a=2, b=2, **kwargs):

        super().__init__(*args, **kwargs)

        self.a = a
        self.b = b

    def setup_basis(self):
        """The wells are divided by x-axis by default."""

        super().setup_basis()

        self._h = get_smooth_double_well_one_body_elements(
            self.l,
            self.omega,
            self.mass,
            a=self.a,
            b=self.b,
            dtype=np.complex128,
        )


class TwoDimHarmonicOscB(TwoDimensionalHarmonicOscillator):
    """Create 2D harmonic oscillator under homogenous
    magnetic field, using polar coordinates

    Parameters
    ----------
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
    setup_basis()
        Must be called to set up quantum system.
    construct_position_integrals()
        Constructs position integrals. This method is called by
        setup_basis().
    """

    def __init__(self, *args, omega_c=0, **kwargs):
        self.omega_c = omega_c

        super().__init__(*args, **kwargs)

    def setup_basis(self):
        self.omega = np.sqrt(self.omega ** 2 + self.omega_c ** 2 / 4)

        num_orbitals = self.l
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

        self.setup_spf()  # This is maybe not wrong.
        self.construct_position_integrals()
        self.cast_to_complex()
        self.change_module(self.np)

    def get_indices_nm(self, p):
        n, m = self.df.loc[p, ["n", "m"]].values

        return int(n), int(m)
