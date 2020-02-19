import numba
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import scipy.special as spec

from quantum_systems import BasisSet

from quantum_systems.quantum_dots.one_dim.one_dim_potentials import (
    HOPotential,
    DWPotential,
    DWPotentialSmooth,
    SymmetricDWPotential,
    AsymmetricDWPotential,
    GaussianPotential,
    AtomicPotential,
)


@numba.njit(cache=True)
def _trapz(f, x):
    n = len(x)
    delta_x = x[1] - x[0]
    val = 0

    for i in range(1, n):
        val += f[i - 1] + f[i]

    return 0.5 * val * delta_x


@numba.njit(cache=True)
def _shielded_coulomb(x_1, x_2, alpha, a):
    return alpha / np.sqrt((x_1 - x_2) ** 2 + a ** 2)


@numba.njit(cache=True)
def _compute_inner_integral(spf, l, num_grid_points, grid, alpha, a):
    inner_integral = np.zeros((l, l, num_grid_points), dtype=np.complex128)

    for q in range(l):
        for s in range(l):
            for i in range(num_grid_points):
                inner_integral[q, s, i] = _trapz(
                    np.conjugate(spf[q])
                    * _shielded_coulomb(grid[i], grid, alpha, a)
                    * spf[s],
                    grid,
                )

    return inner_integral


@numba.njit(cache=True)
def _compute_orbital_integrals(spf, l, inner_integral, grid):
    u = np.zeros((l, l, l, l), dtype=np.complex128)
    for p in range(l):
        for q in range(l):
            for r in range(l):
                for s in range(l):
                    u[p, q, r, s] = _trapz(
                        np.conjugate(spf[p]) * inner_integral[q, s] * spf[r],
                        grid,
                    )

    return u


class ODHO(BasisSet):
    """Create matrix elements and grid representation associated with the harmonic
    oscillator basis.

    >>> odho = ODHO(20, 11, 201, omega=1)
    >>> odho.l == 20
    True
    >>> abs(0.5 - odho.h[0, 0]) # doctest.ELLIPSIS
    0.0
    """

    def __init__(
        self,
        l,
        grid_length,
        num_grid_points,
        omega=0.25,
        a=0.25,
        alpha=1.0,
        beta=0,
        **kwargs,
    ):

        super().__init__(l, dim=1, **kwargs)

        self.omega = omega
        self.a = a
        self.alpha = alpha

        self.grid_length = grid_length
        self.num_grid_points = num_grid_points
        self.grid = np.linspace(
            -self.grid_length, self.grid_length, self.num_grid_points
        )
        self.beta = beta

        self.setup_basis()

    def setup_basis(self):
        dx = self.grid[1] - self.grid[0]
        self.eigen_energies = self.omega * (np.arange(self.l) + 0.5)
        self.h = np.diag(self.eigen_energies).astype(np.complex128)
        self.s = np.eye(self.l)
        self.spf = np.zeros((self.l, self.num_grid_points))

        for p in range(self.l):
            self.spf[p] = self.ho_function(self.grid, p)

        inner_integral = _compute_inner_integral(
            self.spf,
            self.l,
            self.num_grid_points,
            self.grid,
            self.alpha,
            self.a,
        )

        self.u = _compute_orbital_integrals(
            self.spf, self.l, inner_integral, self.grid
        )

        self.construct_dipole_moment()

    def ho_function(self, x, n):
        return (
            self.normalization(n)
            * np.exp(-0.5 * self.omega * x ** 2)
            * spec.hermite(n)(np.sqrt(self.omega) * x)
        )

    def normalization(self, n):
        return (
            1.0
            / np.sqrt(2 ** n * spec.factorial(n))
            * (self.omega / np.pi) ** 0.25
        )

    def construct_dipole_moment(self):
        self.dipole_moment = np.zeros((1, self.l, self.l), dtype=self.spf.dtype)

        for n in range(self.l - 1):
            Nn = self.normalization(n)
            Nn_up = self.normalization(n + 1)
            dip_mom = (
                Nn
                * Nn_up
                * (n + 1)
                * np.sqrt(np.pi)
                * 2 ** n
                * spec.factorial(n)
                / self.omega
            )
            self.dipole_moment[0, n, n + 1] = dip_mom
            self.dipole_moment[0, n + 1, n] = dip_mom


class ODQD(BasisSet):
    """Create 1D quantum dot system

    Parameters
    ----------
    l : int
        Number of basis functions
    grid_length : int or float
        Space over which to model wavefunction
    num_grid_points : int or float
        Defines resolution of numerical representation
        of wavefunction
    a : float, default 0.25
        Screening parameter in the shielded Coulomb interation.
    alpha : float, default 1.0
        Strength parameter in the shielded Coulomb interaction.
    beta : float, default 0.0
        Strength parameter of the non-dipole term in the laser interaction matrix.

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
        Must be called to set up quantum system.  The method will revert to
        regular harmonic oscillator potential if no potential is provided. It
        is also possible to use double well potentials.
    construct_dipole_moment()
        Constructs dipole moment. This method is called by setup_basis().

    >>> odqd = ODQD(20, 11, 201, potential=ODQD.HOPotential(omega=1))
    >>> odqd.l == 20
    True
    >>> abs(0.5 - odqd.h[0, 0]) # doctest.ELLIPSIS
    0.0003...
    """

    HOPotential = HOPotential
    DWPotential = DWPotential
    DWPotentialSmooth = DWPotentialSmooth
    SymmetricDWPotential = SymmetricDWPotential
    AsymmetricDWPotential = AsymmetricDWPotential
    GaussianPotential = GaussianPotential
    AtomicPotential = AtomicPotential

    def __init__(
        self,
        l,
        grid_length,
        num_grid_points,
        a=0.25,
        alpha=1.0,
        beta=0,
        potential=None,
        **kwargs,
    ):

        super().__init__(l, dim=1, **kwargs)

        self.a = a
        self.alpha = alpha

        self.grid_length = grid_length
        self.num_grid_points = num_grid_points
        self.grid = np.linspace(
            -self.grid_length, self.grid_length, self.num_grid_points
        )
        self.beta = beta

        if potential is None:
            omega = (
                0.25  # Default frequency corresponding to Zanghellini article
            )
            potential = HOPotential(omega)

        self.potential = potential

        self.setup_basis()

    def setup_basis(self):
        dx = self.grid[1] - self.grid[0]

        h_diag = 1.0 / (dx ** 2) + self.potential(self.grid[1:-1])
        h_off_diag = -1.0 / (2 * dx ** 2) * np.ones(self.num_grid_points - 3)

        h = (
            np.diag(h_diag)
            + np.diag(h_off_diag, k=-1)
            + np.diag(h_off_diag, k=1)
        )

        eigen_energies, eigen_states = np.linalg.eigh(h)
        eigen_energies = eigen_energies[: self.l]
        eigen_states = eigen_states[:, : self.l]

        self.spf = np.zeros((self.l, self.num_grid_points), dtype=np.complex128)
        self.spf[:, 1:-1] = eigen_states.T / np.sqrt(dx)
        self.eigen_energies = eigen_energies

        self.h = np.diag(eigen_energies).astype(np.complex128)
        self.s = np.eye(self.l)

        inner_integral = _compute_inner_integral(
            self.spf,
            self.l,
            self.num_grid_points,
            self.grid,
            self.alpha,
            self.a,
        )

        self.u = _compute_orbital_integrals(
            self.spf, self.l, inner_integral, self.grid
        )

        self.construct_dipole_moment()

    def construct_dipole_moment(self):
        self.dipole_moment = np.zeros((1, self.l, self.l), dtype=self.spf.dtype)

        for p in range(self.l):
            for q in range(self.l):
                self.dipole_moment[0, p, q] = np.trapz(
                    self.spf[p].conj()
                    * (self.grid + self.beta * self.grid ** 2)
                    * self.spf[q],
                    self.grid,
                )
