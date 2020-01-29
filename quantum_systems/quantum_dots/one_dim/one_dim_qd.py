import numba
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import scipy.special as spec
import time

from quantum_systems import QuantumSystem

from quantum_systems.system_helper import (
    add_spin_one_body,
    add_spin_two_body,
    anti_symmetrize_u,
)

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


class ODHO(QuantumSystem):
    """
    Create matrix elements and grid representation associated with the harmonic 
    oscillator basis.
    """

    def __init__(
        self,
        n,
        l,
        omega,
        grid_length,
        num_grid_points,
        a=0.25,
        alpha=1.0,
        beta=0,
        **kwargs
    ):

        super().__init__(n, l, **kwargs)

        self.omega = omega
        self.a = a
        self.alpha = alpha

        self.grid_length = grid_length
        self.num_grid_points = num_grid_points
        self.grid = np.linspace(
            -self.grid_length, self.grid_length, self.num_grid_points
        )
        self.beta = beta

    def setup_system(self, potential=None, add_spin=True, anti_symmetrize=True):
        dx = self.grid[1] - self.grid[0]
        self.eigen_energies = self.omega * (np.arange(self.l // 2) + 0.5)
        self._h = np.diag(self.eigen_energies).astype(np.complex128)
        self._s = np.eye(self.l // 2)
        self._spf = np.zeros((self.l // 2, self.num_grid_points))
        for p in range(self.l // 2):
            self._spf[p] = self.ho_function(self.grid, p)

        tic = time.time()
        inner_integral = _compute_inner_integral(
            self._spf,
            self.l // 2,
            self.num_grid_points,
            self.grid,
            self.alpha,
            self.a,
        )

        self._u = _compute_orbital_integrals(
            self._spf, self.l // 2, inner_integral, self.grid
        )
        toc = time.time()
        # print(f"Time computing u_pqrs: {toc-tic}")

        self.construct_dipole_moment()
        self.cast_to_complex()
        self.change_module()

        if add_spin:
            self.change_to_spin_orbital_basis(anti_symmetrize=anti_symmetrize)

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
        self._dipole_moment = np.zeros(
            (1, self.l // 2, self.l // 2), dtype=self._spf.dtype
        )

        for n in range(self.l // 2 - 1):
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
            self._dipole_moment[0, n, n + 1] = dip_mom
            self._dipole_moment[0, n + 1, n] = dip_mom


class ODQD(QuantumSystem):
    """Create 1D quantum dot system

    Parameters
    ----------
    n : int
        Number of electrons
    l : int
        Number of spinorbitals
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
    setup_system(potential=None)
        Must be called to set up quantum system.
        The method will revert to regular harmonic oscillator
        potential if no potential is provided. It is also
        possible to use double well potentials.
    construct_dipole_moment()
        Constructs dipole moment. This method is called by
        setup_system().
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
        n,
        l,
        grid_length,
        num_grid_points,
        a=0.25,
        alpha=1.0,
        beta=0,
        **kwargs
    ):

        super().__init__(n, l, **kwargs)

        self.a = a
        self.alpha = alpha

        self.grid_length = grid_length
        self.num_grid_points = num_grid_points
        self.grid = np.linspace(
            -self.grid_length, self.grid_length, self.num_grid_points
        )
        self.beta = beta

    def setup_system(self, potential=None, add_spin=True, anti_symmetrize=True):
        if potential is None:
            omega = (
                0.25  # Default frequency corresponding to Zanghellini article
            )
            potential = HOPotential(omega)

        self.potential = potential

        dx = self.grid[1] - self.grid[0]

        h_diag = 1.0 / (dx ** 2) + potential(self.grid[1:-1])
        h_off_diag = -1.0 / (2 * dx ** 2) * np.ones(self.num_grid_points - 3)

        h = (
            np.diag(h_diag)
            + np.diag(h_off_diag, k=-1)
            + np.diag(h_off_diag, k=1)
        )

        eigen_energies, eigen_states = np.linalg.eigh(h)
        eigen_energies = eigen_energies[: self.l // 2]
        eigen_states = eigen_states[:, : self.l // 2]

        self._spf = np.zeros(
            (self.l // 2, self.num_grid_points), dtype=np.complex128
        )
        self._spf[:, 1:-1] = eigen_states.T / np.sqrt(dx)
        self.eigen_energies = eigen_energies

        self._h = np.diag(eigen_energies).astype(np.complex128)
        self._s = np.eye(self.l // 2)

        tic = time.time()
        inner_integral = _compute_inner_integral(
            self._spf,
            self.l // 2,
            self.num_grid_points,
            self.grid,
            self.alpha,
            self.a,
        )

        self._u = _compute_orbital_integrals(
            self._spf, self.l // 2, inner_integral, self.grid
        )
        toc = time.time()
        # print(f"Time computing u_pqrs: {toc-tic}")

        self.construct_dipole_moment()
        self.cast_to_complex()
        self.change_module()

        if add_spin:
            self.change_to_spin_orbital_basis(anti_symmetrize=anti_symmetrize)

    def construct_dipole_moment(self):
        self._dipole_moment = np.zeros(
            (1, self.l // 2, self.l // 2), dtype=self._spf.dtype
        )

        for p in range(self.l // 2):
            for q in range(self.l // 2):
                self._dipole_moment[0, p, q] = np.trapz(
                    self._spf[p].conj()
                    * (self.grid + self.beta * self.grid ** 2)
                    * self._spf[q],
                    self.grid,
                )
