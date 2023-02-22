import numba
import numpy as np
import scipy.special
import scipy.linalg

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


@numba.njit
def _trapz_prep(vec, dx):
    # The trapezoidal method applied to a vector can be implemented as multiplying it by dx, halving the ends, and taking the sum.
    # Since we are applying the trapezoidal method to a product of 3 vectors,
    # we can first perform the first two steps on this vector, which is reused, to greatly speed up the calculation.
    prepped_vec = vec * dx
    prepped_vec[0] *= 0.5
    prepped_vec[-1] *= 0.5
    return prepped_vec


@numba.njit
def _shielded_coulomb(x_1, x_2, alpha, a):
    return alpha / np.sqrt((x_1 - x_2) ** 2 + a**2)


@numba.njit
def _compute_inner_integral(spf, l, num_grid_points, grid, alpha, a):
    inner_integral = np.zeros((l, l, num_grid_points), dtype=np.complex128)
    dx = grid[1] - grid[0]

    for i in range(num_grid_points):
        coulomb = _shielded_coulomb(grid[i], grid, alpha, a)
        trapz_prepped_coulomb = _trapz_prep(coulomb, dx)
        for q in range(l):
            trapz_prod = np.conjugate(spf[q]) * trapz_prepped_coulomb
            for s in range(l):
                inner_integral[q, s, i] = np.dot(
                    trapz_prod, spf[s]
                )  # _trapz(np.conjugate(spf[q]) * coulomb * spf[s], grid)

    return inner_integral


@numba.njit
def _compute_orbital_integrals(spf, l, inner_integral, grid):
    u = np.zeros((l, l, l, l), dtype=np.complex128)
    dx = grid[1] - grid[0]

    for q in range(l):
        for s in range(l):
            trapz_prepped_inner = _trapz_prep(inner_integral[q, s], dx)
            for p in range(l):
                trapz_prod = np.conjugate(spf[p]) * trapz_prepped_inner
                for r in range(l):
                    u[p, q, r, s] = np.dot(
                        trapz_prod, spf[r].astype(np.complex128)
                    )  # _trapz(np.conjugate(spf[p]) * inner_integral[q, s] * spf[r], grid)

    return u


class ODHO(BasisSet):
    """Create matrix elements and grid representation associated with the harmonic
    oscillator basis.

    Example
    -------

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

        self.construct_position_integrals()

    def ho_function(self, x, n):
        return (
            self.normalization(n)
            * np.exp(-0.5 * self.omega * x**2)
            * scipy.special.hermite(n)(np.sqrt(self.omega) * x)
        )

    def normalization(self, n):
        return (
            1.0
            / np.sqrt(2**n * scipy.special.factorial(n))
            * (self.omega / np.pi) ** 0.25
        )

    def construct_position_integrals(self):
        self.position = np.zeros((1, self.l, self.l), dtype=self.spf.dtype)

        for n in range(self.l - 1):
            Nn = self.normalization(n)
            Nn_up = self.normalization(n + 1)
            pos = (
                Nn
                * Nn_up
                * (n + 1)
                * np.sqrt(np.pi)
                * 2**n
                * scipy.special.factorial(n)
                / self.omega
            )
            self.position[0, n, n + 1] = pos
            self.position[0, n + 1, n] = pos


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
    construct_position_integrals()
        Constructs position matrix elements. This method is called by
        setup_basis().

    Example
    -------

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

        h_diag = 1.0 / (dx**2) + self.potential(self.grid[1:-1])
        h_off_diag = -1.0 / (2 * dx**2) * np.ones(self.num_grid_points - 3)

        eps, C = scipy.linalg.eigh_tridiagonal(
            h_diag, h_off_diag, select="i", select_range=(0, self.l - 1)
        )

        self.spf = np.zeros((self.l, self.num_grid_points), dtype=np.complex128)
        self.spf[:, 1:-1] = C.T / np.sqrt(dx)
        self.eigen_energies = eps

        self.h = np.diag(eps).astype(np.complex128)
        self.s = np.eye(self.l)

        u = _shielded_coulomb(
            self.grid[None, 1:-1], self.grid[1:-1, None], self.alpha, self.a
        )
        self.u = np.einsum(
            "pa, qb, pc, qd, pq -> abcd", C, C, C, C, u, optimize=True
        )

        self.position = np.zeros((1, self.l, self.l), dtype=self.spf.dtype)
        self.position[0] = np.einsum(
            "pa, p, pb -> ab",
            C,
            self.grid[1:-1] + self.beta * self.grid[1:-1] ** 2,
            C,
            optimize=True,
        )
