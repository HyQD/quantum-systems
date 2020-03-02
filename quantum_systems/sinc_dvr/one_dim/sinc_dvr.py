import numba
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import scipy.special as spec

from ... import BasisSet

from quantum_systems.quantum_dots.one_dim.one_dim_qd import _shielded_coulomb
from quantum_systems.quantum_dots.one_dim.one_dim_potentials import (
    HOPotential,
    DWPotential,
    DWPotentialSmooth,
    SymmetricDWPotential,
    AsymmetricDWPotential,
    GaussianPotential,
    AtomicPotential,
)


class ODSincDVR(BasisSet):
    """Create matrix elements and grid representation of sinc-dvr basis
    functions, to exploit sparsity in coulomb operator

    Parameters
    ----------
    l_dvr : int
        Number of sinc-dvr functions
    grid_length : int or float
        Space over which to model wavefunction
    a : float, default 0.25
        Screening parameter in the shielded Coulomb interation.
    alpha : float, default 1.0
        Strength parameter in the shielded Coulomb interaction.
    beta : float, default 0.0
        Strength parameter of the non-dipole term in the laser interaction matrix.
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
        l_dvr,
        grid_length,
        a=0.25,
        alpha=1.0,
        beta=0,
        potential=None,
        sparse_u=True,
        **kwargs,
    ):
        super().__init__(l_dvr, dim=1, **kwargs)

        self.sparse_u = sparse_u
        self.alpha = alpha
        self.a = a
        self.beta = beta

        self.l_dvr = l_dvr
        self.grid_length = grid_length

        self.grid = np.linspace(-self.grid_length, self.grid_length, self.l_dvr)

        if potential is None:
            omega = (
                0.25  # Default frequency corresponding to Zanghellini article
            )
            potential = HOPotential(omega)

        self.potential = potential

        self.setup_basis()

        # self.hamiltonian = self._setup_hamiltonian()

    def setup_basis(self):
        self.dx = self.grid[1] - self.grid[0]

        self.h = np.zeros((self.l_dvr, self.l_dvr))

        # create multi_dim index for speedy calculations
        ind = np.arange(self.l_dvr)
        diff_grid = ind[:, None] - ind

        # mask diagonal to edit offdiagonal elements
        mask = np.ones(self.h.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        self.h[mask] = (-1.0) ** diff_grid[mask] / (
            self.dx ** 2 * diff_grid[mask] ** 2
        )
        # fill diagonal
        self.h[ind, ind] = np.pi ** 2 / (6 * self.dx ** 2)
        self.h[ind, ind] += self.potential(self.grid)

        self.s = self.construct_s()
        self.spf = self.construct_sinc_grid()

        self.u = self.construct_coulomb_elements()

        self.construct_dipole_moment()

    def construct_sinc_grid(self):
        x = self.grid
        return 1 / np.sqrt(self.dx) * np.sinc((x - x[:, None]) / self.dx)

    def construct_dipole_moment(self):
        self.dipole_moment = np.zeros((1, self.l, self.l), dtype=self.spf.dtype)
        self.dipole_moment[0] = np.diag(self.grid + self.beta * self.grid ** 2)

    def construct_coulomb_elements(self):
        """Computes Sinc-DVR matrix elements of onebody operator h and two-body
        operator u. """
        x = self.grid

        if self.sparse_u:
            self.u = np.zeros((self.l_dvr, self.l_dvr))
        else:
            self.u = np.zeros((self.l_dvr, self.l_dvr, self.l_dvr, self.l_dvr))

        for p in range(self.l_dvr):
            for q in range(self.l_dvr):
                if self.sparse_u:
                    self.u[p, q] = _shielded_coulomb(
                        x[p], x[q], self.alpha, self.a
                    )
                else:
                    self.u[p, q, p, q] = _shielded_coulomb(
                        x[p], x[q], self.alpha, self.a
                    )

        return self.u

    def construct_s(self):
        return np.eye(self.l_dvr)
