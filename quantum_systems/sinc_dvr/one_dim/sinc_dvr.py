import numba
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import scipy.special as spec

import warnings

from quantum_systems import BasisSet

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
    l : int
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
        l,
        grid_length,
        a=0.25,
        alpha=1.0,
        beta=0,
        potential=None,
        u_repr="2d",
        **kwargs,
    ):
        # for backwards compatibility:
        # u_repr = "sparse" if kwargs.pop("sparse_u", False) else u_repr

        if u_repr not in ("2d", "4d"):
            raise ValueError("Invalid u_repr value: '{}'".format(u_repr))

        super().__init__(l, dim=1, **kwargs)

        self.alpha = alpha
        self.a = a
        self.beta = beta

        self.grid_length = grid_length

        self.grid = np.linspace(-self.grid_length, self.grid_length, self.l)
        self.num_grid_points = self.l

        if potential is None:
            omega = (
                0.25  # Default frequency corresponding to Zanghellini article
            )
            potential = HOPotential(omega)

        self.potential = potential

        self.setup_basis(u_repr)

    @property
    def sparse_repr(self):
        return self.u_repr == "2d"

    @property
    def u_repr(self):
        if np.ndim(self.u) == 2:
            return "2d"
        if np.ndim(self.u) == 4:
            return "4d"
        return "unknown"

    def setup_basis(self, u_repr):
        self.dx = self.grid[1] - self.grid[0]

        self.h = np.zeros((self.l, self.l), dtype=np.complex128)

        # create multi_dim index for speedy calculations
        ind = np.arange(self.l)
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

        self.u = self.construct_coulomb_elements(u_repr)

        self.construct_position_integrals()
        self.cast_to_complex()

    def set_u_repr(self, new_repr):
        """Switches representation of coloumb matrix elements between 2d and 4d"""
        coords0 = np.arange(self.l ** 2) % self.l
        coords1 = np.arange(self.l ** 2) // self.l

        if new_repr == self.u_repr:
            print("u repr is already {}, doing nothing".format(new_repr))
        elif new_repr == "4d":
            new_u = np.zeros(
                (self.l, self.l, self.l, self.l), dtype=self.u.dtype
            )
            new_u[coords0, coords1, coords0, coords1] = self.u[coords0, coords1]
        elif new_repr == "2d":
            new_u = np.zeros((self.l, self.l), dtype=self.u.dtype)
            new_u[coords0, coords1] = self.u[coords0, coords1, coords0, coords1]
        else:
            raise ValueError(
                "'{}' is not a valid representation".format(new_repr)
            )

    def construct_sinc_grid(self):
        x = self.grid
        return 1 / np.sqrt(self.dx) * np.sinc((x - x[:, None]) / self.dx)

    def construct_position_integrals(self):
        self.position = np.zeros((1, self.l, self.l), dtype=self.spf.dtype)
        self.position[0] = np.diag(self.grid + self.beta * self.grid ** 2)

    def construct_coulomb_elements(self, u_repr="4d"):
        """Computes Sinc-DVR matrix elements of onebody operator h and two-body
        operator u."""
        x = self.grid

        coords = []
        data = []
        for p in range(self.l):
            for q in range(self.l):
                data.append(_shielded_coulomb(x[p], x[q], self.alpha, self.a))
                coords.append((p, q))
        coords = np.array(coords).T
        data = np.array(data)

        if u_repr == "2d":
            self.u = np.zeros((self.l, self.l))
            self.u[coords[0], coords[1]] = data
        else:
            self.u = np.zeros((self.l, self.l, self.l, self.l))
            self.u[coords[0], coords[1], coords[0], coords[1]] = data
        return self.u

    def construct_s(self):
        return np.eye(self.l)

    def change_to_general_orbital_basis(self, anti_symmetrize=True):
        if anti_symmetrize and self.u_repr == "2d":
            if self.l > 100:
                warnings.warn(
                    "Warning, l large. Change to gos with anti_symmetrize=True forces 4d u."
                )

            self.set_u_repr("4d")
        return super().change_to_general_orbital_basis(
            anti_symmetrize=anti_symmetrize
        )

    def change_module(self, np):
        if self.sparse_repr:

            self.np = np
            warnings.warn(
                "change_module not implemented for sparse u, doing nothing"
            )
        else:
            return super().change_module(np)

    @staticmethod
    def add_spin_two_body(u, np):
        """Class method overwriting the static method of BasisSet."""
        if np.ndim(u) == 2:
            # spin symmetry is equal to dvr symmetry, meaning that the total matrix of
            # spatial and spin elements is doubled with every element kept
            return np.kron(u, np.ones((2, 2)))
        else:
            return super(ODSincDVR, ODSincDVR).add_spin_two_body(u, np)

    @staticmethod
    def anti_symmetrize_u(_u):
        if len(np.shape(_u)) == 2:
            return _u  # _u.transpose((0, 1))[:,::-1]
        else:
            return super(ODSincDVR, ODSincDVR).anti_symmetrize_u(_u)

    def transform_two_body_elements(
        self, u, C, np, anti_symmetrize=False, C_tilde=None
    ):
        """Class method overwriting the static method of BasisSet. Returns a 4d
        u_prime in numpy format, and allows for direct antisymmetrization of transformed
        elements, to compensate for lack of antisymmetrization in 2d representation of u."""
        if self.u_repr == "2d":
            if C_tilde is None:
                C_tilde = C.conj().T
            # get the 2d matrix of nonzero values
            if self.u_repr == "2d":
                _u = u
            else:
                _u = np.zeros(u.shape[:2])
                _u[u.coords[0], u.coords[1]] = u.data
            u_prime = np.einsum(
                "bs,ar,qb,pa,ab->pqrs",
                C,
                C,
                C_tilde,
                C_tilde,
                _u,
                optimize=True,
            )
            if anti_symmetrize:
                u_prime -= np.einsum(
                    "br,as,qb,pa,ab->pqrs",
                    C,
                    C,
                    C_tilde,
                    C_tilde,
                    _u,
                    optimize=True,
                )
            return u_prime
        else:
            assert (
                not anti_symmetrize
            ), "antisymmetrize only valid for sparse storage of u"
            # call static method of superclass
            return super(ODSincDVR, ODSincDVR).transform_two_body_elements(
                u, C, np, C_tilde
            )

    def change_basis(self, *args, **kwargs):
        super().change_basis(*args, **kwargs)
