import abc
import copy

from quantum_systems.system_helper import (
    add_spin_spf,
    add_spin_bra_spf,
    add_spin_one_body,
    add_spin_two_body,
    anti_symmetrize_u,
    check_axis_lengths,
    compute_particle_density,
)


class QuantumSystem(metaclass=abc.ABCMeta):
    """Base class defining some of the common methods used by all the different
    quantum systems.

    Parameters
    ----------
    n : int
        Number of particles.
    l : int
        Number of basis functions.
    np : module
        Matrix and linear algebra module. Currently only `numpy` is supported.
    """

    def __init__(self, n, l, np=None):
        assert n <= l

        if np is None:
            import numpy as np

        self.np = np

        self.set_system_size(n, l)

        self._h = None
        self._u = None
        self._s = None
        self._dipole_moment = None

        self._time_evolution_operator = None

        self._spf = None
        self._bra_spf = None

        self._nuclear_repulsion_energy = 0

    def set_system_size(self, n, l):
        """Function setting the system size. Note that ``l`` should
        correspond to the length of each axis of the matrix elements.

        Parameters
        ----------
        n : int
            The number of particles.
        l : int
            The number of basis functions.
        """

        assert n <= l

        self.n = n
        self.l = l
        self.m = self.l - self.n

        self.o = slice(0, self.n)
        self.v = slice(self.n, self.l)

    @abc.abstractmethod
    def construct_fock_matrix(self, h, u, f=None):
        pass

    @staticmethod
    def change_arr_module(arr, np):
        return np.asarray(arr) if arr is not None else None

    def change_module(self, np):
        """Function converting between modules.

        Parameters
        ----------
        np : module
            Array- and linalg-module.
        """
        self.np = np

        self._h = self.change_arr_module(self._h, self.np)
        self._s = self.change_arr_module(self._s, self.np)
        self._u = self.change_arr_module(self._u, self.np)
        self._spf = self.change_arr_module(self._spf, self.np)
        self._bra_spf = self.change_arr_module(self._bra_spf, self.np)
        self._dipole_moment = self.change_arr_module(
            self._dipole_moment, self.np
        )

    def cast_to_complex(self):
        """Function converting all matrix elements to ``np.complex128``, where
        ``np`` is the member array module.
        """
        np = self.np

        self._h = self._h.astype(np.complex128)
        self._u = self._u.astype(np.complex128)

        if self._s is not None:
            self._s = self._s.astype(np.complex128)

        if self._spf is not None:
            self._spf = self._spf.astype(np.complex128)

        if self._bra_spf is not None:
            self._bra_spf = self._bra_spf.astype(np.complex128)

        if self._dipole_moment is not None:
            self._dipole_moment = self._dipole_moment.astype(np.complex128)

    @staticmethod
    def transform_spf(spf, c, np):
        return np.tensordot(c, spf, axes=((0), (0)))

    @staticmethod
    def transform_bra_spf(bra_spf, c_tilde, np):
        return np.tensordot(c_tilde, bra_spf, axes=((1), (0)))

    @staticmethod
    def transform_one_body_elements(h, c, np, c_tilde=None):
        if c_tilde is None:
            c_tilde = c.conj().T

        return np.dot(c_tilde, np.dot(h, c))

    @staticmethod
    def transform_two_body_elements(u, c, np, c_tilde=None):
        if c_tilde is None:
            c_tilde = c.conj().T

        # abcd, ds -> abcs
        _u = np.tensordot(u, c, axes=(3, 0))
        # abcs, cr -> absr -> abrs
        _u = np.tensordot(_u, c, axes=(2, 0)).transpose(0, 1, 3, 2)
        # abrs, qb -> arsq -> aqrs
        _u = np.tensordot(_u, c_tilde, axes=(1, 1)).transpose(0, 3, 1, 2)
        # pa, aqrs -> pqrs
        _u = np.tensordot(c_tilde, _u, axes=(1, 0))

        return _u

    def change_basis_one_body_elements(self, c, c_tilde=None):
        self._h = self.transform_one_body_elements(
            self._h, c, np=self.np, c_tilde=c_tilde
        )

        if self._s is not None:
            self._s = self.transform_one_body_elements(
                self._s, c, c_tilde=c_tilde, np=self.np
            )

    def change_basis_two_body_elements(self, c, c_tilde=None):
        self._u = self.transform_two_body_elements(
            self._u, c, np=self.np, c_tilde=c_tilde
        )

    def change_basis_dipole_moment(self, c, c_tilde=None):
        dipole_moment = []
        for i in range(self._dipole_moment.shape[0]):
            dipole_moment.append(
                self.transform_one_body_elements(
                    self._dipole_moment[i], c, np=self.np, c_tilde=c_tilde
                )
            )

        self._dipole_moment = self.np.asarray(dipole_moment)

    def change_basis_spf(self, c, c_tilde=None):
        if c_tilde is not None:
            # In case of bi-orthogonal basis sets, we create an extra set
            # of single-particle functions for the bra-side.
            # Note the use of self.bra_spf instead of self._bra_spf in the
            # argument to the helper function. This guarantees that
            # self._bra_spf is not None.
            self._bra_spf = self.transform_bra_spf(
                self.bra_spf, c_tilde, self.np
            )

        self._spf = self.transform_spf(self._spf, c, self.np)

    def change_basis(self, c, c_tilde=None):
        self.change_basis_one_body_elements(c, c_tilde)
        self.change_basis_two_body_elements(c, c_tilde)

        if self._dipole_moment is not None:
            self.change_basis_dipole_moment(c, c_tilde)

        if self._spf is not None:
            self.change_basis_spf(c, c_tilde)

    @abc.abstractmethod
    def change_to_hf_basis(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def compute_reference_energy(self):
        pass

    def compute_particle_density(self, rho_qp, c=None, c_tilde=None):
        r"""Function computing the particle density for a given one-body
        density matrix. This function (optionally) changes the basis of the
        single-particle states for given coefficient matrices. The one-body
        density is given by

        .. math:: \rho(\mathbf{r}) = \tilde{\phi}_{q}(\mathbf{r})
            \rho^{q}_{p} \phi_{p}(\mathbf{r}),

        where :math:`\rho^{q}_{p}` is the one-body density matrix,
        :math:`\phi_p(\mathbf{r})` the spin-orbitals, and
        :math:`\tilde{\phi}_p(\mathbf{r})` the dual spin-orbital.

        Parameters
        ----------
        rho_qp : np.ndarray
            One-body density matrix
        c : np.ndarray
            Coefficient matrix for basis change. Default is `None` and hence no
            transformation occurs.
        c_tilde : np.ndarray
            Bra-state coefficient matrix. Default is `None` which leads to one
            of two situations. If `c != None` `c_tilde` is assumed to be the
            conjugated transpose of `c`. Otherwise, no transformation occurs.

        Returns
        -------
        np.ndarray
            Particle density with the same dimensions as the grid.
        """
        assert (
            self._spf is not None
        ), "Set up single-particle functions prior to calling this function"

        ket_spf = self.spf
        bra_spf = self.bra_spf

        if c is not None:
            ket_spf = self.transform_spf(ket_spf, c, self.np)
            c_tilde = c_tilde if c_tilde is not None else c.conj().T
            bra_spf = self.transform_bra_spf(bra_spf, c_tilde, self.np)

        return compute_particle_density(rho_qp, ket_spf, bra_spf, self.np)

    @property
    def h(self):
        """Getter returning one-body matrix."""
        return self._h

    @property
    def u(self):
        """Getter returning the antisymmetric two-body matrix."""
        return self._u

    @property
    def s(self):
        """Getter returning the overlap matrix of the atomic orbitals. If the
        overlap elements don't yet exist, we assume that the overlap is the
        identity.
        """
        np = self.np

        if self._s is None:
            self._s = np.eye(*self._h.shape)

        return self._s

    @property
    def dipole_moment(self):
        return self._dipole_moment

    @property
    def spf(self):
        """Getter returning the single particle functions, i.e, the eigenstates
        of the non-interacting Hamiltonian.
        """
        return self._spf

    @property
    def bra_spf(self):
        """Getter returning the conjugate single particle functions. This is
        `None`, unless we are working with a bi-variational basis.
        """
        if self._bra_spf is None:
            self._bra_spf = self._spf.conj()

        return self._bra_spf

    @property
    def nuclear_repulsion_energy(self):
        return self._nuclear_repulsion_energy

    def get_transformed_h(self, c):
        return self.transform_one_body_elements(self._h, c, np=self.np)

    def get_transformed_u(self, c):
        return self.transform_two_body_elements(self._u, c, np=self.np)

    def set_time_evolution_operator(self, time_evolution_operator):
        self._time_evolution_operator = time_evolution_operator
        self._time_evolution_operator.set_system(self)

    @property
    def has_one_body_time_evolution_operator(self):
        if self._time_evolution_operator is None:
            return False

        return self._time_evolution_operator.is_one_body_operator

    @property
    def has_two_body_time_evolution_operator(self):
        if self._time_evolution_operator is None:
            return False

        return self._time_evolution_operator.is_two_body_operator

    def h_t(self, current_time):
        if self._time_evolution_operator is None:
            return self._h

        return self._time_evolution_operator.h_t(current_time)

    def u_t(self, current_time):
        if self._time_evolution_operator is None:
            return self._u

        return self._time_evolution_operator.u_t(current_time)

    def copy_system(self):
        """Function creating a deep copy of the current system. This function
        is a hack as we have to temporarily remove the stored module before
        using Python's `copy.deepcopy`-function.

        Returns
        -------
        QuantumSystem
            A deep copy of the current system.
        """

        np = self.np
        self.np = None

        new_system = copy.deepcopy(self)
        new_system.np = np
        self.np = np

        return new_system
