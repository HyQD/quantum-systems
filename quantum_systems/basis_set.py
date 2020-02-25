import warnings

from quantum_systems.system_helper import compute_particle_density


class BasisSet:
    """Class serving as a container for matrix elements for various systems. The
    purpose of ``BasisSet`` after a set has been constructed is to be passed to
    a subclass of ``QuantumSystem``. From there on the ``QuantumSystem`` takes
    ownership of the matrix elements and provides methods and functionality to
    be used by second-quantized many-body methods.

    Parameters
    ----------
    l : int
        Number of basis functions.
    dim : int
        Dimensionality of the systems.
    np : module
        Matrix and linear algebra module. Currently only `numpy` is supported.
    includes_spin : bool
        Toggle to determine if the basis set to be created includes explicitly
        in the basis functions. By default we assume that ``BasisSet`` only
        contains spatial orbitals thus there are no explicit spin-dependence in
        the elements.
    anti_symmetrized_u : bool
        Toggle to determine if the two-body matrix elements have been
        anti_symmetrized. By default we assume that this is not the case.
    """

    def __init__(
        self, l, dim, np=None, includes_spin=False, anti_symmetrized_u=False
    ):
        if np is None:
            import numpy as np

        self.np = np

        self.l = l
        self.dim = dim

        self._grid = None

        self._h = None
        self._u = None
        self._s = None
        self._dipole_moment = None

        self._spf = None
        self._bra_spf = None

        self._nuclear_repulsion_energy = 0

        self._includes_spin = includes_spin
        self._anti_symmetrized_u = anti_symmetrized_u

    @property
    def includes_spin(self):
        return self._includes_spin

    @property
    def anti_symmetrized_u(self):
        return self._anti_symmetrized_u

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, grid):
        self._grid = grid

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, h):
        assert all(self.check_axis_lengths(h, self.l))

        self._h = h

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, u):
        assert all(self.check_axis_lengths(u, self.l))

        self._u = u

    @property
    def s(self):
        return self._s

    @s.setter
    def s(self, s):
        assert all(self.check_axis_lengths(s, self.l))

        self._s = s

    @property
    def dipole_moment(self):
        return self._dipole_moment

    @dipole_moment.setter
    def dipole_moment(self, dipole_moment):
        assert len(dipole_moment) == self.dim

        for i in range(self.dim):
            assert all(self.check_axis_lengths(dipole_moment[i], self.l))

        self._dipole_moment = dipole_moment

    @property
    def spf(self):
        return self._spf

    @spf.setter
    def spf(self, spf):
        if not spf is None:
            assert spf.shape[0] == self.l
            assert len((*spf.shape[1:],)) == self.dim

        self._spf = spf

    @property
    def bra_spf(self):
        if self._bra_spf is None and self._spf is not None:
            # Return complex conjugate in case of a Hermitian basis set.
            self._bra_spf = self._spf.conj()

        return self._bra_spf

    @bra_spf.setter
    def bra_spf(self, bra_spf):
        if not bra_spf is None:
            assert bra_spf.shape[0] == self.l
            assert len((*bra_spf.shape[1:],)) == self.dim

        self._bra_spf = bra_spf

    @property
    def nuclear_repulsion_energy(self):
        return self._nuclear_repulsion_energy

    @nuclear_repulsion_energy.setter
    def nuclear_repulsion_energy(self, nuclear_repulsion_energy):
        self._nuclear_repulsion_energy = nuclear_repulsion_energy

    @staticmethod
    def change_arr_module(arr, np):
        return np.asarray(arr) if arr is not None else None

    def change_module(self, np):
        """Function converting all matrix elements to a new module.

        Parameters
        ----------
        np : module
            Array- and linalg-module.
        """
        self.np = np

        self.h = self.change_arr_module(self.h, self.np)
        self.s = self.change_arr_module(self.s, self.np)
        self.u = self.change_arr_module(self.u, self.np)
        self.spf = self.change_arr_module(self.spf, self.np)
        self.bra_spf = self.change_arr_module(self.bra_spf, self.np)
        self.dipole_moment = self.change_arr_module(self.dipole_moment, self.np)

    def cast_to_complex(self):
        """Function converting all matrix elements to ``np.complex128``, where
        ``np`` is the member array module.
        """
        np = self.np

        self.h = self.h.astype(np.complex128)
        self.u = self.u.astype(np.complex128)

        if self.s is not None:
            self.s = self.s.astype(np.complex128)

        if self.spf is not None:
            self.spf = self.spf.astype(np.complex128)

        if self.bra_spf is not None:
            self.bra_spf = self.bra_spf.astype(np.complex128)

        if self.dipole_moment is not None:
            self.dipole_moment = self.dipole_moment.astype(np.complex128)

    @staticmethod
    def transform_spf(spf, C, np):
        return np.tensordot(C, spf, axes=((0), (0)))

    @staticmethod
    def transform_bra_spf(bra_spf, C_tilde, np):
        return np.tensordot(C_tilde, bra_spf, axes=((1), (0)))

    @staticmethod
    def transform_one_body_elements(h, C, np, C_tilde=None):
        if C_tilde is None:
            C_tilde = C.conj().T

        return np.dot(C_tilde, np.dot(h, C))

    @staticmethod
    def transform_two_body_elements(u, C, np, C_tilde=None):
        if C_tilde is None:
            C_tilde = C.conj().T

        # abcd, ds -> abcs
        _u = np.tensordot(u, C, axes=(3, 0))
        # abcs, cr -> absr -> abrs
        _u = np.tensordot(_u, C, axes=(2, 0)).transpose(0, 1, 3, 2)
        # abrs, qb -> arsq -> aqrs
        _u = np.tensordot(_u, C_tilde, axes=(1, 1)).transpose(0, 3, 1, 2)
        # pa, aqrs -> pqrs
        _u = np.tensordot(C_tilde, _u, axes=(1, 0))

        return _u

    def get_transformed_h(self, C):
        return self.transform_one_body_elements(self.h, C, np=self.np)

    def get_transformed_u(self, C):
        return self.transform_two_body_elements(self.u, C, np=self.np)

    def _change_basis_one_body_elements(self, C, C_tilde=None):
        self.h = self.transform_one_body_elements(
            self.h, C, np=self.np, C_tilde=C_tilde
        )

        if self.s is not None:
            self.s = self.transform_one_body_elements(
                self.s, C, C_tilde=C_tilde, np=self.np
            )

    def _change_basis_two_body_elements(self, C, C_tilde=None):
        self.u = self.transform_two_body_elements(
            self.u, C, np=self.np, C_tilde=C_tilde
        )

    def _change_basis_dipole_moment(self, C, C_tilde=None):
        dipole_moment = []
        for i in range(self.dipole_moment.shape[0]):
            dipole_moment.append(
                self.transform_one_body_elements(
                    self.dipole_moment[i], C, np=self.np, C_tilde=C_tilde
                )
            )

        self.dipole_moment = self.np.asarray(dipole_moment)

    def _change_basis_spf(self, C, C_tilde=None):
        if C_tilde is not None:
            # In case of bi-orthogonal basis sets, we create an extra set
            # of single-particle functions for the bra-side.
            # Note the use of self.bra_spf instead of self.bra_spf in the
            # argument to the helper function. This guarantees that
            # self.bra_spf is not None.
            self.bra_spf = self.transform_bra_spf(
                self.bra_spf, C_tilde, self.np
            )

        self.spf = self.transform_spf(self.spf, C, self.np)

    def change_basis(self, C, C_tilde=None):
        r"""Function using coefficient matrices to change basis of the matrix
        elements and the single-particle functions. This function also supports
        rectangular coefficient matrices meaning that the number of basis
        functions can be updated. Let :math`\lvert \chi_{\alpha} \rangle`
        (:math:`\langle \tilde{\chi}_{\alpha} \rvert`) be the current basis set
        with :math:`\alpha \in \{1, \dots, N_{\alpha}\}`, and :math:`\lvert
        \phi_p \rangle` (:math:`\langle \tilde{\phi}_p \rvert`)the new basis
        with :math:`p \in \{1, \dots, N_p\}`. Then, for a given coefficient
        matrix :math:`C \in \mathbb{C}^{N_{p}\times N_{\alpha}}`
        (:math:`\tilde{C} \in \mathbb{C}^{N_{\alpha} \times N_p}`) the basis
        change for a single particle state is given by

        .. math:: \lvert \phi_p \rangle = C^{\alpha}_{p}
            \lvert\chi_{\alpha}\rangle,

        where the Einstein summation convention is assumed. Similarly for the
        dual state we have

        .. math:: \langle\tilde{\phi}_p\rvert = \tilde{C}^{p}_{\alpha}
            \langle\tilde{\chi}_{\alpha}\rvert.

        In case of a Hermitian basis we have :math:`\tilde{C} = C^{\dagger}`,
        and the dual states are the Hermitian adjoint of one another.

        Parameters
        ----------
        C : np.ndarray
            Coefficient matrix for "ket"-states.
        C_tilde : np.ndarray
            Coefficient matrix for "bra"-states, by default ``C_tilde`` is
            ``None`` and we treat ``C_tilde`` as the Hermitian conjugate of
            ``C``.
        """
        # Update basis set size
        self.l = C.shape[1]

        self._change_basis_one_body_elements(C, C_tilde)
        self._change_basis_two_body_elements(C, C_tilde)

        if self.dipole_moment is not None:
            self._change_basis_dipole_moment(C, C_tilde)

        if self.spf is not None:
            self._change_basis_spf(C, C_tilde)

    def compute_particle_density(self, rho_qp, C=None, C_tilde=None):
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
        C : np.ndarray
            Coefficient matrix for basis change. Default is ``None`` and hence
            no transformation occurs.
        C_tilde : np.ndarray
            Bra-state coefficient matrix. Default is ``None`` which leads to
            one of two situations. If ``C != None`` ``C_tilde`` is assumed to
            be the conjugated transpose of ``C``. Otherwise, no transformation
            occurs.

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

        if C is not None:
            ket_spf = self.transform_spf(ket_spf, C, self.np)
            C_tilde = C_tilde if C_tilde is not None else C.conj().T
            bra_spf = self.transform_bra_spf(bra_spf, C_tilde, self.np)

        return compute_particle_density(rho_qp, ket_spf, bra_spf, self.np)

    def anti_symmetrize_two_body_elements(self):
        r"""Function making the two-body matrix elements anti-symmetric. This
        corresponds to

        .. math:: \langle \phi_p \phi_q \rvert \hat{u}
            \lvert \phi_r \phi_s \rangle_{AS}
            \equiv
            \langle \phi_p \phi_q \rvert \hat{u} \lvert \phi_r \phi_s \rangle
            -
            \langle \phi_p \phi_q \rvert \hat{u} \lvert \phi_s \phi_r \rangle.
        """
        if not self._anti_symmetrized_u:
            self.u = self.anti_symmetrize_u(self.u)
            self._anti_symmetrized_u = True

    def change_to_general_orbital_basis(self, anti_symmetrize=True):
        r"""Function converting a spatial orbital basis set to a general orbital
        basis. That is, the function duplicates every element by adding a
        spin-function to each spatial orbital. This leads to an
        orbital-restricted spin-orbital where each spatial orbital is included
        in each spin-direction.

        .. math:: \phi_p(\mathbf{r}) \to
            \psi_p(x) = \phi_p(\mathbf{r}) \sigma(m_s),

        where :math:`x = (\mathbf{r}, m_s)` is a generalized coordinate of both
        position :math:`\mathbf{r}` and spin :math:`m_s`. Here
        :math:`\phi_p(\mathbf{r})` is a spatial orbital and :math:`\sigma(m_s)`
        a spin-function with :math:`\sigma \in \{\alpha, \beta\}`. The
        conversion happens in place.

        Parameters
        ----------
        anti_symmetrize : bool
            Whether or not to anti-symmetrize the elements in the two-body
            Hamiltonian. By default we perform an anti-symmetrization.
        """

        if self._includes_spin:
            warnings.warn(
                "The basis has already been spin-doubled. Avoiding a second "
                + "doubling."
            )
            return

        self._includes_spin = True

        self.l = 2 * self.l

        self.h = self.add_spin_one_body(self.h, np=self.np)
        self.s = self.add_spin_one_body(self.s, np=self.np)
        self.u = self.add_spin_two_body(self.u, np=self.np)

        if anti_symmetrize:
            self.anti_symmetrize_two_body_elements()

        if not self.dipole_moment is None:
            dipole_moment = [
                self.add_spin_one_body(self.dipole_moment[i], np=self.np)
                for i in range(len(self.dipole_moment))
            ]

            self.dipole_moment = self.np.array(dipole_moment)

        if not self.spf is None:
            self.spf = self.add_spin_spf(self.spf, self.np)

            # Make sure that we check if _bra_spf is not None, otherwise we
            # potentially construct the dual state.
            if not self._bra_spf is None:
                self.bra_spf = self.add_spin_bra_spf(self._bra_spf, self.np)

        return self

    @staticmethod
    def add_spin_spf(spf, np):
        new_shape = [spf.shape[0] * 2, *spf.shape[1:]]
        new_spf = np.zeros(tuple(new_shape), dtype=spf.dtype)

        new_spf[::2] = spf
        new_spf[1::2] = spf

        return new_spf

    @staticmethod
    def add_spin_bra_spf(bra_spf, np):
        if bra_spf is None:
            return None

        return BasisSet.add_spin_spf(bra_spf, np)

    @staticmethod
    def add_spin_one_body(h, np):
        return np.kron(h, np.eye(2))

    @staticmethod
    def add_spin_two_body(_u, np):
        u = _u.transpose(1, 3, 0, 2)
        u = np.kron(u, np.eye(2))
        u = u.transpose(2, 3, 0, 1)
        u = np.kron(u, np.eye(2))
        u = u.transpose(0, 2, 1, 3)

        return u

    @staticmethod
    def anti_symmetrize_u(_u):
        return _u - _u.transpose(0, 1, 3, 2)

    @staticmethod
    def check_axis_lengths(arr, length):
        return [length == axis for axis in arr.shape]

    def copy_basis(self):
        """Function creating a deep copy of the current basis. This function
        is a hack as we have to temporarily remove the stored module before
        using Python's ``copy.deepcopy``-function.

        Returns
        -------
        BasisSet
            A deep copy of the current basis.
        """

        np = self.np
        self.np = None

        new_basis = copy.deepcopy(self)

        new_basis.change_module(np)
        self.change_module(np)

        assert self.np is np

        return new_basis
