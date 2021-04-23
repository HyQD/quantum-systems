import warnings
import copy

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
        self._position = None
        self._momentum = None

        self._sigma_x = None
        self._sigma_y = None
        self._sigma_z = None

        self._spin_x = None
        self._spin_y = None
        self._spin_z = None
        self._spin_2 = None
        self._spin_2_tb = None

        self._spf = None
        self._bra_spf = None

        self._nuclear_repulsion_energy = 0
        # We assume negative charge by default, i.e., electrons
        self.particle_charge = -1

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
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        assert len(position) == self.dim

        for i in range(self.dim):
            assert all(self.check_axis_lengths(position[i], self.l))

        self._position = position

    @property
    def dipole_moment(self):
        return self.particle_charge * self.position

    @property
    def momentum(self):
        return self._momentum

    @momentum.setter
    def momentum(self, momentum):
        assert len(momentum) == self.dim

        for i in range(self.dim):
            assert all(self.check_axis_lengths(momentum[i], self.l))

        self._momentum = momentum

    @property
    def spin_x(self):
        return self._spin_x

    @spin_x.setter
    def spin_x(self, spin_x):
        assert self.includes_spin
        assert all(self.check_axis_lengths(spin_x, self.l))

        self._spin_x = spin_x

    @property
    def spin_y(self):
        return self._spin_y

    @spin_y.setter
    def spin_y(self, spin_y):
        assert self.includes_spin
        assert all(self.check_axis_lengths(spin_y, self.l))

        self._spin_y = spin_y

    @property
    def spin_z(self):
        return self._spin_z

    @spin_z.setter
    def spin_z(self, spin_z):
        assert self.includes_spin
        assert all(self.check_axis_lengths(spin_z, self.l))

        self._spin_z = spin_z

    @property
    def spin_2(self):
        return self._spin_2

    @spin_2.setter
    def spin_2(self, spin_2):
        assert self.includes_spin
        assert all(self.check_axis_lengths(spin_2, self.l))

        self._spin_2 = spin_2

    @property
    def spin_2_tb(self):
        return self._spin_2_tb

    @spin_2_tb.setter
    def spin_2_tb(self, spin_2_tb):
        assert self.includes_spin
        assert all(self.check_axis_lengths(spin_2_tb, self.l))

        self._spin_2_tb = spin_2_tb

    @property
    def sigma_x(self):
        return self._sigma_x

    @sigma_x.setter
    def sigma_x(self, sigma_x):
        assert self.includes_spin

        self._sigma_x = sigma_x

    @property
    def sigma_y(self):
        return self._sigma_y

    @sigma_y.setter
    def sigma_y(self, sigma_y):
        assert self.includes_spin

        self._sigma_y = sigma_y

    @property
    def sigma_z(self):
        return self._sigma_z

    @sigma_z.setter
    def sigma_z(self, sigma_z):
        assert self.includes_spin

        self._sigma_z = sigma_z

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

        for name, arr in [
            ("_h", self.h),
            ("_s", self.s),
            ("_u", self.u),
            ("_spf", self.spf),
            ("_bra_spf", self.bra_spf),
            ("_position", self.position),
            ("_momentum", self.momentum),
            ("_spin_x", self.spin_x),
            ("_spin_y", self.spin_y),
            ("_spin_z", self.spin_z),
            ("_spin_2", self.spin_2),
            ("_spin_2_tb", self.spin_2_tb),
        ]:
            setattr(self, name, self.change_arr_module(arr, self.np))

    def cast_to_complex(self):
        """Function converting all matrix elements to ``np.complex128``, where
        ``np`` is the member array module.
        """
        np = self.np

        for name, arr in [
            ("_h", self.h),
            ("_s", self.s),
            ("_u", self.u),
            ("_spf", self.spf),
            ("_bra_spf", self.bra_spf),
            ("_position", self.position),
            ("_momentum", self.momentum),
            ("_spin_x", self.spin_x),
            ("_spin_y", self.spin_y),
            ("_spin_z", self.spin_z),
            ("_spin_2", self.spin_2),
            ("_spin_2_tb", self.spin_2_tb),
        ]:
            if arr is not None:
                setattr(self, name, arr.astype(np.complex128))

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

    def _change_basis_one_body_elements(self, C, C_tilde):
        self.h = self.transform_one_body_elements(
            self.h, C, np=self.np, C_tilde=C_tilde
        )

        if self.s is not None:
            self.s = self.transform_one_body_elements(
                self.s, C, C_tilde=C_tilde, np=self.np
            )

        for spin in [self.spin_x, self.spin_y, self.spin_z, self.spin_2]:
            if spin is not None:
                spin = self.transform_one_body_elements(
                    spin, C, C_tilde=C_tilde, np=self.np
                )

    def _change_basis_two_body_elements(self, C, C_tilde):
        self.u = self.transform_two_body_elements(
            self.u, C, np=self.np, C_tilde=C_tilde
        )

        if self.spin_2_tb is not None:
            self.spin_2_tb = self.transform_two_body_elements(
                self.spin_2_tb, C, np=self.np, C_tilde=C_tilde
            )

    def _change_basis_position_elements(self, C, C_tilde):
        position = []

        for i in range(self.position.shape[0]):
            position.append(
                self.transform_one_body_elements(
                    self.position[i], C, np=self.np, C_tilde=C_tilde
                )
            )

        self.position = self.np.asarray(position)

    def _change_basis_momentum_elements(self, C, C_tilde):
        momentum = []

        for i in range(self.momentum.shape[0]):
            momentum.append(
                self.transform_one_body_elements(
                    self.momentum[i], C, np=self.np, C_tilde=C_tilde
                )
            )

        self.momentum = self.np.asarray(momentum)

    def _change_basis_spf(self, C, C_tilde):
        self.bra_spf = self.transform_bra_spf(self.bra_spf, C_tilde, self.np)

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

        # Ensure that C_tilde is not None
        if C_tilde is None:
            C_tilde = C.conj().T

        self._change_basis_one_body_elements(C, C_tilde)
        self._change_basis_two_body_elements(C, C_tilde)

        if self.position is not None:
            self._change_basis_position_elements(C, C_tilde)

        if self.momentum is not None:
            self._change_basis_momentum_elements(C, C_tilde)

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

            if self.spin_2_tb is not None:
                self.spin_2_tb = self.anti_symmetrize_u(self.spin_2_tb)

            self._anti_symmetrized_u = True

    def change_to_general_orbital_basis(
        self, a=[1, 0], b=[0, 1], anti_symmetrize=True
    ):
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
        conversion happens in-place.

        Parameters
        ----------
        a : list, np.array
            The :math:`\alpha` (up) spin basis vector. Default is :math:`\alpha
            = (1, 0)^T`.
        b : list, np.array
            The :math:`\beta` (down) spin basis vector. Default is :math:`\beta
            = (0, 1)^T`. Note that ``a`` and ``b`` are assumed orthonormal.
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

        overlap = self.s.copy()

        self.h = self.add_spin_one_body(self.h, np=self.np)
        self.s = self.add_spin_one_body(self.s, np=self.np)
        self.u = self.add_spin_two_body(self.u, np=self.np)

        # temporary change to allow 2d representations of two-body operators, such as
        # for dvr. A 2d representation is necessary for large basis sets, in which case
        # self.spin_2 also becomes huge. Until a better representation of self.spin_2
        # can be found, we've removed the spin functionality.
        if getattr(self, "u_repr", "4d") != "2d":
            self.a = self.np.array(a).astype(self.np.complex128).reshape(-1, 1)
            self.b = self.np.array(b).astype(self.np.complex128).reshape(-1, 1)

            # Check that spin basis elements are orthonormal
            assert abs(self.np.dot(self.a.conj().T, self.a) - 1) < 1e-12
            assert abs(self.np.dot(self.b.conj().T, self.b) - 1) < 1e-12
            assert abs(self.np.dot(self.a.conj().T, self.b)) < 1e-12

            (
                self.sigma_x,
                self.sigma_y,
                self.sigma_z,
            ) = self.setup_pauli_matrices(self.a, self.b, self.np)

            self.spin_x = 0.5 * self.np.kron(overlap, self.sigma_x)
            self.spin_y = 0.5 * self.np.kron(overlap, self.sigma_y)
            self.spin_z = 0.5 * self.np.kron(overlap, self.sigma_z)

            self.spin_2, self.spin_2_tb = self.setup_spin_squared_operator(
                self.spin_x, self.spin_y, self.spin_z, self.s, self.np
            )

        if anti_symmetrize:
            self.anti_symmetrize_two_body_elements()

        if not self.position is None:
            position = [
                self.add_spin_one_body(self.position[i], np=self.np)
                for i in range(len(self.position))
            ]

            self.position = self.np.array(position)

        if not self.momentum is None:
            momentum = [
                self.add_spin_one_body(self.momentum[i], np=self.np)
                for i in range(len(self.momentum))
            ]

            self.momentum = self.np.array(momentum)

        if not self.spf is None:
            self.spf = self.add_spin_spf(self.spf, self.np)

            # Make sure that we check if _bra_spf is not None, otherwise we
            # potentially construct the dual state.
            if not self._bra_spf is None:
                self.bra_spf = self.add_spin_bra_spf(self._bra_spf, self.np)

        # Due to inherently complex matrices for spin we cast all elements to
        # complex numbers.
        self.cast_to_complex()

        return self

    @staticmethod
    def setup_pauli_matrices(a, b, np):
        r"""Static method computing matrix elements of the Pauli spin-matrices
        in a given orthonormal basis of spin functions :math:`\{\alpha,
        \beta\}`.

        .. math:: (\sigma_i)^{\rho}_{\gamma}
            = \langle \rho \rvert \hat{\sigma}_i \lvert \gamma \rangle,

        where :math:`\rho, \gamma \in \{\alpha, \beta\}` and :math:`i \in \{x,
        y, z\}` for the three Pauli matrices.

        Parameters
        ----------
        a : np.ndarray
            The :math:`\alpha` basis vector as a column vector.
        b : np.ndarray
            The :math:`\beta` basis vector as a column vector. Note that the
            two basis vectors are assumed to be orthonormal.
        np : module
            An appropriate array and linalg module.

        Returns
        -------
        tuple
            The three Pauli matrices in the order :math:`\sigma_x, \sigma_y,
            \sigma_z`.

        Example
        -------

        >>> import numpy as np
        >>> a = np.array([1, 0]).reshape(-1, 1)
        >>> b = np.array([0, 1]).reshape(-1, 1)
        >>> sigma_x, sigma_y, sigma_z = BasisSet.setup_pauli_matrices(a, b, np)
        >>> print(sigma_x)
        [[0.+0.j 1.+0.j]
         [1.+0.j 0.+0.j]]
        >>> print(sigma_y)
        [[0.+0.j 0.-1.j]
         [0.+1.j 0.+0.j]]
        >>> print(sigma_z)
        [[ 1.+0.j  0.+0.j]
         [ 0.+0.j -1.+0.j]]
        """
        sigma_x_mat = np.array([[0, 1], [1, 0]]).astype(np.complex128)
        sigma_y_mat = np.array([[0, -1j], [1j, 0]])
        sigma_z_mat = np.array([[1, 0], [0, -1]]).astype(np.complex128)

        sigma_x = np.zeros_like(sigma_x_mat)
        sigma_y = np.zeros_like(sigma_y_mat)
        sigma_z = np.zeros_like(sigma_z_mat)

        for i, s in enumerate([a, b]):
            for j, t in enumerate([a, b]):
                sigma_x[i, j] = np.dot(s.conj().T, np.dot(sigma_x_mat, t))
                sigma_y[i, j] = np.dot(s.conj().T, np.dot(sigma_y_mat, t))
                sigma_z[i, j] = np.dot(s.conj().T, np.dot(sigma_z_mat, t))

        return sigma_x, sigma_y, sigma_z

    @staticmethod
    def setup_spin_squared_operator(spin_x, spin_y, spin_z, overlap, np):
        r"""Static method computing the matrix elements of the one- and
        two-body spin squared operator, :math:`\hat{S}^2`. The operator is
        computed by

        .. math:: \hat{S}^2 = \hat{S}_x^2 + \hat{S}_y^2 + \hat{S}_z^2,

        where each squared direction :math:`\hat{S}_i^2` can be written

        .. math:: \hat{S}_i^2
            = (S_i)^{p}_{r} (S_i)^{r}_{q}
            \hat{c}^{\dagger}_{p} \hat{c}_{q}
            + (S_i)^{p}_{r} (S_i)^{q}_{s}
            \hat{c}^{\dagger}_{p} \hat{c}^{\dagger}_{q}
            \hat{c}_{s} \hat{c}_{r},

        in the second quantization formulation.

        Parameters
        ----------
        spin_x : np.ndarray
            Spin-matrix in :math:`x`-direction including orbital overlap.
        spin_y : np.ndarray
            Spin-matrix in :math:`y`-direction including orbital overlap.
        spin_z : np.ndarray
            Spin-matrix in :math:`z`-direction including orbital overlap.
        overlap : np.ndarray
            The overlap matrix elements between the spin-orbitals.
        np : module
            An appropriate array and linalg module.

        Returns
        -------
        (np.ndarray, np.ndarray)
            The spin-squared operator as two arrays on the form ``(l, l)`` and
            ``(l, l, l, l)``, where ``l`` is the number of spin-orbitals. The
            former corresponds to the one-body part of the spin-squared
            operator whereas the latter is the two-body part.
        """

        l = len(spin_x)

        spin_2 = np.zeros_like(spin_x)
        spin_2_tb = np.zeros((l, l, l, l), dtype=spin_2.dtype)

        for s_i in [spin_x, spin_y, spin_z]:
            spin_2 += s_i @ overlap @ s_i
            spin_2_tb += np.einsum("pr, qs -> pqrs", s_i, s_i)

        return spin_2, spin_2_tb

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
        return np.kron(_u, np.einsum("pr, qs -> pqrs", np.eye(2), np.eye(2)))

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
