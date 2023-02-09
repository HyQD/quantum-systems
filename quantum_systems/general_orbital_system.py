from quantum_systems import QuantumSystem


class GeneralOrbitalSystem(QuantumSystem):
    r"""Quantum system containing matrix elements represented as general
    spin-orbitals, i.e.,

    .. math:: \psi(x, t) = \psi^{\alpha}(\mathbf{r}, t) \alpha(m_s)
        + \psi^{\beta}(\mathbf{r}, t) \beta(m_s),

    where :math:`x = (\mathbf{r}, m_s)` is a generalized coordinate of position
    and spin, and :math:`\alpha(m_s)` and :math:`\beta(m_s)` are the
    two-dimensional spin basis states. Using general spin-orbitals allows
    mixing between different spin-directions, furthermore the spatial orbitals
    :math:`\psi^{\alpha}(\mathbf{r}, t)` and :math:`\psi^{\beta}(\mathbf{r},
    t)` are allowed to vary freely. The spin-orbital representation is the most
    general representation of the single-particle states, but note that the
    system size grows quickly. Furthermore, we are unable to categorize a
    general spin-orbital as having a definitive spin-direction.

    Parameters
    ----------
    a : list, np.array
        The :math:`\alpha` (up) spin basis vector. Default is :math:`\alpha =
        (1, 0)^T`.
    b : list, np.array
        The :math:`\beta` (down) spin basis vector. Default is :math:`\beta =
        (0, 1)^T`. Note that ``a`` and ``b`` are assumed orthonormal.
    anti_symmetrize : bool
        Whether or not to create the anti-symmetrized two-body elements.
        Default is ``True``.

    See Also
    -------
    QuantumSystem
        Parent class with constructor and main interface functions.
    """

    def __init__(
        self, n, basis_set, a=[1, 0], b=[0, 1], anti_symmetrize=True, **kwargs
    ):
        if not basis_set.includes_spin:
            basis_set = basis_set.change_to_general_orbital_basis(
                a=a, b=b, anti_symmetrize=anti_symmetrize
            )

        if anti_symmetrize:
            # Anti-symmetrize in case of a basis set containing general
            # orbitals (i.e., basis_set.includes_spin == True), without the
            # two-body elements being anti-symmetric.
            basis_set.anti_symmetrize_two_body_elements()

        super().__init__(n, basis_set, **kwargs)

    @property
    def spin_x(self):
        return self._basis_set.spin_x

    @property
    def spin_y(self):
        return self._basis_set.spin_y

    @property
    def spin_z(self):
        return self._basis_set.spin_z

    @property
    def spin_2(self):
        return self._basis_set.spin_2

    @property
    def spin_2_tb(self):
        return self._basis_set.spin_2_tb

    def compute_reference_energy(self, h=None, u=None):
        r"""Function computing the reference energy in a general spin-orbital
        system.  This is given by

        .. math:: E_0 = \langle \Phi_0 \rvert \hat{H} \lvert \Phi_0 \rangle
            = h^{i}_{i} + \frac{1}{2} u^{ij}_{ij} + E_n,

        where :math:`\lvert \Phi_0 \rangle` is the reference determinant,
        :math:`i, j` are occupied indices, :math:`h^{i}_{j}` the matrix
        elements of the one-body Hamiltonian, :math:`u^{ij}_{kl}` the matrix
        elements of the two-body Hamiltonian, and :math:`E_n` the nuclear
        repulsion energy, i.e., the constant term in the Hamiltonian.

        Parameters
        ----------
        h : np.ndarray
            The one-body matrix elements. When `h=None` `self.h` is used.
            If a custom `h` is passed in, the function assumes that `h` is at
            least a `(n, n)`-array, where `n` is the number of occupied
            indices. Default is `h=None`.
        u : np.ndarray
            The two-body matrix elements. When `u=None` `self.u` is used.
            If a custom `u` is passed in, the function assumes that `u` is at
            least a `(n, n, n, n)`-array, where `n` is the number of occupied
            indices. Default is `u=None`.

        Returns
        -------
        complex
            The reference energy.
        """

        o, v = self.o, self.v

        h = self.h if h is None else h
        u = self.u if u is None else u

        return (
            self.np.trace(h[o, o])
            + 0.5
            * self.np.trace(self.np.trace(u[o, o, o, o], axis1=1, axis2=3))
            + self.nuclear_repulsion_energy
        )

    def construct_fock_matrix(self, h, u, f=None):
        """Function setting up the Fock matrix, that is, the normal-ordered
        one-body elements, in a general spin-orbital basis. This function
        assumes that the two-body elements are anti-symmetrized.

        In a general spin-orbital basis we compute:

        .. math:: f^{p}_{q} = h^{p}_{q} + u^{pi}_{qi},

        where :math:`p, q, r, s, ...` run over all indices and :math:`i, j, k,
        l, ...` correspond to the occupied indices.

        Parameters
        ----------
        h : np.ndarray
            The one-body matrix elements.
        u : np.ndarray
            The two-body matrix elements.
        f : np.ndarray
            An empty array of the same shape as ``h`` to be filled with the
            Fock matrix elements. Default is ``None`` which means that we
            allocate a new matrix.

        Returns
        -------
        np.ndarray
            The filled Fock matrix.
        """

        np = self.np
        o, v = (self.o, self.v)

        if f is None:
            f = np.zeros_like(h)

        f.fill(0)

        f += h
        f += np.einsum("piqi -> pq", u[:, o, :, o])

        return f

    def change_to_hf_basis(self, *args, **kwargs):
        # from tdhf import HartreeFock

        # hf = HartreeFock(system=self, verbose=verbose, np=self.np)
        # c = hf.scf(*args, **kwargs)
        # self.change_basis(c)

        # TODO: Change to GHF-basis.
        raise NotImplementedError("There is currently no GHF implementation")
