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

    def compute_reference_energy(self):
        r"""Function computing the reference energy in a general spin-orbital
        system.  This is given by

        .. math:: E_0 = \langle \Phi_0 \rvert \hat{H} \lvert \Phi_0 \rangle
            = h^{i}_{i} + \frac{1}{2} u^{ij}_{ij},

        where :math:`\lvert \Phi_0 \rangle` is the reference determinant, and
        :math:`i, j` are occupied indices.

        Returns
        -------
        complex
            The reference energy.
        """

        o, v = self.o, self.v

        return (
            self.np.trace(self.h[o, o])
            + 0.5
            * self.np.trace(self.np.trace(self.u[o, o, o, o], axis1=1, axis2=3))
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
