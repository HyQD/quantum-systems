from quantum_systems import QuantumSystem


class SpinOrbitalSystem(QuantumSystem):
    r"""Quantum system containing matrix elements represented as spin-orbitals,
    i.e.,

    .. math:: \psi(x, t) = \psi_{\alpha}(\mathbf{r}, t) \alpha(m_s)
        + \psi_{\beta}(\mathbf{r}, t) \beta(m_s),

    where :math:`x = (\mathbf{r}, m_s)` is a generalized coordinate of position
    and spin, and :math:`\alpha(m_s)` and :math:`\beta(m_s)` are the
    two-dimensional spin basis states. Using spin-orbitals allows mixing
    between different spin-directions, furthermore the spatial orbitals
    :math:`\psi_{\alpha}(\mathbf{r}, t)` and :math:`\psi_{\beta}(\mathbf{r},
    t)` are allowed to vary freely. The spin-orbital representation is the most
    general representation of the single-particle states, but note that the
    system size grows quickly.
    """

    def __init__(self, *args, n_a=None, **kwargs):
        self.n_a = n_a
        super().__init__(*args, **kwargs)

    def set_system_size(self, n, l):
        super().set_system_size(n, l)

        if self.n_a is None:
            self.n_a = n // 2

        assert self.n_a <= self.n
        assert self.l % 2 == 0
        assert self.n_a <= self.l // 2
        assert self.n - self.n_a <= self.l // 2

        self.n_b = self.n - self.n_a
        self.l_a = self.l // 2
        self.l_b = self.l // 2
        self.m_a = self.l_a - self.n_a
        self.m_b = self.l_b - self.n_b

        self.o_a, self.o_b, self.v_a, self.v_b = self.get_spin_block_slices(
            self.n, self.n_a, self.l
        )

    @staticmethod
    def get_spin_block_slices(n, n_a, l):
        """Function computing the spin-block slicing. We assume that there is
        an equal amount of alpha spin and beta spin orbitals, i.e., ``l == l_a
        * 2 == l_b * 2``, but allow for unequal number of occupied orbitals.

        We use an even-odd ordering of the spin-orbitals where alpha orbitals
        are given even indices and beta orbitals are given odd indices.

        Parameters
        ----------
        n : int
            The number of occupied orbitals.
        n_a : int
            The number of occupied orbitals with alpha spin. Note that we
            require ``n >= n_a``.
        l : int
            The total number of spin-orbitals.

        Returns
        -------
        tuple
            Collection of the spin-block slices in the ordering (occupied
            alpha, occupied beta, virtual alpha, virtual down).
        """

        assert n >= n_a

        n_b = n - n_a
        l_a = l_b = l // 2

        m_a = l_a - n_a
        m_b = l_b - n_b

        o_a = slice(0, 2 * n_a, 2)
        o_b = slice(1, 2 * n_b + 1, 2)
        v_a = slice(2 * n_a, l, 2)
        v_b = slice(2 * n_b + 1, l, 2)

        assert (o_a.stop - o_a.start) // 2 == n_a
        assert (o_b.stop - o_b.start) // 2 == n_b
        assert (v_a.stop - v_a.start) // 2 == m_a
        assert (v_b.stop - v_b.start) // 2 + 1 == m_b

        return o_a, o_b, v_a, v_b

    def compute_reference_energy(self):
        r"""Function computing the reference energy in a spin-orbital system.
        This is given by

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

        return self.np.trace(self._h[o, o]) + 0.5 * self.np.trace(
            self.np.trace(self._u[o, o, o, o], axis1=1, axis2=3)
        )

    def construct_fock_matrix(self, h, u, f=None):
        """Function setting up the Fock matrix, that is, the normal-ordered
        one-body elements, in a spin-orbital basis. This function assumes that
        the two-body elements are anti-symmetrized.

        In a spin-orbital basis we compute:

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
            An empty array of the same shape as `h` to be filled with the Fock
            matrix elements. Default is `None` which means that we allocate a
            new matrix.

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
