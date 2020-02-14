from quantum_systems import QuantumSystem, SpinOrbitalSystem


class OrbitalSystem(QuantumSystem):
    r"""Quantum system containing orbital matrix elements, i.e., we only keep
    the spatial orbitals as they are degenerate in each spin direction. This
    means that

    .. math:: \psi(x, t) = \psi(\mathbf{r}, t) \sigma(m_s),

    where :math:`x = (\mathbf{r}, m_s)` is a generalized coordinate of position
    and spin, and :math:`sigma(m_s)` is either :math:`\alpha(m_s)` or
    :math:`\beta(m_s)` as the two-dimensional spin basis states. This means
    that we only store :math:`psi(mathbf{r}, t)`.
    """

    def __init__(self, n, *args, **kwargs):
        assert (
            n % 2 == 0
        ), "n must be divisable by 2 to be a closed-shell system"

        super().__init__(n // 2, *args, **kwargs)

    # def set_system_size(self, n, l):
    #     """Function setting the number of occupied orbitals `n // 2` and the
    #     number of orbital basis functions `l`. Note that `n` corresponds to
    #     the number of particles, i.e., the number of occupied _spin-orbitals_.
    #     This means that the occupied orbital indices in matrix elements
    #     correspond to `n // 2`, and we therefore require that `n % 2 == 0` to
    #     be a closed-shell system.

    #     Parameters
    #     ----------
    #     n : int
    #         The number of particles.
    #     l : int
    #         The number of orbital functions.

    #     See Also
    #     --------
    #     QuantumSystem.set_system_size
    #     """

    #     assert n % 2, "n must be divisable by 2 to be a closed-shell system"

    #     super().set_system_size(n // 2, l)

    def change_to_spin_orbital_basis(self, anti_symmetrize=True, n_a=None):
        r"""Function converting ``OrbitalSystem`` to a ``SpinOrbitalSystem`` by
        duplicating every basis element. That is,

        .. math:: \psi(\mathbf{r}, t)
            \to \psi(x, t) = \psi(\mathbf{r}, t) \sigma(m_2),

        where $x = (\mathbf{r}, m_s)$ is a generalized coordinate of both
        position $\mathbf{r}$ and spin $m_s$, with $\sigma(m_s)$ one of the two
        spin-functions.

        Parameters
        ----------
        anti_symmetrize : bool
            Whether or not to create the anti-symmetrized two-body elements.
            Default is ``True``.
        n_a : int
            Number of occupied particles with :math:`\alpha`-spin.

        Returns
        -------
        SpinOrbitalSystem
            The doubly degenerate spin-orbital system.
        """

        so_system = SpinOrbitalSystem(self.n * 2, self.l * 2, n_a, np=self.np)
        so_system.set_h(self.h, add_spin=True)
        so_system.set_u(self.u, add_spin=True, anti_symmetrize=anti_symmetrize)
        so_system.set_s(self.s, add_spin=True)
        so_system.set_dipole_moment(self.dipole_moment, add_spin=True)
        so_system.set_spf(self.spf, add_spin=True)
        so_system.set_bra_spf(self.bra_spf, add_spin=True)
        so_system.set_nuclear_repulsion_energy(self.nuclear_repulsion_energy)

        return so_system

    def compute_reference_energy(self):
        r"""Function computing the reference energy in an orbital system.
        This is given by

        .. math:: E_0 = \langle \Phi_0 \rvert \hat{H} \lvert \Phi_0 \rangle
            = 2 h^{i}_{i} + 2 u^{ij}_{ij} - u^{ij}_{ji},

        where :math:`\lvert \Phi_0 \rangle` is the reference determinant, and
        :math:`i, j` are occupied indices.

        Returns
        -------
        complex
            The reference energy.
        """

        o, v = self.o, self.v

        return (
            2 * self.np.trace(self._h[o, o])
            + 2
            * self.np.trace(
                self.np.trace(self._u[o, o, o, o], axis1=1, axis2=3)
            )
            - self.np.trace(
                self.np.trace(self._u[o, o, o, o], axis1=1, axis2=2)
            )
        )

    def construct_fock_matrix(self, h, u, f=None):
        r"""Function setting up the restricted Fock matrix in a closed-shell
        orbital basis.

        In an orbital basis we compute:

        .. math:: f^{p}_{q} = h^{p}_{q} + 2 u^{pi}_{qi} - u^{pi}_{iq},

        where :math:`p, q, r, s, ...` run over all indices and :math:`i, j, k,
        l, ...` correspond to the occupied indices. The two-body elements are
        assumed to not be anti-symmetric.

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
        f += 2 * np.einsum("piqi -> pq", u[:, o, :, o])
        f -= np.einsum("piiq -> pq", u[:, o, o, :])

        return f

    def change_to_hf_basis(self, *args, **kwargs):
        # TODO: Change to RHF-basis.
        raise NotImplementedError("There is currently no RHF implementation")
