from quantum_systems import QuantumSystem


class SpinOrbitalSystem(QuantumSystem):
    r"""Quantum system containing matrix elements represented as spin-orbitals,
    i.e.,

    .. math:: \psi(x, t) = \psi_{\alpha}(\mathbf{r}, t) \alpha(m_s)
        + \psi_{\beta}(\mathbf{r}, t) \beta(m_s),

    where $x = (\mathbf{r}, m_s)$ is a generalized coordinate of position and
    spin, and $\alpha(m_s)$ and $\beta(m_s)$ are the two-dimensional spin basis
    states. Using spin-orbitals allows mixing between different spin-directions,
    furthermore the spatial orbitals $\psi_{\alpha}(\mathbf{r}, t)$ and
    $\psi_{\beta}(\mathbf{r}, t)$ are allowed to vary freely. The spin-orbital
    representation is the most general representation of the single-particle states,
    but note that the system size grows quickly.
    """

    def __init__(self, *args, n_a=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_a = n_a

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

        self.o_a, self.o_b, self.v_a, self.v_b = get_spin_block_slices(
            self.n, self.n_a, self.l
        )

    @staticmethod
    def get_spin_block_slices(n, n_a, l):
        """Function computing the spin-block slicing. We assume that there is
        an equal amount of alpha spin and beta spin orbitals, i.e., `l == l_a *
        2 == l_b * 2`, but allow for unequal number of occupied orbitals.

        We use an even-odd ordering of the spin-orbitals where alpha orbitals
        are given even indices and beta orbitals are given odd indices.

        Parameters
        ----------
        n : int
            The number of occupied orbitals.
        n_a : int
            The number of occupied orbitals with alpha spin. Note that we
            require `n >= n_a`.
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
