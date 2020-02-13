from quantum_systems import QuantumSystem


class OrbitalSystem(QuantumSystem):
    r"""Quantum system containing orbital matrix elements, i.e., we only keep
    the spatial orbitals as they are degenerate in each spin direction. This
    means that

    .. math:: \psi(x, t) = \psi(\mathbf{r}, t) \sigma(m_s),

    where $x = (\mathbf{r}, m_s)$ is a generalized coordinate of position and
    spin, and $\sigma(m_s)$ is either $\alpha(m_s)$ or $\beta(m_s)$ as the
    two-dimensional spin basis states. This means that we only store
    $\psi(mathbf{r}, t)$.
    """
    pass
