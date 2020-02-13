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
    pass
