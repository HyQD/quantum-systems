import numpy as np
import math

from quantum_systems import QuantumSystem
from quantum_systems.quantum_dots.two_dim.two_dim_helper import (
    get_double_well_one_body_elements,
    get_coulomb_elements,
    spf_state,
)
from quantum_systems.system_helper import (
    add_spin_one_body,
    add_spin_two_body,
    anti_symmetrize_u,
    transform_two_body_elements,
)


class TwoDimensionalDoubleWell(QuantumSystem):
    """System constructing two-dimensional quantum dots with a double well
    potential barrier.

    Parameters
    ----------
    n : int
        Number of occupied spin-orbitals.
    l : int
        Number of spin-orbitals.
    radius : float
        Length of radial coordinate.
    num_grid_points : int
        Discretization of spatial coordinates. This includes both the radial and
        the angular part of the room.
    barrier_strength: float
        Barrier strength in the double well potential.
    l_ho_factor : float
        Factor of harmonic oscillator basis functions compared to the number of
        double-well functions. Note that l_ho_factor >= 1, as we need more
        harmonic oscillator basis functions than double-well functions in
        order to get a good representation of the basis elements.
    omega: float
        Frequency of the oscillator potential.
    mass: float
        Mass of the particles.

    Returns
    -------
    None
    """

    def __init__(
        self,
        n,
        l,
        radius,
        num_grid_points,
        barrier_strength=1,
        l_ho_factor=1.0,
        omega=1,
        mass=1,
    ):
        assert l_ho_factor >= 1, (
            "Number of harmonic oscillator functions must be higher than the"
            + " number of double-well basis functions"
        )
        super().__init__(n, l)

        self.omega = omega
        self.mass = mass
        self.barrier_strength = barrier_strength
        self.l_ho = math.floor(self.l * l_ho_factor)

        self.radius = radius
        self.num_grid_points = num_grid_points

        self.radius = np.linspace(0, self.radius, self.num_grid_points)
        self.theta = np.linspace(0, 2 * np.pi, self.num_grid_points)

    def setup_system(self, axis=0):
        """Function setting up the one- and two-body elements, the
        single-particle functions, dipole moments and other quantities used by
        second quantization methods.
        Parameters
        ----------
        axis : int
            Specifies which axis the barrier should be set to. For axis == 0
            this corresponds to the x-axis and axis == 1 the y-axis. That is,
                axis == 0 -> <p||x||q>,
                axis == 1 -> <p||y||q>.

        Returns
        -------
        None
        """

        __h = get_double_well_one_body_elements(
            self.l_ho // 2,
            self.omega,
            self.mass,
            self.barrier_strength,
            dtype=np.complex128,
            axis=axis,
        )
        __u = np.sqrt(self.omega) * get_coulomb_elements(
            self.l_ho // 2, dtype=np.complex128
        )

        self.epsilon, C = np.linalg.eigh(__h)

        # Construct one-body matrix from the eigenenergies of the single
        # particle eigenstates for the double well. We only include the
        # states we deem converged.
        _h = np.diagflat(self.epsilon[: self.l // 2])
        self._h = add_spin_one_body(_h, np=np)

        # Transform Coulomb elements to new double-well basis
        _u = transform_two_body_elements(__u, C[:, : self.l // 2], np=np)
        self._u = anti_symmetrize_u(add_spin_two_body(_u, np=np))

        # Create harmonic oscillator single-particle functions
        _spf_ho = self.setup_spf()
        # Transform to double-well basis
        _spf_dw = np.tensordot(C[:, : self.l // 2], _spf_ho, axes=((0), (0)))

        # Create spin-degenerate single-particle functions
        self._spf = np.zeros((self.l, *self.R.shape), dtype=np.complex128)
        self._spf[::2] += _spf_dw
        self._spf[1::2] += _spf_dw

    def setup_spf(self):
        self.R, self.T = np.meshgrid(self.radius, self.theta)
        spf = np.zeros((self.l_ho // 2, *self.R.shape), dtype=np.complex128)

        for p in range(self.l_ho // 2):
            spf[p, :] += spf_state(self.R, self.T, p, self.mass, self.omega)

        return spf
