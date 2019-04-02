import numpy as np
import math

from quantum_systems import QuantumSystem
from quantum_systems.quantum_dots.two_dim.two_dim_helper import (
    get_one_body_elements,
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
        l_ho_factor=1.3,
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

        self.radius = np.linspace(0, self.radius_length, self.num_grid_points)
        self.theta = np.linspace(0, 2 * np.pi, self.num_grid_points)

    def setup_system(self, axis=0):
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
