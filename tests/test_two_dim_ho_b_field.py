import numpy as np
from quantum_systems.quantum_dots.two_dim.two_dim_ho import (
    TwoDimensionalHarmonicOscillator as tdho,
)
from quantum_systems.quantum_dots.two_dim_b_field.two_dim_ho import (
    TwoDimHarmonicOscB as tdho_b,
)


def test_two_body_elements_compare():
    n = 2
    l = 6
    grid_length = 5
    num_grid_points = 1001
    mass = 1
    omega = 1
    omega_c = 0

    two_dim_ho = tdho(
        n, l, grid_length, num_grid_points, mass=mass, omega=omega
    )
    two_dim_ho.setup_system()
    two_dim_ho_b = tdho_b(
        n,
        l,
        grid_length,
        num_grid_points,
        mass=mass,
        omega_0=omega,
        omega_c=omega_c,
    )
    two_dim_ho_b.setup_system()

    np.testing.assert_allclose(two_dim_ho._u, two_dim_ho_b._u, atol=1e-8)
