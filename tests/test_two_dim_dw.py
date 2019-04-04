import numpy as np

from quantum_systems import (
    TwoDimensionalDoubleWell,
    TwoDimensionalHarmonicOscillator,
)
from quantum_systems.quantum_dots.two_dim.two_dim_helper import (
    get_double_well_one_body_elements,
)
from quantum_systems.system_helper import add_spin_one_body


def test_zero_barrier():
    """Test checking if we can reproduce the regular two-dimensional harmonic
    oscillator system when setting the barrier strength to zero.
    """

    n = 2
    l = 12
    radius = 10
    num_grid_points = 401

    tddw = TwoDimensionalDoubleWell(
        n, l, radius, num_grid_points, l_ho_factor=1, barrier_strength=0
    )
    tddw.setup_system(axis=0)

    tdho = TwoDimensionalHarmonicOscillator(n, l, radius, num_grid_points)
    tdho.setup_system()

    np.testing.assert_allclose(tddw.h, tdho.h, atol=1e-7)
    np.testing.assert_allclose(tddw.u, tdho.u, atol=1e-7)
    np.testing.assert_allclose(tddw.spf, tdho.spf, atol=1e-7)


def test_spf_energies():
    test_energies = np.array(
        [0.81129823, 1.37162083, 1.93581042, 2.21403823, 2.37162083, 2.93581042]
    )
    n = 2
    l = 6
    radius = 10
    num_grid_points = 401

    # Test barrier in x-direction
    tddw = TwoDimensionalDoubleWell(
        n, l, radius, num_grid_points, l_ho_factor=2, barrier_strength=2
    )
    tddw.setup_system(axis=0)
    np.testing.assert_allclose(tddw.epsilon, test_energies)

    # Test barrier in y-direction
    tddw = TwoDimensionalDoubleWell(
        n, l, radius, num_grid_points, l_ho_factor=2, barrier_strength=2
    )
    tddw.setup_system(axis=1)
    np.testing.assert_allclose(tddw.epsilon, test_energies)


def test_change_of_basis():
    # We try to construct a two-dimensional double-well system from a
    # two-dimensional harmonic oscillator system by using the change-of-basis
    # function with C found from diagonalizing the double-well one-body
    # Hamiltonian.
    n = 2
    l = 12
    l_ho_factor = 2
    l_ho = int(l * l_ho_factor)
    omega = 1
    mass = 1
    barrier_strength = 3
    radius = 10
    num_grid_points = 401
    axis = 0

    tdho = TwoDimensionalHarmonicOscillator(
        n, l_ho, radius, num_grid_points, omega=omega
    )
    tdho.setup_system()

    h_dw = get_double_well_one_body_elements(
        l_ho // 2, omega, mass, barrier_strength, dtype=np.complex128, axis=axis
    )

    epsilon, C_dw = np.linalg.eigh(h_dw)
    C = add_spin_one_body(C_dw, np=np)

    tdho.change_basis(C[:, :l])

    tddw = TwoDimensionalDoubleWell(
        n,
        l,
        radius,
        num_grid_points,
        omega=omega,
        mass=mass,
        barrier_strength=barrier_strength,
        l_ho_factor=l_ho_factor,
    )
    tddw.setup_system(axis=axis)

    np.testing.assert_allclose(tdho.u, tddw.u, atol=1e-7)
    np.testing.assert_allclose(tdho.spf, tddw.spf, atol=1e-7)
