import numpy as np

from quantum_systems import (
    TwoDimensionalDoubleWell,
    TwoDimensionalHarmonicOscillator,
)


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

    tddw = TwoDimensionalDoubleWell(
        n, l, radius, num_grid_points, l_ho_factor=2, barrier_strength=2
    )
    tddw.setup_system(axis=1)

    np.testing.assert_allclose(tddw.epsilon, test_energies)
