import os
import pytest
import numpy as np
from quantum_systems import (
    GeneralOrbitalSystem,
    TwoDimensionalHarmonicOscillator,
    TwoDimHarmonicOscB,
)


@pytest.fixture(scope="module")
def get_tdhob():
    n = 2
    l = 10

    radius = 5
    num_grid_points = 201
    omega_c = 0.5

    tdhob = GeneralOrbitalSystem(
        n, TwoDimHarmonicOscB(l, radius, num_grid_points, omega_c=omega_c)
    )

    return tdhob


def test_tdhob(get_tdhob):
    tdhob = get_tdhob

    dip = np.load(os.path.join("tests", "dat", "tdhob_dipole_moment.npy"))
    np.testing.assert_allclose(dip, tdhob.position, atol=1e-10)

    h = np.load(os.path.join("tests", "dat", "tdhob_h.npy"))
    np.testing.assert_allclose(h, tdhob.h, atol=1e-10)

    u = np.load(os.path.join("tests", "dat", "tdhob_u.npy"))
    np.testing.assert_allclose(u, tdhob.u, atol=1e-10)

    spf = np.load(os.path.join("tests", "dat", "tdhob_spf.npy"))
    np.testing.assert_allclose(spf, tdhob.spf, atol=1e-10)


def test_two_body_elements_compare():
    n = 2
    l = 6
    grid_length = 5
    num_grid_points = 201
    mass = 1
    omega = 1
    omega_c = 0

    two_dim_ho = GeneralOrbitalSystem(
        n,
        TwoDimensionalHarmonicOscillator(
            l, grid_length, num_grid_points, mass=mass, omega=omega
        ),
    )
    two_dim_ho_b = GeneralOrbitalSystem(
        n,
        TwoDimHarmonicOscB(
            l,
            grid_length,
            num_grid_points,
            mass=mass,
            omega=omega,
            omega_c=omega_c,
        ),
    )

    np.testing.assert_allclose(two_dim_ho.u, two_dim_ho_b.u, atol=1e-8)
