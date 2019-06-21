import os
import pytest
import numpy as np

from quantum_systems import OneDimensionalHarmonicOscillator


@pytest.fixture(scope="module")
def get_odho():
    n = 2
    l = 20

    grid_length = 5
    num_grid_points = 1001
    mass = 1
    omega = 1

    odho = OneDimensionalHarmonicOscillator(
        n, l, grid_length, num_grid_points, mass=mass, omega=omega
    )
    odho.setup_system()

    return odho


def test_dipole_moment(get_odho):
    odho = get_odho

    dip = np.load(os.path.join("tests", "dat", "odho_dipole_moment.npy"))
    np.testing.assert_allclose(dip, odho.dipole_moment)


def test_h(get_odho):
    odho = get_odho

    h = np.load(os.path.join("tests", "dat", "odho_h.npy"))
    np.testing.assert_allclose(h, odho.h)


def test_u(get_odho):
    odho = get_odho

    u = np.load(os.path.join("tests", "dat", "odho_u.npy"))
    np.testing.assert_allclose(u, odho.u)


def test_spf(get_odho):
    odho = get_odho

    spf = np.load(os.path.join("tests", "dat", "odho_spf.npy"))
    np.testing.assert_allclose(spf, odho.spf)
