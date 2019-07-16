import os
import pytest
import numpy as np

from quantum_systems import ODQD
from quantum_systems.quantum_dots.one_dim.one_dim_potentials import (
    HOPotential,
    DWPotential,
    GaussianPotential,
    DWPotentialSmooth,
)


@pytest.fixture(scope="module")
def get_odho():
    n = 2
    l = 20

    grid_length = 5
    num_grid_points = 1001
    omega = 1

    odho = ODQD(n, l, grid_length, num_grid_points)
    odho.setup_system(potential=HOPotential(omega))

    return odho


@pytest.fixture(scope="module")
def get_oddw():
    n = 2
    l = 20

    grid_length = 6
    num_grid_points = 1001

    omega = 1
    length_of_dw = 5

    oddw = ODQD(n, l, grid_length, num_grid_points)
    oddw.setup_system(potential=DWPotential(omega, length_of_dw))

    return oddw


@pytest.fixture(scope="module")
def get_odgauss():
    n = 2
    l = 20

    grid_length = 20
    num_grid_points = 1001

    weight = 1
    center = 0
    deviation = 2.5

    odgauss = ODQD(n, l, grid_length, num_grid_points)
    odgauss.setup_system(
        potential=GaussianPotential(weight, center, deviation, np=np)
    )

    return odgauss


@pytest.fixture(scope="module")
def get_oddw_smooth():
    n = 2
    l = 20

    grid_length = 5
    num_grid_points = 1001
    a = 5

    oddw_smooth = ODQD(n, l, grid_length, num_grid_points)
    oddw_smooth.setup_system(potential=DWPotentialSmooth(a=a))

    return oddw_smooth


def test_odho(get_odho):
    odho = get_odho

    dip = np.load(os.path.join("tests", "dat", "odho_dipole_moment.npy"))
    np.testing.assert_allclose(dip, odho.dipole_moment, atol=1e-10)

    h = np.load(os.path.join("tests", "dat", "odho_h.npy"))
    np.testing.assert_allclose(h, odho.h, atol=1e-10)

    u = np.load(os.path.join("tests", "dat", "odho_u.npy"))
    np.testing.assert_allclose(u, odho.u, atol=1e-10)

    spf = np.load(os.path.join("tests", "dat", "odho_spf.npy"))
    np.testing.assert_allclose(spf, odho.spf, atol=1e-10)


def test_oddw(get_oddw):
    oddw = get_oddw

    dip = np.load(os.path.join("tests", "dat", "oddw_dipole_moment.npy"))
    np.testing.assert_allclose(dip, oddw.dipole_moment, atol=1e-10)

    h = np.load(os.path.join("tests", "dat", "oddw_h.npy"))
    np.testing.assert_allclose(h, oddw.h, atol=1e-10)

    u = np.load(os.path.join("tests", "dat", "oddw_u.npy"))
    np.testing.assert_allclose(u, oddw.u, atol=1e-10)

    spf = np.load(os.path.join("tests", "dat", "oddw_spf.npy"))
    np.testing.assert_allclose(spf, oddw.spf, atol=1e-10)


def test_odgauss(get_odgauss):
    odgauss = get_odgauss

    dip = np.load(os.path.join("tests", "dat", "odgauss_dipole_moment.npy"))
    np.testing.assert_allclose(dip, odgauss.dipole_moment, atol=1e-10)

    h = np.load(os.path.join("tests", "dat", "odgauss_h.npy"))
    np.testing.assert_allclose(h, odgauss.h, atol=1e-10)

    u = np.load(os.path.join("tests", "dat", "odgauss_u.npy"))
    np.testing.assert_allclose(u, odgauss.u, atol=1e-10)

    spf = np.load(os.path.join("tests", "dat", "odgauss_spf.npy"))
    np.testing.assert_allclose(spf, odgauss.spf, atol=1e-10)


def test_oddw_smooth(get_oddw_smooth):
    oddw_smooth = get_oddw_smooth

    dip = np.load(os.path.join("tests", "dat", "oddw_smooth_dipole_moment.npy"))
    np.testing.assert_allclose(dip, oddw_smooth.dipole_moment, atol=1e-10)

    h = np.load(os.path.join("tests", "dat", "oddw_smooth_h.npy"))
    np.testing.assert_allclose(h, oddw_smooth.h, atol=1e-10)

    u = np.load(os.path.join("tests", "dat", "oddw_smooth_u.npy"))
    np.testing.assert_allclose(u, oddw_smooth.u, atol=1e-10)

    spf = np.load(os.path.join("tests", "dat", "oddw_smooth_spf.npy"))
    np.testing.assert_allclose(spf, oddw_smooth.spf, atol=1e-10)
