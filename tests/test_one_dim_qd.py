import os
import pytest
import numpy as np

from quantum_systems import ODQD, GeneralOrbitalSystem, SpatialOrbitalSystem


@pytest.fixture(scope="module")
def get_odho():
    n = 2
    l = 10

    grid_length = 5
    num_grid_points = 1001
    omega = 1

    odho = GeneralOrbitalSystem(
        n,
        ODQD(
            l, grid_length, num_grid_points, potential=ODQD.HOPotential(omega)
        ),
    )

    return odho


@pytest.fixture(scope="module")
def get_odho_ao():
    n = 2
    l = 20

    grid_length = 5
    num_grid_points = 1001
    omega = 1

    odho = SpatialOrbitalSystem(
        n,
        ODQD(
            l, grid_length, num_grid_points, potential=ODQD.HOPotential(omega)
        ),
    )

    return odho


@pytest.fixture(scope="module")
def get_oddw():
    n = 2
    l = 10

    grid_length = 6
    num_grid_points = 1001

    omega = 1
    length_of_dw = 5

    oddw = GeneralOrbitalSystem(
        n,
        ODQD(
            l,
            grid_length,
            num_grid_points,
            potential=ODQD.DWPotential(omega, length_of_dw),
        ),
    )

    return oddw


@pytest.fixture(scope="module")
def get_odgauss():
    n = 2
    l = 10

    grid_length = 20
    num_grid_points = 1001

    weight = 1
    center = 0
    deviation = 2.5

    odgauss = GeneralOrbitalSystem(
        n,
        ODQD(
            l,
            grid_length,
            num_grid_points,
            potential=ODQD.GaussianPotential(weight, center, deviation, np=np),
        ),
    )

    return odgauss


@pytest.fixture(scope="module")
def get_oddw_smooth():
    n = 2
    l = 10

    grid_length = 5
    num_grid_points = 1001
    a = 5

    oddw_smooth = GeneralOrbitalSystem(
        n,
        ODQD(
            l,
            grid_length,
            num_grid_points,
            potential=ODQD.DWPotentialSmooth(a=a),
        ),
    )

    return oddw_smooth


def test_odho(get_odho):
    odho = get_odho

    dip = np.load(os.path.join("tests", "dat", "odho_dipole_moment.npy"))
    np.testing.assert_allclose(dip, odho.position, atol=1e-10)

    h = np.load(os.path.join("tests", "dat", "odho_h.npy"))
    np.testing.assert_allclose(h, odho.h, atol=1e-10)

    u = np.load(os.path.join("tests", "dat", "odho_u.npy"))
    np.testing.assert_allclose(u, odho.u, atol=1e-10)

    spf = np.load(os.path.join("tests", "dat", "odho_spf.npy"))
    np.testing.assert_allclose(spf, odho.spf, atol=1e-10)


def test_oddw(get_oddw):
    oddw = get_oddw

    dip = np.load(os.path.join("tests", "dat", "oddw_dipole_moment.npy"))
    np.testing.assert_allclose(dip, oddw.position, atol=1e-10)

    h = np.load(os.path.join("tests", "dat", "oddw_h.npy"))
    np.testing.assert_allclose(h, oddw.h, atol=1e-10)

    u = np.load(os.path.join("tests", "dat", "oddw_u.npy"))
    np.testing.assert_allclose(u, oddw.u, atol=1e-10)

    spf = np.load(os.path.join("tests", "dat", "oddw_spf.npy"))
    np.testing.assert_allclose(spf, oddw.spf, atol=1e-10)


def test_odgauss(get_odgauss):
    odgauss = get_odgauss

    dip = np.load(os.path.join("tests", "dat", "odgauss_dipole_moment.npy"))
    np.testing.assert_allclose(dip, odgauss.position, atol=1e-10)

    h = np.load(os.path.join("tests", "dat", "odgauss_h.npy"))
    np.testing.assert_allclose(h, odgauss.h, atol=1e-10)

    u = np.load(os.path.join("tests", "dat", "odgauss_u.npy"))
    np.testing.assert_allclose(u, odgauss.u, atol=1e-10)

    spf = np.load(os.path.join("tests", "dat", "odgauss_spf.npy"))
    np.testing.assert_allclose(spf, odgauss.spf, atol=1e-10)


def test_oddw_smooth(get_oddw_smooth):
    oddw_smooth = get_oddw_smooth

    dip = np.load(os.path.join("tests", "dat", "oddw_smooth_dipole_moment.npy"))
    np.testing.assert_allclose(dip, oddw_smooth.position, atol=1e-10)

    h = np.load(os.path.join("tests", "dat", "oddw_smooth_h.npy"))
    np.testing.assert_allclose(h, oddw_smooth.h, atol=1e-10)

    u = np.load(os.path.join("tests", "dat", "oddw_smooth_u.npy"))
    np.testing.assert_allclose(u, oddw_smooth.u, atol=1e-10)

    spf = np.load(os.path.join("tests", "dat", "oddw_smooth_spf.npy"))
    np.testing.assert_allclose(spf, oddw_smooth.spf, atol=1e-10)


def test_anti_symmetric_two_body_symmetry_odho(get_odho):
    odho = get_odho

    l = odho.l
    u = odho.u

    for p in range(l):
        for q in range(l):
            for r in range(l):
                for s in range(l):
                    assert abs(u[p, q, r, s] + u[p, q, s, r]) < 1e-8
                    assert abs(u[p, q, r, s] + u[q, p, r, s]) < 1e-8
                    assert abs(u[p, q, r, s] - u[q, p, s, r]) < 1e-8


def test_anti_symmetric_two_body_symmetry_oddw(get_oddw):
    oddw = get_oddw

    l = oddw.l
    u = oddw.u

    for p in range(l):
        for q in range(l):
            for r in range(l):
                for s in range(l):
                    assert abs(u[p, q, r, s] + u[p, q, s, r]) < 1e-8
                    assert abs(u[p, q, r, s] + u[q, p, r, s]) < 1e-8
                    assert abs(u[p, q, r, s] - u[q, p, s, r]) < 1e-8


def test_two_body_symmetry_odho(get_odho_ao):
    odho = get_odho_ao

    l = odho.l // 2
    u = odho.u

    for p in range(l):
        for q in range(l):
            for r in range(l):
                for s in range(l):
                    assert abs(u[p, q, r, s] - u[q, p, s, r]) < 1e-8
