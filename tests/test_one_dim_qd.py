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


@pytest.fixture
def get_odqd_systems(get_odho, get_oddw, get_odgauss, get_oddw_smooth):
    return [
        ("odho", get_odho),
        ("oddw", get_oddw),
        ("odgauss", get_odgauss),
        ("oddw_smooth", get_oddw_smooth),
    ]


def test_odqd_systems(get_odqd_systems):
    for sys in get_odqd_systems:
        name, odqd = sys

        dip = np.load(os.path.join("tests", "dat", f"{name}_dipole_moment.npy"))
        np.testing.assert_allclose(
            np.abs(dip), np.abs(odqd.position), atol=1e-9
        )

        h = np.load(os.path.join("tests", "dat", f"{name}_h.npy"))
        np.testing.assert_allclose(h, odqd.h, atol=1e-10)

        u = np.load(os.path.join("tests", "dat", f"{name}_u.npy"))
        np.testing.assert_allclose(np.abs(u), np.abs(odqd.u), atol=1e-10)

        spf = np.load(os.path.join("tests", "dat", f"{name}_spf.npy"))
        np.testing.assert_allclose(np.abs(spf), np.abs(odqd.spf), atol=1e-10)


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
