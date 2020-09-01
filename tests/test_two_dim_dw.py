import os
import pytest
import numpy as np

from quantum_systems import (
    BasisSet,
    GeneralOrbitalSystem,
    TwoDimensionalDoubleWell,
    TwoDimensionalHarmonicOscillator,
)
from quantum_systems.quantum_dots.two_dim.two_dim_helper import (
    get_double_well_one_body_elements,
    theta_1_tilde_integral,
    theta_2_tilde_integral,
)


def theta_1_tilde_integral_wolfram(m_p, m_q):
    if abs(m_p - m_q) == 1:
        return 0

    integral = (
        -1j
        * (
            -m_q
            + m_p
            + (m_q - m_p) * np.exp(1j * np.pi * (m_q - m_p))
            - 2 * 1j * np.exp(1j * np.pi * (m_q - m_p) / 2)
        )
        * (1 + np.exp(1j * np.pi * (m_q - m_p)))
        / ((m_q - m_p) ** 2 - 1)
    )

    return integral


def theta_2_tilde_integral_wolfram(m_p, m_q):
    if abs(m_p - m_q) == 1:
        return 0

    integral = -((1 + np.exp(1j * np.pi * (m_q - m_p))) ** 2) / (
        (m_q - m_p) ** 2 - 1
    )

    return integral


def test_theta_1_tilde_integral():
    for m_p in range(-100, 101):
        for m_q in range(-100, 101):
            assert (
                abs(
                    theta_1_tilde_integral_wolfram(m_p, m_q)
                    - theta_1_tilde_integral(m_p, m_q)
                )
                < 1e-10
            )


def test_theta_2_tilde_integral():
    for m_p in range(-100, 101):
        for m_q in range(-100, 101):
            assert (
                abs(
                    theta_2_tilde_integral_wolfram(m_p, m_q)
                    - theta_2_tilde_integral(m_p, m_q)
                )
                < 1e-10
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
        l, radius, num_grid_points, barrier_strength=0, axis=0
    )

    tdho = TwoDimensionalHarmonicOscillator(l, radius, num_grid_points)

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
    omega = 1
    mass = 1
    barrier_strength = 2
    axis = 1

    h_dw = get_double_well_one_body_elements(
        l, omega, mass, barrier_strength, dtype=np.complex128, axis=axis
    )

    epsilon, C = np.linalg.eigh(h_dw)
    np.testing.assert_allclose(epsilon[: len(test_energies)], test_energies)


def test_change_of_basis():
    # We try to construct a two-dimensional double-well system from a
    # two-dimensional harmonic oscillator system by using the change-of-basis
    # function with C found from diagonalizing the double-well one-body
    # Hamiltonian.
    n = 2
    l = 12
    omega = 1
    mass = 1
    barrier_strength = 3
    radius = 10
    num_grid_points = 401
    axis = 0

    tdho = GeneralOrbitalSystem(
        n,
        TwoDimensionalHarmonicOscillator(
            l, radius, num_grid_points, omega=omega
        ),
    )

    h_dw = get_double_well_one_body_elements(
        l, omega, mass, barrier_strength, dtype=np.complex128, axis=axis
    )

    epsilon, C_dw = np.linalg.eigh(h_dw)
    C = BasisSet.add_spin_one_body(C_dw, np=np)

    tdho.change_basis(C)

    tddw = GeneralOrbitalSystem(
        n,
        TwoDimensionalDoubleWell(
            l,
            radius,
            num_grid_points,
            omega=omega,
            mass=mass,
            barrier_strength=barrier_strength,
            axis=axis,
        ),
    )
    tddw.change_basis(C)

    np.testing.assert_allclose(tdho.u, tddw.u, atol=1e-7)
    np.testing.assert_allclose(tdho.spf, tddw.spf, atol=1e-7)


@pytest.fixture(scope="module")
def get_tddw():
    n = 2
    l = 10
    axis = 0

    radius = 8
    num_grid_points = 201
    barrier_strength = 3
    omega = 0.8

    tddw = GeneralOrbitalSystem(
        n,
        TwoDimensionalDoubleWell(
            l,
            radius,
            num_grid_points,
            barrier_strength=barrier_strength,
            omega=omega,
            axis=axis,
        ),
    )

    return tddw


def test_tddw(get_tddw):
    tddw = get_tddw

    h_dw = get_double_well_one_body_elements(
        tddw.l // 2,
        tddw._basis_set.omega,
        tddw._basis_set.mass,
        tddw._basis_set.barrier_strength,
        dtype=np.complex128,
        axis=0,
    )

    epsilon, C_dw = np.linalg.eigh(h_dw)
    C = BasisSet.add_spin_one_body(C_dw, np=np)

    tddw.change_basis(C[:, : tddw.l])

    dip = np.load(os.path.join("tests", "dat", "tddw_dipole_moment.npy"))
    np.testing.assert_allclose(
        np.abs(dip), np.abs(tddw.dipole_moment), atol=1e-10
    )

    h = np.load(os.path.join("tests", "dat", "tddw_h.npy"))
    np.testing.assert_allclose(h, tddw.h, atol=1e-10)

    u = np.load(os.path.join("tests", "dat", "tddw_u.npy"))
    np.testing.assert_allclose(np.abs(u), np.abs(tddw.u), atol=1e-10)

    spf = np.load(os.path.join("tests", "dat", "tddw_spf.npy"))
    np.testing.assert_allclose(np.abs(spf), np.abs(tddw.spf), atol=1e-10)
