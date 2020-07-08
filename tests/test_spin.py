import pytest
import numpy as np

from quantum_systems import (
    BasisSet,
    RandomBasisSet,
    GeneralOrbitalSystem,
    SpatialOrbitalSystem,
)


def test_add_spin_spf():
    spf = (np.arange(15) + 1).reshape(3, 5).T

    n = 3
    n_a = 2
    n_b = n - n_a

    l = 2 * spf.shape[0]
    assert l == 10

    m_a = l // 2 - n_a
    assert m_a == 3

    m_b = l // 2 - n_b
    assert m_b == 4

    new_spf = BasisSet.add_spin_spf(spf, np)

    # Occupied spin-up
    np.testing.assert_allclose(spf[0], new_spf[0])

    np.testing.assert_allclose(spf[1], new_spf[2])

    # Occupied spin-down
    np.testing.assert_allclose(spf[0], new_spf[1])

    # Virtual spin-up
    np.testing.assert_allclose(spf[2], new_spf[4])

    np.testing.assert_allclose(spf[3], new_spf[6])

    np.testing.assert_allclose(spf[4], new_spf[8])

    # Virtual spin-down
    np.testing.assert_allclose(spf[1], new_spf[3])

    np.testing.assert_allclose(spf[2], new_spf[5])

    np.testing.assert_allclose(spf[3], new_spf[7])

    np.testing.assert_allclose(spf[4], new_spf[9])


def test_gos_spin_matrices():
    n = 4
    l = 10
    dim = 2

    spas = SpatialOrbitalSystem(n, RandomBasisSet(l, dim))
    gos = spas.construct_general_orbital_system()

    for spin, sigma in zip(
        [gos.spin_x, gos.spin_y, gos.spin_z],
        [
            gos._basis_set.sigma_x,
            gos._basis_set.sigma_y,
            gos._basis_set.sigma_z,
        ],
    ):
        spin_2 = np.zeros((gos.l, gos.l), dtype=np.complex128)

        for i in range(gos.l):
            a = i % 2
            g = i // 2

            for j in range(gos.l):
                b = j % 2
                d = j // 2

                spin_2[i, j] = 0.5 * spas.s[g, d] * sigma[a, b]

        np.testing.assert_allclose(spin, spin_2)


def test_spin_up_down():
    n = 4
    l = 10
    dim = 2

    spas = SpatialOrbitalSystem(n, RandomBasisSet(l, dim))
    gos = spas.construct_general_orbital_system()

    a = gos._basis_set.a
    b = gos._basis_set.b

    sigma_up = 0.5 * (gos._basis_set.sigma_x + 1j * gos._basis_set.sigma_y)
    sigma_down = 0.5 * (gos._basis_set.sigma_x - 1j * gos._basis_set.sigma_y)

    np.testing.assert_allclose(a, sigma_up @ b)
    np.testing.assert_allclose(np.zeros_like(a), sigma_up @ a)
    np.testing.assert_allclose(np.zeros_like(b), sigma_down @ b)
    np.testing.assert_allclose(b, sigma_down @ a)
    np.testing.assert_allclose(a, sigma_up @ sigma_down @ a)
    np.testing.assert_allclose(b, sigma_down @ sigma_up @ b)
