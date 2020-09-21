import pytest
import numpy as np

from quantum_systems import (
    ODQD,
    BasisSet,
    RandomBasisSet,
    GeneralOrbitalSystem,
    SpatialOrbitalSystem,
    construct_pyscf_system_ao,
    construct_pyscf_system_rhf,
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

    S_up = np.kron(spas.s, sigma_up)
    S_down = np.kron(spas.s, sigma_down)

    np.testing.assert_allclose(S_up, gos.spin_x + 1j * gos.spin_y)
    np.testing.assert_allclose(S_down, gos.spin_x - 1j * gos.spin_y)


def test_overlap_squared():
    n = 4
    l = 10
    dim = 2

    spas = SpatialOrbitalSystem(n, RandomBasisSet(l, dim))

    overlap = spas.s
    overlap_sq = np.einsum("pr, qs -> pqrs", overlap, overlap)

    for p in range(spas.l):
        for q in range(spas.l):
            for r in range(spas.l):
                for s in range(spas.l):
                    np.testing.assert_allclose(
                        overlap_sq[p, q, r, s], overlap[p, r] * overlap[q, s]
                    )


def test_spin_squared():
    n = 2
    l = 2
    dim = 2

    spas = SpatialOrbitalSystem(n, RandomBasisSet(l, dim))
    spas = SpatialOrbitalSystem(
        n, ODQD(l, 8, 1001, potential=ODQD.HOPotential(1))
    )
    gos = spas.construct_general_orbital_system(a=[1, 0], b=[0, 1])

    overlap = spas.s
    overlap_sq = np.einsum("pr, qs -> pqrs", overlap, overlap)

    a = gos._basis_set.a
    b = gos._basis_set.b

    aa = np.kron(a, a)
    ab = np.kron(a, b)
    ba = np.kron(b, a)
    bb = np.kron(b, b)

    triplets = [aa, 1 / np.sqrt(2) * (ab + ba), bb]
    singlet = [1 / np.sqrt(2) * (ab - ba)]

    # S^2 with alpha = [1, 0]^T and beta = [0, 1]^T
    S_sq_spin = np.zeros((4, 4))
    S_sq_spin[0, 0] = 2
    S_sq_spin[3, 3] = 2
    S_sq_spin[1, 1] = 1
    S_sq_spin[2, 2] = 1
    S_sq_spin[1, 2] = 1
    S_sq_spin[2, 1] = 1
    S_sq_spin = S_sq_spin.reshape(2, 2, 2, 2)

    for trip in triplets:
        # Check that the eigenvalue of all triplet states is 2
        np.testing.assert_allclose(trip.T @ S_sq_spin.reshape(4, 4) @ trip, 2)

    # Check that the eigenvalue of the singlet state is 0
    np.testing.assert_allclose(
        singlet[0].T @ S_sq_spin.reshape(4, 4) @ singlet[0], 0
    )

    # S_sq = np.kron(overlap_sq, S_sq_spin)
    # assert S_sq.shape == gos.u.shape

    # np.testing.assert_allclose(S_sq, 0.5 * gos.spin_2_tb)

    # for P in range(gos.l):
    #     p = P // 2
    #     sigma = P % 2

    #     for Q in range(gos.l):
    #         q = Q // 2
    #         tau = Q % 2

    #         for R in range(gos.l):
    #             r = R // 2
    #             gamma = R % 2

    #             for S in range(gos.l):
    #                 s = S // 2
    #                 delta = S % 2

    #                 np.testing.assert_allclose(
    #                     overlap[p, r]
    #                     * overlap[q, s]
    #                     * S_sq_spin[sigma, tau, gamma, delta],
    #                     S_sq[P, Q, R, S],
    #                 )


def test_spin_squared_constructions():
    # TODO: Try to make this test applicable for non-orthonomal basis sets.

    # n = 2
    # l = 10
    # system = GeneralOrbitalSystem(n, RandomBasisSet(l, 3))
    # system = GeneralOrbitalSystem(
    #     n, ODQD(l, 10, 1001, potential=ODQD.HOPotential(1))
    # )

    # system = construct_pyscf_system_ao("he")

    system = construct_pyscf_system_rhf("he", basis="cc-pVTZ")

    spin_dir_tb_orig = []
    spin_dir_tb_pm = []

    spin_p = system.spin_x + 1j * system.spin_y
    spin_m = system.spin_x - 1j * system.spin_y
    spin_z = system.spin_z
    s = system.s

    np.testing.assert_allclose(
        spin_p @ s @ spin_m - spin_m @ s @ spin_p, 2 * spin_z, atol=1e-10
    )

    # S^2 = S_- * S_+ + S_z + S_z^2
    spin_2_mp = spin_m @ spin_p + spin_z + spin_z @ spin_z
    spin_2_tb_mp = (
        0.5 * np.einsum("pr, qs -> pqrs", spin_m, spin_p)
        + 0.5 * np.einsum("pr, qs -> pqrs", spin_p, spin_m)
        + np.einsum("pr, qs -> pqrs", spin_z, spin_z)
    )
    spin_2_tb_mp = system._basis_set.anti_symmetrize_u(spin_2_tb_mp)

    # S^2 = S_+ * S_- - S_z + S_z^2
    spin_2_pm = spin_p @ spin_m - spin_z + spin_z @ spin_z
    spin_2_tb_pm = (
        0.5 * np.einsum("pr, qs -> pqrs", spin_m, spin_p)
        + 0.5 * np.einsum("pr, qs -> pqrs", spin_p, spin_m)
        + np.einsum("pr, qs -> pqrs", spin_z, spin_z)
    )
    spin_2_tb_pm = system._basis_set.anti_symmetrize_u(spin_2_tb_pm)

    np.testing.assert_allclose(spin_2_mp, spin_2_pm, atol=1e-10)
    np.testing.assert_allclose(spin_2_tb_mp, spin_2_tb_pm)

    np.testing.assert_allclose(spin_2_mp, system.spin_2, atol=1e-10)
    np.testing.assert_allclose(spin_2_tb_mp, system.spin_2_tb)

    np.testing.assert_allclose(spin_2_pm, system.spin_2, atol=1e-10)
    np.testing.assert_allclose(spin_2_tb_pm, system.spin_2_tb)
