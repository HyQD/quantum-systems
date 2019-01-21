import numpy as np
from quantum_systems.system_helper import (
    add_spin_one_body,
    add_spin_two_body,
    spin_delta,
    anti_symmetrize_u,
)


def test_spin_delta():
    n = 100

    for p in range(n):
        for q in range(n):
            assert spin_delta(p, q) == ((p % 2) == (q % 2))


def test_add_spin_one_body():
    l_half = 10
    h = np.random.random((l_half, l_half))
    l = l_half * 2
    h_spin = np.zeros((l, l))

    for p in range(l):
        for q in range(l):
            h_spin[p, q] = spin_delta(p, q) * h[p // 2, q // 2]

    np.testing.assert_allclose(h_spin, add_spin_one_body(h), atol=1e-10)


def test_spin_two_body():
    l_half = 10
    u = np.random.random((l_half, l_half, l_half, l_half))
    l = l_half * 2
    u_spin = np.zeros((l, l, l, l))

    for p in range(l):
        for q in range(l):
            for r in range(l):
                for s in range(l):
                    u_spin[p, q, r, s] = (
                        spin_delta(p, r)
                        * spin_delta(q, s)
                        * u[p // 2, q // 2, r // 2, s // 2]
                    )

    np.testing.assert_allclose(u_spin, add_spin_two_body(u), atol=1e-10)


def test_anti_symmetrize_u():
    l_half = 10
    u = np.random.random((l_half, l_half, l_half, l_half))
    l = l_half * 2
    u_spin = np.zeros((l, l, l, l))

    for p in range(l):
        for q in range(l):
            for r in range(l):
                for s in range(l):
                    u_spin[p, q, r, s] = (
                        spin_delta(p, r)
                        * spin_delta(q, s)
                        * u[p // 2, q // 2, r // 2, s // 2]
                    )
                    u_spin[p, q, r, s] -= (
                        spin_delta(p, s)
                        * spin_delta(q, r)
                        * u[p // 2, q // 2, s // 2, r // 2]
                    )

    np.testing.assert_allclose(
        u_spin, anti_symmetrize_u(add_spin_two_body(u)), atol=1e-10
    )