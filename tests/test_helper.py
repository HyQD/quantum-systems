import numpy as np
from quantum_systems.system_helper import (
    add_spin_one_body,
    add_spin_two_body,
    spin_delta,
    anti_symmetrize_u,
    transform_one_body_elements,
    transform_two_body_elements,
)


def test_spin_delta():
    n = 100

    for p in range(n):
        for q in range(n):
            assert spin_delta(p, q) == ((p % 2) == (q % 2))


def test_transform_one_body_elements():
    l = 10
    h = np.random.random((l, l)) + 1j * np.random.random((l, l))
    c = np.random.random((l, l)) + 1j * np.random.random((l, l))

    h_transformed = np.einsum("ip, jq, ij", c.conj(), c, h, optimize=True)

    np.testing.assert_allclose(
        h_transformed, transform_one_body_elements(h, c, np=np), atol=1e-10
    )

    _h = np.dot(h, c)
    _h = np.dot(c.conj().T, _h)

    np.testing.assert_allclose(_h, transform_one_body_elements(h, c, np))
    np.testing.assert_allclose(
        _h, transform_one_body_elements(h, c, np, c_tilde=c.conj().T)
    )


def test_transform_two_body_elements():
    l = 10
    u = np.random.random((l, l, l, l)) + 1j * np.random.random((l, l, l, l))
    c = np.random.random((l, l)) + 1j * np.random.random((l, l))

    u_transformed = np.einsum(
        "ls, kr, jq, ip, ijkl -> pqrs",
        c,
        c,
        c.conj(),
        c.conj(),
        u,
        optimize=True,
    )

    np.testing.assert_allclose(
        u_transformed, transform_two_body_elements(u, c, np=np), atol=1e-10
    )

    _u = np.dot(u, c)
    _u = np.tensordot(_u, c, axes=(2, 0)).transpose(0, 1, 3, 2)
    _u = np.tensordot(_u, c.conj(), axes=(1, 0)).transpose(0, 3, 1, 2)
    _u = np.tensordot(c.conj().T, _u, axes=(1, 0))

    np.testing.assert_allclose(_u, transform_two_body_elements(u, c, np))
    np.testing.assert_allclose(
        _u, transform_two_body_elements(u, c, np, c_tilde=c.conj().T)
    )


def test_add_spin_one_body():
    l_half = 10
    h = np.random.random((l_half, l_half))
    l = l_half * 2
    h_spin = np.zeros((l, l))

    for p in range(l):
        for q in range(l):
            h_spin[p, q] = spin_delta(p, q) * h[p // 2, q // 2]

    np.testing.assert_allclose(h_spin, add_spin_one_body(h, np=np), atol=1e-10)


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

    np.testing.assert_allclose(u_spin, add_spin_two_body(u, np=np), atol=1e-10)


def test_anti_symmetrize_u():
    l_half = 10
    u = np.random.random((l_half, l_half, l_half, l_half))
    # Make u symmetric
    u = u + u.transpose(1, 0, 3, 2)
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
        u_spin, anti_symmetrize_u(add_spin_two_body(u, np=np)), atol=1e-10
    )


def test_anti_symmetric_properties():
    l_half = 10
    u = np.random.random((l_half, l_half, l_half, l_half))
    # Make u symmetric
    u = u + u.transpose(1, 0, 3, 2)
    u = anti_symmetrize_u(add_spin_two_body(u, np=np))

    np.testing.assert_allclose(u, -u.transpose(0, 1, 3, 2), atol=1e-10)
    np.testing.assert_allclose(u, -u.transpose(1, 0, 2, 3), atol=1e-10)
    np.testing.assert_allclose(u, u.transpose(1, 0, 3, 2), atol=1e-10)
