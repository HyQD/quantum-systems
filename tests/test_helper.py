import numpy as np
from quantum_systems import BasisSet
from quantum_systems.system_helper import spin_delta


def test_spin_delta():
    n = 100

    for p in range(n):
        for q in range(n):
            assert spin_delta(p, q) == ((p % 2) == (q % 2))


def test_transform_one_body_elements():
    l = 10
    h = np.random.random((l, l)) + 1j * np.random.random((l, l))
    C = np.random.random((l, l)) + 1j * np.random.random((l, l))

    h_transformed = np.einsum("ip, jq, ij", C.conj(), C, h, optimize=True)

    np.testing.assert_allclose(
        h_transformed,
        BasisSet.transform_one_body_elements(h, C, np=np),
        atol=1e-10,
    )

    _h = np.dot(h, C)
    _h = np.dot(C.conj().T, _h)

    np.testing.assert_allclose(
        _h, BasisSet.transform_one_body_elements(h, C, np)
    )
    np.testing.assert_allclose(
        _h, BasisSet.transform_one_body_elements(h, C, np, C_tilde=C.conj().T)
    )


def test_transform_two_body_elements():
    l = 10
    u = np.random.random((l, l, l, l)) + 1j * np.random.random((l, l, l, l))
    C = np.random.random((l, l)) + 1j * np.random.random((l, l))

    u_transformed = np.einsum(
        "ls, kr, jq, ip, ijkl -> pqrs",
        C,
        C,
        C.conj(),
        C.conj(),
        u,
        optimize=True,
    )

    np.testing.assert_allclose(
        u_transformed,
        BasisSet.transform_two_body_elements(u, C, np=np),
        atol=1e-10,
    )

    _u = np.dot(u, C)
    _u = np.tensordot(_u, C, axes=(2, 0)).transpose(0, 1, 3, 2)
    _u = np.tensordot(_u, C.conj(), axes=(1, 0)).transpose(0, 3, 1, 2)
    _u = np.tensordot(C.conj().T, _u, axes=(1, 0))

    np.testing.assert_allclose(
        _u, BasisSet.transform_two_body_elements(u, C, np)
    )
    np.testing.assert_allclose(
        _u, BasisSet.transform_two_body_elements(u, C, np, C_tilde=C.conj().T)
    )


def test_add_spin_one_body():
    l_half = 10
    h = np.random.random((l_half, l_half))
    l = l_half * 2
    h_spin = np.zeros((l, l))

    for p in range(l):
        for q in range(l):
            h_spin[p, q] = spin_delta(p, q) * h[p // 2, q // 2]

    np.testing.assert_allclose(
        h_spin, BasisSet.add_spin_one_body(h, np=np), atol=1e-10
    )


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

    np.testing.assert_allclose(
        u_spin, BasisSet.add_spin_two_body(u, np=np), atol=1e-10
    )


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
        u_spin,
        BasisSet.anti_symmetrize_u(BasisSet.add_spin_two_body(u, np=np)),
        atol=1e-10,
    )


def test_anti_symmetric_properties():
    l_half = 10
    u = np.random.random((l_half, l_half, l_half, l_half))
    # Make u symmetric
    u = u + u.transpose(1, 0, 3, 2)
    u = BasisSet.anti_symmetrize_u(BasisSet.add_spin_two_body(u, np=np))

    np.testing.assert_allclose(u, -u.transpose(0, 1, 3, 2), atol=1e-10)
    np.testing.assert_allclose(u, -u.transpose(1, 0, 2, 3), atol=1e-10)
    np.testing.assert_allclose(u, u.transpose(1, 0, 3, 2), atol=1e-10)
