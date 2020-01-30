import pytest
import numpy as np

from quantum_systems import ODQD
from quantum_systems.system_helper import (
    add_spin_spf,
    get_spin_block_slices,
    add_spin_one_body,
)


def test_get_spin_block_slices():
    n = 7
    l = 20
    n_a = 3

    ind = np.arange(l)

    o_a, o_b, v_a, v_b = get_spin_block_slices(n, n_a, l)

    assert len(ind[o_a]) == n_a
    assert len(ind[o_b]) == n - n_a

    assert len(ind[v_a]) == l // 2 - n_a
    assert len(ind[v_b]) == l // 2 - (n - n_a)

    np.testing.assert_allclose(
        ind[o_a], np.arange(0, 2 * n_a, 2),
    )

    np.testing.assert_allclose(
        ind[o_b], np.arange(1, 2 * (n - n_a), 2),
    )

    np.testing.assert_allclose(
        ind[v_a], np.arange(2 * n_a, l, 2),
    )

    np.testing.assert_allclose(
        ind[v_b], np.arange(2 * (n - n_a) + 1, l, 2),
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

    new_spf = add_spin_spf(spf, np)

    # Occupied spin-up
    np.testing.assert_allclose(
        spf[0], new_spf[0],
    )

    np.testing.assert_allclose(
        spf[1], new_spf[2],
    )

    # Occupied spin-down
    np.testing.assert_allclose(
        spf[0], new_spf[1],
    )

    # Virtual spin-up
    np.testing.assert_allclose(
        spf[2], new_spf[4],
    )

    np.testing.assert_allclose(
        spf[3], new_spf[6],
    )

    np.testing.assert_allclose(
        spf[4], new_spf[8],
    )

    # Virtual spin-down
    np.testing.assert_allclose(
        spf[1], new_spf[3],
    )

    np.testing.assert_allclose(
        spf[2], new_spf[5],
    )

    np.testing.assert_allclose(
        spf[3], new_spf[7],
    )

    np.testing.assert_allclose(
        spf[4], new_spf[9],
    )


def test_add_spin_one_body():
    n = 7
    l = 20

    for n_a in range(0, n):
        n_b = n - n_a

        o_a, o_b, v_a, v_b = get_spin_block_slices(n, n_a, l)
        h = np.arange((l // 2) ** 2).reshape(l // 2, l // 2) + 1.0
        new_h = add_spin_one_body(h, np)

        np.testing.assert_allclose(np.sum(h) * 2, np.sum(new_h))

        np.testing.assert_allclose(
            new_h[o_a, o_a], h[:n_a, :n_a],
        )

        np.testing.assert_allclose(
            new_h[o_b, o_b], h[:n_b, :n_b],
        )

        np.testing.assert_allclose(
            new_h[v_a, v_a], h[n_a:, n_a:],
        )

        np.testing.assert_allclose(
            new_h[v_b, v_b], h[n_b:, n_b:],
        )

        np.testing.assert_allclose(
            new_h[o_a, o_b], np.zeros(h[:n_a, :n_b].shape),
        )

        np.testing.assert_allclose(
            new_h[o_b, o_a], np.zeros(h[:n_b, :n_a].shape),
        )

        np.testing.assert_allclose(
            new_h[o_a, v_b], np.zeros(h[:n_a, n_b : l // 2].shape),
        )

        np.testing.assert_allclose(
            new_h[v_b, o_a], np.zeros(h[n_b : l // 2, :n_a].shape),
        )

        np.testing.assert_allclose(
            new_h[o_b, v_a], np.zeros(h[:n_b, n_a : l // 2].shape),
        )

        np.testing.assert_allclose(
            new_h[v_a, o_b], np.zeros(h[n_a : l // 2, :n_b].shape),
        )

        np.testing.assert_allclose(
            new_h[v_a, v_b], np.zeros(h[n_a : l // 2, n_b : l // 2].shape),
        )

        np.testing.assert_allclose(
            new_h[v_b, v_a], np.zeros(h[n_b : l // 2, n_a : l // 2].shape),
        )


def test_restricted_odho():
    n = 4
    l = 20

    odho_res = ODQD(n, l, 11, 201)
    odho_res.setup_system(add_spin=False, anti_symmetrize=False)

    odho = ODQD(n, l, 11, 201)
    odho.setup_system()

    assert odho.h.shape == tuple(2 * s for s in odho_res.h.shape)
    assert odho.u.shape == tuple(2 * s for s in odho_res.u.shape)

    np.testing.assert_allclose(
        odho.spf[odho.o_a], odho.spf[odho.o_b],
    )

    np.testing.assert_allclose(
        odho.spf[odho.v_a], odho.spf[odho.v_b],
    )

    np.testing.assert_allclose(
        odho.h[odho.o_a, odho.o_a], odho.h[odho.o_b, odho.o_b]
    )

    np.testing.assert_allclose(
        odho.h[odho.v_a, odho.v_a], odho.h[odho.v_b, odho.v_b]
    )

    np.testing.assert_allclose(
        np.zeros((n // 2, n // 2)), odho.h[odho.o_b, odho.o_a]
    )

    np.testing.assert_allclose(
        np.zeros((n // 2, n // 2)), odho.h[odho.o_a, odho.o_b]
    )

    np.testing.assert_allclose(
        odho.u[odho.o_a, odho.o_a, odho.o_a, odho.o_a],
        odho.u[odho.o_b, odho.o_b, odho.o_b, odho.o_b],
    )

    np.testing.assert_allclose(
        odho.u[odho.v_a, odho.v_a, odho.v_a, odho.v_a],
        odho.u[odho.v_b, odho.v_b, odho.v_b, odho.v_b],
    )

    o_res = slice(0, odho_res.n // 2)
    v_res = slice(odho_res.n // 2, odho_res.l // 2)

    np.testing.assert_allclose(
        odho_res.spf[o_res], odho.spf[odho.o_a],
    )

    np.testing.assert_allclose(
        odho_res.spf[v_res], odho.spf[odho.v_a],
    )

    np.testing.assert_allclose(
        odho_res.h[o_res, o_res], odho.h[odho.o_a, odho.o_a]
    )

    np.testing.assert_allclose(
        odho_res.h[o_res, o_res], odho.h[odho.o_b, odho.o_b]
    )

    np.testing.assert_allclose(
        odho_res.h[v_res, v_res], odho.h[odho.v_a, odho.v_a]
    )

    np.testing.assert_allclose(
        odho_res.h[v_res, v_res], odho.h[odho.v_b, odho.v_b]
    )


def test_open_odqd():
    n = 5
    n_a = 3
    n_b = n - n_a
    l = 20

    odho_res = ODQD(n, l, 11, 201, n_a=n_a)
    odho_res.setup_system(add_spin=False, anti_symmetrize=False)

    odho = ODQD(n, l, 11, 201, n_a=n_a)
    odho.setup_system()

    np.testing.assert_allclose(
        odho_res.spf[:n_a], odho.spf[odho.o_a],
    )

    np.testing.assert_allclose(
        odho_res.spf[: n - n_a], odho.spf[odho.o_b],
    )

    np.testing.assert_allclose(
        odho_res.h[:n_a, :n_a], odho.h[odho.o_a, odho.o_a]
    )

    np.testing.assert_allclose(
        odho_res.h[: n - n_a, : n - n_a], odho.h[odho.o_b, odho.o_b]
    )

    np.testing.assert_allclose(np.zeros((n_a, n_b)), odho.h[odho.o_a, odho.o_b])

    np.testing.assert_allclose(np.zeros((n_b, n_a)), odho.h[odho.o_b, odho.o_a])
