import pytest
import numpy as np

from quantum_systems import ODQD


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
        odho.spf[odho.o_up], odho.spf[odho.o_down],
    )

    np.testing.assert_allclose(
        odho.spf[odho.v_up], odho.spf[odho.v_down],
    )

    np.testing.assert_allclose(
        odho.h[odho.o_up, odho.o_up], odho.h[odho.o_down, odho.o_down]
    )

    np.testing.assert_allclose(
        odho.h[odho.v_up, odho.v_up], odho.h[odho.v_down, odho.v_down]
    )

    np.testing.assert_allclose(
        np.zeros((n // 2, n // 2)), odho.h[odho.o_down, odho.o_up]
    )

    np.testing.assert_allclose(
        np.zeros((n // 2, n // 2)), odho.h[odho.o_up, odho.o_down]
    )

    np.testing.assert_allclose(
        odho.u[odho.o_up, odho.o_up, odho.o_up, odho.o_up],
        odho.u[odho.o_down, odho.o_down, odho.o_down, odho.o_down],
    )

    np.testing.assert_allclose(
        odho.u[odho.v_up, odho.v_up, odho.v_up, odho.v_up],
        odho.u[odho.v_down, odho.v_down, odho.v_down, odho.v_down],
    )

    o_res = slice(0, odho_res.n // 2)
    v_res = slice(odho_res.n // 2, odho_res.l // 2)

    np.testing.assert_allclose(
        odho_res.spf[o_res], odho.spf[odho.o_up],
    )

    np.testing.assert_allclose(
        odho_res.spf[v_res], odho.spf[odho.v_up],
    )

    np.testing.assert_allclose(
        odho_res.h[o_res, o_res], odho.h[odho.o_up, odho.o_up]
    )

    np.testing.assert_allclose(
        odho_res.h[o_res, o_res], odho.h[odho.o_down, odho.o_down]
    )

    np.testing.assert_allclose(
        odho_res.h[v_res, v_res], odho.h[odho.v_up, odho.v_up]
    )

    np.testing.assert_allclose(
        odho_res.h[v_res, v_res], odho.h[odho.v_down, odho.v_down]
    )


def test_open_odqd():
    n = 5
    n_up = 3
    n_down = n - n_up
    l = 20

    odho_res = ODQD(n, l, 11, 201, n_up=n_up)
    odho_res.setup_system(add_spin=False, anti_symmetrize=False)

    odho = ODQD(n, l, 11, 201, n_up=n_up)
    odho.setup_system()

    np.testing.assert_allclose(
        odho_res.spf[:n_up], odho.spf[odho.o_up],
    )

    np.testing.assert_allclose(
        odho_res.spf[: n - n_up], odho.spf[odho.o_down],
    )

    np.testing.assert_allclose(
        odho_res.h[:n_up, :n_up], odho.h[odho.o_up, odho.o_up]
    )

    np.testing.assert_allclose(
        odho_res.h[: n - n_up, : n - n_up], odho.h[odho.o_down, odho.o_down]
    )

    np.testing.assert_allclose(
        np.zeros((n_up, n_down)), odho.h[odho.o_up, odho.o_down]
    )

    np.testing.assert_allclose(
        np.zeros((n_down, n_up)), odho.h[odho.o_down, odho.o_up]
    )
