import os
import sys
import pytest
import numpy as np

from quantum_systems.quantum_dots.two_dim.two_dim_helper import (
    get_index_p,
    get_indices_nm,
    get_one_body_elements,
    get_coulomb_elements,
)

from quantum_systems import (
    GeneralOrbitalSystem,
    TwoDimensionalHarmonicOscillator,
    BasisSet,
)


def test_two_body_symmetry():
    l = 12

    u = get_coulomb_elements(l)

    for p in range(l):
        for q in range(l):
            for r in range(l):
                for s in range(l):
                    assert abs(u[p, q, r, s] - u[q, p, s, r]) < 1e-8


def test_tdho_caching():
    l = 12

    pre_cache = os.listdir()

    os.environ["QS_CACHE_TDHO"] = "1"
    u = get_coulomb_elements(l)

    post_cache = os.listdir()

    assert len(set(post_cache) - set(pre_cache)) == 1

    u_2 = get_coulomb_elements(l)
    np.testing.assert_allclose(u, u_2)

    filename = (set(post_cache) - set(pre_cache)).pop()
    os.remove(filename)

    assert len(set(os.listdir()) - set(pre_cache)) == 0


def test_p_index(index_map):
    for p, (n, m) in enumerate(index_map):
        assert p == get_index_p(n, m)


def test_nm_indices(index_map):
    for p, (n, m) in enumerate(index_map):
        assert get_indices_nm(p) == (n, m)


def test_one_body_elements(hi):
    l = len(hi)
    _hi = get_one_body_elements(l)

    np.testing.assert_allclose(hi, _hi, atol=1e-6, rtol=1e-6)


def test_antisymmetric_one_body_elements(h):
    l = len(h)
    _h = BasisSet.add_spin_one_body(get_one_body_elements(l // 2), np=np)

    np.testing.assert_allclose(h, _h, atol=1e-6, rtol=1e-6)


def test_two_body_elements(orbital_integrals):
    l = len(orbital_integrals)
    oi = get_coulomb_elements(l)

    np.testing.assert_allclose(orbital_integrals, oi, atol=1e-6, rtol=1e-6)


def test_antisymmetric_two_body_elements(u):
    l = len(u)
    _u = BasisSet.anti_symmetrize_u(
        BasisSet.add_spin_two_body(get_coulomb_elements(l // 2), np=np)
    )

    np.testing.assert_allclose(u, _u, atol=1e-6, rtol=1e-6)


def test_spf(spf_2dho):
    n, l, radius, num_grid_points, spf_test = spf_2dho

    tdho = TwoDimensionalHarmonicOscillator(l, radius, num_grid_points)

    for p in range(l // 2):
        np.testing.assert_allclose(spf_test[p], tdho.spf[p])
