import pytest
import numpy as np

from quantum_systems.quantum_dots.two_dim.two_dim_helper import (
    get_index_p,
    get_indices_nm,
    get_one_body_elements,
    get_coulomb_elements,
)

from quantum_systems.system_helper import (
    add_spin_one_body,
    add_spin_two_body,
    anti_symmetrize_u,
)


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
    _h = add_spin_one_body(get_one_body_elements(l // 2), np=np)

    np.testing.assert_allclose(h, _h, atol=1e-6, rtol=1e-6)


def test_two_body_elements(orbital_integrals):
    l = len(orbital_integrals)
    oi = get_coulomb_elements(l)

    np.testing.assert_allclose(orbital_integrals, oi, atol=1e-6, rtol=1e-6)


def test_antisymmetric_two_body_elements(u):
    l = len(u)
    _u = anti_symmetrize_u(
        add_spin_two_body(get_coulomb_elements(l // 2), np=np)
    )

    np.testing.assert_allclose(u, _u, atol=1e-6, rtol=1e-6)
