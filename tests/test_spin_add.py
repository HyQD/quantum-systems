import pytest
import numpy as np

from quantum_systems import BasisSet


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
