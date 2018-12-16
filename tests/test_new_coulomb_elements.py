from quantum_systems.quantum_dots.two_dim.two_dim_interface import (
    get_coulomb_element,
    get_indices_nm,
)
from quantum_systems.quantum_dots.two_dim.coulomb_elements import new_coulomb_ho


def test_comparison():
    l = 90
    for i in range(l):
        n_i, m_i = get_indices_nm(i)
        for j in range(l):
            n_j, m_j = get_indices_nm(j)
            for k in range(l):
                n_k, m_k = get_indices_nm(k)
                for l in range(l):
                    n_l, m_l = get_indices_nm(l)

                    args = [n_i, m_i, n_j, m_j, n_k, m_k, n_l, m_l]

                    assert (
                        abs(get_coulomb_element(*args) - new_coulomb_ho(*args))
                        < 1e-8
                    )
