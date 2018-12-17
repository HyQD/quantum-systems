from quantum_systems.quantum_dots.two_dim.two_dim_interface import (
    get_coulomb_element,
    get_indices_nm,
)
from quantum_systems.quantum_dots.two_dim.coulomb_elements import new_coulomb_ho


def test_comparison():
    l = 20
    for p in range(l):
        n_p, m_p = get_indices_nm(p)
        for q in range(l):
            n_q, m_q = get_indices_nm(q)
            for r in range(l):
                n_r, m_r = get_indices_nm(r)
                for s in range(l):
                    n_s, m_s = get_indices_nm(s)

                    args = [n_p, m_p, n_q, m_q, n_r, m_r, n_s, m_s]

                    assert (
                        abs(get_coulomb_element(*args) - new_coulomb_ho(*args))
                        < 1e-8
                    )
