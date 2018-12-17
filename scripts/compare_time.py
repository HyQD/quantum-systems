from quantum_systems.quantum_dots.two_dim.two_dim_interface import (
    get_coulomb_element,
    get_indices_nm,
    get_coulomb_elements,
)
from quantum_systems.quantum_dots.two_dim.coulomb_elements import new_coulomb_ho
from quantum_systems.quantum_dots.two_dim.two_dim_helper import (
    new_get_coulomb_elements
)

import time
import numpy as np


def fetch_elements(l, foo):
    t0 = time.time()
    for p in range(l):
        n_p, m_p = get_indices_nm(p)
        for q in range(l):
            n_q, m_q = get_indices_nm(q)
            for r in range(l):
                n_r, m_r = get_indices_nm(r)
                for s in range(l):
                    n_s, m_s = get_indices_nm(s)

                    args = [n_p, m_p, n_q, m_q, n_r, m_r, n_s, m_s]
                    foo(*args)

    t1 = time.time()
    print("Time spent running {0}: {1} sec".format(foo.__name__, t1 - t0))


def fetch_all_elements(l, foo):
    t0 = time.time()
    oi = foo(l)
    t1 = time.time()

    print("Time spent running {0}: {1} sec".format(foo.__name__, t1 - t0))

    return oi


l = 20
for i in range(10):
    fetch_elements(l, get_coulomb_element)
for i in range(10):
    fetch_elements(l, new_coulomb_ho)


oi_c = fetch_all_elements(l // 2, get_coulomb_elements)
oi_n = np.complex128(fetch_all_elements(l // 2, new_get_coulomb_elements))

np.testing.assert_allclose(oi_c, oi_n)
