import cython
from cython.parallel cimport prange, parallel

import numpy as np
cimport numpy as np


def delta(int p, int q):
    return _delta(p, q)


cdef int _delta(int p, int q) nogil:
    return p == q


def transform_two_body_elements(u, c):
    _u = np.einsum("ls, ijkl -> ijks", c, u)
    _u = np.einsum("kr, ijks -> ijrs", c, _u)
    _u = np.einsum("jq, ijrs -> iqrs", c.conj(), _u)
    _u = np.einsum("ip, iqrs -> pqrs", c.conj(), _u)

    return _u


cdef int spin_delta(int p, int q) nogil:
    return ((p & 0x1) ^ (q & 0x1)) ^ 0x1


def get_antisymmetrized_one_body_elements(
        np.ndarray[np.complex128_t, ndim=2] h
):
    cdef int p, q, _p, _q, l
    cdef np.ndarray[np.complex128_t, ndim=2] _h

    l = len(h) * 2
    _h = np.zeros((l, l), dtype=np.complex128)

    for p in range(l):
        _p = p // 2
        for q in range(l):
            _q = q // 2

            _h[p, q] = spin_delta(p, q) * h[_p, _q]

    return _h


def get_antisymmetrized_two_body_elements(
        np.ndarray[np.complex128_t, ndim=4] orbital_integrals
):
    cdef int p, q, r, s, l, _p, _q, _r, _s
    cdef np.ndarray[np.complex128_t, ndim=4] u
    cdef np.complex128_t u_pqrs, u_pqsr

    l = len(orbital_integrals) * 2

    u = np.zeros((l, l, l, l), dtype=np.complex128)

    with nogil, parallel():
        for p in prange(l, schedule="dynamic"):
            _p = p // 2
            for q in range(l):
                _q = q // 2
                for r in range(l):
                    _r = r // 2
                    for s in range(l):
                        _s = s // 2

                        u_pqrs = (
                            spin_delta(p, r) * spin_delta(q, s)
                            * orbital_integrals[_p, _q, _r, _s]
                        )

                        u_pqsr = (
                            spin_delta(p, s) * spin_delta(q, r)
                            * orbital_integrals[_p, _q, _s, _r]
                        )

                        u[p, q, r, s] = u_pqrs - u_pqsr

    return u
