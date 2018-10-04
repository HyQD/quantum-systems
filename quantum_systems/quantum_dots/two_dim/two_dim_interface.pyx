import cython
from cython.parallel cimport prange, parallel

import numpy as np
cimport numpy as np

from libc.math cimport fabs, floor


def get_index_p(int n, int m):
    return _get_index_p(n, m)


def get_indices_nm(int p):
    cdef int n, m

    n, m = 0, 0
    _get_indices_nm(p, &n, &m)

    return (n, m)


cdef int _get_index_p(int n, int m):
    cdef int num_shells, current_shell, previous_shell, i, p

    num_shells = 2 * n + int(fabs(m)) + 1

    previous_shell = 0
    for i in range(1, num_shells):
        previous_shell += i

    current_shell = previous_shell + num_shells

    if m == 0:
        if n == 0:
            return 0

        p = previous_shell + (current_shell - previous_shell) // 2

        return p

    elif m < 0:
        return previous_shell + n

    else:
        return current_shell - (n + 1)


cdef void _get_indices_nm(int p, int *n, int *m) nogil:
    cdef int previous_shell, current_shell, shell_counter
    cdef double middle

    previous_shell = 0
    current_shell = 1
    shell_counter = 1

    while current_shell <= p:
        shell_counter += 1
        previous_shell = current_shell
        current_shell = previous_shell + shell_counter

    middle = (current_shell - previous_shell) / 2 + previous_shell

    if (
        (current_shell - previous_shell) & 0x1 == 1
        and fabs(p - floor(middle)) < 1e-8
    ):
        n[0] = shell_counter // 2
        m[0] = 0

        return

    if p < middle:
        n[0] = p - previous_shell
        m[0] = -((shell_counter - 1) - 2 * n[0])

    else:
        n[0] = (current_shell - 1) - p
        m[0] = (shell_counter - 1) - 2 * n[0]

    return


def get_coulomb_elements(int num_orbitals):
    # NOTE: If we are to look at systems with complex numbers we need to change
    # the dtype of u.
    cdef np.ndarray[np.complex128_t, ndim=4] u
    cdef int p, q, r, s, n_p, m_p, n_q, m_q, n_r, m_r, n_s, m_s

    u = np.zeros(
        (num_orbitals, num_orbitals, num_orbitals, num_orbitals),
        dtype=np.complex128
    )

    with nogil, parallel():
        n_p, m_p = 0, 0
        n_q, m_q = 0, 0
        n_r, m_r = 0, 0
        n_s, m_s = 0, 0

        for p in prange(num_orbitals, schedule="dynamic"):
            _get_indices_nm(p, &n_p, &m_p)
            for q in range(num_orbitals):
                _get_indices_nm(q, &n_q, &m_q)
                for r in range(num_orbitals):
                    _get_indices_nm(r, &n_r, &m_r)
                    for s in range(num_orbitals):
                        _get_indices_nm(s, &n_s, &m_s)

                        u[p, q, r, s] = coulomb_ho(
                            n_p, m_p,
                            n_q, m_q,
                            n_r, m_r,
                            n_s, m_s
                        )

    return u


def get_shell_energy(int n, int m):
    return _get_shell_energy(n, m)


cdef double _get_shell_energy(int n, int m) nogil:
    return 2 * n + fabs(m) + 1


def get_one_body_elements(int num_orbitals):
    cdef np.ndarray[np.complex128_t, ndim=2] h
    cdef int p, n, m

    h = np.zeros((num_orbitals, num_orbitals), dtype=np.complex128)

    for p in range(num_orbitals):
        n, m = get_indices_nm(p)
        h[p, p] = _get_shell_energy(n, m)

    return h
