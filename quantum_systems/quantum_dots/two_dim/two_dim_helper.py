import numpy as np
import numba
import math

from quantum_systems.quantum_dots.two_dim.coulomb_elements import new_coulomb_ho


@numba.njit(cache=True, nogil=True)
def new_get_index_p(n, m):
    num_shells = 2 * n + abs(m) + 1

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


@numba.njit(cache=True, nogil=True)
def new_get_indices_nm(p):
    n, m = 0, 0
    previous_shell = 0
    current_shell = 1
    shell_counter = 1

    while current_shell <= p:
        shell_counter += 1
        previous_shell = current_shell
        current_shell = previous_shell + shell_counter

    middle = (current_shell - previous_shell) / 2 + previous_shell

    if (current_shell - previous_shell) & 0x1 == 1 and abs(
        p - math.floor(middle)
    ) < 1e-8:
        n = shell_counter // 2
        m = 0

        return n, m

    if p < middle:
        n = p - previous_shell
        m = -((shell_counter - 1) - 2 * n)

    else:
        n = (current_shell - 1) - p
        m = (shell_counter - 1) - 2 * n

    return n, m


@numba.njit(cache=True, nogil=True)
def new_get_shell_energy(n, m):
    return 2 * n + abs(m) + 1


@numba.njit(cache=True, nogil=True)
def new_get_one_body_elements(num_orbitals, dtype=np.float64):
    h = np.zeros((num_orbitals, num_orbitals), dtype=dtype)

    for p in range(num_orbitals):
        n, m = new_get_indices_nm(p)
        h[p, p] = new_get_shell_energy(n, m)

    return h


@numba.njit(fastmath=True, nogil=True, parallel=True)
def new_get_coulomb_elements(num_orbitals, dtype=np.float64):

    shape = (num_orbitals, num_orbitals, num_orbitals, num_orbitals)
    u = np.zeros(shape, dtype=dtype)

    for p in numba.prange(num_orbitals):
        n_p, m_p = new_get_indices_nm(p)
        for q in range(num_orbitals):
            n_q, m_q = new_get_indices_nm(q)
            for r in range(num_orbitals):
                n_r, m_r = new_get_indices_nm(r)
                for s in range(num_orbitals):
                    n_s, m_s = new_get_indices_nm(s)

                    u[p, q, r, s] = new_coulomb_ho(
                        n_p, m_p, n_q, m_q, n_r, m_r, n_s, m_s
                    )

    return u
