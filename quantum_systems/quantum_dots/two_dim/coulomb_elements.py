import numpy as np
import math
import numba


@numba.njit(cache=True, nogil=True)
def coulomb_ho(n_i, m_i, n_j, m_j, n_k, m_k, n_l, m_l):
    element = 0

    if m_i + m_j != m_k + m_l:
        return 0

    M_i = 0.5 * (abs(m_i) + m_i)
    dm_i = 0.5 * (abs(m_i) - m_i)

    M_j = 0.5 * (abs(m_j) + m_j)
    dm_j = 0.5 * (abs(m_j) - m_j)

    M_k = 0.5 * (abs(m_k) + m_k)
    dm_k = 0.5 * (abs(m_k) - m_k)

    M_l = 0.5 * (abs(m_l) + m_l)
    dm_l = 0.5 * (abs(m_l) - m_l)

    n = np.array([n_i, n_j, n_k, n_l])
    m = np.array([m_i, m_j, m_k, m_l])
    j = np.array([0, 0, 0, 0])
    l = np.array([0, 0, 0, 0])
    g = np.array([0, 0, 0, 0])

    for j_1 in range(n_i):
        j[0] = j_1
        for j_2 in range(n_j):
            j[1] = j_2
            for j_3 in range(n_l):
                j[2] = j_3
                for j_4 in range(n_k):
                    j[3] = j_4

                    g[0] = j_1 + j_4 + M_i + dm_k
                    g[1] = j_2 + j_3 + M_j + dm_l
                    g[2] = j_3 + j_2 + M_l + dm_j
                    g[3] = j_4 + j_1 + M_k + dm_i

                    G = np.sum(g)
                    ratio_1 = log_ratio_1(j)
                    prod_2 = log_product_2(n, m, j)
                    ratio_2 = log_ratio_2(G)

                    temp = 0
                    for l_1 in range(g[0]):
                        l[0] = l_1
                        for l_2 in range(g[1]):
                            l[1] = l_2
                            for l_3 in range(g[2]):
                                l[2] = l_3
                                for l_4 in range(g[3]):
                                    l[3] = l_4

                                    if l_1 + l_2 != l_3 + l_4:
                                        continue

                                    L = np.sum(l)

                                    temp += (
                                        -2 * ((g[1] + g[2] - l[1] - l[2]) & 0x1) + 1
                                    ) * np.exp(
                                        log_product_3(l, g)
                                        + math.lgamma(1.0 + 0.5 * L)
                                        + math.lgamma(0.5 * (G - L + 1.0))
                                    )

                    element += (
                        -2 * (np.sum(j) & 0x1) + 1
                    ) * np.exp(ratio_1 + prod_2 + ratio_2) * temp

    element *= log_product_1(n, m)

    return element


@numba.njit(cache=True, nogil=True)
def log_factorial(n):
    fac = 0

    for a in range(2, n + 1):
        fac += np.log(a)

    return fac


@numba.njit(cache=True, nogil=True)
def log_ratio_1(n_arr):
    ratio = 0

    for i in range(len(n_arr)):
        ratio -= log_factorial(n_arr[i])

    return ratio


@numba.njit(cache=True, nogil=True)
def log_ratio_2(G):
    return -0.5 * (G + 1) * np.log(2)


@numba.njit(cache=True, nogil=True)
def log_product_1(n, m):
    prod = 0

    for i in range(len(n)):
        prod += log_factorial(n[i])
        prod -= log_factorial(n[i] + abs(m[i]))

    return np.exp(0.5 * prod)


@numba.njit(cache=True, nogil=True)
def log_product_2(n, m, j):
    prod = 0

    for i in range(len(n)):
        prod += log_factorial(n[i] + abs(m[i]))
        prod -= log_factorial(n[i] - j[i])
        prod -= log_factorial(j[i] + abs(m[i]))

    return prod


@numba.njit(cache=True, nogil=True)
def log_product_3(l, g):
    prod = 0

    for i in range(len(l)):
        prod += log_factorial(g[i])
        prod -= log_factorial(l[i])
        prod -= log_factorial(g[i] - l[i])

    return prod
