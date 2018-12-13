import numpy as np
import numba


def add_spin_h(h):
    return np.kron(h, np.eye(2))


def add_spin_u(_u):
    u = _u.transpose(1, 3, 0, 2)
    u = np.kron(u, np.eye(2))
    u = u.transpose(2, 3, 0, 1)
    u = np.kron(u, np.eye(2))
    u = u.transpose(0, 2, 1, 3)

    return u


def antisymmetrize_u(_u):
    return _u - _u.transpose(0, 1, 3, 2)
