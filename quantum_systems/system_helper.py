import numpy as np
import numba


@numba.njit(cache=True)
def delta(p, q):
    return p == q


@numba.njit(cache=True)
def spin_delta(p, q):
    return ((p & 0x1) ^ (q & 0x1)) ^ 0x1


def new_transform_one_body_elements(h, c):
    _h = np.dot(h, c)
    _h = np.dot(c.conj().T, _h)

    return _h


def new_transform_two_body_elements(u, c):
    _u = np.dot(u, c)
    _u = np.tensordot(_u, c, axes=(2, 0)).transpose(0, 1, 3, 2)
    _u = np.tensordot(_u, c.conj(), axes=(1, 0)).transpose(0, 3, 1, 2)
    _u = np.tensordot(c.conj().T, _u, axes=(1, 0))

    return _u


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
