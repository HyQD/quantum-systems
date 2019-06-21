import numba


def check_axis_lengths(arr, length):
    return [length == axis for axis in arr.shape]


def change_module(arr, np):
    return np.asarray(arr) if arr is not None else None


@numba.njit(cache=True)
def delta(p, q):
    return p == q


@numba.njit(cache=True)
def spin_delta(p, q):
    return ((p & 0x1) ^ (q & 0x1)) ^ 0x1


def transform_one_body_elements(h, c, np, c_tilde=None):
    if c_tilde is None:
        c_tilde = c.conj().T

    return c_tilde @ h @ c


def transform_two_body_elements(u, c, np, c_tilde=None):
    if c_tilde is None:
        c_tilde = c.conj().T

    # abcd, ds -> abcs
    _u = np.tensordot(u, c, axes=(3, 0))
    # abcs, cr -> absr -> abrs
    _u = np.tensordot(_u, c, axes=(2, 0)).transpose(0, 1, 3, 2)
    # abrs, qb -> arsq -> aqrs
    _u = np.tensordot(_u, c_tilde, axes=(1, 1)).transpose(0, 3, 1, 2)
    # pa, aqrs -> pqrs
    _u = np.tensordot(c_tilde, _u, axes=(1, 0))

    return _u


def add_spin_one_body(h, np):
    return np.kron(h, np.eye(2))


def add_spin_two_body(_u, np):
    u = _u.transpose(1, 3, 0, 2)
    u = np.kron(u, np.eye(2))
    u = u.transpose(2, 3, 0, 1)
    u = np.kron(u, np.eye(2))
    u = u.transpose(0, 2, 1, 3)

    return u


def anti_symmetrize_u(_u):
    return _u - _u.transpose(0, 1, 3, 2)
