import numba


def check_axis_lengths(arr, length):
    return [length == axis for axis in arr.shape]


@numba.njit(cache=True)
def delta(p, q):
    return p == q


@numba.njit(cache=True)
def spin_delta(p, q):
    return ((p & 0x1) ^ (q & 0x1)) ^ 0x1


def add_spin_spf(spf, np):
    new_shape = [spf.shape[0] * 2, *spf.shape[1:]]
    new_spf = np.zeros(tuple(new_shape), dtype=spf.dtype)

    new_spf[::2] = spf
    new_spf[1::2] = spf

    return new_spf


def add_spin_bra_spf(bra_spf, np):
    if bra_spf is None:
        return None

    return add_spin_spf(bra_spf, np)


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


def compute_particle_density(rho_qp, ket_spf, bra_spf, np):
    assert bra_spf.shape == ket_spf.shape
    assert bra_spf.dtype == ket_spf.dtype

    rho = np.zeros(ket_spf.shape[1:], dtype=ket_spf.dtype)

    for p in range(bra_spf.shape[0]):
        phi_tilde_p = bra_spf[p]

        for q in range(ket_spf.shape[0]):
            phi_q = ket_spf[q]
            rho += phi_tilde_p * rho_qp[q, p] * phi_q

    return rho
