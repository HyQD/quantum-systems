import numba


@numba.njit(cache=True)
def delta(p, q):
    return p == q


@numba.njit(cache=True)
def spin_delta(p, q):
    return ((p & 0x1) ^ (q & 0x1)) ^ 0x1


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
