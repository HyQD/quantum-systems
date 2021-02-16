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

    # Note: Is the ordering correct?
    # I think maybe it should be phi_tilde_q rho[q, p] phi_p instead.
    for p in range(bra_spf.shape[0]):
        phi_tilde_p = bra_spf[p]

        for q in range(ket_spf.shape[0]):
            phi_q = ket_spf[q]
            rho += phi_tilde_p * rho_qp[q, p] * phi_q

    return rho


def compute_two_body_particle_density(rho_rspq, ket_spf, bra_spf, np):
    assert bra_spf.shape == ket_spf.shape
    assert bra_spf.dtype == ket_spf.dtype

    rho = np.zeros(
        (*ket_spf.shape[1:], *ket_spf.shape[1:]), dtype=ket_spf.dtype
    )

    # for p in range(ket_spf.shape[0]):
    #     phi_p = ket_spf[p]

    #     for q in range(ket_spf.shape[0]):
    #         phi_q = ket_spf[q]

    #         for r in range(bra_spf.shape[0]):
    #             phi_tilde_r = bra_spf[r]

    #             for s in range(bra_spf.shape[0]):
    #                 phi_tilde_s = bra_spf[s]

    #                 rho += (
    #                     phi_tilde_s
    #                     * phi_tilde_r
    #                     * rho_rspq[r, s, p, q]
    #                     * phi_q
    #                     * phi_p
    #                 )

    # rho_sq = np.zeros(ket_spf.shape[1:], dtype=ket_spf.dtype)

    # for r in range(bra_spf.shape[0]):
    #     phi_tilde_r = bra_spf[r]

    #     for p in range(ket_spf.shape[0]):
    #         phi_p = ket_spf[p]
    #         rho_sq += phi_tilde_r * rho_rspq[r, :, p, :] * phi_p

    # for s in range(bra_spf.shape[0]):
    #     phi_tilde_s = bra_spf[s]

    #     for q in range(ket_spf.shape[0]):
    #         phi_q = ket_spf[q]
    #         rho += phi_tilde_s * rho_sq[s, q] * phi_q

    wat
    return 0.5 * rho
