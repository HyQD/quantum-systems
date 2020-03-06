def add_spin_two_body(_u, np):
    u = np.kron(_u, np.eye(2))
    return u

def anti_symmetrize_u(_u):
    return _u - _u.transpose(0, 1)

def transform_two_body_elements(u, C, np, C_tilde=None):
    if C_tilde is None:
        C_tilde = C.conj().T

    _u = np.einsum(
        "bs,ar,qb,pa,ab->pqrs", C, C, C_tilde, C_tilde, u, optimize=True
    )

    return _u
