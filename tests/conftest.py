import os
import pytest
import numpy as np
import numba

l = 72
filename_one_body = os.path.join(
    "tests", "dat", "two_dim_quantum_dots_one_body_elements.dat"
)
filename_two_body = os.path.join(
    "tests", "dat", "two_dim_quantum_dots_coulomb_elements.dat"
)


@numba.njit(cache=True)
def spin_delta(p, q):
    return ((p & 0x1) ^ (q & 0x1)) ^ 0x1


@numba.njit(cache=True)
def antisymmetrize_orbital_integrals(oi):
    l = len(oi) * 2

    u = np.zeros((l, l, l, l), dtype=np.complex128)

    for p in range(l):
        _p = p // 2
        for q in range(l):
            _q = q // 2
            for r in range(l):
                _r = r // 2
                for s in range(l):
                    _s = s // 2

                    u_pqrs = (
                        spin_delta(p, r) * spin_delta(q, s) * oi[_p, _q, _r, _s]
                    )

                    u_pqsr = (
                        spin_delta(p, s) * spin_delta(q, r) * oi[_p, _q, _s, _r]
                    )

                    u[p, q, r, s] = u_pqrs - u_pqsr

    return u


def get_file_orbital_integrals(l, filename):
    orbital_integrals = np.zeros((l, l, l, l), dtype=np.complex128)

    with open(filename, "r") as f:
        for line in f:
            line = line.split()

            if not line:
                continue

            p, q, r, s, val = line

            if any(filter(lambda x: int(x) >= l, (p, q, r, s))):
                continue

            orbital_integrals[int(p), int(q), int(r), int(s)] = float(val)

    return orbital_integrals


_orbital_integrals = get_file_orbital_integrals(l // 2, filename_two_body)
_u = antisymmetrize_orbital_integrals(_orbital_integrals)


def get_file_one_body_elements(l, filename):
    hi = np.zeros((l, l), dtype=np.complex128)

    with open(filename, "r") as f:
        f.readline()
        for line in f:
            line = line.split()

            if not line:
                continue

            p, val = line

            if int(p) >= l:
                continue

            hi[int(p), int(p)] = float(val)

    return hi


@numba.njit(cache=True)
def get_antisymmetrized_one_body_elements(hi):
    l = len(hi) * 2

    h = np.zeros((l, l), dtype=np.complex128)
    for p in range(l):
        _p = p // 2
        for q in range(l):
            _q = q // 2

            h[p, q] = spin_delta(p, q) * hi[_p, _q]

    return h


_hi = get_file_one_body_elements(l // 2, filename_one_body)
_h = get_antisymmetrized_one_body_elements(_hi)


def read_index_map():
    path = os.path.join("tests", "dat", "index_map.dat")
    index_map = []

    with open(path, "r") as f:
        f.readline()
        for line in f:
            p, n, m = map(lambda x: int(x), line.split())
            index_map.append((n, m))

            assert p == len(index_map) - 1

    return index_map


_index_map = read_index_map()


@pytest.fixture
def index_map():
    return _index_map


@pytest.fixture
def hi():
    return _hi


@pytest.fixture
def h():
    return _h


@pytest.fixture
def orbital_integrals():
    return _orbital_integrals


@pytest.fixture
def u():
    return _u


@pytest.fixture
def spf_2dho():
    n = 2
    l = 30
    radius = 4
    num_grid_points = 101

    spf = []

    for p in range(l // 2):
        filename = os.path.join("tests", "dat", f"2d-ho-qd-spf-p={p}.dat")
        spf.append(np.loadtxt(filename).view(complex))

    return n, l, radius, num_grid_points, np.asarray(spf)
