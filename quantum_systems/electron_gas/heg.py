import numpy as np
import numba
import pandas as pd

from quantum_systems import QuantumSystem
from quantum_systems.system_helper import add_spin_one_body


def _shell_level(max_n, num_dimensions):
    assert num_dimensions > 0

    table = []

    for n_x in range(-max_n, max_n + 1):
        if num_dimensions == 1:
            table.append([n_x])
            continue

        lower = _shell_level(max_n, num_dimensions - 1)
        table.extend(list(map(lambda x: [n_x, *x], lower)))

    return table


def _generate_shells(num_shells, num_dimensions):
    max_n = int(np.sqrt(num_shells)) + 1
    table = pd.DataFrame(_shell_level(max_n, num_dimensions))
    table.columns = ["n_%d" % i for i in range(1, num_dimensions + 1)]
    n_columns = table.columns
    table["shell"] = sum([table[col] ** 2 for col in n_columns])
    table = table.sort_values(by=["shell", *n_columns])
    table = table[table.shell <= num_shells]
    table = table.reset_index(drop=True)

    return table


def _eigen_energy(level, length, mass):
    return level * (2 * np.pi) ** 2 / (2 * mass * length ** 2)


@numba.njit(cache=True)
def _delta_momentum_conservation(n_p, n_q, n_r, n_s):
    return np.all(n_p + n_q == n_r + n_s)


@numba.njit(cache=True)
def _delta_momentum(n_p, n_q):
    return np.all(n_p == n_q)


@numba.njit(cache=True)
def spin_delta(p, q):
    return ((p & 0x1) ^ (q & 0x1)) ^ 0x1


@numba.njit(cache=True)
def _construct_coulomb_elements(l, n, length):
    u = np.zeros((l, l, l, l), dtype=np.complex128)

    for p in range(l):
        _p = p // 2
        for q in range(p, l):
            _q = q // 2
            for r in range(l):
                _r = r // 2
                for s in range(r, l):
                    _s = s // 2
                    if not _delta_momentum_conservation(
                        n[_p], n[_q], n[_r], n[_s]
                    ):
                        continue

                    val = 0
                    if (
                        spin_delta(p, r)
                        and spin_delta(q, s)
                        and not _delta_momentum(n[_p], n[_r])
                    ):
                        val += (length / (2 * np.pi)) ** 2 / np.sum(
                            (n[_r] - n[_p]) ** 2
                        )

                    if (
                        spin_delta(p, s)
                        and spin_delta(q, r)
                        and not _delta_momentum(n[_p], n[_s])
                    ):
                        val -= (length / (2 * np.pi)) ** 2 / np.sum(
                            (n[_s] - n[_p]) ** 2
                        )

                    val *= 4 * np.pi / length ** 3

                    u[p, q, r, s] = val
                    u[p, q, s, r] = -val
                    u[q, p, r, s] = -val
                    u[q, p, s, r] = val

    return u


class HomogeneousElectronGas(QuantumSystem):
    def __init__(
        self,
        n,
        num_shells,
        length=1,
        num_grid_points=0,
        mass=1,
        num_dimensions=3,
    ):
        assert num_shells >= 0

        self.num_dimensions = num_dimensions
        self.num_shells = num_shells
        self.table = _generate_shells(self.num_shells, self.num_dimensions)
        self.n_columns = list(
            filter(lambda x: x.startswith("n_"), self.table.columns)
        )
        l = 2 * len(self.table)

        super().__init__(n, l)

        self.length = length
        self.mass = mass

    def setup_system(self):
        __h = np.zeros((self.l // 2, self.l // 2), dtype=np.complex128)
        for p in range(self.l // 2):
            __h[p, p] = _eigen_energy(
                self.table[self.table.index == p].shell, self.length, self.mass
            )

        self._h = add_spin_one_body(__h)
        self._u = _construct_coulomb_elements(
            self.l,
            self.table[self.n_columns].values.astype(np.int),
            self.length,
        )

        self._f = self.construct_fock_matrix(self._h, self._u)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cmap_args = dict(vmin=-1.0, vmax=1.0, cmap="seismic")

    heg = HomogeneousElectronGas(
        2, num_shells=0, length=1.0, num_grid_points=1001, num_dimensions=3
    )
    print(heg.table)
    print(heg.table[heg.n_columns])
    print(heg.table[heg.n_columns].values)
    print(heg.table[heg.n_columns].values.dtype)
    print(heg.l)
    heg.setup_system()

    plt.imshow(heg.u.real.reshape(heg.l ** 2, heg.l ** 2), **cmap_args)
    plt.show()
