import os
import numpy as np
import matplotlib.pyplot as plt

from quantum_systems import TwoDimensionalDoubleWell
from quantum_systems.quantum_dots.two_dim.two_dim_helper import get_indices_nm

n = 2
l = 20
radius = 8
num_grid_points = 201
l_ho_factor = 1
barrier_strength = 3
omega = 0.8

tddw = TwoDimensionalDoubleWell(
    n,
    l,
    radius,
    num_grid_points,
    l_ho_factor=l_ho_factor,
    barrier_strength=barrier_strength,
    omega=omega,
)
tddw.setup_system(axis=0)

dip = np.load(os.path.join("tests", "dat", "tddw_dipole_moment.npy"))

for p in range(l):
    for q in range(l):
        if abs(dip[0, p, q] - tddw.dipole_moment[0, p, q]) > 1e-10:
            print(f"m_p = {get_indices_nm(p)[1]}, m_q = {get_indices_nm(q)[1]}")
            print(
                f"dip[0, {p}, {q}] = {dip[0, p, q]} != dip2[0, {p}, {q}] = {tddw.dipole_moment[0, p, q]}"
            )
        if abs(dip[1, p, q] - tddw.dipole_moment[1, p, q]) > 1e-10:
            print(f"m_p = {get_indices_nm(p)[1]}, m_q = {get_indices_nm(q)[1]}")
            print(
                f"dip[1, {p}, {q}] = {dip[1, p, q]} != dip2[1, {p}, {q}] = {tddw.dipole_moment[1, p, q]}"
            )
        # assert abs(dip[1, p, q] - tddw.dipole_moment[1, p, q]) < 1e-10

# print(np.diag(tddw.h))
#
# fig = plt.figure(figsize=(16, 12))
# fig.suptitle(r"Probability density $\langle \phi_p \vert \phi_p \rangle$")
#
# for p in range(l // 2):
#    ax = fig.add_subplot(3, 5, p + 1, polar=True)
#    ax.set_title(fr"$p = {p}$", loc="left")
#    plt.contourf(
#        tddw.T, tddw.R, np.abs(tddw.spf[2 * p] * tddw.spf[2 * p].conj())
#    )
#
# plt.show()
