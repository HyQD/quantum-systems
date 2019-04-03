import numpy as np
import matplotlib.pyplot as plt

from quantum_systems import TwoDimensionalDoubleWell

n = 2
l = 30
radius = 10
num_grid_points = 401
l_ho_factor = 1
barrier_strength = 2
omega = 1

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

print(np.diag(tddw.h))

fig = plt.figure(figsize=(16, 12))
fig.suptitle(r"Probability density $\langle \phi_p \vert \phi_p \rangle$")

for p in range(l // 2):
    ax = fig.add_subplot(3, 5, p + 1, polar=True)
    ax.set_title(fr"$p = {p}$", loc="left")
    plt.contourf(
        tddw.T, tddw.R, np.abs(tddw.spf[2 * p] * tddw.spf[2 * p].conj())
    )

plt.show()
