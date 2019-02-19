import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from quantum_systems import TwoDimensionalHarmonicOscillator


n = 2
l = 30

radius_length = 4
num_grid_points = 101

tdho = TwoDimensionalHarmonicOscillator(n, l, radius_length, num_grid_points)
tdho._setup_spf()


fig = plt.figure(figsize=(16, 12))
fig.suptitle(r"Probability density $\langle \phi_p \vert \phi_p \rangle$")

for p in range(l // 2):
    ax = fig.add_subplot(3, 5, p + 1, polar=True)
    ax.set_title(fr"$p = {p}$", loc="left")
    plt.contourf(tdho.T, tdho.R, np.abs(tdho.spf[p] * tdho.spf[p].conj()))


fig = plt.figure(figsize=(16, 12))
fig.suptitle(r"$\Re(\phi_p(\mathbf{r}))^2$")

for p in range(l // 2):
    ax = fig.add_subplot(3, 5, p + 1, polar=True)
    ax.set_title(fr"$p = {p}$", loc="left")
    plt.contourf(tdho.T, tdho.R, (tdho.spf[p] * tdho.spf[p]).real)


fig = plt.figure(figsize=(16, 12))
fig.suptitle(r"$\Im(\phi_p(\mathbf{r}))$")

for p in range(l // 2):
    ax = fig.add_subplot(3, 5, p + 1, polar=True)
    ax.set_title(fr"$p = {p}$", loc="left")
    plt.contourf(tdho.T, tdho.R, (tdho.spf[p] * tdho.spf[p]).imag)

plt.show()


# X, Y = tdho.R * np.cos(tdho.T), tdho.R * np.sin(tdho.T)
#
# for p in range(l // 2):
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection="3d")
#
#    #ax.plot_surface(X, Y, tdho.spf[p].real, cmap=cm.hot)
#    #ax.plot_surface(X, Y, tdho.spf[p].imag, cmap=cm.cool)
#    ax.plot_surface(X, Y, np.abs(tdho.spf[p] * tdho.spf[p].conj()), cmap=cm.coolwarm)
#
# plt.show()
