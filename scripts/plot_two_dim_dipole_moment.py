import numpy as np
import matplotlib.pyplot as plt

from quantum_systems import TwoDimensionalHarmonicOscillator

n = 2
l = 30

radius_length = 4
num_grid_points = 101

tdho = TwoDimensionalHarmonicOscillator(n, l, radius_length, num_grid_points)
tdho.construct_dipole_moment()

plt.subplot(2, 2, 1)
plt.imshow(tdho.dipole_moment[0].real)
plt.title(r"$\Re(\langle p | \hat{x} | q \rangle)$")
plt.xlabel(r"$q$")
plt.ylabel(r"$p$")
ax = plt.gca()
ax.set_xticks(np.arange(0, l))
ax.set_yticks(np.arange(0, l))

ax.set_xticklabels(np.arange(0, l))
ax.set_yticklabels(np.arange(0, l))

ax.set_xticks(np.arange(-0.5, l - 0.5), minor=True)
ax.set_yticks(np.arange(-0.5, l - 0.5), minor=True)

ax.grid(which="minor", color="w", linewidth=1)

plt.subplot(2, 2, 2)
plt.imshow(tdho.dipole_moment[0].imag)
plt.title(r"$\Im(\langle p | \hat{x} | q \rangle)$")
plt.xlabel(r"$q$")
plt.ylabel(r"$p$")
ax = plt.gca()
ax.set_xticks(np.arange(0, l))
ax.set_yticks(np.arange(0, l))

ax.set_xticklabels(np.arange(0, l))
ax.set_yticklabels(np.arange(0, l))

ax.set_xticks(np.arange(-0.5, l - 0.5), minor=True)
ax.set_yticks(np.arange(-0.5, l - 0.5), minor=True)

ax.grid(which="minor", color="w", linewidth=1)


plt.subplot(2, 2, 3)
plt.imshow(tdho.dipole_moment[1].real)
plt.title(r"$\Re(\langle p | \hat{y} | q \rangle)$")
plt.xlabel(r"$q$")
plt.ylabel(r"$p$")
ax = plt.gca()
ax.set_xticks(np.arange(0, l))
ax.set_yticks(np.arange(0, l))

ax.set_xticklabels(np.arange(0, l))
ax.set_yticklabels(np.arange(0, l))

ax.set_xticks(np.arange(-0.5, l - 0.5), minor=True)
ax.set_yticks(np.arange(-0.5, l - 0.5), minor=True)

ax.grid(which="minor", color="w", linewidth=1)


plt.subplot(2, 2, 4)
plt.imshow(tdho.dipole_moment[1].imag)
plt.title(r"$\Im(\langle p | \hat{y} | q \rangle)$")
plt.xlabel(r"$q$")
plt.ylabel(r"$p$")
ax = plt.gca()
ax.set_xticks(np.arange(0, l))
ax.set_yticks(np.arange(0, l))

ax.set_xticklabels(np.arange(0, l))
ax.set_yticklabels(np.arange(0, l))

ax.set_xticks(np.arange(-0.5, l - 0.5), minor=True)
ax.set_yticks(np.arange(-0.5, l - 0.5), minor=True)

ax.grid(which="minor", color="w", linewidth=1)


plt.show()
