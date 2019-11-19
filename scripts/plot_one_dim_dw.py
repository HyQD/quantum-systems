import numpy as np
import matplotlib.pyplot as plt

from quantum_systems import ODQD


n = 2
l = 10


odqd = ODQD(n, l, 4, 201)
odqd.setup_system(potential=ODQD.DWPotentialSmooth(a=4.5))

plt.plot(odqd.grid, odqd.potential(odqd.grid))

for i in range(l // 2):
    plt.plot(odqd.grid, odqd.eigen_energies[i] + np.abs(odqd.spf[i * 2]) ** 2)

plt.show()
