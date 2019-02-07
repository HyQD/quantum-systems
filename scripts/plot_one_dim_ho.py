import numpy as np
import matplotlib.pyplot as plt

from quantum_systems import OneDimensionalHarmonicOscillator


n = 2
l = 6
grid_length = 10
num_grid_points = 1001

odho = OneDimensionalHarmonicOscillator(n, l, grid_length, num_grid_points)
odho.setup_system()

plt.plot(odho.grid, odho.spf[0].real ** 2)
plt.plot(odho.grid, odho.spf[1].real ** 2)
plt.plot(odho.grid, odho.spf[2].real ** 2)
plt.show()

plt.plot(odho.grid, odho.potential(odho.grid))
plt.show()
