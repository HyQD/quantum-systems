import matplotlib.pyplot as plt

from quantum_systems import TwoDimensionalHarmonicOscillator

n = 2
l = 30

radius_length = 4
num_grid_points = 101

tdho = TwoDimensionalHarmonicOscillator(n, l, radius_length, num_grid_points)
tdho.construct_dipole_moment()

plt.imshow(tdho.dipole_moment[0].real)
plt.show()

plt.imshow(tdho.dipole_moment[1].imag)
plt.show()
