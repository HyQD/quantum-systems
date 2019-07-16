import abc
import numpy as np


class OneDimPotential(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, x):
        pass


class HOPotential(OneDimPotential):
    def __init__(self, omega):
        self.omega = omega

    def __call__(self, x):
        return 0.5 * self.omega ** 2 * x ** 2


class DWPotential(HOPotential):
    def __init__(self, omega, l):
        super().__init__(omega)
        self.l = l

    def __call__(self, x):
        return super().__call__(x) + 0.5 * self.omega ** 2 * (
            0.25 * self.l ** 2 - self.l * abs(x)
        )


class DWPotentialSmooth(OneDimPotential):
    """
    This is the double-well potential used by J. Kryvi and S. Bøe in their
    thesis work. See Eq. [13.11] in Bøe: https://www.duo.uio.no/handle/10852/37170
    """

    def __init__(self, a=4):
        self.a = a

    def __call__(self, x):
        return (
            (1.0 / (2 * self.a ** 2))
            * (x + 0.5 * self.a) ** 2
            * (x - 0.5 * self.a) ** 2
        )


class GaussianPotential(OneDimPotential):
    def __init__(self, weight, center, deviation, np):
        self.weight = weight
        self.center = center
        self.deviation = deviation
        self.np = np

    def __call__(self, x):
        return -self.weight * self.np.exp(
            -(x - self.center) ** 2 / (2.0 * self.deviation ** 2)
        )


class AtomicPotential(OneDimPotential):
    def __init__(self, Za=2, c=0.54878464):
        self.Za = Za
        self.c = c

    def __call__(self, x):
        return -self.Za / np.sqrt(x ** 2 + self.c)
