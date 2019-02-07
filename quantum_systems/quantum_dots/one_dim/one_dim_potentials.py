import abc


class OneDimPotential(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, x):
        pass


class HOPotenial(OneDimPotential):
    def __init__(self, mass, omega):
        self.mass = mass
        self.omega = omega

    def __call__(self, x):
        return 0.5 * self.mass * self.omega ** 2 * x ** 2


class DWPotential(HOPotenial):
    def __init__(self, mass, omega, l):
        super().__init__(mass, omega)
        self.l = l

    def __call__(self, x):
        return super().__call__(x) + 0.5 * self.mass * self.omega ** 2 * (
            0.25 * self.l ** 2 - self.l * abs(x)
        )
