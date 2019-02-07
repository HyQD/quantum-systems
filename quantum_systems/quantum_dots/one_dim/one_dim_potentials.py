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
