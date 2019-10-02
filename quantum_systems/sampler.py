import os
import abc


class SampleCollector(metaclass=abc.ABCMeta):
    """Abstract base class defining common methods for all sampler classes. The
    purpose of a derived class of `SampleCollector` is to define all sampler
    classes and pass them to the constructor of `SampleCollector`.

    Parameters
    ----------
    samplers : iterable
        Container with all sampler classes.
    """

    def __init__(self, samplers):
        if type(samplers) not in [list, tuple, set]:
            samplers = list(samplers)

        self.samplers = samplers
        self.samples = dict()

    @abc.abstractmethod
    def sample(self, step):
        """Function calling all samplers for a given time step.

        Parameters
        ----------
        step : float
            Time step to sample.
        """

        for sampler in self.samplers:
            sampler.sample(step)

    def dump(self, path="dump", filename=None, exist_ok=True):
        """Function storing sampled data. As of now this function dumps the full
        `samples`-dictionary using `np.save`.

        Parameters
        ----------
        path : str
            Directory path to storage location. This path is created if it does
            not exist. Default is `"dump"`.
        filename : str
            Name of samples dictionary output. Default is `None` which results
            in the name `"tmp_{self.__class__.__name__}.samples"`.
        exist_ok : bool
            Whether or not to create directory path if it already exists.
            Default is `True` as the path does not overwrite any existing data.
        """

        if filename is None:
            filename = f"tmp_{self.__class__.__name__}.samples"

        os.makedirs(path, exist_ok=exist_ok)

        for sampler in self.samplers:
            self.samples = sampler.dump(self.samples)

        filename = os.path.join(path, filename)
        np.save(filename, self.samples)

    def add_sample(self, key, sample):
        self.samples[key] = sample

    def get_samples(self):
        return self.samples


class Sampler(metaclass=abc.ABCMeta):
    """Abstract base class defining the skeleton of a sampler class."""

    @abc.abstractmethod
    def sample(self, step):
        pass

    @abc.abstractmethod
    def dump(self, samples):
        pass
