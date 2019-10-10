import os
import abc


class SampleCollector:
    """Base class defining common methods for all sampler classes. The purpose
    of a derived class of ``SampleCollector`` is to define all sampler classes
    and pass them to the constructor of ``SampleCollector``.

    Parameters
    ----------
    samplers : iterable
        Container with all sampler classes.
    np : linalg module
        For example, ``numpy`` or ``cupy``.
    """

    def __init__(self, samplers, np):
        self.np = np

        if type(samplers) not in [list, tuple, set]:
            samplers = list(samplers)

        self.samplers = samplers
        self.samples = dict()

    def sample(self, step):
        """Function calling all samplers for a given time step.

        Parameters
        ----------
        step : float
            Time step to sample.
        """

        for sampler in self.samplers:
            sampler.sample(step)

    def dump(
        self, path="dump", filename=None, exist_ok=True, save_samples=True
    ):
        """Function storing sampled data. As of now this function dumps the full
        ``samples``-dictionary using ``np.save``.

        Parameters
        ----------
        path : str
            Directory path to storage location. This path is created if it does
            not exist. Default is ``"dump"``.
        filename : str
            Name of samples dictionary output. Default is ``None`` which results
            in the name ``"tmp_{self.__class__.__name__}.samples"``.
        exist_ok : bool
            Whether or not to create directory path if it already exists.
            Default is ``True`` as the path does not overwrite any existing data.
        save_samples : bool
            Whether or not to write the samples to disk. Default is ``True``.

        Returns
        -------
        dict
            Dictionary with all sampled values.
        """

        for sampler in self.samplers:
            self.samples = sampler.dump(self.samples)

        if not save_samples:
            return self.samples

        if filename is None:
            filename = f"tmp_{self.__class__.__name__}.samples"

        os.makedirs(path, exist_ok=exist_ok)

        filename = os.path.join(path, filename)
        self.np.save(filename, self.samples)

        return self.samples

    def add_sample(self, key, sample):
        self.samples[key] = sample

    def get_samples(self):
        return self.samples


class Sampler(metaclass=abc.ABCMeta):
    """Abstract base class defining the skeleton of a sampler class."""

    def __init__(self, solver, num_samples, np):
        self.solver = solver
        self.system = self.solver.system
        self.num_samples = num_samples
        self.np = np

    @abc.abstractmethod
    def sample(self, step):
        pass

    @abc.abstractmethod
    def dump(self, samples):
        pass
