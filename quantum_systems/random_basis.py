from quantum_systems import BasisSet


class RandomBasisSet(BasisSet):
    """A basis set consisting of random matrix elements satisfying the
    symmetries of the second quantized integral elements. The purpose of this
    system is for unit testing of purely mathematical methods ignoring inherent
    symmetries in specific systems.

    >>> l, dim = 20, 3
    >>> rbs = RandomBasisSet(l, dim)
    >>> rbs.dipole_moment.shape
    (3, 20, 20)

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_basis()

    def setup_basis(self):
        self.h = self.make_hermitian(
            self.get_random_elements((self.l, self.l), self.np)
        )
        self.s = self.make_hermitian(
            self.get_random_elements((self.l, self.l), self.np)
        )
        self.u = self.make_two_body_symmetry(
            self.get_random_elements((self.l, self.l, self.l, self.l), self.np)
        )
        self.position = self.make_position_elements_hermitian(
            self.get_random_elements((self.dim, self.l, self.l), self.np)
        )
        self.nuclear_repulsion_energy = self.np.random.random()
        self.charge = self.np.random.choice([-1, 1])

    @staticmethod
    def make_hermitian(h):
        return 0.5 * (h + h.conj().T)

    @staticmethod
    def make_position_elements_hermitian(position):
        for i in range(len(position)):
            position[i] = RandomBasisSet.make_hermitian(position[i])

        return position

    @staticmethod
    def make_two_body_symmetry(u):
        return 0.5 * (u + u.transpose(1, 0, 3, 2))

    @staticmethod
    def get_random_elements(shape, np):
        """Function creating a complex array representing random matrix
        elements of a given shape.

        Parameters
        ----------
        shape : tuple or int
            The shape of the array.
        np : array module

        Returns
        -------
        out : ndarray
            A complex, random array of given shape.
        """

        return np.random.random(shape) + 1j * np.random.random(shape)
