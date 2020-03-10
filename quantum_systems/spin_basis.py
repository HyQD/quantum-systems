class SpinBasis:
    r"""

    >>> import numpy as np
    >>> sb = SpinBasis(np)
    >>> print(sb.a)
    [1.+0.j 0.+0.j]
    >>> print(sb.b)
    [0.+0.j 1.+0.j]
    >>> sq_2 = 1 / np.sqrt(2)
    >>> sb = SpinBasis(np, a=[sq_2, sq_2], b=[sq_2, -sq_2])
    >>> np.allclose(sb.I_2, sb._I_2_mat)
    True
    """

    DIRECTIONS = [0, 1, 2, 3]

    def __init__(self, np, a=None, b=None):
        self.np = np

        if a is None:
            a = [1, 0]

        self.a = self.np.array(a).astype(self.np.complex128)

        if b is None:
            b = [0, 1]

        self.b = self.np.array(b).astype(self.np.complex128)

        # Check that spin basis elements are orthonormal
        assert abs(self.np.dot(self.a.conj().T, self.a) - 1) < 1e-12
        assert abs(self.np.dot(self.b.conj().T, self.b) - 1) < 1e-12
        assert abs(self.np.dot(self.a.conj().T, self.b)) < 1e-12

        self._setup_pauli_matrices()
        self._setup_spin_matrix_elements()

        self.MATRICES = {
            0: self.I_2,
            1: self.sigma_x,
            2: self.sigma_y,
            3: self.sigma_z,
        }

    def _setup_pauli_matrices(self):
        """
        >>> import numpy as np
        >>> sb = SpinBasis(np)
        >>> print(sb._I_2_mat)
        [[1.+0.j 0.+0.j]
         [0.+0.j 1.+0.j]]
        >>> print(sb._sigma_x_mat)
        [[0.+0.j 1.+0.j]
         [1.+0.j 0.+0.j]]
        >>> print(sb._sigma_y_mat)
        [[ 0.+0.j -0.-1.j]
         [ 0.+1.j  0.+0.j]]
        >>> print(sb._sigma_z_mat)
        [[ 1.+0.j  0.+0.j]
         [ 0.+0.j -1.+0.j]]
        """

        self._I_2_mat = self.np.eye(2).astype(self.np.complex128)
        self._sigma_x_mat = self.np.array([[0, 1], [1, 0]]).astype(
            self.np.complex128
        )
        self._sigma_y_mat = self.np.array([[0, -1j], [1j, 0]])
        self._sigma_z_mat = self.np.array([[1, 0], [0, -1]]).astype(
            self.np.complex128
        )

    def _setup_spin_matrix_elements(self):
        """
        >>> sb = SpinBasis(__import__("numpy"))
        >>> print(sb.I_2)
        [[1.+0.j 0.+0.j]
         [0.+0.j 1.+0.j]]
        >>> print(sb.sigma_x)
        [[0.+0.j 1.+0.j]
         [1.+0.j 0.+0.j]]
        >>> print(sb.sigma_y)
        [[0.+0.j 0.-1.j]
         [0.+1.j 0.+0.j]]
        >>> print(sb.sigma_z)
        [[ 1.+0.j  0.+0.j]
         [ 0.+0.j -1.+0.j]]
        """

        self._I_2 = self.np.zeros_like(self._I_2_mat)
        self._sigma_x = self.np.zeros_like(self._sigma_x_mat)
        self._sigma_y = self.np.zeros_like(self._sigma_y_mat)
        self._sigma_z = self.np.zeros_like(self._sigma_z_mat)

        for i, s in enumerate([self.a, self.b]):
            for j, t in enumerate([self.a, self.b]):
                self._I_2[i, j] = self.np.dot(
                    s.conj().T, self.np.dot(self._I_2_mat, t)
                )
                self._sigma_x[i, j] = self.np.dot(
                    s.conj().T, self.np.dot(self._sigma_x_mat, t)
                )
                self._sigma_y[i, j] = self.np.dot(
                    s.conj().T, self.np.dot(self._sigma_y_mat, t)
                )
                self._sigma_z[i, j] = self.np.dot(
                    s.conj().T, self.np.dot(self._sigma_z_mat, t)
                )

    @property
    def I_2(self):
        return self._I_2

    @property
    def sigma_x(self):
        return self._sigma_x

    @property
    def sigma_y(self):
        return self._sigma_y

    @property
    def sigma_z(self):
        return self._sigma_z

    def construct_spas_to_gos_spin_matrix_elements(self, overlap, direction):
        r"""
        Parameters
        ----------
        overlap : np.ndarray
            The spatial orbital overlap matrix elements.
        direction : int
            An integer in ``[0, 1, 2, 3]``, where the number refers to the
            different Pauli matrices (``0`` is the two-by-two identity matrix).

        Returns
        -------
        np.ndarray
            The spin matrix elements :math:`(\sigma_i)^p_q` where :math:`i` is
            the direction index.

        >>> import numpy as np
        >>> sb = SpinBasis(np)
        >>> l = 6
        >>> for direction in sb.DIRECTIONS:
        ...     s = np.arange(l ** 2).reshape(l, l) + 1
        ...     sigma = sb.construct_spas_to_gos_spin_matrix_elements(s, direction)
        ...     sigma_2 = np.zeros((l * 2, l * 2), dtype=np.complex128)
        ...     for i in range(l * 2):
        ...         a = i % 2
        ...         g = i // 2
        ...         for j in range(l * 2):
        ...             b = j % 2
        ...             d = j // 2
        ...             sigma_2[i, j] = s[g, d] * sb.MATRICES[direction][a, b]
        ...     np.allclose(sigma, sigma_2)
        True
        True
        True
        True
        """

        assert direction in self.DIRECTIONS

        sigma = self.MATRICES[direction]

        return self.np.kron(overlap, sigma)
