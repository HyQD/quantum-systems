cdef extern from "coulomb_elements.h":
    cdef double coulomb_ho(
        int ni, int mi,
        int nj, int mj,
        int nk, int mk,
        int nl, int ml
    ) nogil
