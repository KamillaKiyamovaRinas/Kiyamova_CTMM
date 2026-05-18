# distutils: language = c
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

import numpy as np
cimport numpy as np
from cython.parallel import prange
cimport cython
from libc.math cimport sqrt

cdef double G = 6.67430e-11 * 1e-9 * (2.592e6 ** 2)

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_accelerations_openmp(double[:, :] positions, double[:] masses):
    # Используем typed memoryview вместо numpy array для производительности
    
    cdef int N = positions.shape[0]
    cdef double[:, :] acc = np.zeros((N, 2), dtype=np.float64)
    
    cdef int i, j
    cdef double dx, dy, dist, inv_dist3
    
    for i in prange(N, nogil=True):
        for j in range(N):
            if i == j:
                continue
            
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            
            dist = sqrt(dx*dx + dy*dy)
            if dist == 0:
                continue
            
            inv_dist3 = 1.0 / (dist * dist * dist)
            
            acc[i, 0] += G * masses[j] * dx * inv_dist3
            acc[i, 1] += G * masses[j] * dy * inv_dist3
    
    return np.asarray(acc)