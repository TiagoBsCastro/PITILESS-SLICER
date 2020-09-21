cimport cython
cimport numpy as np
import numpy as np

cdef void _wrapPositions (float *array, int size) nogil:

  cdef Py_ssize_t i;

  for i in range(size):
    if array[i]<0:
      array[i] += 1
    elif array[i]>1:
      array[i] -= 1
    else:
      continue

def wrapPositions (float [:,::1] pos):

    _wrapPositions(& pos[0, 0], pos.size)
