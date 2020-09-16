cimport cython
cimport numpy as np
from libc.math cimport floor, acos, pi
from libc.stdlib cimport rand, RAND_MAX

# Hardcoding the NFW profile
cdef extern from "NFW.h":

    float u [100]
    float c [100]
    float inv_u [10000]

def random_nfw (int[:] npart, float[:] conc, float[:] r, float[:] theta, float[:] phi):

  _random_nfw(&npart[0], &conc[0], conc.shape[0], &r[0], &theta[0], &phi[0])

cdef float getr(float x, float y) nogil:

  return bilinear_interpolation (y, x, &u[0], &c[0], &inv_u[0], 100)

cdef float bilinear_interpolation (float x, float y, float *gridx, float *gridy, float *v, int dim) nogil:

  # Get dx and dy
  cdef float dx = gridx[1] - gridx[0];
  cdef float dy = gridy[1] - gridy[0];
  # Get x0 and y0
  cdef float x0 = gridx[0];
  cdef float y0 = gridy[0];

  # Get x and y indexes
  cdef int i = <int>floor((x-x0)/dx);
  cdef int j = <int>floor((y-y0)/dy);

  # Get Points
  cdef float x1  = gridx[i];
  cdef float x2  = gridx[i + 1];
  cdef float y1  = gridy[j];
  cdef float y2  = gridy[j+1];
  cdef float v12 = v[ j*dim + i +1 ];
  cdef float v11 = v[ j*dim + i ];
  cdef float v21 = v[ (j+1)*dim + i ];
  cdef float v22 = v[ (j+1)*dim + i + 1];

  return 1.0/(x2-x1)/(y2-y1)*( v11*(x2-x)*(y2-y) + v21*(x-x1)*(y2-y) + v12*(x2-x)*(y-y1) + v22*(x-x1)*(y-y1) )

cdef void _random_nfw (int *npart, float *conc, int dim, float *r, float *theta, float *phi) nogil:

  cdef int i,j;
  cdef int nhalos=0;
  cdef float u;

  for i in range(dim):

    for j in range(npart[i]):

      u = (<float>rand())/(RAND_MAX);
      r[nhalos]     = getr(conc[i], u);
      theta[nhalos] = acos( 2.0*((<float>rand())/(RAND_MAX)-0.5) )
      phi[nhalos]   = 2.0 * pi *(<float>rand())/(RAND_MAX)
      nhalos = nhalos + 1
