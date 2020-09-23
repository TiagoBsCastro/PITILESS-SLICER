cimport cython
import numpy as np
from libc.math cimport floor, acos, pi, log
from libc.stdlib cimport rand, RAND_MAX

# Hardcoding the NFW profile
cdef extern from "NFW.h":

    int NDIM;          # Dimension of the hard-coded NFW profile
    int NDIM2;         # Dimension^2 of the hard-coded NFW profile
    float u [];
    float logu [];
    float c [];
    float inv_u [];
    float inv_logu [];

def random_nfw (int[:] npart, float[:] conc, float[:] rdelta, float[:] r, float[:] theta, float[:] phi):
  '''
  Random sampling particles inside halos following a NFW profile.

  npart: np.int32 array
    Array with the number of particle for different halos.
  conc: np.float32 array
    Array with the concentration parameter for different halos.
  rdelta: np.float32 array
    Array with the boundary radius for different halos.
  r: np.float32
    Empty array to be filled with the sampled particle radii coordinates.
  theta: np.float32
    Empty array to be filled with the sampled particle theta coordinates.
  phi: np.float32
    Empty array to be filled with the sampled particle phi coordinates.
  '''
  if (npart.size != conc.size) or (npart.size != rdelta.size):
    raise RuntimeError("Halo arrays should all have the same size!")
  elif r.size != np.sum(npart):
    raise RuntimeError("Particle array size different then the total number of particles")
  elif (r.size != theta.size) or (r.size != phi.size):
    raise RuntimeError("Particle arrays should all have the same size!")
  else:
    _random_nfw(&npart[0], &conc[0], &rdelta[0], npart.size, &r[0], &theta[0], &phi[0])

# Linearly interpolating on u
cdef float getrlin(float x, float y) nogil:

  return bilinear_interpolation (y, x, &u[0], &c[0], &inv_u[0], NDIM)

def _getrlin (float x, float y):

  return getrlin(x, y)

# Geometricaly interpolating on u
#cdef float getrgeom(float x, float y) nogil:
#
#  return bilinear_interpolation (log(y), x, &logu[0], &c[0], &inv_logu[0], NDIM)

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

  # Boundary check
  if (i<0) or (j<0) or (i>=NDIM-1) or (j>=NDIM-1):

    return 0.0

  # Get Points
  cdef float x1  = gridx[i];
  cdef float x2  = gridx[i + 1];
  cdef float y1  = gridy[j];
  cdef float y2  = gridy[j+1];
  cdef float v11 = v[ j*dim + i ];
  cdef float v21 = v[ j*dim + i +1 ];
  cdef float v12 = v[ (j+1)*dim + i ];
  cdef float v22 = v[ (j+1)*dim + i + 1];

  return 1.0/(x2-x1)/(y2-y1)*( v11*(x2-x)*(y2-y) + v21*(x-x1)*(y2-y) + v12*(x2-x)*(y-y1) + v22*(x-x1)*(y-y1) )

cdef void _random_nfw (int *npart, float *conc, float *rdelta, int dim, float *r, float *theta, float *phi) nogil:

  cdef Py_ssize_t i,j;
  cdef int nhalos=0;
  cdef float ru;

  for i in range(dim):

    for j in range(npart[i]):

      ru = (<float>rand())/(RAND_MAX);

      r[nhalos]     = getrlin(conc[i], ru) * rdelta[i];
      theta[nhalos] = acos( 2.0*((<float>rand())/(RAND_MAX)-0.5) )
      phi[nhalos]   = 2.0 * pi *(<float>rand())/(RAND_MAX)
      nhalos = nhalos + 1
