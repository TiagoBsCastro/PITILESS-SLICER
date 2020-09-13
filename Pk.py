import matplotlib.pyplot as plt
import numpy as np
import MAS_library as MASL
import readsnap as rs
import Pk_library as PKL
import params
import cosmology

z        = 19.0
a        = 1.0/(1.0+z)
snapshot = 'pinocchio.example.{0:5.4f}.out'.format(z)
grid     = 256
ptypes   = [1]
MAS      = 'CIC'
do_RSD   = False
axis     = 0
BoxSize  = params.boxsize
threads  = 1

# define the array hosting the density field
delta = np.zeros((grid,grid,grid), dtype=np.float32)
# compute density field
pos = rs.read_block(snapshot, "POS ")
MASL.MA(pos.astype(np.float32), delta, BoxSize, MAS)
# compute overdensity field
delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0

Pk = PKL.Pk(delta, BoxSize, axis, MAS, threads)

plt.loglog(Pk.k3D * params.h0true/100, Pk.Pk[:, 0] * Pk.k3D**3)

growth = np.interp(a, cosmology.a, cosmology.D)
plt.loglog(cosmology.k, cosmology.Pk * growth**2 * cosmology.k**3)
plt.show()
