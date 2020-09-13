import numpy as np
import MAS_library as MASL
import readsnap as rs
import Pk_library as PKL

snapshot = 'pinocchio.example.0.0000.out'
grid     = 128
ptypes   = [1]
MAS      = 'CIC'
do_RSD   = False
axis     = 0
BoxSize  = 500.0
threads  = 1

# define the array hosting the density field
delta = np.zeros((grid,grid,grid), dtype=np.float32)
# compute density field
pos = rs.read_block(snapshot, "POS ")
MASL.MA(pos.astype(np.float32), delta, BoxSize, MAS)
# compute overdensity field
delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0

Pk = PKL.Pk(delta, BoxSize, axis, MAS, threads)
