import numpy as np
import MAS_library as MASL
import readsnap as rs
import Pk_library as PKL
import params

grid     = 512
ptypes   = [1]
MAS      = 'CIC'
do_RSD   = False
axis     = 0
BoxSize  = params.boxsize
threads  = 4

for folder in ['/beegfs/tcastro/TestRuns/minmass10/']:

    for z in [0.0]:

        snapshot = folder + 'pinocchio.minmass10.{0:5.4f}.out'.format(z)

        # define the array hosting the density field
        delta = np.zeros((grid,grid,grid), dtype=np.float32)
        # compute density field
        pos = rs.read_block(snapshot, "POS ")
        MASL.MA(pos.astype(np.float32), delta, BoxSize, MAS)
        # compute overdensity field
        delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0

        Pk = PKL.Pk(delta, BoxSize, axis, MAS, threads)

        np.savetxt(folder+"/Pk.{0:5.4f}.txt".format(z), np.transpose([Pk.k3D, Pk.Pk[:,0]]))
