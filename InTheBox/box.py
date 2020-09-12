import params
from IO import snapshot, Snapshot3
import numpy as np
import cosmology
from copy import deepcopy

#######################  Reading Timeless Snapshot  #######################
snap = snapshot.Timeless_Snapshot(params.pintlessfile)

#getting the ordered Map:
dummy_head = deepcopy(snap.snap.Header)
dummy_map  = deepcopy(snap.snap.Map)
omap       = np.argsort([x.offset[0] for x in dummy_map])[:5]
dummy_map  = [dummy_map[i] for i in omap]
dummy_map[-2].name = b"POS "; dummy_map[-1].name = b"VEL "
###########################################################################

for z in params.redshifts:

    pos = np.transpose(snap.snapPos(z, zcentered=False)) + params.boxsize/2
    vel = np.transpose(snap.snapVel(z))

    dummy_head.redshift = z
    dummy_head.time     = 1.0/(1.0+z)
    # Copying the Map from timeless snapshot
    zsnap = Snapshot3.Init(params.pintlessfile.replace("t_snapshot", "{0:5.4f}".format(z)), -1, ToWrite=True, override=True)

    zsnap.write_header(dummy_head)
    zsnap.write_info_block(dummy_map)
    zsnap.write_block(b"ID  ", np.dtype('uint32'), params.nparticles, snap.ID.astype(np.uint32))
    zsnap.write_block(b"POS ", np.dtype('float32'), 3*params.nparticles, pos.astype(np.float32))
    zsnap.write_block(b"VEL ", np.dtype('float32'), 3*params.nparticles, vel.astype(np.float32))
