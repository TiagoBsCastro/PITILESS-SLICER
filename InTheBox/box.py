import params
from IO import snapshot
import numpy as np
import cosmology

#######################  Reading Timeless Snapshot  #######################
snap = snapshot.Timeless_Snapshot(params.pintlessfile)

pos  = qPos 

###########################################################################

for z in params.redshifts:
