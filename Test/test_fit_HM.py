import numpy as np
import cosmology
import matplotlib.pyplot as plt
from IO.Pinocchio.ReadPinocchio import mf, catalog
from NFW import HM
from IO.Utils.fixLowMasses import get_mass_correction
from scipy.optimize import fsolve

# BoxSize
boxsize = 150.0
# number of snapshots
nsnap = 64
# Redshift
z = 0.0
cosmology.hmf_w13.update(z=z)
cosmology.hmf_t10.update(z=z)

mf0 = mf("/beegfs/tcastro/TestRuns/pinocchio.{0:5.4f}.example.catalog.out".format(z), "catalog", nsnap=nsnap, boxsize=boxsize)

# Checking the k which P1HT~0.1 Plin
Paux = HM.P1H(cosmology.k, z, cosmology.hmf_t10)/cosmology.Pk / np.interp(1.0/(1.0+z), cosmology.a, cosmology.D)**2 - 0.1
#kmin = fsolve(lambda k : np.interp(k, cosmology.k, Paux), [1.0])[0]
kmin = 0.1

k   = np.geomspace(kmin, 3e1)
sol = HM.fitConc(k, z, mf0, cosmology.hmf_t10)

Pkw13 = HM.P1H(k, z, mf0, model='generic', A=sol.x[0], B=sol.x[1], Mpv=sol.x[2], C=sol.x[3])
Pkt08 = HM.P1H(k, z, cosmology.hmf_t10)

plt.loglog(k, Pkw13)
plt.loglog(k, Pkt08)
plt.loglog(cosmology.k, cosmology.Pk * np.interp(1.0/(1.0+z), cosmology.a, cosmology.D)**2)
plt.show()
