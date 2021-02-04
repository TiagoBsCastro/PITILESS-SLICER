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
z = 1.9775
cosmology.hmf_w13.update(z=z)
cosmology.hmf_t10.update(z=z)

mf0 = mf("../TestRuns/pinocchio.{0:5.4f}.example.catalog.out".format(z), "catalog", nsnap=nsnap, boxsize=boxsize)

# Checking the k which P1HT~0.1 Plin
#Paux = HM.P1H(cosmology.k, z, cosmology.hmf_t10)/cosmology.Pk / np.interp(1.0/(1.0+z), cosmology.a, cosmology.D)**2 - 0.1
#kmin = fsolve(lambda k : np.interp(k, cosmology.k, Paux), [1.0])[0]
kmin = 0.1
kmax = 10.0

k   = np.geomspace(kmin, kmax)
sol, brute, log10Mpv = HM.fitConc(k, z, mf0, cosmology.hmf_w13)

if brute:
    Pkpin = HM.P1H(k, z, mf0, model='generic', A=sol[0], B=sol[1], C=sol[2], log10Mpv=sol[3])
else:
    Pkpin = HM.P1H(k, z, mf0, model='generic', A=sol.x[0], B=sol.x[1], C=sol.x[2], log10Mpv=log10Mpv)

Pktar = HM.P1H(k, z, cosmology.hmf_w13)

plt.loglog(k, Pkpin)
plt.loglog(k, Pktar)
plt.loglog(cosmology.k, cosmology.Pk * np.interp(1.0/(1.0+z), cosmology.a, cosmology.D)**2)
plt.show()
