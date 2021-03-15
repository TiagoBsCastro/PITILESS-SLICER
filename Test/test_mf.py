import numpy as np
from IO.Pinocchio.ReadPinocchio import mf
import matplotlib.pyplot as plt

# Box Size
boxsize = 150.0

aux = mf("../TestRuns/pinocchio.1.9775.example.mf.out", "mf")
mf0 = mf("../TestRuns/pinocchio.1.9775.example.catalog.out", "catalog", 64, boxsize)
mf0.dndm_teo   = np.interp(mf0.m, aux.m, aux.dndm_teo)
mf0.dndlnm_teo = np.interp(mf0.m, aux.m, aux.dndlnm_teo)

plt.loglog(mf0.m, mf0.dndlnm, label='dn/dm Pin.')
plt.loglog(mf0.m, mf0.dndlnm_teo, label='dn/dm Teo.')
plt.legend()
plt.show()

plt.plot(mf0.m, mf0.dndlnm/mf0.dndlnm_teo, label='dn/dlnm Pin.')
plt.axhline(1.0)
plt.ylim([0.5, 1.05])
plt.xscale('log')
plt.legend()
plt.show()
