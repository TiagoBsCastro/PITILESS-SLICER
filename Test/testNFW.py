import params
import cosmology
from IO.Pinocchio.ReadPinocchio import catalog
from IO.Pinocchio.TimelessSnapshot import timeless_snapshot
import readsnap as rs
import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology as colossus
from colossus.halo import profile_nfw
from colossus.halo import concentration

# Setting cosmology for concentrations
pdict = {'flat': True, 'H0': params.h0true, 'Om0': params.omega0, 'Ob0': params.omegabaryon,\
          'sigma8': cosmology.sigma8, 'ns': params.ns}
colossus.addCosmology('myCosmo', pdict)
colossus.setCosmology('myCosmo')

# Load Catalog
cat = catalog("pinocchio.0.0000.example.catalog.out")
rhoc = cosmology.lcdm.critical_density(0.0).to("M_sun/Mpc^3").value
rDelta = (3*cat.Mass/4/np.pi/200/rhoc)**(1.0/3)

# Getting particle positions and particle mass
pos = rs.read_block("pinocchio.example.0.0000.out", "POS ")
mp  = rs.snapshot_header("pinocchio.example.0.0000.out").massarr[1] * 1e10

# Getting the Index of the i-th passive object in catalog
idx = np.argsort(cat.Mass)[-100]

# Getting the first and last icdx of the particles inside the most massive halo
idxp1 = np.sum(cat.Npart[:idx])
idxp2 = idxp1 + cat.Npart[idx]

plt.scatter(pos[idxp1:idxp2][:,0], pos[idxp1:idxp2][:,1], s=0.1)
plt.show()

plt.scatter(pos[idxp1:idxp2][:,0], pos[idxp1:idxp2][:,2], s=0.1)
plt.show()

plt.scatter(pos[idxp1:idxp2][:,1], pos[idxp1:idxp2][:,2], s=0.1)
plt.show()

# Getting Radius
r = ((pos[idxp1:idxp2] - cat.pos[idx])**2).sum(axis=1)**0.5
rhist = np.geomspace(r.min(), r.max(), 10)
mhist, aux = np.histogram(r, bins=rhist)
mhist = mp * mhist
mhist = np.cumsum(mhist)
mhist = np.append([0], mhist)

profile = profile_nfw.NFWProfile(M = cat.Mass[idx], mdef = '200c', z = 0.0, c = 4.0)

fit = profile.fit(rhist*1e3, mhist, 'M')

c = concentration.concentration(cat.Mass, '200c', 0.0, model = 'bhattacharya13')
print(c[idx], rDelta[idx]*1e3/fit['x'][1])

plt.plot(rhist, fit['q_fit'])
plt.plot(rhist, mhist)
plt.show()
