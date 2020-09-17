import numpy as np
from NFW import NFWx
import params
import cosmology
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology as colossus
from colossus.halo import profile_nfw
from colossus.halo import concentration

nhalos = 100000

# Setting cosmology for concentrations
pdict = {'flat': True, 'H0': params.h0true, 'Om0': params.omega0, 'Ob0': params.omegabaryon,\
          'sigma8': cosmology.sigma8, 'ns': params.ns}
colossus.addCosmology('myCosmo', pdict)
colossus.setCosmology('myCosmo')
rhoc = cosmology.lcdm.critical_density(0.0).to("M_sun/Mpc^3").value

npart  = np.random.randint(10, 5000, nhalos, dtype=np.int32)
mpart  = npart * 1e10
rdelta = (3*mpart/4/np.pi/200/rhoc)**(1.0/3)
conc   = 3 + 5 * np.random.random(nhalos)
conc   = conc.astype(np.float32)
r      = np.empty(npart.sum(), dtype=np.float32)
theta  = np.empty(npart.sum(), dtype=np.float32)
phi    = np.empty(npart.sum(), dtype=np.float32)

NFWx.random_nfw(npart, conc, r, theta, phi)

# Getting the Index of the most passive object in catalog
idx = np.argmax(npart)

# Getting the first and last icdx of the particles inside the most massive halo
idxp1 = np.sum(npart[:idx])
idxp2 = idxp1 + npart[idx]

plt.scatter(pos[idxp1:idxp2][:,0], pos[idxp1:idxp2][:,1])
plt.show()

plt.scatter(pos[idxp1:idxp2][:,0], pos[idxp1:idxp2][:,2])
plt.show()

plt.scatter(pos[idxp1:idxp2][:,1], pos[idxp1:idxp2][:,2])
plt.show()

# Getting Radius
r = ((pos[idxp1:idxp2] - cat.pos[idx])**2).sum(axis=1)**0.5
rhist = np.geomspace(r.min(), r.max(), 10)
mhist, aux = np.histogram(r, bins=rhist)
mhist = mp * mhist
mhist = np.cumsum(mhist)
mhist = np.append([0], mhist)

profile = profile_nfw.NFWProfile(M = mpart[idx], mdef = '200c', z = 0.0, c = 4.0)

fit = profile.fit(rhist*1e3, mhist, 'M')

c = concentration.concentration(mpart, '200c', 0.0, model = 'bhattacharya13')
print(c[idx], rdelta[idx]*1e3/fit['x'][1])

plt.plot(rhist, fit['q_fit'])
plt.plot(rhist, mhist)
plt.show()
