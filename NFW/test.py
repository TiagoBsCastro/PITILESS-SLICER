import numpy as np
import NFWx

nhalos = 100000

npart = np.random.randint(10, 5000, nhalos, dtype=np.int32)
conc = 3 + 5 * np.random.random(nhalos)
conc = conc.astype(np.float32)
r = np.empty(npart.sum(), dtype=np.float32)
theta = np.empty(npart.sum(), dtype=np.float32)
phi = np.empty(npart.sum(), dtype=np.float32)

NFWx.random_nfw(npart, conc, r, theta, phi)
