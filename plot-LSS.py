import params
import matplotlib.pyplot as plt
import numpy as np
import cosmology
from IO.ReadPinocchio import catalog
from IO import snapshot

snap = snapshot.Timeless_Snapshot(params.pintlessfile)

posp = np.transpose(snap.snapPos(0.0, zcentered=False)) + params.boxsize/2

cat = catalog("pinocchio.0.0000.example.catalog.out")

cut = posp[:,2] < 20
plt.scatter(posp[cut, 0], posp[cut, 1], s=0.001)
plt.figure()
cut = (cat.pos[:, 2] < 20) & (cat.Mass > 1e14)
plt.scatter(cat.pos[cut, 0], cat.pos[cut, 1], s=0.1, c='r')

plt.show()
