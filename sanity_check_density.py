import numpy as np
from scipy import stats
from mayavi import mlab
import ReadPinocchio as rp

plc = rp.plc("pinocchio.cdm.plc.out")
auxcut = (plc.redshift<0.68) & (plc.redshift>0.55)

x = plc.pos[auxcut,0]
y = plc.pos[auxcut,1]
z = plc.pos[auxcut,2]

xyz = np.vstack([x,y,z])
kde = stats.gaussian_kde(xyz)
density = kde(xyz)

# Plot scatter with mayavi
figure = mlab.figure('DensityPlot')
pts = mlab.points3d(x, y, z, density, scale_mode='none', scale_factor=0.07)
mlab.axes()
mlab.show()
