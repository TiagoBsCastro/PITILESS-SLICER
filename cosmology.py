import params
import ReadPinocchio as rp
import numpy as np
from astropy.constants import c
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy.units import Mpc

################# Cosmological Model ##################
lcdm  = FlatLambdaCDM(H0=params.h0, Om0=params.omega0)
cspeed = c.to('km/s').to_value()

zinterp = np.linspace(0, params.zsource + 0.1, 100)
ainterp = 1.0/(1.0+zinterp)
Dinterp = lcdm.comoving_distance(zinterp).value

##################### Lens Planes #####################

lensthickness = params.boxsize/params.nlensperbox
nreplications = int(lcdm.comoving_distance(params.zsource).value/lensthickness + 1 )
zl    = [0] + [ z_at_value(lcdm.comoving_distance, n*lensthickness*Mpc, zmax = 1e4) for n in range(1, nreplications) ]
zlsup = zl[1:]
zlinf = zl[:-1]

########## Pinocchio Cosmological Quantities ##########

(a, t, D, D2, D31, D32) = np.loadtxt(params.pincosmofile, usecols=(0,1,2,3,4,5), unpack=True)

########## Pinocchio Past Light Cone of Halos ##########

plc = rp.plc(params.pinplcfile)
plc.Mass = (plc.Mass/plc.Mass.min()*params.minhalomass).astype(int)
