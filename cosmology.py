import params
import ReadPinocchio as rp
import numpy as np
from astropy.constants import c
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy.units import Mpc

################# Cosmological Model ##################
lcdm  = FlatLambdaCDM(H0=params.h0, Om0=params.omega0)
cspeed = c.to('km/s').value

zinterp = np.linspace(0, params.zsource + 0.1, 1000)
ainterp = 1.0/(1.0+zinterp)
Dinterp = lcdm.comoving_distance(zinterp).value

##################### Lens Planes #####################

lensthickness = params.boxsize/params.nlensperbox
nreplications = int(lcdm.comoving_distance(params.zsource).value/lensthickness + 1 )
zl    = [0] + [ z_at_value(lcdm.comoving_distance, n*lensthickness*Mpc, zmax = 1e4) for n in range(1, nreplications) ] + [params.zsource]
zlsup = zl[1:]
zlinf = zl[:-1]

########## Pinocchio Cosmological Quantities ##########

(a, t, D, D2, D31, D32) = np.loadtxt(params.pincosmofile, usecols=(0,1,2,3,4,5), unpack=True)

class PolyFitOrderTooLow (Exception):
    pass

def getWisePolyFit (x, y, dtype=np.float32):

    try:

        for norder in range(1, params.norder+1):

            if norder >= x.size:

               raise PolyFitOrderTooLow

            P = np.polyfit( x , y, norder )
            prec = np.abs( np.mean( np.polyval( P, x[y!=0] )/y[y!=0] - 1) )
            acc  = np.std( np.polyval(P, x[y!=0])/y[y!=0] )
            if prec < 1e-2 and acc < 1e-2:

                return np.array(([0 for i in range(params.norder - norder)] + P.tolist())[::-1], dtype=dtype)

        raise PolyFitOrderTooLow

    except:

        raise PolyFitOrderTooLow

########## Pinocchio Past Light Cone of Halos ##########

plc = rp.plc(params.pinplcfile)
plc.Mass = (plc.Mass/plc.Mass.min()*params.minhalomass).astype(int)
