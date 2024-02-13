import params
from IO.Pinocchio import ReadPinocchio as rp
import numpy as np
from astropy.constants import c
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy.units import Mpc
from colossus.halo import profile_nfw, concentration
from colossus.cosmology import cosmology as colossus
from hmf import MassFunction     # The main hmf class
from hmf.mass_function.fitting_functions import Tinker10, Watson_FoF

########################### Cosmological Model ##########################
lcdm  = FlatLambdaCDM(H0=params.h0, Om0=params.omega0)
cspeed = c.to('km/s').value

_lcdm_for_hmf  = FlatLambdaCDM(H0=params.h0true, Om0=params.omega0, \
                               Ob0=params.omegabaryon, Tcmb0=params.TCMB)

hmf_w13 = MassFunction(cosmo_model=_lcdm_for_hmf, n=params.ns, \
                   sigma_8=params.sigma8, transfer_model=params.transfer_model,\
                   hmf_model=Watson_FoF, Mmin=10.0, Mmax=16.0)

hmf_t10 = MassFunction(cosmo_model=_lcdm_for_hmf, n=params.ns, \
                   sigma_8=params.sigma8, transfer_model=params.transfer_model,\
                   hmf_model=Tinker10, Mmin=10.0, Mmax=16.0)

zinterp = np.linspace(0, params.zsource + 0.1, 1000)
ainterp = 1.0/(1.0+zinterp)
Dinterp = lcdm.comoving_distance(zinterp).value

############################# Lens Planes ###############################

if params.nlensperbox != 0:
  lensthickness = params.boxsize/params.nlensperbox
else:
  lensthickness = params.lensthickness

nreplications = int(lcdm.comoving_distance(params.zsource).value/lensthickness + 1 )
zl    = [0] + [ z_at_value(lcdm.comoving_distance, n*lensthickness*Mpc, zmax = 1e4) for n in range(1, nreplications) ] + [params.zsource]
zlsup = zl[1:  ]
zlinf = zl[ :-1]

################### Pinocchio Cosmological Quantities ###################

# Checking the pinocchio cosmology version
_ = np.loadtxt(params.pincosmofile)
if _.shape[1] == 20:
   (a, t, D, D2, D31, D32) = np.loadtxt(params.pincosmofile, usecols=(0, 1, 6, 7, 8, 9), unpack=True)
   (R, DeltaRGauss, k, Pk) = np.loadtxt(params.pincosmofile, usecols=(14, 15, 18, 19), unpack=True)
else:
   (a, t, D, D2, D31, D32) = np.loadtxt(params.pincosmofile, usecols=(0, 1, 2, 3, 4, 5), unpack=True)
   (R, DeltaRGauss, k, Pk) = np.loadtxt(params.pincosmofile, usecols=(9, 10, 12, 13), unpack=True)

k  = k*100/params.h0true
Pk = Pk/(100/params.h0true)**3

# Setting cosmology for concentrations
pdict = {'flat': True, 'H0': params.h0true, 'Om0': params.omega0, 'Ob0': params.omegabaryon,\
        'sigma8': params.sigma8, 'ns': params.ns, 'relspecies': params.relspecies}
colossus.addCosmology('myCosmo', pdict)
colossus.setCosmology('myCosmo'); del pdict

# PowerLaw Pk case
if params.ns < 0.0:
    colossus.setCosmology(f'powerlaw_{params.ns}', flat = True, H0=params.h0true, Om0=params.omega0, Ob0=params.omegabaryon, sigma8=params.sigma8, ns=params.ns, relspecies=params.relspecies)
else:
# General case
    colossus.setCosmology('myCosmo'); del pdict
    
class PolyFitOrderTooLow (Exception):
    pass

class PolyFitIsNotMonotonic (Exception):
    pass

def getWisePolyFit (x, y, dtype=np.float32):

    for norder in range(1, params.norder+1):

        if norder >= x.size:

            raise PolyFitOrderTooLow("The number of points inside the range is smaller than the polynomial order")

        P = np.polyfit( x , y, norder )
        prec = np.abs( np.mean( np.polyval( P, x[y!=0] )/y[y!=0] - 1) )
        acc  = np.std( np.polyval(P, x[y!=0])/y[y!=0] )
        if prec < 1e-2 and acc < 1e-2:

            p_prime = np.polyder(P,1)
            roots   = np.roots(p_prime)
            roots   = roots[ np.isreal(roots) ]

            if any( (roots>x.min()) & (roots<x.max()) ):

                raise PolyFitIsNotMonotonic("The Polynomial is not monotonic inside the fit range")

            return np.array(([0 for i in range(params.norder - norder)] + P.tolist())[::-1], dtype=dtype)

    raise PolyFitOrderTooLow
