import params
from IO import ReadPinocchio as rp
import numpy as np
from astropy.constants import c
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy.units import Mpc

########################### Cosmological Model ##########################
lcdm  = FlatLambdaCDM(H0=params.h0, Om0=params.omega0)
cspeed = c.to('km/s').value

zinterp = np.linspace(0, params.zsource + 0.1, 1000)
ainterp = 1.0/(1.0+zinterp)
Dinterp = lcdm.comoving_distance(zinterp).value

################################ Utils ##################################

def w_tophat(k,r):
    '''
    Returns the Fourier Transform of the Spherical Top-Hat function
    with radius r
    '''
    return ( 3.*( np.sin(k * r) - k*r*np.cos(k * r) )/((k * r)**3.))

def ssq (r, k, Dk):
    '''
    Returns the integrand of sigma_r^2 given k, r and Dk using the trapezoidal method

    k:  Array
        Fourier frequency
    r:  Float
        Value of the Lagrangean patch corresponding to the smoothing radius
    Dk: Array
        Adimensional matter-power spectrum Dk(k)
    '''
    return np.trapz( ((w_tophat(k,r))**2)*Dk/k, x=k )

############################# Lens Planes ###############################

lensthickness = params.boxsize/params.nlensperbox
nreplications = int(lcdm.comoving_distance(params.zsource).value/lensthickness + 1 )
zl    = [0] + [ z_at_value(lcdm.comoving_distance, n*lensthickness*Mpc, zmax = 1e4) for n in range(1, nreplications) ] + [params.zsource]
zlsup = zl[1:  ]
zlinf = zl[ :-1]

################### Pinocchio Cosmological Quantities ###################

(a, t, D, D2, D31, D32) = np.loadtxt(params.pincosmofile, usecols=(0,1,2,3,4,5), unpack=True)
(R, DeltaRGauss, k, Pk) = np.loadtxt(params.pincosmofile, usecols=(9, 10, 12, 13), unpack=True)

sigma8 = ssq(8.0 * 100/params.h0true, k, Pk * k**3/2/np.pi**2 )**0.5

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
