import numpy as np
from scipy import optimize
from scipy.interpolate import interp2d

# Maximum radii that particles will be considered in R_Delta units
RMAX = 1.0
# NUmber of points for the interpolator
NMAX = 50

def nfwFac (c, r):
    '''
    Navarro-Frenk-White averaged profile factor:
        1/r^3 x ( 1/(1 + cxr) -1 + ln 1+cxr)
    '''
    return (1.0/r**3) * ( 1.0/(1.0+c*r) - 1.0 + np.log(1.0 + c*r) )

def rVirial (cx, DeltavOverDeltax):
    '''
    Returns the radius r=R/Rx of a Delta_x SO halo which
    the enclosed mass is equals to M_V.

    cx is the concentration of the Delta_x SO halo
    DeltavOverDeltax is the ratio of the x-threshold
    with respect to the desired threshold. (eg. 200 rho_c/ 500 rho_c)
    '''
    f = lambda r: nfwFac(cx, r)/nfwFac(cx, 1.0) - DeltavOverDeltax

    x0 = 0.8 if DeltavOverDeltax > 1.0 else 1.0

    return optimize.fsolve(f, x0)

def normalizedNFWMass (cx, r):
    '''
    Returns the ratio of the halo mass at radius r=R/Rx and r=1

    cx is the concentration of the Delta_x SO halo
    '''
    return r**3 * nfwFac(cx, r)/nfwFac(cx, 1.0) if r > 0 else 0.0

def dnormalizedNFWMass (cx, r):
    '''
    Returns the derivative of the ratio of the halo mass at radius r=R/Rx and r=1 as a function of r

    cx is the concentration of the Delta_x SO halo
    '''
    return  cx**2 * (1.0 + cx) *r / (1.0 + cx*r) ** 2 / (-cx + (1.0+cx) * np.log(1.0+cx))

# Calculating the CDF (_u) for several values of radii _r and concentration _c
_r     = np.linspace(0.0,  RMAX, NMAX)
_c     = np.linspace(2.0, 30.0, NMAX)
_u     = np.linspace(0.0,  1.0, NMAX)
_inv_u = np.array( [ [ optimize.fsolve( lambda r: normalizedNFWMass (c, r) * normalizedNFWMass (c, 1.0)/normalizedNFWMass (c, RMAX) - u, u)[0] for c in _c ] for u in _u]  )
_inv_u = interp2d(_c, _u, _inv_u)

_logu      = np.log( np.geomspace(1e-3,  1.0, NMAX) )
_inv_log_u = np.array( [ [ optimize.fsolve( lambda r: normalizedNFWMass (c, r) * normalizedNFWMass (c, 1.0)/normalizedNFWMass (c, RMAX) - np.exp(u), np.exp(u))[0] for c in _c ] for u in _logu]  )
_inv_log_u = interp2d(_c, _logu, _inv_log_u)

def getr (cx, u):
    '''
    Returns the  radius r=R/Rx where the enclosed mass fraction is u with respect to
    the total mass at the enclosed mass by RMAX x R_Delta

    cx is the concentration of the Delta_x SO halo
    '''
    if (u < 0) or (u > 1.0):

        raise ValueError("The CDF should be bounded to [0,1]")

    return _inv_u(cx, u)

def randomr (cx, n=1):
    '''
    Returns n (default=1) random value of r=R/Rx~NFW

    cx is the concentration of the Delta_x SO halo
    '''

    u = np.random.rand(n)

    if n == 1:

        return getr(cx, u)[0]

    else:

        return np.array( [getr(cx, x)[0] for x in u] )
