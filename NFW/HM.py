import numpy as np
from scipy.integrate import quad, trapz, cumtrapz
import params
from cosmology import colossus, concentration, profile_nfw
from scipy.special import sici
from scipy.optimize import minimize
from scipy.optimize import dual_annealing, brute
from scipy.optimize import fsolve
import sys

rhoc = 2.7753663e11

try:

    cvir = concentration.concentration(1E12, 'vir', 0.0, model = 'bullock01')

except:

    raise RuntimeError("Attempt to run HM.py without setting cosmology!")

def W (k, M, z, model = 'diemer19', A=None, B=None, C=None, log10Mpv=None):
    '''
    W(k, M) is the normalized Fourier transform of the halo density profile. Eq. (9)

    k : float
        Fourier Mode in (Mpc/h)^-1
    M : float
        Halo mass in Msun/h
    z : float
        Redshift
    model : Colossus concentration model
        str (default diemer19)
    A : float (Only required if model == 'generic')
        c-M relation amplitude parameter
    B : float (Only required if model == 'generic')
        c-M relation power-law parameter
    C: float (Only required if model == 'generic')
        C-M relation minimum value
    log10Mpv : float (Only required if model == 'generic')
        c-M relation log10-mass pivot value
    '''
    if model == 'generic':

        try:

            conc = A * (M/(10**log10Mpv)) ** B + C

        except TypeError:

            raise RuntimeError("A, B, C, Mpv have to be explictly set if model is generic!")

    else:

        conc  = concentration.concentration(M, '200c', z, model = model)

    # Getting the comoving parameters assuming 200c
    R200c = ( 3 * M /( 4*np.pi * 200 * rhoc ) )**(1.0/3)
    rs    = R200c/conc*(1+z)
    rhos  = conc**3/3 * 200 * rhoc / ( np.log(conc+1) - conc/(conc+1) ) / (1+z)**3

    # Analytical Integration
    aux1 = k[:, np.newaxis] * (1.0+conc)*rs
    aux2 = k[:, np.newaxis] * rs
    aux3 = k[:, np.newaxis] * conc * rs
    si1, ci1 = sici(aux1)
    si2, ci2 = sici(aux2)

    return 4.0 * np.pi * rhos * rs**3 *\
           ( np.sin( aux2 ) * (si1-si2) - np.sinc(aux3/np.pi) * (conc/(1.0+conc)) +\
             np.cos( aux2 ) * (ci1-ci2) )


def P1H (k, z, mf, model='diemer19', A=None, B=None, C=None, log10Mpv=None):
    '''
    One-halo term contribution to the non-linear matter P(k). Eq. (8)/(k/2pi)**3

    k : numpy array of floats
        Fourier Mode in (Mpc/h)^-1
    z : float
        Redshift
    mf : Mass Function object
        Mass function produced by Pinocchio as read by ReadPinocchio
    model : Colossus concentration model
        str (default diemer19)
    A : float (Only required if model == 'generic')
        c-M relation amplitude parameter
    B : float (Only required if model == 'generic')
        c-M relation power-law parameter
    C : float (Only required if model == 'generic')
        c-M relation minimum value
    Log10Mpv : float (Only required if model == 'generic')
        c-M relation log10-mass pivot value
    '''

    try:

        integrand = mf.dndm * W(k, mf.m, z, model, A=A, B=B, C=C, log10Mpv=log10Mpv) ** 2

        return 1.0 / (colossus.current_cosmo.rho_m(0.0) * 1e9)**2 * trapz( integrand, x=mf.m, axis=1)

    except IndexError:

        raise RuntimeError('k should be a numpy array!!')

def fitConc (k, z, mf0, mf1, model='diemer19'):
    '''
    Fits a generic mass-concentration relation by fitting the one
    halo term.

    k : numpy array of floats
        Fourier Mode in (Mpc/h)^-1
    z : float
        Redshift
    mf0 : Mass Function object
        The observed mass function
    mf1 : Mass Function object
        The target mass function
    model : Colossus concentration model
        str (default diemer19)
    '''
    # Checking if the cumulative mf of the target (mf1) has
    # more objects than the obs (mf0)
    totmass = np.trapz(mf0.dndm * mf0.m**2, x=mf0.m)
    while ( np.trapz(mf1.dndm * mf1.m**2, x=mf1.m) <= totmass ):

        mf1.update(Mmin=np.log10(mf1.m.min()/2))
        if np.log10(mf1.m.min()/2) < 8:

            print("The target HMF never reach Pinocchio's MF!")
            print("I will stop here!")
            sys.exit(-1)

    # Getting the Mass equality
    cmf1 = np.trapz(mf1.dndm * mf1.m**2, x=mf1.m) - cumtrapz(mf1.dndm * mf1.m**2, x=mf1.m)
    cmf1 = np.append(cmf1, [0.0])

    f    = lambda x: np.interp(x, mf1.m, cmf1) - totmass
    Mmin = fsolve(f, [mf0.m.min()])
    mf1.update(Mmin=np.log10(Mmin))

    # The target P1H term
    target = P1H(k, z, mf1, model=model)

    # Residual
    # Defining the generic c-M relation (Duffry)
    # Getting the Mass equality
    cmf0 = totmass - cumtrapz(mf0.dndm * mf0.m**2, x=mf0.m)
    cmf0 = np.append(cmf0, [0.0])
    f    = lambda x: np.interp(x, mf0.m, cmf0) - totmass/2
    Mpv = fsolve(f, [mf0.m.min()])
    p1h  = lambda x: np.sum((P1H (k, z, mf0, model='generic', A=x[0], B=x[1], C=x[2], log10Mpv=np.log10(Mpv))/target - 1)**2)
    # Minimizing
    res = dual_annealing(p1h, bounds=[(0.01, 20), (-4.0, 0.0), (2.0, 6.0)])

    if res.message[0] == 'Maximum number of iteration reached':

        print("Dual Annealing returned: ", res.message[0])
        print("Trying different solution with Brute Force optimization: ")
        p1h  = lambda x: np.sum((P1H (k, z, mf0, model='generic', A=x[0], B=x[1], C=x[2], log10Mpv=x[3])/target - 1)**2)
        res = brute(p1h, ranges=[(0.01, 20), (-4.0, 0.0), (2.0, 6.0), (10, 13)], finish=None)
        brute_bool = 1

    else:

        brute_bool = 0

    return res, brute_bool, np.log10(Mpv)
