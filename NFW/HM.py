import numpy as np
from scipy.integrate import quad, trapz, cumtrapz
import params
from cosmology import colossus, concentration, profile_nfw
from scipy.special import sici
from scipy.optimize import minimize
from scipy.optimize import dual_annealing
from scipy.optimize import fsolve

try:

    cvir = concentration.concentration(1E12, 'vir', 0.0, model = 'bullock01')

except:

    raise RuntimeError("Attempt to run HM.py without setting cosmology!")

def W (k, M, z, model = 'bhattacharya13', A=None, B=None, Mpv=None):
    '''
    W(k, M) is the normalized Fourier transform of the halo density profile. Eq. (9)

    k : float
        Fourier Mode in (Mpc/h)^-1
    M : float
        Halo mass in Msun/h
    z : float
        Redshift
    model : Colossus concentration model
        str (default bhattacharya13)
    A : float (Only required if model == 'generic')
        c-M relation amplitude parameter
    B : float (Only required if model == 'generic')
        c-M relation power-law parameter
    Mpv: float (Only required if model == 'generic')
        c-M relation pivot parameter
    '''
    # Converting k to (kpc/h)^-1
    kinkpc = k/1e3

    if model == 'generic':

        try:

            conc = A * (M/Mpv) ** B

        except TypeError:

            raise RuntimeError("A, B, Mpv have to be explictly set if model is generic!")

    else:

        conc  = concentration.concentration(M, '200c', z, model = model)

    p_nfw = profile_nfw.NFWProfile(M = M, c = conc, z = z, mdef = '200c')
    # Getting the comoving parameters
    rs    = p_nfw.par['rs']*(1+z)
    rhos  = p_nfw.par['rhos']/(1+z)**3

    # Numerical Integration
    #w = lambda r: np.sinc(kinkpc*r/np.pi) * r**2 * p_nfw.rho(p_nfw.par['rhos'], r/p_nfw.par['rs'])
    #return 4.0 * np.pi * quad(w, 0, conc * p_nfw.par['rs'])[0]

    # Analytical Integration
    si1, ci1 = sici( (1.0+conc)*kinkpc*rs )
    si2, ci2 = sici( kinkpc*rs )
    return 4.0 * np.pi * rhos * rs**3 *\
           ( np.sin( kinkpc*rs ) * (si1-si2) - np.sinc(conc*kinkpc*rs/np.pi)*conc/(1.0+conc) +\
             np.cos( kinkpc*rs ) * (ci1-ci2) )

def P1H (k, z, mf, model='bhattacharya13', A=None, B=None, Mpv=None):
    '''
    One-halo term contribution to the non-linear matter P(k). Eq. (8)/(k/2pi)**3

    k : numpy array of floats
        Fourier Mode in (Mpc/h)^-1
    z : float
        Redshift
    mf : Mass Function object
        Mass function produced by Pinocchio as read by ReadPinocchio
    model : Colossus concentration model
        str (default bhattacharya13)
    A : float (Only required if model == 'generic')
        c-M relation amplitude parameter
    B : float (Only required if model == 'generic')
        c-M relation power-law parameter
    Mpv: float (Only required if model == 'generic')
        c-M relation pivot parameter
    '''

    try:

        integrand = [ dndm*W(k, m, z, model, A=A, B=B, Mpv=Mpv) ** 2 for m, dndm in zip(mf.m, mf.dndm) ]

        return 1.0 / (colossus.current_cosmo.rho_m(0.0) * 1e9)**2 * trapz( np.transpose(integrand), x=mf.m, axis=1)

    except IndexError:

        raise RuntimeError('k should be a numpy array!!')

def fitConc (k, z, mf0, mf1, model='bhattacharya13'):
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
        str (default bhattacharya13)
    '''
    # Checking if the cumulative mf of the target (mf1) has
    # more objects than the obs (mf0)
    totmass = np.trapz(mf0.dndm * mf0.m**2, x=mf0.m)
    while ( np.trapz(mf1.dndm * mf1.m**2, x=mf1.m) <= totmass ):

        mf1.update(Mmin=np.log10(mf1.m.min()/2))

    # Getting the Mass equality
    cmf1 = np.trapz(mf1.dndm * mf1.m**2, x=mf1.m) - cumtrapz(mf1.dndm * mf1.m**2, x=mf1.m)
    cmf1 = np.append([0.0], cmf1)
    f    = lambda x: np.interp(x, mf1.m, cmf1) - totmass
    Mmin = fsolve(f, [mf0.m.min()])
    mf1.update(Mmin=np.log10(Mmin))

    # The target P1H term
    target = P1H(k, z, mf1, model=model)

    # Residual
    # Defining the generic c-M relation (Duffry)
    p1h = lambda x: np.sum((P1H (k, z, mf0, model='generic', A=x[0], B=x[1], Mpv=x[2]) - target)**2)

    # Minimizing
    res = minimize(p1h, [5.74, -0.097, 2e12], method='Nelder-Mead', tol=1e-3, bounds=[(0.0, 20), (-1.0, 0.0), (1e10, 1e14)])
    #res = dual_annealing(p1h, bounds=[(0.0, 20), (-1.0, 0.0), (1e10, 1e14)])

    return res
