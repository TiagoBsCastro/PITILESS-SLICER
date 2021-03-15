import numpy as np
from scipy.integrate import quad, trapz, cumtrapz
import params
from cosmology import colossus, concentration, profile_nfw
from scipy.special import sici
import lmfit 
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

            conc = (A * (M/(10**log10Mpv)) ** B) + C

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

        integrand = mf.dndm * W(k, mf.m, z, model=model, A=A, B=B, C=C, log10Mpv=log10Mpv) ** 2

        return 1.0 / (colossus.current_cosmo.rho_m(0.0) * 1e9)**2 * trapz( integrand, x=mf.m, axis=1)

    except IndexError:

        raise RuntimeError('k should be a numpy array!!')

def fitConc (k, Pk, Pklin, sigma, z, mf0, verbose=True):
    '''
    Fits a generic mass-concentration relation by fitting the one
    halo term.

    k : numpy array of floats
        Fourier Mode in (Mpc/h)^-1
    Pk : numpy array
        Target power-spectrum
    Pklin : numpy array
        Linear Power-Spectrum to be used as the 2ht
    sigma: numpy array
        Error on the Pk
    z : float
        Redshift
    mf0 : Mass Function object
        The observed mass function

    returns:

      sol: solution instance of the method applied
          Best-fit
      log10Mpv: float
          The pivot scale used for the CM relation
    '''
    # Getting the Mass Pivot
    totmass = np.trapz(mf0.dndm * mf0.m**2, x=mf0.m)
    cmf0 = totmass - cumtrapz(mf0.dndm * mf0.m**2, x=mf0.m)
    cmf0 = np.append(cmf0, [0.0])
    f    = lambda x: np.interp(x, mf0.m, cmf0) / totmass - 0.5
    Mpv = fsolve(f, [2 * mf0.m.min() * mf0.m.max() / (mf0.m.min() + mf0.m.max())])
    if ( np.abs(f(Mpv)) >= 1e-3  ):

        print("I could not find a solution for the mass pivot!")
        sys.exit()

    def res (pars):

        A        = pars['A'].value
        B        = pars['B'].value
        C        = pars['C'].value
        log10Mpv = pars['log10Mpv'].value
        return np.sum(( ( P1H (k, z, mf0, model='generic', A=A, B=B, C=C, log10Mpv=log10Mpv) + Pklin - Pk)/sigma )**2)

    # Parameters
    p = lmfit.Parameters()
    p.add_many(('A', 1.0, True, 0.01, 100.0), ('B', -1.0, True, -4.0, 0.0), ('C', 2.5, True, 2.0, 6.0), ('log10Mpv', np.log10(Mpv), False))

    # Minimizing
    mi = lmfit.minimize(res, p, method='ampgo', nan_policy='omit')

    if verbose:
        lmfit.printfuncs.report_fit(mi.params, min_correl=0.5)

    return mi
