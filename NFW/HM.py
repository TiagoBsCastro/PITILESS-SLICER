'''
Based in Mead et al. 2015
'''
import numpy as np
from scipy.integrate import quad, trapz
import params
from cosmology import colossus, concentration, profile_nfw
from scipy.special import sici

try:

    cvir = concentration.concentration(1E12, 'vir', 0.0, model = 'bullock01')

except:

    raise RuntimeError("Attempt to run HM.py without setting cosmology!")

def W (k, M, z, model = 'bhattacharya13'):
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
    '''
    # Converting k to (kpc/h)^-1
    kinkpc = k/1e3

    conc  = concentration.concentration(M, '200c', z, model = model)
    p_nfw = profile_nfw.NFWProfile(M = M, c = conc, z = z, mdef = '200c')

    # Numerical Integration
    #w = lambda r: np.sinc(kinkpc*r/np.pi) * r**2 * p_nfw.rho(p_nfw.par['rhos'], r/p_nfw.par['rs'])
    #return 4.0 * np.pi * quad(w, 0, conc * p_nfw.par['rs'])[0]

    # Analytical Integration
    si1, ci1 = sici( (1.0+conc)*kinkpc*p_nfw.par['rs'] )
    si2, ci2 = sici( kinkpc*p_nfw.par['rs'] )
    return 4.0 * np.pi * p_nfw.par['rhos'] * p_nfw.par['rs']**3 *\
           ( np.sin( kinkpc*p_nfw.par['rs'] ) * (si1-si2) - np.sinc(conc*kinkpc*p_nfw.par['rs']/np.pi)*conc/(1.0+conc) +\
             np.cos( kinkpc*p_nfw.par['rs'] ) * (ci1-ci2) )

def P1H (k, z, mf, model='bhattacharya13'):
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
    '''

    try:

        integrand = [ dndm*W(k, m, z, model) ** 2 for m, dndm in zip(mf.m, mf.dndm) ]

        return 1.0 / (colossus.current_cosmo.rho_m(0.0) * 1e9)**2 * trapz( np.transpose(integrand), x=mf.m, axis=1)

    except IndexError:

        raise RuntimeError('k should be a numpy array!!')
