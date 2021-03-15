import numpy as np
import params
import cosmology
from scipy.interpolate import interp1d
from scipy.integrate import quad

######################## UTILS ################################

c = 299792.458
fundamental = 2 * np.pi/params.boxsize
cut = (cosmology.k>= fundamental)
Pk0 = interp1d(cosmology.k[cut], cosmology.Pk[cut], fill_value=(0.0, 0.0), bounds_error=False)
D   = interp1d(1/cosmology.a[::-1]-1, cosmology.D[::-1])

z   = np.linspace(0, 5, 1000)
a   = 1.0/(z + 1)

chi = np.array([cosmology.lcdm.comoving_distance(zz).value for zz in z])
ang = np.array([cosmology.lcdm.angular_diameter_distance(zz).value for zz in z])

def integrand (xx, zz, ll):

    return( (np.interp( zz, z, chi ) - xx)/(np.interp( xx, chi, a ) * np.interp( zz, z, chi )) ) ** 2\
            * Pk0(ll/xx) * D(np.interp(xx, chi, z)) 

###############################################################

def Cls (ll, zz):
    '''
    Computes the linear-theory expectation for the convergence angular Power-Spectrum.

    ll: int or array-like (since limber approximation is used floats are accepted as well).
      multipole
    zz: float
      redshift
    '''
    try:

        return 9*100.0**4*params.omega0**2/4.0/c**4 * quad( integrand, 0.0, np.interp(zz, z, chi), args=(zz, ll)  )[0]

    except TypeError:

        return np.array([ 9*100.0**4*params.omega0**2/4.0/c**4 * quad( integrand, 0.0, np.interp(zz, z, chi), args=(zz, l)  )[0] for l in ll])
