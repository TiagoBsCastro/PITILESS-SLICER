import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d
from astropy.cosmology import z_at_value

################# Spherical Overdensity critical overdensity ###################

# Importing Virial Threshold (_Da) as a Function of the matter density (_oma)
#_oma, _Da   = np.loadtxt('flat.data',unpack=True,usecols=[0,1],comments='#')
# Creating the Interpolator
#interp1d(_oma,_Da,kind='linear')

def VirialDelta (Om):

    x = Om - 1.0
    return (18 * np.pi * np.pi + 82 * x - 39 * x * x) / Om

def deltaC(Om, z=None):
    '''
    Returns the delta_c using Bryan and Norman fit
    Om: Float or Function
        If Float returns the Bryan and Norman for the given value of Om.
        For this behaviour z has to be left untouched (None)
        If a vectorized function z has to be a Float or Array like.
    z:  Float or Array (Optional)
        If not None Om is assumed to be a vectorized function that returns
        the value of the matter density (Omega_m) as a function of z
    '''

    if z is None:
        return 3.0/20.0*(12.0*np.pi)**(2.0/3.0)*( 1.0+0.012299*np.log10( Om ) )
    else:
        return 3.0/20.0*(12.0*np.pi)**(2.0/3.0)*( 1.0+0.012299*np.log10( Om(z) ) )

def zStar(cosmo, halo_def):
    '''
    Returns the redshift where the virial definition coincides with the halo_def.
    If the definitions do not cross the function returns -1.0
    cosmo: Instance of Colossus cosmology like
           object with atribute Om.
    halo_def: String
              The halo definition. Should be a 4 character string ending where
              the first three must be numeric denoting the threshold 'Delta'
              and the last one must be 'c' or ('b|m') respectively for critical and
              mean overdensities.
    '''
    if len(halo_def) !=4:
        raise ValueError('halo_def should be a 4 character string. The passed value is {}'.format(halo_def))
    else:

        # Defining the threshold value and with respect to which energy it has been defined
        Delta_val = int(halo_def[:3])
        if (halo_def[3] == 'b') or (halo_def[3] == 'm'):
            Delta_wrt = lambda z: cosmo.Om(z)
        elif halo_def[3] == 'c':
            Delta_wrt = lambda z: 1.0
        else:
            raise ValueError('The Delta definition could not be interpreted. The passed value is {}'.format(halo_def))

        # Checking if there is a solution in the range z = (0.0, 99.0)
        if (Delta_val * Delta_wrt(99.0) - VirialDelta(cosmo.Om(99.0))) *\
                  (Delta_val * Delta_wrt(0.0) - VirialDelta(cosmo.Om(0.0))) < 0:

            f = lambda x: VirialDelta(x) - Delta_val * Delta_wrt(x)
            x = optimize.root(f, x0=cosmo.Om(1.0))

            return z_at_value(cosmo.Om, x.x)

        else:

            return -1.0

########################## Defining Pert. Integrands ###########################

def w_tophat(k,r):
    '''
    Returns the Fourier Transform of the Spherical Top-Hat function
    with radius r
    '''
    return ( 3.*( np.sin(k * r) - k*r*np.cos(k * r) )/((k * r)**3.))

def dw_tophat_sq_dr (k,r):
    '''
    Returns the r-derivative of Fourier Transform of the Spherical Top-Hat function
    with radius r
    '''
    return (-18.*(k*r*np.cos(k*r) - np.sin(k*r)) * (3.*k*r*np.cos(k*r) + (-3. + (k**2)*(r**2))*np.sin(k*r) ) )/((k**6)*(r**7))

def s_rsq_integrand_given_Dk(k,r,Dk):
    '''
    Returns the integrand of sigma_r^2 given k, r and Dk
    k:  Float or Array
        Fourier frequency
    r:  Float
        Value of the top-hat smoothing
    Dk: Float or Array
        Adimensional matter-power spectrum Dk(k)
    '''
    return ((w_tophat(k,r))**2)*Dk/k

def d_s_rsq_integrand_given_Dk(k,r,Dk):
    '''
    Returns the r-derivative or the integrand of sigma_r^2 given k, r and Dk
    k:  Float or Array
        Fourier frequency
    r:  Float
        Value of the top-hat smoothing
    Dk: Float or Array
        Adimensional matter-power spectrum Dk(k)
    '''
    return (dw_tophat_sq_dr(k,r))*Dk/k

def ssq_given_Dk (m, rhob, k, Dk):
    '''
    Returns the integrand of sigma_r^2 given k, r and Dk using the trapezoidal method
    k:  Array
        Fourier frequency
    m:  Float
        Value of the mass of the Lagrangean patch corresponding to the smoothing radius
    Dk: Array
        Adimensional matter-power spectrum Dk(k)
    '''
    r = m_get_r(m, rhob)
    s_sq = np.trapz( s_rsq_integrand_given_Dk(k,r,Dk), x=k)
    return s_sq

def d_s_given_Dk (m, rhob, k, Dk):
    '''
    Returns the integrand of d sigma_r^2/dr given k, r and Dk using the trapezoidal method
    k:  Array
        Fourier frequency
    m:  Float
        Value of the mass of the Lagrangean patch corresponding to the smoothing radius
    Dk: Array
        Adimensional matter-power spectrum Dk(k)
    '''
    r   = m_get_r(m, rhob)
    d_s = np.trapz( d_s_rsq_integrand_given_Dk(k,r,Dk), x=k)
    return d_s/2.0/np.sqrt(ssq_given_Dk (m, rhob, k, Dk))

def m_get_r (m, rhob):
    '''
    Returns the smoothing radius corresponding to the the mass of the Lagrangean patch.
    m: Float
        Value of the mass of the Lagrangean patch corresponding to the smoothing radius
    rhob: Float
        Matter Density of the Universe
    '''
    return ( 3.0*m/4.0/rhob/np.pi )**(1.0/3)

def dr_dm (m, rhob):
    '''
    Returns the derivative of the smoothing radius corresponding to the the mass
    of the Lagrangean patch.
    m: Float
        Value of the mass of the Lagrangean patch corresponding to the smoothing radius
    rhob: Float
        Matter Density of the Universe
    '''
    return ( ( 3.0/4.0/rhob/np.pi )**(1.0/3) ) * (1.0/3) * m**(-2.0/3)
