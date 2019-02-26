import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import c
from scipy.interpolate import interp1d
import healpy as hp

Omega0 = 0.276
cosmo  = FlatLambdaCDM(H0=100, Om0=Omega0)
cspeed = c.to('km/s').to_value()
hubble = 0.724
zinterp = np.linspace(0.0,2.0, 100)
Dinterp = cosmo.comoving_distance(zinterp).value

lstar = 0.33*np.pi/hp.pixelfunc.nside2resol(hp.pixelfunc.npix2nside(12*2**14))

k, Pk, G, a = np.loadtxt('pinocchio.cdm.cosmology.out',usecols=(12,13,2,0),unpack=True)
k *= hubble
Pk *= hubble**3
Pk = interp1d(k, Pk )
G = interp1d(1.0/a-1, G)

zsource = 1.0
chisource = cosmo.comoving_distance(zsource).value

def lensingKernel (chisource, chidummy):

    return ( (chisource-chidummy)/(chisource * chidummy**2) )**2

z = np.linspace(0.01, zsource, 1000)
chiz = cosmo.comoving_distance(z).to_value()
wz = lensingKernel(chisource, chiz)

Cell = lambda l : (l*(l+1))**2 * np.trapz( wz * Pk( (l+0.5)/chiz ) * G(z)**2, x=chiz )*np.exp(-l/lstar)
ell = np.linspace(1, 512)
Cl = [Cell(l)*(3.0 * cosmo.Om0*cosmo.H0**2/2.0/cspeed**2).to_value() for l in ell]

plt.loglog(ell, Cl)
ell, Cl = np.loadtxt("Maps/Cls_kappa_z1.0.txt",unpack=True, usecols=(0,2))
plt.loglog(ell, Cl)

plt.xlabel(r"$l$")
plt.ylabel(r"$C_{\kappa}(l)$")
plt.show()
