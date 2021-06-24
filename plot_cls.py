import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
from PLC import Cl
import params
from glob import glob
from plot_style import *
import pyccl as ccl

# Create new Cosmology object with a given set of parameters. This keeps track
# of previously-computed cosmological functions
cosmo = ccl.Cosmology(
    Omega_c=params.omega0-params.omegabaryon, Omega_b=params.omegabaryon, h=params.h0true/100.0, sigma8=params.sigma8, n_s=0.963)

# Define a simple binned galaxy number density curve as a function of redshift
z_n = np.linspace(0.0, params.zsource, 50)
n = np.zeros(z_n.shape); n[-1] = 1

# Create objects to represent tracers of the weak lensing signal with this
# number density (with has_intrinsic_alignment=False)
lens = ccl.WeakLensingTracer(cosmo, dndz=(z_n, n))

# Calculate the angular cross-spectrum of the two tracers as a function of ell
ell = np.arange(2, 1000)
cls = ccl.angular_cl(cosmo, lens, lens, ell)

for name in ["lowres1"]:

  for i, fname in enumerate(glob("Maps/kappa_{}_*".format(name))):

    print(fname)

    if i == 0:

        kappa = hp.read_map(fname, dtype=np.float32)
        mask  = (kappa != hp.UNSEEN)
        norm  = mask.size/mask.sum()

    else:

        kappa[mask] += hp.read_map(fname, dtype=np.float32)[mask]

  cls_pin = hp.anafast(kappa, lmax=1000) * norm
  l_pin   = np.arange(cls_pin.size)

  plt.loglog(l_pin, l_pin * (l_pin+1) * cls_pin, label="Pinocchio "+name)
  np.savetxt("Maps/Cls_{}.txt".format(name), np.transpose([l_pin, cls_pin]))

# Lin. Theory
l = np.logspace(0, np.log10(l_pin).max())
plt.loglog(l, l*(l+1)*Cl.Cls(l, params.zsource), label="Lin. Theory")
plt.loglog(ell, ell*(ell+1)*cls, label="Halofit")
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\ell\,(\ell+1)\times C_{\kappa}(\ell)}$")
plt.legend()
plt.show()
