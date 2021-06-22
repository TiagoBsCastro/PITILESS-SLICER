import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
from PLC import Cl
import params
from glob import glob
from plot_style import *

for name in ["lowres0"]:

  for i, fname in enumerate(glob("Maps/kappa_{}_*".format(name))):

    print(fname)

    if i == 0:

        kappa = hp.read_map(fname, dtype=np.float32)
        mask  = (kappa != hp.UNSEEN)
        norm  = mask.size/mask.sum()

    else:

        kappa[mask] += hp.read_map(fname, dtype=np.float32)[mask]

  cls_pin = hp.anafast(kappa) * norm
  l_pin   = np.arange(cls_pin.size)

  plt.loglog(l_pin, l_pin * (l_pin+1) * cls_pin, label="Pinocchio "+name)
  np.savetxt("Maps/Cls_{}.txt".format(name), np.transpose([l_pin, cls_pin]))

# Lin. Theory
l = np.logspace(0, np.log10(l_pin).max())
plt.loglog(l, l*(l+1)*Cl.Cls(l, params.zsource), label="Lin. Theory")
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\ell\,(\ell+1)\times C_{\kappa}(\ell)}$")
plt.legend()
plt.show()

