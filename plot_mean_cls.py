import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
from PLC import Cl
import params
from glob import glob
from plot_style import *

for j, name in enumerate(["lowres0", "lowres1", "lowres2", "lowres3", "lowres4", "lowres5"]):

  for i, fname in enumerate(glob("Maps/kappa_{}_*".format(name))):

    if i == 0:

        kappa = hp.read_map(fname, dtype=np.float32)

    else:

        kappa += hp.read_map(fname, dtype=np.float32)

  cls_pin_j = hp.anafast(kappa)
  l_pin     = np.arange(cls_pin_j.size)

  np.savetxt("Maps/Cls_{}.txt".format(name), np.transpose([l_pin, cls_pin_j]))

  if j == 0:

    cls_pin    = np.copy(cls_pin_j)
    cls_pin_sq = cls_pin_j**2

  else:

    cls_pin    += cls_pin_j
    cls_pin_sq += cls_pin_j**2

cls_pin    /= (j+1)
cls_pin_err = np.sqrt(cls_pin_sq/(j+1) - cls_pin**2)
plt.errorbar(l_pin, l_pin * (l_pin+1) * cls_pin, l_pin * (l_pin+1) * cls_pin_err, label="Pinocchio mean")
# Lin. Theory
l = np.logspace(0, np.log10(l_pin).max())
plt.loglog(l, l*(l+1)*Cl.Cls(l, params.zsource), label="Lin. Theory")
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\ell\,(\ell+1)\times C_{\kappa}(\ell)}$")
plt.legend()
plt.show()
