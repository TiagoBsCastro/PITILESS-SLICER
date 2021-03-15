import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
from PLC import Cl
from glob import glob
from plot_style import *

for i, fname in enumerate(glob("Maps/kappa*")):

    if i == 0:

        kappa = hp.read_map(fname)

    else:

        kappa += hp.read_map(fname)

cls_pin = hp.anafast(kappa)
l_pin   = np.arange(cls_pin.size)

plt.loglog(l_pin, l_pin * (l_pin+1) * cls_pin, label="Pinocchio")

# Lin. Theory
l = np.logspace(0, np.log10(l_pin).max())
plt.loglog(l, l*(l+1)*Cl.Cls(l, 1.0), label="Lin. Theory")
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\ell\,(\ell+1)\times C_{\kappa}(\ell)}$")
plt.legend()
plt.show()
