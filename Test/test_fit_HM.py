import numpy as np
import cosmology
import matplotlib.pyplot as plt
from IO.Pinocchio.ReadPinocchio import mf, catalog
from NFW import HM
from IO.Utils.fixLowMasses import get_mass_correction
from scipy.optimize import fsolve
import sys

# BoxSize
boxsize = 150.0
# number of snapshots
nsnap = 64
# Redshift

#tgl = np.loadtxt("z=0.00/Pk.txt", skiprows=1)
tgl = np.loadtxt("z=1.98/Pk.txt", skiprows=1)

#for z in [0.0]:
for z in [1.9775]:

    plt.figure(str(z))

    cosmology.hmf_w13.update(z=z)
    cosmology.hmf_t10.update(z=z)

    mf0 = mf("/beegfs/tcastro/TestRuns/pinocchio.{0:5.4f}.example.catalog.out".format(z), "catalog", nsnap=nsnap, boxsize=boxsize)

    Pk     = np.loadtxt("/beegfs/tcastro/Alice/nbody/uhr/Pk_matter_z={0:4.3f}.dat".format(z))
    k      = Pk[:,  0]
    Nk     = Pk[:, -1][k<=2]
    Pk     = Pk[:,  1][k<=2]
    k      = k[k<=2]
    sigma  = np.sqrt( 2.0/Nk + (0.05)**2 ) * Pk
    Pklin  = np.interp(k, cosmology.k, cosmology.Pk) * np.interp(1.0/(1+z), cosmology.a, cosmology.D)**2  

    #sol = HM.fitConc(k, Pk, Pklin, sigma, z, mf0)

    #Pkpin  = HM.P1H(k, z, mf0, model='generic', A=sol.params['A'].value, B=sol.params['B'].value, C=sol.params['C'].value, log10Mpv=sol.params['log10Mpv'].value)
    Pkbhat = HM.P1H(k, z, cosmology.hmf_w13, model='bhattacharya13')
    
    #plt.loglog(k, Pklin + Pkpin, label="Best-Fit")
    plt.loglog(k, Pklin + Pkbhat, label="Bhattacharya + Watson")
    plt.errorbar(k, Pk, sigma, label="Target")
    plt.loglog(k, Pklin, label="Linear")
    if z<1.5:

        plt.loglog(tgl[:, 0], tgl[:, -1], label="Halo Fit")

    plt.loglog(tgl[:, 0], tgl[:, 1], label="TGL")
    plt.xlabel("$k\,[h/Mpc]$")
    plt.ylabel("$P(k)\,[(Mpc/h)^{3}]$")

plt.legend()
plt.show()
