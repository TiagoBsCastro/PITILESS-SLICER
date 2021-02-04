import numpy as np
from IO.Pinocchio.ReadPinocchio import catalog
from IO.Utils.fixLowMasses import get_mass_correction
from IO.Pinocchio.ReadPinocchio import mf
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Box Size
boxsize = 150.0
# Minimum number of particles for halos
nmin    = 10
# Mass threshold for mass_correction
mthreshold = 1e12

aux = mf("../TestRuns/pinocchio.1.9775.example.mf.out", "mf")
mf0 = mf("../TestRuns/pinocchio.1.9775.example.catalog.out", "catalog", 64, boxsize)
mf0.dndm_teo   = np.interp(mf0.m, aux.m, aux.dndm_teo)
mf0.dndlnm_teo = np.interp(mf0.m, aux.m, aux.dndlnm_teo)
cat = []
m_orig   = mf0.m
dndm_teo = mf0.dndm_teo
for i in range(64):

    cat   += catalog("../TestRuns/pinocchio.1.9775.example.catalog.out.{}".format(i)).Mass.tolist()

cat   = np.sort(cat)
aux_bins, m_corr  = get_mass_correction(cat, mf0, mthreshold, linear=False)
for i in range(10):

    try:

        m_bins  = np.geomspace(m_corr.min(), m_corr.max(), 51);
        m_bins[-1] = m_corr.max()
        delta_m = m_bins[1:]-m_bins[:-1]
        hist, m_bins = np.histogram(m_corr, bins=m_bins)
        m_weighted, bins = np.histogram(m_corr, bins=m_bins, weights=m_corr)
        m_weighted[hist>0]  = m_weighted[hist>0]/hist[hist>0]
        m_weighted[hist==0] = ((m_bins[1:]+m_bins[:-1])/2)[hist==0]
        dndm = hist/delta_m/boxsize**3

        mf0.dndm_teo = [np.interp(m, m_orig, dndm_teo) for m in m_weighted]
        mf0.dndm = dndm
        mf0.m = m_weighted
        mf0.dndlnm_teo = mf0.dndm_teo/m_weighted
        mf0.dndlnm = mf0.dndm/m_weighted

        aux_bins, m_corr  = get_mass_correction(m_corr, mf0, mthreshold, linear=False)

    except ValueError:

        continue

m_bins  = np.geomspace(m_corr.min(), m_corr.max(), mf0.m.size + 1)
delta_m = m_bins[1:]-m_bins[:-1]
hist, bins = np.histogram(m_corr, bins=m_bins)
m_weighted, bins = np.histogram(m_corr, bins=m_bins, weights=m_corr)

m_weighted = m_weighted/hist
dndm = hist/delta_m/boxsize**3

plt.loglog(m_weighted, dndm/m_weighted, label='dn/dlnm corr.')
plt.loglog(mf0.m, mf0.dndlnm, label='dn/dm Pin.')
plt.loglog(mf0.m, mf0.dndlnm_teo, label='dn/dm Teo.')
plt.legend()
plt.show()

plt.plot(m_weighted, dndm/m_weighted/np.interp(m_weighted, mf0.m, mf0.dndlnm_teo), label='dn/dlnm corr.')
plt.plot(mf0.m, mf0.dndlnm/mf0.dndlnm_teo, label='dn/dlnm Pin.')
plt.axhline(1.0)
plt.ylim([0.5, 1.05])
plt.xscale('log')
plt.xlim([mf0.m[0], mthreshold])
plt.legend()
plt.show()
