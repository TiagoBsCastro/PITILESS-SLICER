import numpy as np
import healpy as hp
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

matter = hp.read_map("Maps/delta_field_fullsky_0.2922.fits")
halos = hp.read_map("Maps/delta_halos_fullsky_0.2922.fits")

print("Pearson correlation coef.: {} and p-value {} for the density map.".format(*pearsonr(matter, halos)) )

hp.mollview(hp.pixelfunc.ud_grade(halos, 128))
hp.mollview(hp.pixelfunc.ud_grade(matter, 128))
plt.show()

clm = hp.anafast(matter)
clh = hp.anafast(halos)

plt.loglog(clm)
plt.loglog(clh)
plt.show()

print( "Pearson correlation coef.: {} and p-value {} for the angular Power-Spectrum.".format(*pearsonr(clm, clh)) )
