import numpy as np
import healpy as hp
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from glob import glob

matter = glob("Maps/delta_cdm_512_field_fullsky_*.fits")
halos  = glob("Maps/delta_cdm_512_halos_fullsky_*.fits")

for m, h in zip(matter, halos):

    m_map = hp.read_map(m)
    h_map = hp.read_map(h)
    hp.visufunc.cartview(hp.pixelfunc.ud_grade(h_map, 128), min=-1)
    plt.savefig(h.replace('.fits','_cartview.pdf'))
    plt.close()
    hp.mollview(hp.pixelfunc.ud_grade(h_map, 128), min=-1)
    plt.savefig(h.replace('.fits','_mollview.pdf'))
    plt.close()

    hp.visufunc.cartview(hp.pixelfunc.ud_grade(m_map, 128), min=-1)
    plt.savefig(m.replace('.fits','_cartview.pdf'))
    plt.close()
    hp.mollview(hp.pixelfunc.ud_grade(m_map, 128), min=-1)
    plt.savefig(m.replace('.fits','_mollview.pdf'))
    plt.close()

quit()

#print("Pearson correlation coef.: {} and p-value {} for the density map.".format(*pearsonr(matter, halos)) )

#hp.visufunc.cartview(hp.pixelfunc.ud_grade(halos, 128))
#hp.visufunc.cartview(hp.pixelfunc.ud_grade(matter, 128))
#hp.visufunc.cartview(halos, lonra=[-30,30], latra=[-30,30])
#hp.visufunc.cartview(matter, lonra=[-30,30], latra=[-30,30])
#plt.show()

hp.mollview(hp.pixelfunc.ud_grade(halos, 64))
hp.mollview(hp.pixelfunc.ud_grade(matter, 64))
plt.show()

exit()

clm = hp.anafast(matter)
clh = hp.anafast(halos)
clmh = hp.anafast(halos, matter)

plt.loglog(clm)
plt.loglog(clh)
plt.loglog(clmh)
plt.show()

print( "Pearson correlation coef.: {} and p-value {} for the angular Power-Spectrum.".format(*pearsonr(clm, clh)) )
