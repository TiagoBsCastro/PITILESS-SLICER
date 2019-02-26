import ReadPinocchio
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from glob import glob
from astropy.coordinates import cartesian_to_spherical

npixels = 12*2**14
npixels_plot = 12*2**14
plc = ReadPinocchio.plc("pinocchio.cdm.plc.out")

cut = (plc.redshift >= 0.135) & (plc.redshift < 0.15)
#pos = plc.pos[cut]
#x, y, z = np.transpose(pos)
#r, theta, phi = cartesian_to_spherical(y,z,x)
#theta = theta.to_value()
#phi = phi.to_value()
theta = plc.theta[cut]
phi = plc.phi[cut]
mass = plc.Mass[cut]
thetatest = plc.theta[cut][plc.name[cut] == 451867]
phitest = plc.phi[cut][plc.name[cut] == 451867]

pixels = hp.ang2pix(hp.npix2nside(npixels),theta*np.pi/180.0+np.pi/2.0,phi*np.pi/180.0)
pixelstest = hp.ang2pix(hp.npix2nside(npixels),thetatest*np.pi/180.0+np.pi/2.0,phitest*np.pi/180.0)
#pixels = hp.ang2pix(hp.npix2nside(npixels),theta + np.pi/2.0, phi)
mapa = np.histogram(pixels,bins=np.linspace(0, npixels_plot, npixels_plot+1))[0]
hdensity = mapa/mapa.mean() - 1.0
hdensity[pixelstest] += 1000000
hp.mollview(hdensity)

#ktab = glob("Maps/*fullsky*")
ktab = ["Maps/delta_field_fullsky_0.14.fits"]
kappa = np.zeros(npixels)
for k in ktab:

   kappa += fits.getdata(k)['T'].flatten()

#hp.mollview(hp.pixelfunc.ud_grade(kappa, hp.npix2nside(npixels_plot)))
hp.mollview(kappa)
plt.show()
