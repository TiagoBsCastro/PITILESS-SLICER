import numpy as np
import matplotlib.pyplot as plt
import Snapshot as S
import ReadPinocchio as rp
import copy
import os
import healpy as hp
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy.coordinates import cartesian_to_spherical
from astropy.io import fits
from astropy.constants import c
from scipy import optimize as opt

################ Cosmology ###################

cosmo  = FlatLambdaCDM(H0=H0, Om0=Omega0)
cspeed = c.to('km/s').to_value()

################   PLC   ######################

zlsup = np.linspace(0,zsource,10)[1:]
zlinf = np.linspace(0,zsource,10)[:-1]

zinterp = np.linspace(0, zsource + deltazplc, int(zsource//deltazplc) + 1 )
ainterp = 1.0/(1.0+zinterp)
Dinterp = cosmo.comoving_distance(zinterp).value

########### Timeless Snapshot #################

snap=S.Init("pinocchio.cdm.t_snapshot.out",-1)
ID=snap.read_block('ID')
V1=snap.read_block('VZEL')
V2=snap.read_block('V2')
V31=snap.read_block('V3_1')
V32=snap.read_block('V3_2')
F=snap.read_block('FMAX')
R=snap.read_block('RMAX')
Zacc=snap.read_block('ZACC')
Npart=len(ID)
NG=np.int(np.float(Npart)**(1./3.)+0.5)
Lbox=snap.Header.boxsize
Cell=Lbox/float(NG)

############### Cosmology ######################

(a, t, D, D2, D31, D32) = np.loadtxt('pinocchio.cdm.cosmology.out',usecols=(0,1,2,3,4,5),unpack=True)

#################### PLC #######################

plc = rp.plc('pinocchio.cdm.plc.out')
MinHaloMass = 10
plc.Mass = (plc.Mass/plc.Mass.min()*MinHaloMass).astype(int)

############### Randomization ##################

#center = np.random.uniform(0,1,3)
#face = int(np.ceil(np.random.uniform(0,6)))
#sgn = (2*( np.ceil(np.random.uniform(-1,1,3))-0.5 )).astype(int)
center = [0.0,0.0,0.0]
face = 1
sgn  = [1, 1, 1]


###################################################

if __name__ == "__main__":

   qPos = np.array([ (ID-1)%NG,((ID-1)//NG)%NG,((ID-1)//NG**2)%NG ]).transpose().astype(float) * Cell + Cell/2.
   qPos = randomizePositions(center, face, sgn, qPos/Lbox)*Lbox
   qPos[:,2] -= Lbox/2.0
   V1  = randomizeVelocities(face, sgn, V1)
   V2  = randomizeVelocities(face, sgn, V2)
   V31 = randomizeVelocities(face, sgn, V31)
   V32 = randomizeVelocities(face, sgn, V32)

   zplc, sphericalcoord = buildPLCFiner (0.0, zsource, zcentered=False)
   cut = ~np.isnan(sphericalcoord[:,1]) & (zplc >  Zacc)

   geometry = np.loadtxt("pinocchio.cdm.geometry.out",usecols=(1,2,3,4,5),dtype={'names':('x','y','z','nearestpoint','farthestpoint'),'formats':('i8','i8','i8','f8','f8')})
   geometry['nearestpoint'][(geometry['x']==0) & (geometry['y']==0) & (geometry['z']==0)] = 0.0
   geometry['nearestpoint'] *= Cell
   geometry['farthestpoint'] *= Cell
   kappa = np.zeros(npixels)

   print("\n++++++++++++++++++++++\n")
   for i,(z1,z2) in enumerate( zip(zlinf, zlsup) ):

       print("Lens plane from z=[{0:.3f},{1:.3f}]".format(z1,z2))
       dlinf = cosmo.comoving_distance(z1).to_value()
       dlsup = cosmo.comoving_distance(z2).to_value()
       zl = (z1 + z2)/2.0
       replicationsinside = geometry[ (geometry['nearestpoint'] < dlsup) & (geometry['farthestpoint'] >= dlinf) ]

       deltai = np.zeros(npixels)
       print(" Replications inside:")
       for repi in replicationsinside:

          print(" * {} {} {}".format(*repi[['x','y','z']]))
          zplc, sphericalcoord = buildPLC(z1, z2, repi[['x','y','z']], zcentered=False)
          cut = ~np.isnan(sphericalcoord[:,1]) & (zplc >  Zacc)
          theta, phi, zplc = sphericalcoord[:,1][cut] + np.pi/2.0, sphericalcoord[:,2][cut], zplc[cut]
          pixels = hp.pixelfunc.ang2pix(hp.pixelfunc.npix2nside(npixels), theta, phi)
          del sphericalcoord, theta, phi
          deltai += np.histogram(pixels, bins=np.linspace(0,npixels,npixels+1).astype(int))[0]

       groupsinplane = (plc.redshift <= z2) & (plc.redshift > z1)
       pixels = hp.pixelfunc.ang2pix(hp.pixelfunc.npix2nside(npixels), plc.theta[groupsinplane]*np.pi/180.0+np.pi/2.0, plc.phi[groupsinplane]*np.pi/180.0)
       deltai += np.histogram(pixels, bins=np.linspace(0,npixels,npixels+1).astype(int), weights=plc.Mass[groupsinplane])[0]
       deltai = deltai/deltai.mean() - 1.0

       kappai = (1.0+zl) * ( ( 1.0 - cosmo.comoving_distance(zl)/cosmo.comoving_distance(zsource) ) *\
                                  cosmo.comoving_distance(zl) * ( cosmo.comoving_distance(z2) - cosmo.comoving_distance(z1) ) ).to_value() * deltai
       kappai *= (3.0 * cosmo.Om0*cosmo.H0**2/2.0/cspeed**2).to_value()
       hp.fitsfunc.write_map('Maps/delta_field_fullsky_{}.fits'.format(str(round(zl,4))), deltai, overwrite=True)
       hp.fitsfunc.write_map('Maps/kappa_field_fullsky_{}.fits'.format(str(round(zl,4))), kappai, overwrite=True)
       kappa += kappai

       print("\n++++++++++++++++++++++\n")

   hp.mollview(kappa)
   plt.show()

   cl = hp.anafast(kappa, lmax=512)
   ell = np.arange(len(cl))
   np.savetxt("Maps/Cls_kappa_z{}.txt".format(zsource), np.transpose([ell, cl, ell * (ell+1) * cl]))
