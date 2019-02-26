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
import scipy.optimize as opt

################ Cosmology ###################

Omega0 = 0.276
cosmo  = FlatLambdaCDM(H0=100, Om0=Omega0)
cspeed = c.to('km/s').to_value()

################   PLC   ######################

fovindeg = 60.0
fovinradians = fovindeg * np.pi/180.0
npixels = 12*2**14
zsource = 1.00
deltazplc = 0.05

zlsup = np.linspace(0,zsource,10)[1:]
zlinf = np.linspace(0,zsource,10)[:-1]

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

################################################

def wrapPositions (xx):

   xxoutofbox = (xx < 0.0) | (xx > 1.0)
   xx[xxoutofbox] = np.abs(1.0 - np.abs( xx[xxoutofbox] ))
   del xxoutofbox

   return xx

def snapPos (z, zcentered=True, filter=None):
   '''
   Returns the particles Position at z
   '''
   if filter is None:
      filter=np.ones(Npart).astype(bool)

   thisa   = 1.0/(1.0+z)
   thisD   = np.interp(thisa,a,D)
   thisD2  = np.interp(thisa,a,D2)
   thisD31 = np.interp(thisa,a,D31)
   thisD32 = np.interp(thisa,a,D32)

   xx, yy, zz = np.transpose(qPos[filter] +  Cell * (thisD * V1[filter] + thisD2 * V2[filter] + thisD31 * V31[filter] + thisD32 * V32[filter]))

   if zcentered:

      xx = ( wrapPositions(xx/Lbox +0.5 ) -0.5 )*Lbox; yy = ( wrapPositions(yy/Lbox +0.5 ) -0.5 )*Lbox; zz = wrapPositions(zz/Lbox)*Lbox;

   else:

      xx = ( wrapPositions(xx/Lbox +0.5 ) -0.5 )*Lbox; yy = ( wrapPositions(yy/Lbox +0.5 ) -0.5 )*Lbox; zz = (wrapPositions(zz/Lbox + 0.5) -0.5 )*Lbox;

   return xx, yy, zz

def snapPos_1part (pos, z, zcentered=True):
   '''
   Returns the particles Position at z
   '''

   thisa   = 1.0/(1.0+z)
   thisD   = np.interp(thisa,a,D)
   thisD2  = np.interp(thisa,a,D2)
   thisD31 = np.interp(thisa,a,D31)
   thisD32 = np.interp(thisa,a,D32)

   xx, yy, zz = qPos[pos] +  Cell * (thisD * V1[pos] + thisD2 * V2[pos] + thisD31 * V31[pos] + thisD32 * V32[pos])

   if zcentered:

      xx = ( wrapPositions(xx/Lbox +0.5 ) -0.5 )*Lbox; yy = ( wrapPositions(yy/Lbox +0.5 ) -0.5 )*Lbox; zz = wrapPositions(zz/Lbox)*Lbox;

   else:

     xx = ( wrapPositions(xx/Lbox +0.5 ) -0.5 )*Lbox; yy = ( wrapPositions(yy/Lbox +0.5 ) -0.5 )*Lbox; zz = (wrapPositions(zz/Lbox + 0.5) -0.5 )*Lbox;

   return xx, yy, zz

############### Randomization #####################

#center = np.random.uniform(0,1,3)
#face = int(np.ceil(np.random.uniform(0,6)))
#sgn = (2*( np.ceil(np.random.uniform(-1,1,3))-0.5 )).astype(int)
center = [0.0,0.0,0.0]
face = 1
sgn  = [1, 1, 1]

def randomizePositions (center, face, sgn, pos):

   '''Randomize the positions acording to the SLICER
      random variables center, face, and sgn'''

   xb, yb, zb = (sgn*pos).T
   xb = wrapPositions(xb); yb = wrapPositions(yb); zb = wrapPositions(zb);

   if face == 1:
      xx = xb
      yy = yb
      zz = zb
   elif face == 2:
      xx = xb
      yy = zb
      zz = yb
   elif face == 3:
      xx = yb
      yy = zb
      zz = xb
   elif face == 4:
      xx = yb
      yy = xb
      zz = zb
   elif face == 5:
      xx = zb
      yy = xb
      zz = yb
   elif face == 6:
      xx = zb
      yy = yb
      zz = xb

   del xb, yb, zb

   xx -= center[0]; yy -= center[1]; zz -= center[2];
   xx = wrapPositions(xx); yy = wrapPositions(yy); zz = wrapPositions(zz);

   return np.transpose([xx - 0.5, yy - 0.5, zz])

def randomizeVelocities (face, sgn, vel):

   '''Randomize velocities acording to the SLICER
      random variables center, face, and sgn'''

   xb, yb, zb = (vel*sgn).T

   if face == 1:
      xx = xb
      yy = yb
      zz = zb
   elif face == 2:
      xx = xb
      yy = zb
      zz = yb
   elif face == 3:
      xx = yb
      yy = zb
      zz = xb
   elif face == 4:
      xx = yb
      yy = xb
      zz = zb
   elif face == 5:
      xx = zb
      yy = xb
      zz = yb
   elif face == 6:
      xx = zb
      yy = yb
      zz = xb

   return np.transpose([xx, yy, zz])


def tofind_crossingz(pos,z,zc=True):

   xx, yy, zz=snapPos_1part(pos,z,zcentered=zc)
   zcross=np.interp(np.sqrt(xx**2 + yy**2 +zz**2), Dinterp, zinterp)
   return zcross-z


def buildPLC_finer (zmin, zmax, zcentered=True):
   '''
   Builds the Past Light Cone from z in [zmin,zmax]
   Returns zplc, and the comoving [r, theta, phi]
   '''

   zinterp = np.linspace(zmin,1.1*zmax, 100)
   Dinterp = cosmo.comoving_distance(zinterp).value

   crossedinthepast = np.zeros(Npart).astype(bool)
   sphericalcoord = np.empty((Npart,3))
   zplc = np.empty(Npart)
   pointer=np.arange(Npart)

   for zet in np.linspace(zmin, zmax, np.floor( (zmax-zmin)/deltazplc )):

      notcrossed=~crossedinthepast
      xx, yy, zz = snapPos(zet, zcentered, notcrossed)
      comovdistance = np.sqrt(xx**2 + yy**2 +zz**2)
      ztrial = np.interp(comovdistance, Dinterp, zinterp)
      justcrossed=ztrial<=zet+deltazplc
      for pos  in pointer[notcrossed][justcrossed]:
         zplc[pos]=brentq(tofind_crossingz, zet, zet+deltazplc)
         if zcentered:
            sphericalcoord[pos] = np.transpose( cartesian_to_spherical(xx[pos], yy[pos], zz[pos]) )
         else:
            sphericalcoord[pos] = np.transpose( cartesian_to_spherical(yy[pos], zz[pos], xx[pos]) )
         crossedinthepast[pos]=True

   zplc[~crossedinthepast] = np.inf
   sphericalcoord[~crossedinthepast] = np.nan

   return zplc, sphericalcoord

def buildPLC (zmin, zmax, repetition, zcentered=True):
   '''
   Builds the Past Light Cone from z in [zmin,zmax]
   Returns zplc, and the comoving [r, theta, phi]
   '''

   zinterp = np.linspace(zmin,1.1*zmax, 100)
   Dinterp = cosmo.comoving_distance(zinterp).value

   crossedinthepast = np.zeros(Npart).astype(bool)
   sphericalcoord = np.empty((Npart,3))
   zplc = np.empty(Npart)

   zettab = np.linspace(zmin, zmax, int( (zmax-zmin)/deltazplc ))
   for zinf, zsup in zip(zettab[:-1],zettab[1:]):

      xx, yy, zz = snapPos(zsup, zcentered=zcentered, filter = ~crossedinthepast)
      xx += repetition[0]*Lbox; yy += repetition[1]*Lbox; zz += repetition[2]*Lbox;
      comovdistance = np.sqrt(xx**2 + yy**2 +zz**2)
      ztrial = np.interp(comovdistance, Dinterp, zinterp)
      crossednow = (ztrial > zinf) & (ztrial <= zsup)
      crossed = np.nonzero(~crossedinthepast)[0][np.nonzero(crossednow)[0]]
      if not zcentered:
         theta = -np.arccos(zz[crossednow]/comovdistance[crossednow]) + np.pi/2.0;
         phi   = np.arctan2(yy[crossednow],xx[crossednow]);
         sphericalcoord[crossed] = np.transpose((comovdistance[crossednow],theta,phi))
      else:
         sphericalcoord[crossed] = np.transpose( cartesian_to_spherical(yy[crossednow], zz[crossednow], xx[crossednow]) )

      zplc[crossed] = ztrial[crossednow]
      crossedinthepast[ crossed ] = True

   zplc[~crossedinthepast] = np.inf
   sphericalcoord[~crossedinthepast] = np.nan

   return zplc, sphericalcoord

###################################################

if __name__ == "__main__":

   qPos = np.array([ (ID-1)%NG,((ID-1)//NG)%NG,((ID-1)//NG**2)%NG ]).transpose().astype(float) * Cell + Cell/2.
   qPos = randomizePositions(center, face, sgn, qPos/Lbox)*Lbox
   qPos[:,2] -= Lbox/2.0
   V1  = randomizeVelocities(face, sgn, V1)
   V2  = randomizeVelocities(face, sgn, V2)
   V31 = randomizeVelocities(face, sgn, V31)
   V32 = randomizeVelocities(face, sgn, V32)

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
   '''
   print("FoV:{} Center:{} Face:{} Sgn:{}".format(fovindeg, center, face, sgn))

   # comoving Mpc
   qPos = np.array([ (ID-1)%NG,((ID-1)//NG)%NG,((ID-1)//NG**2)%NG ]).transpose().astype(float) * Cell + Cell/2.
   qPos = randomizePositions(center, face, sgn, qPos/Lbox)*Lbox
   qPos[:,2] -= Lbox/2.0
   V1  = randomizeVelocities(face, sgn, V1)
   V2  = randomizeVelocities(face, sgn, V2)
   V31 = randomizeVelocities(face, sgn, V31)
   V32 = randomizeVelocities(face, sgn, V32)
   #dist_aux = qPos[:,0]**2 + qPos[:,1]**2 + qPos[:,2]**2
   #argsort = np.argsort(dist_aux)
   #qPos = qPos[argsort]
   #V1  = randomizeVelocities(face, sgn,  V1[argsort])
   #V2  = randomizeVelocities(face, sgn,  V2[argsort])
   #V31 = randomizeVelocities(face, sgn, V31[argsort])
   #V32 = randomizeVelocities(face, sgn, V32[argsort])
   #print("Initial conditions randomized and sorted")

   zplc, sphericalcoord = buildPLC(0.0, zsource, zcentered=False)
   cut = ~np.isnan(sphericalcoord[:,1]) & (zplc >  Zacc)
   theta, phi, zplc = sphericalcoord[:,1][cut] + np.pi/2.0, sphericalcoord[:,2][cut], zplc[cut]
   pixels = hp.pixelfunc.ang2pix(hp.pixelfunc.npix2nside(npixels), theta, phi)
   del sphericalcoord, theta, phi

   for i,(z1,z2) in enumerate( zip(zlinf, zlsup) ):

      lensi = (zplc <= z2) & (zplc > z1)
      pixelsi, zplci = pixels[lensi], zplc[lensi]
      zl = (z1+z2)/2.0
      deltai = np.histogram(pixelsi, bins=np.linspace(0,npixels,npixels+1).astype(int))[0]
      deltai = deltai/deltai.mean() - 1.0
      kappai = (1.0+zl) * ( ( 1.0 - cosmo.comoving_distance(zl)/cosmo.comoving_distance(zsource) ) *\
                              cosmo.comoving_distance(zl) * ( cosmo.comoving_distance(z2) - cosmo.comoving_distance(z1) ) ).to_value() * deltai
      kappai *= (3.0 * cosmo.Om0*cosmo.H0**2/2.0/cspeed**2).to_value()
      deltai[pixelstest] += 1000
      hp.fitsfunc.write_map('Maps/delta_field_fullsky_{}.fits'.format(str(round(zl,2))), deltai, overwrite=True)
      hp.fitsfunc.write_map('Maps/kappa_field_fullsky_{}.fits'.format(str(round(zl,2))), kappai, overwrite=True)

   # comoving Mpc
   qPos = np.array([ (ID-1)%NG,((ID-1)//NG)%NG,((ID-1)//NG**2)%NG ]).transpose().astype(float) * Cell + Cell/2.
   qPos = randomizePositions(center, face, sgn, qPos/Lbox)*Lbox
   V1 = randomizeVelocities(face, sgn, V1)
   V2 = randomizeVelocities(face, sgn, V2)
   V31 = randomizeVelocities(face, sgn, V31)
   V32 = randomizeVelocities(face, sgn, V32)

   zplc, sphericalcoord = buildPLC(0.0,zsource)
   cut = ~np.isnan(sphericalcoord[:,1]) & ( np.abs(sphericalcoord[:,1]) <= fovinradians/2.0 ) & (np.abs(sphericalcoord[:,2] - np.pi/2.0) <= fovinradians/2.0) & (zplc < Zacc)

   theta, phi, zplc = sphericalcoord[:,1][cut], sphericalcoord[:,2][cut], zplc[cut]

   for i,(z1,z2) in enumerate( zip(zlinf, zlsup) ):

      lensi = (zplc <= z2) & (zplc > z1)
      thetai, phii, zplci = theta[lensi], phi[lensi], zplc[lensi]
      deltai = np.histogram2d(thetai, phii, bins = npixels, range = [[-fovinradians/2.0,fovinradians/2.0],[np.pi/2.0-fovinradians/2.0, np.pi/2.0+fovinradians/2.0]])[0]
      deltai = deltai/deltai.mean() - 1.0
      zl = (z1+z2)/2.0
      kappai = (1+zl) * ( ( 1.0 - cosmo.comoving_distance(zl)/cosmo.comoving_distance(zsource) ) *\
                              cosmo.comoving_distance(zl) * ( cosmo.comoving_distance(z2) - cosmo.comoving_distance(z1) ) ).to_value() * deltai
      kappai *= (3.0 * cosmo.Om0*cosmo.H0**2/2.0/cspeed).to_value()
      hdu = fits.PrimaryHDU(kappai)
      hdu.writeto('kappa_field_{}.fits'.format(str(round(zl,2))))
   '''
