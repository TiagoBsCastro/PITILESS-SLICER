import numpy as np
from astropy.coordinates import cartesian_to_spherical
import matplotlib.pyplot as plt
import healpy as hp
import  builder
import cosmology
import params
from randomization import randomizePositions, randomizeVelocities
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if params.nparticles%size:

    print("The data cannot be scattered on {} processes!".format(size))
    exit(-1)

if rank == 0:
    print("Rank 0 is reading the data!")
    from snapshot import ID, V1, V2, V31, V32, Cell, Lbox, NG, Zacc
    print("All done! Let's go!")

comm.barrier()

###################################################

sgn = 1
face = 1
center = [0.5, 0.5, 0.5]

if __name__ == "__main__":

   if rank == 0:

      qPos = np.array([ (ID-1)%NG,((ID-1)//NG)%NG,((ID-1)//NG**2)%NG ]).transpose().astype(float) * Cell + Cell/2.
      qPos = randomizePositions(center, face, sgn, qPos/Lbox)*Lbox
      qPos[:,2] -= Lbox/2.0
      qPos = qPos.astype(np.float32)
      V1  = Cell*randomizeVelocities(face, sgn, V1)
      V2  = Cell*randomizeVelocities(face, sgn, V2)
      V31 = Cell*randomizeVelocities(face, sgn, V31)
      V32 = Cell*randomizeVelocities(face, sgn, V32)

   else:

      qPos = np.empty((params.nparticles, 3), dtype = np.float32)
      V1   = np.empty((params.nparticles, 3), dtype = np.float32)
      V2   = np.empty((params.nparticles, 3), dtype = np.float32)
      V31  = np.empty((params.nparticles, 3), dtype = np.float32)
      V32  = np.empty((params.nparticles, 3), dtype = np.float32)

   qPosslice = np.empty((params.nparticles//size, 3), dtype = np.float32)
   V1slice   = np.empty((params.nparticles//size, 3), dtype = np.float32)
   V2slice   = np.empty((params.nparticles//size, 3), dtype = np.float32)
   V31slice  = np.empty((params.nparticles//size, 3), dtype = np.float32)
   V32slice  = np.empty((params.nparticles//size, 3), dtype = np.float32)

   comm.Scatterv(qPos, qPosslice,root=0)
   comm.Scatterv(V1  ,V1slice   ,root=0)
   comm.Scatterv(V2  ,V2slice   ,root=0)
   comm.Scatterv(V31 ,V31slice  ,root=0)
   comm.Scatterv(V32 ,V32slice  ,root=0)

   del qPos, V1, V2, V31, V32

   norder = 5
   amin = 0.5
   amax = 1.0
   aplcslice = np.empty(params.nparticles//size).astype(np.float32)
   cut  = (cosmology.a >= amin) & (cosmology.a < amax)
   D    = np.polyfit(cosmology.a[cut], cosmology.D[cut], norder)[::-1].astype(np.float32)
   D2   = np.polyfit(cosmology.a[cut], cosmology.D2[cut], norder)[::-1].astype(np.float32)
   D31  = np.polyfit(cosmology.a[cut], cosmology.D31[cut], norder)[::-1].astype(np.float32)
   D32  = np.polyfit(cosmology.a[cut], cosmology.D32[cut], norder)[::-1].astype(np.float32)
   DPLC = np.polyfit(cosmology.ainterp, cosmology.Dinterp, norder)[::-1].astype(np.float32)


   builder.getCrossingScaleParameter (qPosslice, V1slice, V2slice, V31slice, V32slice, aplcslice,
                                          params.nparticles//size, DPLC, D, D2, D31, D32, norder, amin, amax)

   aplc = np.empty(params.nparticles, dtype = np.float32)

   comm.Gatherv(aplcslice, aplc, root=0)

   if rank:

       exit(0)

   quit()

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
