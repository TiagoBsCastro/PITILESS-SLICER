import numpy as np
from astropy.coordinates import cartesian_to_spherical
import matplotlib.pyplot as plt
import healpy as hp
import  builder
import cosmology
import params
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if params.nparticles%size:

    print("The data cannot be scattered on {} processes!".format(size))
    exit(-1)

if rank == 0:
    print("Rank 0 is reading the data!")
    from snapshot import qPos, V1, V2, V31, V32, Cell, Lbox, NG, Zacc
    print("All done! Let's go!")

comm.Barrier()

###################################################

center = [0.5, 0.5, 0.5]
face = 1
sgn = [1, 1, 1]

if __name__ == "__main__":

   if rank:

      qPos = None
      V1   = None
      V2   = None
      V31  = None
      V32  = None
      Zacc = None

   qPosslice = np.empty((params.nparticles//size,3), dtype = np.float32)
   V1slice   = np.empty((params.nparticles//size,3), dtype = np.float32)
   V2slice   = np.empty((params.nparticles//size,3), dtype = np.float32)
   V31slice  = np.empty((params.nparticles//size,3), dtype = np.float32)
   V32slice  = np.empty((params.nparticles//size,3), dtype = np.float32)
   Zaccslice = np.empty(params.nparticles//size, dtype = np.float32)
   aplcslice = np.empty(params.nparticles//size, dtype = np.float32)
   skycoordslice = np.empty((params.nparticles//size, 3), dtype = np.float32)

   comm.Scatterv(qPos, qPosslice,root=0)
   comm.Scatterv(V1  ,V1slice   ,root=0)
   comm.Scatterv(V2  ,V2slice   ,root=0)
   comm.Scatterv(V31 ,V31slice  ,root=0)
   comm.Scatterv(V32 ,V32slice  ,root=0)
   comm.Scatterv(Zacc ,Zaccslice,root=0)

   del qPos, V1, V2, V31, V32, Zacc

   geometry = np.loadtxt("pinocchio.cdm.geometry.out",usecols=(1,2,3,4,5),dtype={'names':['x','y','z','nearestpoint','farthestpoint'],'formats':['f8','f8','f8','f8','f8']})
   geometry['nearestpoint'][(geometry['x']==0) & (geometry['y']==0) & (geometry['z']==0)] = 0.0
   geometry['nearestpoint'] *= Cell
   geometry['farthestpoint'] *= Cell
   kappa = np.zeros(params.npixels)

   print("\n++++++++++++++++++++++\n")
   for i,(z1,z2) in enumerate( zip(cosmology.zlinf, cosmology.zlsup) ):

      print("Lens plane from z=[{0:.3f},{1:.3f}]".format(z1,z2))
      dlinf = cosmology.lcdm.comoving_distance(z1).to_value()
      dlsup = cosmology.lcdm.comoving_distance(z2).to_value()
      zl = (z1 + z2)/2.0
      amin = 1.0/(1.0+z2)
      amax = 1.0/(1.0+z1)

      auxcut  = (cosmology.a >= amin) & (cosmology.a <= amax)
      D       = cosmology.getWisePolyFit(cosmology.a[auxcut], cosmology.D[auxcut])
      D2      = cosmology.getWisePolyFit(cosmology.a[auxcut], cosmology.D2[auxcut])
      D31     = cosmology.getWisePolyFit(cosmology.a[auxcut], cosmology.D31[auxcut])
      D32     = cosmology.getWisePolyFit(cosmology.a[auxcut], cosmology.D32[auxcut])
      auxcut  = (cosmology.ainterp >= amin) & (cosmology.ainterp <= amax)
      DPLC    = cosmology.getWisePolyFit(cosmology.ainterp[auxcut], cosmology.Dinterp[auxcut])

      replicationsinside = geometry[ (geometry['nearestpoint'] < dlsup) & (geometry['farthestpoint'] >= dlinf) ]
      deltai = np.zeros(params.npixels)

      print(" Replications inside:")
      for repi in replicationsinside:

         print(" * {} {} {} ".format(*repi[['x','y','z']]))

         shift = params.boxsize * np.array(repi[['x','y','z']].tolist())
         builder.getCrossingScaleParameterNewtonRaphson (qPosslice + shift.astype(np.float32), V1slice, V2slice, V31slice, V32slice, aplcslice,
                                                             params.nparticles//size, DPLC, D, D2, D31, D32, params.norder, amin, amax)


         aplcslice[ 1.0/aplcslice -1  < Zaccslice ] = -1.0
         builder.getSkyCoordinates(qPosslice + shift.astype(np.float32), V1slice, V2slice, V31slice, V32slice, aplcslice, skycoordslice, params.nparticles//size, D, D2, D31, D32, params.norder)

         if rank:
             zplc     = None
             skycoord = None
         else:
             zplc     = np.empty(params.nparticles, dtype = np.float32)
             skycoord = np.empty((params.nparticles,3), dtype = np.float32)

         comm.Gatherv(1.0/aplcslice -1, zplc, root=0)
         comm.Gatherv(skycoordslice, skycoord, root=0)

         if rank == 0:

            cut = skycoord[:,0] > 0
            theta, phi = skycoord[:,1][cut] + np.pi/2.0, skycoord[:,2][cut]
            pixels = hp.pixelfunc.ang2pix(hp.pixelfunc.npix2nside(params.npixels), theta, phi)
            deltai += np.histogram(pixels, bins=np.linspace(0,params.npixels,params.npixels+1).astype(int))[0]

         comm.Barrier()

      if rank == 0:

         groupsinplane = (cosmology.plc.redshift <= z2) & (cosmology.plc.redshift > z1)
         pixels = hp.pixelfunc.ang2pix(hp.pixelfunc.npix2nside(params.npixels), \
                                     cosmology.plc.theta[groupsinplane]*np.pi/180.0+np.pi/2.0, cosmology.plc.phi[groupsinplane]*np.pi/180.0)
         deltai += np.histogram(pixels, bins=np.linspace(0,params.npixels,params.npixels+1).astype(int), weights=cosmology.plc.Mass[groupsinplane])[0]
         deltai = deltai/deltai.mean() - 1.0
         kappai = (1.0+zl) * ( ( 1.0 - cosmology.lcdm.comoving_distance(zl)/cosmology.lcdm.comoving_distance(params.zsource) ) *\
                                 cosmology.lcdm.comoving_distance(zl) * ( cosmology.lcdm.comoving_distance(z2) - cosmology.lcdm.comoving_distance(z1) ) ).to_value() * deltai
         kappai *= (3.0 * cosmology.lcdm.Om0*cosmology.lcdm.H0**2/2.0/cosmology.cspeed**2).to_value()
         hp.fitsfunc.write_map('Maps/delta_field_fullsky_{}.fits'.format(str(round(zl,4))), deltai, overwrite=True)
         hp.fitsfunc.write_map('Maps/kappa_field_fullsky_{}.fits'.format(str(round(zl,4))), kappai, overwrite=True)
         kappa += kappai

         print("\n++++++++++++++++++++++\n")

      comm.Barrier()

   if rank:

       exit(0)

   hp.mollview(kappa)
   plt.show()

   cl = hp.anafast(kappa, lmax=512)
   ell = np.arange(len(cl))
   np.savetxt("Maps/Cls_kappa_z{}.txt".format(params.zsource), np.transpose([ell, cl, ell * (ell+1) * cl]))
