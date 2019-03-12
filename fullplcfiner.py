import numpy as np
from astropy.coordinates import cartesian_to_spherical
import matplotlib.pyplot as plt
import healpy as hp
import  builder
import cosmology
import params
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if params.nparticles%size:

    print("The data cannot be scattered on {} processes!".format(size))
    exit(-1)

if rank == 0:
    print("Rank 0 is reading the data!")
    from snapshot import qPos, ID, V1, V2, V31, V32, Cell, Lbox, NG, Zacc
    print("All done! Let's go!")

    try:
        os.mkdir("Maps/")
    except FileExistsError:
        pass

else:
    Cell = None

Cell = comm.bcast(Cell, root=0)

###################################################

if __name__ == "__main__":

   if rank:

      qPos = None
      V1   = None
      V2   = None
      V31  = None
      V32  = None
      Zacc = None

   qPosslice     = np.empty((params.nparticles//size,3), dtype = np.float32)
   V1slice       = np.empty((params.nparticles//size,3), dtype = np.float32)
   V2slice       = np.empty((params.nparticles//size,3), dtype = np.float32)
   V31slice      = np.empty((params.nparticles//size,3), dtype = np.float32)
   V32slice      = np.empty((params.nparticles//size,3), dtype = np.float32)
   Zaccslice     = np.empty(params.nparticles//size, dtype = np.float32)
   aplcslice     = np.empty(params.nparticles//size, dtype = np.float32)
   skycoordslice = np.empty((params.nparticles//size, 3), dtype = np.float32)

   comm.Scatterv(qPos , qPosslice ,root=0)
   comm.Scatterv(V1   , V1slice   ,root=0)
   comm.Scatterv(V2   , V2slice   ,root=0)
   comm.Scatterv(V31  , V31slice  ,root=0)
   comm.Scatterv(V32  , V32slice  ,root=0)
   comm.Scatterv(Zacc , Zaccslice ,root=0)

   del qPos, V1, V2, V31, V32, Zacc

   geometry = np.loadtxt("pinocchio.cdm.geometry.out",usecols=(1,2,3,4,5),dtype={'names':['x','y','z','nearestpoint','farthestpoint'],'formats':['f8','f8','f8','f8','f8']})
   geometry['nearestpoint'][(geometry['x']==0) & (geometry['y']==0) & (geometry['z']==0)] = 0.0
   geometry['nearestpoint'] *= Cell
   geometry['farthestpoint'] *= Cell

   if not rank:
      print("\n++++++++++++++++++++++\n")
      kappa = np.zeros(params.npixels)
   for i,(z1,z2) in enumerate( zip(cosmology.zlinf, cosmology.zlsup) ):

      if not rank:
         print("Lens plane from z=[{0:.3f},{1:.3f}]".format(z1,z2))
      dlinf = cosmology.lcdm.comoving_distance(z1).to_value()
      dlsup = cosmology.lcdm.comoving_distance(z2).to_value()
      zl = (z1 + z2)/2.0
      amin = 1.0/(1.0+z2)
      amax = 1.0/(1.0+z1)

      auxcut  = (cosmology.a >= amin) & (cosmology.a <= amax)
      if not rank:
         print( "Number of points inside the redshift range is {} values!".format(np.sum(auxcut) ) )
      if auxcut.sum() == 1:
          index = auxcut.argmax()
          if index < auxcut.size:
              auxcut[index + 1] = True
          if index > 0:
              auxcut[index - 1] = True
      if auxcut.sum() == 0:
          index = ( (cosmology.a - (amax-amin)/2.0)**2 ).argmin()
          if index < auxcut.size:
              auxcut[index + 1] = True
          if index > 0:
              auxcut[index - 1] = True
      if not rank:
         print( "Interpolating the Cosmological functions using {} values!".format(np.sum(auxcut) ) )

      D      = cosmology.getWisePolyFit(cosmology.a[auxcut], cosmology.D[auxcut])
      D2     = cosmology.getWisePolyFit(cosmology.a[auxcut], cosmology.D2[auxcut])
      D31    = cosmology.getWisePolyFit(cosmology.a[auxcut], cosmology.D31[auxcut])
      D32    = cosmology.getWisePolyFit(cosmology.a[auxcut], cosmology.D32[auxcut])
      auxcut = (cosmology.ainterp >= amin) & (cosmology.ainterp <= amax)
      DPLC   = cosmology.getWisePolyFit(cosmology.ainterp[auxcut], cosmology.Dinterp[auxcut])

      replicationsinside = geometry[ (geometry['nearestpoint'] < dlsup) & (geometry['farthestpoint'] >= dlinf) ]

      if not rank:
         deltai  = np.zeros(params.npixels)
         print(" Replications inside:")

      for ii, repi in enumerate(replicationsinside):

         if not rank:
            print(" * [{}/{}] {} {} {} ".format( str(ii + 1).zfill(int(np.log10(replicationsinside.size) + 1)), replicationsinside.size,
                                                               repi['x'], repi['y'], repi['z']) )

         shift = params.boxsize * np.array(repi[['x','y','z']].tolist())
         builder.getCrossingScaleParameterNewtonRaphson (qPosslice + shift.astype(np.float32), V1slice, V2slice, V31slice, V32slice, aplcslice,
                                                             params.nparticles//size, DPLC, D, D2, D31, D32, params.norder, amin, amax)


         #aplcslice[ 1.0/aplcslice -1  > Zaccslice ] = -1.0
         builder.getSkyCoordinates(qPosslice + shift.astype(np.float32), V1slice, V2slice, V31slice, V32slice, aplcslice, skycoordslice, params.nparticles//size, D, D2, D31, D32, params.norder)

         for ranki in range(size):

            if (rank == ranki) and ranki > 0:

               comm.Send(skycoordslice, dest=0)

            if rank == 0:

               if ranki:
                  print(" Rank: 0 receving slice from Rank: {}".format(ranki))
                  comm.Recv(skycoordslice, source=ranki)
               else:
                  print(" Rank: 0 working on its own load".format(ranki))

               cut = skycoordslice[:,0] > 0
               theta, phi = skycoordslice[:,1][cut] + np.pi/2.0, skycoordslice[:,2][cut]
               pixels = hp.pixelfunc.ang2pix(hp.pixelfunc.npix2nside(params.npixels), theta, phi)
               deltai += np.histogram(pixels, bins=np.linspace(0,params.npixels,params.npixels+1).astype(int))[0]

            comm.Barrier()

      if rank == 0:

         groupsinplane = (cosmology.plc.redshift <= z2) & (cosmology.plc.redshift > z1)
         pixels = hp.pixelfunc.ang2pix(hp.pixelfunc.npix2nside(params.npixels), \
                                   cosmology.plc.theta[groupsinplane]*np.pi/180.0+np.pi/2.0, cosmology.plc.phi[groupsinplane]*np.pi/180.0)
         deltahi = np.histogram(pixels, bins=np.linspace(0,params.npixels,params.npixels+1).astype(int))[0]
         deltahi = deltahi/deltahi.mean() - 1.0
         deltai = deltai/deltai.mean() - 1.0
         kappai = (1.0+zl) * ( ( 1.0 - cosmology.lcdm.comoving_distance(zl)/cosmology.lcdm.comoving_distance(params.zsource) ) *\
                                 cosmology.lcdm.comoving_distance(zl) * ( cosmology.lcdm.comoving_distance(z2) - cosmology.lcdm.comoving_distance(z1) ) ).to_value() * deltai
         kappai *= (3.0 * cosmology.lcdm.Om0*cosmology.lcdm.H0**2/2.0/cosmology.cspeed**2).to_value()

         theta, phi = [skycoordslice[ID==id,[1,2]] for id in plc.name[groupsinplane]]
         theta += np.pi/2.0
         pixels = hp.pixelfunc.ang2pix(hp.pixelfunc.npix2nside(params.npixels), theta, phi)
         delta_sanity += np.histogram(pixels, bins=np.linspace(0,params.npixels,params.npixels+1).astype(int))[0]

         hp.fitsfunc.write_map('Maps/delta_field_fullsky_{}.fits'.format(str(round(zl,4))), deltai, overwrite=True)
         hp.fitsfunc.write_map('Maps/delta_sanity_fullsky_{}.fits'.format(str(round(zl,4))), delta_sanity, overwrite=True)
         hp.fitsfunc.write_map('Maps/delta_halos_fullsky_{}.fits'.format(str(round(zl,4))), deltahi, overwrite=True)
         hp.fitsfunc.write_map('Maps/kappa_field_fullsky_{}.fits'.format(str(round(zl,4))), kappai, overwrite=True)
         kappa += kappai

         print("\n++++++++++++++++++++++\n")

      comm.Barrier()

   if rank == 0:

      hp.mollview(kappa)
      plt.show()

      cl = hp.anafast(kappa, lmax=512)
      ell = np.arange(len(cl))
      np.savetxt("Maps/Cls_kappa_z{}.txt".format(params.zsource), np.transpose([ell, cl, ell * (ell+1) * cl]))
