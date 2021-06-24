from mpi4py import MPI
from IO.Utils.print import print
import params
import numpy as np
import healpy as hp
from PLC import builder
import cosmology
import os
import IO.Pinocchio.TimelessSnapshot as snapshot
import IO.Pinocchio.ReadPinocchio as rp
import astropy.coordinates as ap
import NFW.NFWx as NFW
from time import time

# Bunch class for easy MPI handling
class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

# MPI comunicatior, rank and size of procs
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:

    print("Rank 0 is Working on the Geometry!")
    from PLC.geometry import geometry

else:

   geometry = None

geometry = comm.bcast(geometry)

# Check if the simplistic work balance will do
if (params.nparticles/params.numfiles)%size:

    print("The data cannot be scattered on {} processes!".format(size))
    comm.Abort()

# particle mass
mpart = (2.7753663e11 * params.omega0 * params.boxsize ** 3)/params.nparticles

# Start loop on the snapshot files
for snapnum in range(params.numfiles):

   if rank == 0:

       print("Rank 0 is reading the data! {}/{}".format(snapnum+1, params.numfiles))
       if params.numfiles == 1:
          ts = snapshot.timeless_snapshot(params.pintlessfile, -1, ready_to_bcast=True)
       else:
          ts = snapshot.timeless_snapshot(params.pintlessfile+".{}".format(snapnum), -1, ready_to_bcast=True)

       print("All done! Let's go!")

       try:

           os.mkdir("Maps/")

       except FileExistsError:

           pass

   ###################################################

   if rank:
      # Defining ts to the other ranks
      ts   = Bunch(qPos = None, V1   = None, V2   = None, V31  = None, V32  = None, Zacc = None, Npart = None)

   npart = comm.bcast(ts.Npart)
   qPosslice = np.empty((npart//size,3), dtype = np.float32)
   V1slice   = np.empty((npart//size,3), dtype = np.float32)
   V2slice   = np.empty((npart//size,3), dtype = np.float32)
   V31slice  = np.empty((npart//size,3), dtype = np.float32)
   V32slice  = np.empty((npart//size,3), dtype = np.float32)
   Zaccslice = np.empty(npart//size, dtype = np.float32)
   aplcslice = np.empty(npart//size, dtype = np.float32)
   skycoordslice = np.empty((npart//size, 3), dtype = np.float32)

   comm.Scatterv(ts.qPos, qPosslice,root=0); qPosslice -= np.array([0.5, 0.5, 0.5]).dot(params.change_of_basis);# Re-centering the snapshot on [0.5, 0.5, 0.5]
   comm.Scatterv(ts.V1  ,V1slice   ,root=0)
   comm.Scatterv(ts.V2  ,V2slice   ,root=0)
   comm.Scatterv(ts.V31 ,V31slice  ,root=0)
   comm.Scatterv(ts.V32 ,V32slice  ,root=0)
   comm.Scatterv(ts.Zacc ,Zaccslice,root=0)

   del ts

   if not rank:
      print("++++++++++++++++++++++")

   for i,(z1,z2) in enumerate( zip(cosmology.zlinf, cosmology.zlsup) ):

      zl = (z1 + z2)/2.0

      if not rank:

         # If working on this lens plane for the first time create an empy (zeros) delta map
         if not snapnum:
            deltai = np.zeros(params.npixels, dtype=np.int64)
         # Else load it from disk
         else:
            print("Reopening density map:", 'Maps/delta_'+params.runflag+'_field_fullsky_{}.fits'.format(str(round(zl,4))))
            deltai = hp.read_map('Maps/delta_'+params.runflag+'_field_fullsky_{}.fits'.format(str(round(zl,4))), dtype=np.int64)

         print("Lens plane from z=[{0:.3f},{1:.3f}]".format(z1,z2))

      # Lens distances
      dlinf = cosmology.lcdm.comoving_distance(z1).to_value()
      dlsup = cosmology.lcdm.comoving_distance(z2).to_value()
      # Range in the scale factor compressed by dlinf and dlsup taking into account the buffer region
      amin  = 1.0/(1.0+cosmology.z_at_value(cosmology.lcdm.comoving_distance, dlsup*(1+params.beta_buffer)*cosmology.Mpc))
      if dlinf == 0.0:
          amax = 1.0
      else:
          amax = 1.0/(1.0+cosmology.z_at_value(cosmology.lcdm.comoving_distance, dlinf*(1-params.beta_buffer)*cosmology.Mpc))

      # Fitting the cosmological functions inside the range
      # Select points inside the range
      auxcut  = (cosmology.a >= amin) & (cosmology.a <= amax)
      if not rank:
         print( "Number of points inside the redshift range is {} values!".format(np.sum(auxcut) ) )

      # If there is less than the interpolation order points inside the range use also the neighbours points
      if auxcut.sum() <= params.norder:
          index = auxcut.argmax()
          if index < auxcut.size:
              auxcut[index: index + params.norder + 1] = True
          if index > 0:
              auxcut[index: index - params.norder - 1] = True
      if not rank:
         print( "Interpolating the Cosmological functions using {} values!".format(np.sum(auxcut) ) )

      D      = cosmology.getWisePolyFit(cosmology.a[auxcut], cosmology.D[auxcut])
      D2     = cosmology.getWisePolyFit(cosmology.a[auxcut], cosmology.D2[auxcut])
      D31    = cosmology.getWisePolyFit(cosmology.a[auxcut], cosmology.D31[auxcut])
      D32    = cosmology.getWisePolyFit(cosmology.a[auxcut], cosmology.D32[auxcut])
      auxcut = (cosmology.ainterp >= amin) & (cosmology.ainterp <= amax)
      DPLC   = cosmology.getWisePolyFit(cosmology.ainterp[auxcut], cosmology.Dinterp[auxcut]/params.boxsize)

      # Check which replications are compressed by the lens
      replicationsinside = geometry[ (geometry['nearestpoint'] < dlsup*(1+params.beta_buffer)) &
                                     (geometry['farthestpoint'] >= dlinf*(1-params.beta_buffer)) ]

      if not rank:
         print(" Replications inside:")
      # Loop on the replications
      for ii, repi in enumerate(replicationsinside):

         if not rank:
            print(" * Replication [{}/{}] of snap [{}/{}] {} {} {} "\
                       .format( str(ii + 1).zfill(int(np.log10(replicationsinside.size) + 1)), replicationsinside.size,
                                str(snapnum + 1).zfill(int(np.log10(params.numfiles) + 1)), params.numfiles,
                                repi['x'], repi['y'], repi['z']))

         # Set the 1st guess for plc-crossing to a-collapse
         aplcslicei = np.copy(aplcslice)
         aplcslicei[ Zaccslice != -1 ] = 1.0/(Zaccslice[ Zaccslice != -1 ] + 1)
         aplcslicei[ Zaccslice == -1 ] = 1.0

         # Position shift of the replication
         shift = (np.array(repi[['x','y','z']].tolist()).dot(params.change_of_basis)).astype(np.float32)
         # Get the scale parameter of the moment that the particle crossed the PLC
         if not rank:
             t0 = time()
         if params.optimizer == "NewtonRaphson":
             builder.getCrossingScaleParameterNewtonRaphson (qPosslice + shift, V1slice, V2slice,\
                                                             V31slice, V32slice, aplcslicei, npart//size, DPLC, D, D2,\
                                                             D31, D32, params.norder, amin, amax)
         elif params.optimizer == "Bisection":
             builder.getCrossingScaleParameterBissection (qPosslice + shift, V1slice, V2slice,\
                                                         V31slice, V32slice, aplcslicei, npart//size, DPLC, D, D2,\
                                                         D31, D32, params.norder, amin, amax)
         elif params.optimizer == "Fast":
             builder.getCrossingScaleParameterFast (qPosslice + shift, V1slice, V2slice,\
                                                    V31slice, V32slice, aplcslicei, npart//size, DPLC, D, D2,\
                                                    D31, D32, params.norder, amin, amax)
         else:

             raise NotImplementedError("Optimizer {} is not implemented!".format(params.optimizer))

         if not rank:
             t0 -= time()

         # If the accretion redshift is smaller than the redshift crossing
         # ignore the particle
         aplcslicei[ (aplcslicei < amin) | (aplcslicei > amax) ] = -1.0

         if not rank:
             t1 = time()
         builder.getSkyCoordinates(qPosslice, shift, V1slice, V2slice, V31slice, V32slice, aplcslicei,\
                                                                 skycoordslice,npart//size, D, D2, D31, D32, params.norder)

         cut = skycoordslice[:,0] > 0
         theta, phi = np.pi/2.0 - skycoordslice[:,1][cut], skycoordslice[:,2][cut]
         phi    = phi[theta <= params.fovinradians]
         theta  = theta[theta <= params.fovinradians]
         pixels = hp.pixelfunc.ang2pix(hp.pixelfunc.npix2nside(params.npixels), theta, phi)
         if rank:

           deltaii = np.bincount(pixels, minlength=params.npixels).astype(np.int64)

         else:

           print(" Rank: 0 working on its own load.")
           deltaii = np.empty_like(deltai)
           deltai += np.bincount(pixels, minlength=params.npixels).astype(np.int64)

         del theta, phi, cut, pixels

         if not rank:
             t1 -= time()
             print(" Rank: 0 spent {0:4.3f}s for getting the PLC crossing and {1:4.3f}s for computing the sky coordinates.".format(-t0, -t1))

         # Collect data from the other ranks
         for ranki in range(size):

            if (rank == ranki) and ranki > 0:
               comm.Send(deltaii, dest=0)
               #print(deltaii)

            if rank == 0:

               if ranki:
                  print(" Rank: 0 receving slice from Rank: {}".format(ranki))
                  comm.Recv(deltaii, source=ranki)
                  # Rank 0 update the map
                  #print(deltaii)
                  deltai += deltaii

            comm.Barrier()

         del deltaii

      if rank == 0:
         # Rank 0 writes the collected map
         hp.fitsfunc.write_map('Maps/delta_'+params.runflag+'_field_fullsky_{}.fits'.format(str(round(zl,4))), deltai, overwrite=True, dtype=np.int64)
         print("++++++++++++++++++++++")

      comm.Barrier()

# Everything done for the particles
# Constructs the density maps for halos
# and convergence maps for particles
print("All done for uncollapsed particles PLC", rank=rank)
if not rank:

   print("Proceeding serially:")
   print("++++++++++++++++++++++")
   if params.fovindeg < 180.0:

       pixels = np.arange(params.npixels)
       mask   = hp.pix2ang( hp.pixelfunc.npix2nside(params.npixels), pixels)[0] * 180.0/np.pi
       mask   = (mask <= params.fovindeg)

   else:

       mask   = np.ones(params.npixels, dtype=bool)

   kappa = np.zeros(params.npixels)

   for z1, z2 in zip(cosmology.zlinf, cosmology.zlsup):

      print("Lens plane from z=[{0:.3f},{1:.3f}]".format(z1,z2))

      zl = 1.0/2.0*(z1+z2)

      deltahi = np.zeros(params.npixels)
      deltai = hp.fitsfunc.read_map('Maps/delta_'+params.runflag+'_field_fullsky_{}.fits'.format(str(round(zl,4))), dtype=np.int64).astype(np.float64)

      if params.numfiles > 1:

         for snapnum in range(params.numfiles):

            plc      = rp.plc(params.pinplcfile+".{}".format(snapnum))
            groupsinplane = (plc.redshift <= z2) & (plc.redshift > z1)

            if not np.any(groupsinplane):
                continue

            print(" Updating halo maps")
            pixels = hp.pixelfunc.ang2pix(hp.pixelfunc.npix2nside(params.npixels), \
                                          np.pi/2.0 - plc.theta[groupsinplane]*np.pi/180.0, plc.phi[groupsinplane]*np.pi/180.0)
            deltahi += np.bincount(pixels, minlength=params.npixels)

            print(" Computing the concentration")
            rhoc   = cosmology.lcdm.critical_density(plc.redshift[groupsinplane]).to("M_sun/Mpc^3").value/(1+plc.redshift[groupsinplane])**3
            rDelta = np.ascontiguousarray((3*plc.Mass[groupsinplane]/4/np.pi/200/rhoc)**(1.0/3), dtype=np.float32)
            conc   = np.array( [cosmology.concentration.concentration( m, '200c', z, model = 'bhattacharya13') for m, z in zip(plc.Mass[groupsinplane], plc.redshift[groupsinplane])], dtype=np.float32 )

            print(" Sampling particle on halos")
            N_part = np.ascontiguousarray( np.round(plc.Mass[groupsinplane]/mpart).astype(np.int32) )
            N_tot  = np.sum(N_part)

            r      = np.empty( N_tot, dtype=np.float32 )
            theta2 = np.empty( N_tot, dtype=np.float32 )
            phi2   = np.empty( N_tot, dtype=np.float32 )

            conv = np.pi/180.
            NFW.random_nfw( N_part, conc, rDelta, r, theta2, phi2)
            r_halos = np.sqrt(plc.pos[:,0][groupsinplane]**2+plc.pos[:,1][groupsinplane]**2+plc.pos[:,2][groupsinplane]**2)
            pos_halos = np.transpose(ap.spherical_to_cartesian(r_halos, plc.theta[groupsinplane]*conv, plc.phi[groupsinplane]*conv))
            pos_part  = np.transpose(ap.spherical_to_cartesian(r, np.pi/2.0-theta2, phi2))
            ang_h     = np.transpose(ap.cartesian_to_spherical(pos_halos[:, 0], pos_halos[:, 1], pos_halos[:, 2]))
            final_pos = np.repeat(pos_halos, N_part, axis=0) + pos_part
            final_ang = np.transpose(ap.cartesian_to_spherical(final_pos[:, 0], final_pos[:, 1], final_pos[:, 2]))

            print(" Updating mass maps")
            pixels = hp.pixelfunc.ang2pix(hp.pixelfunc.npix2nside(params.npixels), np.pi/2.0-final_ang[:, 1], final_ang[:, 2])
            deltai  += np.bincount(pixels, minlength=params.npixels)

      else:

         plc      = rp.plc(params.pinplcfile)
         groupsinplane = (plc.redshift <= z2) & (plc.redshift > z1)

         if np.any(groupsinplane):

             print(" Updating halo maps")
             pixels = hp.pixelfunc.ang2pix(hp.pixelfunc.npix2nside(params.npixels), \
                                           np.pi/2.0 - plc.theta[groupsinplane]*np.pi/180.0, plc.phi[groupsinplane]*np.pi/180.0)
             deltahi += np.bincount(pixels, minlength=params.npixels)

             print(" Computing the concentration")
             rhoc   = cosmology.lcdm.critical_density(plc.redshift[groupsinplane]).to("M_sun/Mpc^3").value/(1+plc.redshift[groupsinplane])**3
             rDelta = np.ascontiguousarray((3*plc.Mass[groupsinplane]/4/np.pi/200/rhoc)**(1.0/3), dtype=np.float32)
             conc   = np.array( [cosmology.concentration.concentration( m, '200c', z, model = 'bhattacharya13') for m, z in zip(plc.Mass[groupsinplane], plc.redshift[groupsinplane])], dtype=np.float32 )

             print(" Sampling particle on halos")
             N_part = np.ascontiguousarray( np.round(plc.Mass[groupsinplane]/mpart).astype(np.int32) )
             N_tot  = np.sum(N_part)

             r      = np.empty( N_tot, dtype=np.float32 )
             theta2 = np.empty( N_tot, dtype=np.float32 )
             phi2   = np.empty( N_tot, dtype=np.float32 )
             conv = np.pi/180.0

             NFW.random_nfw( N_part, conc, rDelta, r, theta2, phi2)
             r_halos = np.sqrt(plc.pos[:,0][groupsinplane]**2+plc.pos[:,1][groupsinplane]**2+plc.pos[:,2][groupsinplane]**2)
             pos_halos = np.transpose( ap.spherical_to_cartesian(r_halos, plc.theta[groupsinplane]*conv, plc.phi[groupsinplane]*conv) )
             pos_part  = np.transpose( ap.spherical_to_cartesian(r, np.pi/2 - theta2, phi2) )
             final_pos = np.repeat(pos_halos, N_part, axis=0) + pos_part
             final_ang = np.transpose( ap.cartesian_to_spherical(final_pos[:,0], final_pos[:,1], final_pos[:,2]) )

             print(" Updating mass maps")
             pixels  = hp.pixelfunc.ang2pix(hp.pixelfunc.npix2nside(params.npixels), np.pi/2 - final_ang[:, 1], final_ang[:, 2])
             deltai += np.bincount(pixels, minlength=params.npixels)

      print(" Saving convergence, mass and halo maps")
      deltahi[~mask] = hp.UNSEEN
      deltahi[mask]  = deltahi[mask]/deltahi[mask].mean() - 1.0
      hp.fitsfunc.write_map('Maps/delta_'+params.runflag+'_halos_fullsky_{}.fits'.format(str(round(zl,4))), deltahi, overwrite=True, dtype=np.float32)

      deltai[~mask]  = hp.UNSEEN
      deltai[mask] = deltai[mask]/deltai[mask].mean() - 1.0
      hp.fitsfunc.write_map('Maps/delta_'+params.runflag+'_matter_fullsky_{}.fits'.format(str(round(zl,4))), deltai, overwrite=True, dtype=np.float32)

      kappai = (1.0+zl) * ( ( 1.0 - cosmology.lcdm.comoving_distance(zl)/cosmology.lcdm.comoving_distance(params.zsource) ) *\
                  cosmology.lcdm.comoving_distance(zl) *\
                ( cosmology.lcdm.comoving_distance(z2) - cosmology.lcdm.comoving_distance(z1) ) ).to_value() * deltai
      kappai *= (3.0 * cosmology.lcdm.Om0*cosmology.lcdm.H0**2/2.0/cosmology.cspeed**2).to_value()
      kappa[mask]  += kappai[mask]
      kappai[~mask] = hp.UNSEEN
      hp.fitsfunc.write_map('Maps/kappa_'+params.runflag+'_field_fullsky_{}.fits'.format(str(round(zl,4))), kappai, overwrite=True, dtype=np.float32)
      print("++++++++++++++++++++++")

   print("Computing convergence Cl")
   cl = hp.anafast(kappa, lmax=512)
   ell = np.arange(len(cl))
   np.savetxt("Maps/Cls_kappa_z{}.txt".format(params.zsource), np.transpose([ell, cl, ell * (ell+1) * cl]))
   print("All done!")
