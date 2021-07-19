from mpi4py import MPI
from IO.Utils.print import print
from IO.Params.readparams import getValueFromFile, typeArrayFromString, checkIfBoolExists
import os
import numpy as np
import sys
from hmf.density_field.transfer_models import EH_BAO, CAMB
from scipy.linalg import inv

# MPI comunicatior, rank and size of procs
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

###############################################################
###############################################################
#################### Cosmological Parameters ##################
###############################################################

TCMB = 2.7255 # Only used if IC Pk was given by camb

###############################################################
######################### C-M Parameters ######################
###############################################################

cmmodel = 'diemer19' # colossus models: https://bdiemer.bitbucket.io/colossus/halo_concentration.html

###############################################################
################## Past Light Cone Parameters #################
###############################################################

npixels       = 12*2**20
zsource       = 0.3
nlensperbox   = 0
lensthickness = 250  # Only used if nlensperbox == 0
norder        = 2
beta_buffer   = 1e-3 # to account for particles that go out of the box
theta_buffer  = 5e-2
optimizer     = 'NewtonRaphson'

###############################################################
#################### Pinocchio Parameters #####################
#### Reads the parameters from Pinocchio's parameters file ####
###############################################################

paramfilename = "/beegfs/pmonaco/Pinocchio/LargePLC_timeless/SmallerPLC/SmallerPLC_0.params"
directoryname = "/beegfs/pmonaco/Pinocchio/LargePLC_timeless/SmallerPLC/"

###############################################################
########### Under the Hood from this point Forward ############
###############################################################
###############################################################

if nlensperbox == 0:

    beta_buffer = 0.0

if os.path.isfile(paramfilename):

   paramfile = open(paramfilename,"r").read()

   try:

      omega0       = getValueFromFile("Omega0", paramfile, float, rank)
      omegabaryon  = getValueFromFile("OmegaBaryon", paramfile, float, rank)
      h0true       = getValueFromFile("Hubble100", paramfile, float, rank)*100
      ns           = getValueFromFile("PrimordialIndex", paramfile, float, rank)
      sigma8       = getValueFromFile("Sigma8", paramfile, float, rank)
      h0           = 100
      boxsize      = getValueFromFile("BoxSize", paramfile, float, rank)
      minhalomass  = getValueFromFile("MinHaloMass", paramfile, int, rank)
      ngrid        = getValueFromFile("GridSize", paramfile, int, rank)
      nparticles   = getValueFromFile("GridSize", paramfile, int, rank)**3
      fovindeg     = getValueFromFile("PLCAperture", paramfile, float, rank)
      if fovindeg > 180.0:
            
            fovindeg = 180.0
            
      fovinradians = fovindeg * np.pi/180.0
      runflag      = getValueFromFile("RunFlag", paramfile, str, rank)
      outputlist   = getValueFromFile("OutputList", paramfile, str, rank)
      redshifts    = np.loadtxt(outputlist)
      if redshifts.size == 1:
          redshifts = redshifts.reshape(1)
      plcstartingz = getValueFromFile("StartingzForPLC", paramfile, float, rank)
      pintlessfile = os.path.join(directoryname,"pinocchio."+runflag+".t_snapshot.out")
      pincosmofile = os.path.join(directoryname,"pinocchio."+runflag+".cosmology.out")
      pingeofile   = os.path.join(directoryname,"pinocchio."+runflag+".geometry.out")
      pinplcfile   = os.path.join(directoryname,"pinocchio."+runflag+".plc.out")
      pincatfile   = os.path.join(directoryname,"pinocchio.{0:5.4f}."+runflag+".catalog.out")
      pinmffile    = os.path.join(directoryname,"pinocchio.{0:5.4f}."+runflag+".mf.out")

      analyticmf = getValueFromFile("AnalyticMassFunction", paramfile, int, rank)

      if analyticmf != 9:

          raise RuntimeError("Pinocchio has been run with a HMF different than the Analytic Mass Function used for calibration!")

      try:

          numfiles = getValueFromFile("NumFiles", paramfile, int)
          print("NumFiles = {}".format(numfiles), rank=rank)

      except ParameterNotFound:

          numfiles = 1

      if numfiles == 1:

          for z in redshifts:

              if os.path.isfile(pintlessfile) and os.path.isfile(pincosmofile) \
                 and os.path.isfile(pingeofile) and os.path.isfile(pinplcfile) \
                 and os.path.isfile(pincatfile.format(z)) and os.path.isfile(pinmffile.format(z)):
                  pass
              else:
                  print("Pinocchio files not found! Check the run!", rank=rank)
                  raise FileNotFoundError

      else:

          for z in redshifts:

              for snapnum in range(numfiles):

                  if not os.path.isfile(pintlessfile+".{}".format(snapnum)):
                      print(pintlessfile+".{}".format(snapnum), rank=rank)
                      print("Pinocchio timeless files not found! Check the run!", rank=rank)
                      raise FileNotFoundError
                  if not os.path.isfile(pincosmofile):
                      print("Pinocchio cosmology files not found! Check the run!", rank=rank)
                      raise FileNotFoundError
                  if not os.path.isfile(pingeofile):
                      print("Pinocchio geometric files not found! Check the run!", rank=rank)
                      raise FileNotFoundError
                  if not os.path.isfile(pinmffile.format(z)):
                      print("Pinocchio mf files not found! Check the run!", rank=rank)
                      raise FileNotFoundError
                  if not os.path.isfile( pinplcfile+".{0:d}".format(snapnum)):
                      print("Pinocchio plc files not found! Check the run!", rank=rank)
                      raise FileNotFoundError
                  if not os.path.isfile( (pincatfile+".{1:d}").format(z, snapnum)):
                      print("Pinocchio catalogs files not found! Check the run!", rank=rank)
                      raise FileNotFoundError

      del z

      if checkIfBoolExists("PLCProvideConeData", paramfile, rank):

          plcaxis    = getValueFromFile("PLCAxis", paramfile, typeArrayFromString(float), rank)
          plcaxis   /= np.sqrt( (plcaxis**2).sum() )
          print(plcaxis)
          plccenter  = getValueFromFile("PLCCenter", paramfile, typeArrayFromString(float), rank)
          plccenter /= boxsize

          if plcaxis[2] == 1.0:
              plcx  = np.array([1.0, 0.0, 0.0])
              plcy  = np.array([0.0, 1.0, 0.0])
          else:
              plcx  = np.cross(plcaxis, [0.0, 0.0, 1.0])
              plcx /= np.sqrt( (plcx**2).sum() )
              plcy  = np.cross(plcaxis, plcx)

          change_of_basis = inv([plcx, plcy, plcaxis])

      else:

          raise RuntimeError("!! Pinocchio was run without specifying the PLC center and axis!! ")

      if checkIfBoolExists("CatalogInAscii", paramfile, rank):

          raise RuntimeError("!! Catalogs were generated in ASCII format !!")

      if plcstartingz < zsource:

          print("StartingzForPLC ({}) is smaller than the source redshift ({}).".format(plcstartingz, zsource), rank=rank)
          print("If this is exactly what you want comment this error Raising in params.py.", rank=rank)
          raise RuntimeError

      # Computing sigma8 instead if it is 0
      if sigma8 == 0:

          transfer_model = CAMB

          k, Pk = np.loadtxt(pincosmofile, usecols=[12, 13], unpack=True)
          Dk    = Pk * k**3/(2*np.pi**2)
          w     = lambda r: ( 3.*( np.sin(k * r) - k*r*np.cos(k * r) )/((k * r)**3.))

          sigma8 = np.trapz( w(8.0*100/h0true)**2 * Dk/k, x=k)**0.5
          print("Sigma8 computed from pinocchio cosmology file: {}".format(sigma8), rank=rank)
          del k, Pk, Dk, w

      else:

          transfer_model = EH_BAO

   except FileNotFoundError:

       sys.exit(-1)

   except RuntimeError:

      '''
      !! Not set yet, wise ass !!
      '''
      raise NotImplementedError

else:

    raise FileNotFoundError("Parameter files not found!")
