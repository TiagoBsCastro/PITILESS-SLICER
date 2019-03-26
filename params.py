from readparams import getValueFromFile, typeArrayFromString, checkIfBoolExists
import os
import numpy as np
import sys

###############################################################
################ Cosmological Parameters ######################
###############################################################

norder = 4
# to account for particles
# that go out of the box
beta_buffer  = 1e-3
theta_buffer = 5e-2

################## Past Light Cone Parameters #################
###############################################################

npixels      = 12*2**14
zsource      = 1.0
nlensperbox  = 5

#################### Pinocchio Parameters #####################
#    Reads the parameters from Pinocchio's parameters file    #
###############################################################

paramfilename = "params_512"

########### Under the Hood from this point Forward ############
###############################################################

if os.path.isfile(paramfilename):

   paramfile = open(paramfilename,"r").read()

   try:

      omega0       = getValueFromFile("Omega0", paramfile, float)
      h0true       = getValueFromFile("Hubble100", paramfile, float)*100
      h0           = 100
      boxsize      = getValueFromFile("BoxSize", paramfile, float)
      minhalomass  = getValueFromFile("MinHaloMass", paramfile, int)
      nparticles   = getValueFromFile("GridSize", paramfile, int)**3
      fovindeg     = getValueFromFile("PLCAperture", paramfile, float)
      fovinradians = fovindeg * np.pi/180.0
      runflag      = getValueFromFile("RunFlag", paramfile, str)
      plcstartingz = getValueFromFile("StartingzForPLC", paramfile, float)
      pintlessfile = "pinocchio."+runflag+".t_snapshot.out"
      pincosmofile = "pinocchio."+runflag+".cosmology.out"
      pingeofile   = "pinocchio."+runflag+".geometry.out"
      pinplcfile   = "pinocchio."+runflag+".plc.out"

      try:

          numfiles = getValueFromFile("NumFiles", paramfile, int)
          print("NumFiles = {}".format(numfiles))

      except ParameterNotFound:

          numfiles = 1

      if numfiles == 1:

          if os.path.isfile(pintlessfile) and os.path.isfile(pincosmofile) and os.path.isfile(pingeofile) and os.path.isfile(pinplcfile):
              pass
          else:
              print("Pinocchio files not found! Check the run!")
              raise FileNotFoundError

      else:

         for snapnum in range(numfiles):

            if os.path.isfile(pintlessfile+".{}".format(snapnum)) and os.path.isfile(pincosmofile) and os.path.isfile(pingeofile) and os.path.isfile(pinplcfile+".{}".format(snapnum)):
               pass
            else:
               print("Pinocchio files not found! Check the run!")
               raise FileNotFoundError

      if checkIfBoolExists("PLCProvideConeData", paramfile):

          plcaxis    = getValueFromFile("PLCAxis", paramfile, typeArrayFromString(float))
          plcaxis   /= np.sqrt( (plcaxis**2).sum() )
          plccenter  = getValueFromFile("PLCCenter", paramfile, typeArrayFromString(float))
          plccenter /= boxsize
          plccenter -= np.array([0.5, 0.5, 0.5])

          if plcaxis[2] == 1.0:
              plcx  = np.array([1.0, 0.0, 0.0])
              plcy  = np.array([0.0, 1.0, 0.0])
          else:
              plcx  = np.cross([0.0, 0.0, 1.0], plcaxis)
              plcx /= np.sqrt( (plcx**2).sum() )
              plcy  = np.cross(plcaxis, plcx)

          change_of_basis = np.transpose([plcx,plcy, plcaxis]).T

      else:

          print("!!                       WARNING                            !!")
          print("!!Pinocchio was run without specifying the PLC center and axis!!\n")
          print("!!                       WARNING                            !!")

      if checkIfBoolExists("CatalogInAscii", paramfile):

          raise RuntimeError("!! Catalogs were generated in ASCII format !!")

      if plcstartingz < zsource:

          print("StartingzForPLC ({}) is smaller than the source redshift ({}).".format(plcstartingz, zsource))
          print("If this is exactly what you want comment this error Raising in params.py.")
          raise RuntimeError

   except FileNotFoundError:

       sys.exit(-1)

   except RuntimeError:

      '''
      !! Not set yet, wise Ass !!
      '''
      raise NotImplementedError
