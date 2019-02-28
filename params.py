from readparams import getValueFromFile, typeArrayFromString, checkIfBoolExists
import os
import numpy as np

###############################################################
################ Cosmological Parameters ######################
# !!Will be overwritten if a Pinocchio params file is given!! #
###############################################################

omega0 = 0.0
h0     = 0.0

################## Past Light Cone Parameters #################
# !!Will be overwritten if a Pinocchio params file is given!! #
###############################################################

fovindeg     = 0
fovinradians = fovindeg * np.pi/180.0
npixels      = 12*2**14
zsource      = 1.0
nlensperbox  = 4

#################### Pinocchio Parameters #####################
#    Reads the parameters from Pinocchio's parameters file    #
#    It will overwrite the parameters set unless it fails     #
#    reading the file                                         #
###############################################################

paramfilename = "params"
boxsize       = 0
nparticles    = 0
plcaxis       = [ 0, 0, 0]
plccenter     = [ 0, 0, 0]
minhalomass   = 0

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
      fovinradians = fovindeg * np.pi/180.0
      pintlessfile = "pinocchio."+runflag+".t_snapshot.out"
      pincosmofile = "pinocchio."+runflag+".cosmology.out"
      pingeofile   = "pinocchio."+runflag+".geometry.out"
      pinplcfile   = "pinocchio."+runflag+".plc.out"

      if os.path.isfile(pintlessfile) and os.path.isfile(pincosmofile) and os.path.isfile(pingeofile) and os.path.isfile(pinplcfile):
          pass
      else:
          print("Pinocchio files not found! Check the run!")
          raise FileNotFoundError

      if checkIfBoolExists("PLCProvideConeData", paramfile):

          plcaxis     = getValueFromFile("PLCAxis", paramfile, typeArrayFromString(float))
          plccenter   = getValueFromFile("PLCCenter", paramfile, typeArrayFromString(float))

      else:

          print("!!                       WARNING                            !!")
          print("!!Pinocchio was run without specifying the PLC center and axis.\nUsing the ones set in params.py!!")
          print("!!                       WARNING                            !!")

      if plcstartingz < zsource:

          print("StartingzForPLC ({}) is smaller than the source redshift ({}).".format(plcstartingz, zsource))
          print("If this is exactly what you want comment this error Raising in params.py.")
          raise RuntimeError

   except:

      '''
      !! Not set yet, wise Ass !!
      '''
      raise NotImplementedError
