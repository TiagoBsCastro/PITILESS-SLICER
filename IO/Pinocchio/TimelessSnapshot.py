import params
import numpy as np
import cosmology as cosmo
from IO.Utils.randomization import randomizePositions, randomizeVelocities
from IO.Utils.wrapPositions import wrapPositions
import g3read
import contextlib
import sys

class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

########################## Timeless Snapshot ############################

class timeless_snapshot:

    def __init__(self, pintlessfile=params.pintlessfile, snapnum=-1, ready_to_bcast = False):

        with nostdout():

           if snapnum == -1:

               fname = pintlessfile

           else:

               fname = pintlessfile+"{}".format(snapnum)

           self.ID    = g3read.read_new(fname, 'ID  ', 1)
           self.V1    = g3read.read_new(fname, 'VZEL', 1)
           self.V2    = g3read.read_new(fname, 'V2  ', 1)
           self.V31   = g3read.read_new(fname, 'V3_1', 1)
           self.V32   = g3read.read_new(fname, 'V3_2', 1)
           self.Zacc  = g3read.read_new(fname, 'ZACC', 1)
           self.Npart = self.ID.size

        self.NG    = params.ngrid
        self.Lbox  = g3read.GadgetFile(fname, is_snap=False).header.BoxSize
        self.Cell  = self.Lbox/float(self.NG)

        face = 1
        sgn  = [1, 1, 1]
        # Recentering the box
        if params.rotatebox:

            self.qPos = np.array([ ((self.ID-1)//self.NG**2)%self.NG , ((self.ID-1)//self.NG)%self.NG, (self.ID-1)%self.NG]).transpose() * self.Cell + self.Cell/2.

        else:

            self.qPos = np.array([ (self.ID-1)%self.NG,((self.ID-1)//self.NG)%self.NG,\
                                  ((self.ID-1)//self.NG**2)%self.NG ]).transpose() * self.Cell + self.Cell/2.

        self.qPos = randomizePositions(params.plccenter, face, sgn, self.qPos/self.Lbox)
        self.V1   = self.Cell*randomizeVelocities(face, sgn, self.V1)/self.Lbox
        self.V2   = self.Cell*randomizeVelocities(face, sgn, self.V2)/self.Lbox
        self.V31  = self.Cell*randomizeVelocities(face, sgn, self.V31)/self.Lbox
        self.V32  = self.Cell*randomizeVelocities(face, sgn, self.V32)/self.Lbox
        # Changing the Basis to PLC basis
        self.qPos = self.qPos.dot(params.change_of_basis)
        self.V1   = self.V1.dot(params.change_of_basis)
        self.V2   = self.V2.dot(params.change_of_basis)
        self.V31  = self.V31.dot(params.change_of_basis)
        self.V32  = self.V32.dot(params.change_of_basis)

        if ready_to_bcast:

            # Reshaping to be MPI.Broadcast friendly
            self.qPos = self.qPos.astype(np.float32).reshape((3,self.Npart))
            self.V1   = self.V1.astype(np.float32).reshape((3,self.Npart))
            self.V2   = self.V2.astype(np.float32).reshape((3,self.Npart))
            self.V31  = self.V31.astype(np.float32).reshape((3,self.Npart))
            self.V32  = self.V32.astype(np.float32).reshape((3,self.Npart))

    def snapPos(self, z, zcentered=True, filter=None):
        """
        Returns the particles Position at z
        """

        thisa   = 1.0 / (1.0 + z)
        thisD   = np.interp(thisa, cosmo.a, cosmo.D)
        thisD2  = np.interp(thisa, cosmo.a, cosmo.D2)
        thisD31 = np.interp(thisa, cosmo.a, cosmo.D31)
        thisD32 = np.interp(thisa, cosmo.a, cosmo.D32)

        if filter is None:

            pos = np.ascontiguousarray(self.qPos + thisD * self.V1 + thisD2 * self.V2 + \
                thisD31 * self.V31 + thisD32 * self.V32, dtype=np.float32)

        else:

            pos = np.ascontiguousarray((self.qPos + thisD * self.V1 + thisD2 * self.V2 + \
                   thisD31 * self.V31 + thisD32 * self.V32)[filter], dtype=np.float32)

        wrapPositions(pos)

        if zcentered:

            return (pos - [0.5, 0.5, 0.0]) * self.Lbox

        else:

            return (pos - [0.5, 0.5, 0.5]) * self.Lbox

    def snapVel(self, z, filter=None):
        """
        Returns the particles Velocities at z
        """

        thisa   = 1.0 / (1.0 + z)
        thisD   = np.interp(thisa, cosmo.a, np.gradient(cosmo.D)/np.gradient(cosmo.a))
        thisD2  = np.interp(thisa, cosmo.a, np.gradient(cosmo.D2)/np.gradient(cosmo.a))
        thisD31 = np.interp(thisa, cosmo.a, np.gradient(cosmo.D31)/np.gradient(cosmo.a))
        thisD32 = np.interp(thisa, cosmo.a, np.gradient(cosmo.D32)/np.gradient(cosmo.a))

        if filter is None:

            vel = ( thisD * self.V1 + thisD2 * self.V2 + \
                thisD31 * self.V31 + thisD32 * self.V32 ) * self.Lbox * \
                thisa * cosmo.lcdm.H(z).value

        else:

            vel = ( thisD * self.V1 + thisD2 * self.V2 + thisD31 * self.V31 +\
                    thisD32 * self.V32 )[filter] * self.Lbox * thisa * cosmo.lcdm.H(z).value

        return vel

#########################################################################
