import params
import numpy as np
import cosmology as cosmo
import Snapshot3 as S
from randomization import randomizePositions, randomizeVelocities
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

############# Timeless Snapshot ###############

class Timeless_Snapshot:

    def __init__(self, pintlessfile, snapnum=-1, ready_to_bcast = False):

        with nostdout():

           snap       = S.Init(pintlessfile, snapnum)
           self.ID    = snap.read_block('ID'  , onlythissnap=True)
           self.V1    = snap.read_block('VZEL', onlythissnap=True)
           self.V2    = snap.read_block('V2'  , onlythissnap=True)
           self.V31   = snap.read_block('V3_1', onlythissnap=True)
           self.V32   = snap.read_block('V3_2', onlythissnap=True)
           self.Zacc  = snap.read_block('ZACC', onlythissnap=True)

        self.Npart      = self.ID.size
        NG         = np.int(np.float(params.nparticles)**(1./3.)+0.5)
        Lbox       = snap.Header.boxsize
        Cell       = Lbox/float(NG)

        face = 1
        sgn  = [1, 1, 1]
        # Recentering the box
        self.qPos = np.array([ (self.ID-1)%NG,((self.ID-1)//NG)%NG,((self.ID-1)//NG**2)%NG ]).transpose() * Cell + Cell/2.
        self.qPos = randomizePositions(params.plccenter, face, sgn, self.qPos/Lbox)
        self.V1   = Cell*randomizeVelocities(face, sgn, self.V1)/Lbox
        self.V2   = Cell*randomizeVelocities(face, sgn, self.V2)/Lbox
        self.V31  = Cell*randomizeVelocities(face, sgn, self.V31)/Lbox
        self.V32  = Cell*randomizeVelocities(face, sgn, self.V32)/Lbox
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

###############################################
