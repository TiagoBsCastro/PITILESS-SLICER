import params
import numpy as np
import cosmology as cosmo
import Snapshot3 as S
from randomization import randomizePositions, randomizeVelocities

############# Timeless Snapshot ###############

snap=S.Init(params.pintlessfile,-1)
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

center = [0.5, 0.5, 0.5]
face = 1
sgn = [1, 1, 1]

qPos = np.array([ (ID-1)%NG,((ID-1)//NG)%NG,((ID-1)//NG**2)%NG ]).transpose() * Cell + Cell/2.
qPos = randomizePositions(center, face, sgn, qPos/Lbox)*Lbox
qPos[:,2] -= Lbox/2.0
qPos = qPos.astype(np.float32).reshape((3,NG**3))
V1  = Cell*randomizeVelocities(face, sgn, V1).astype(np.float32).reshape((3,NG**3))
V2  = Cell*randomizeVelocities(face, sgn, V2).astype(np.float32).reshape((3,NG**3))
V31 = Cell*randomizeVelocities(face, sgn, V31).astype(np.float32).reshape((3,NG**3))
V32 = Cell*randomizeVelocities(face, sgn, V32).astype(np.float32).reshape((3,NG**3))

###############################################

def snapPos (z, zcentered=True, filter=None):
   '''
   Returns the particles Position at z
   '''
   if filter is None:
      filter=np.ones(Npart).astype(bool)

   thisa   = 1.0/(1.0+z)
   thisD   = np.interp(thisa,cosmo.a,cosmo.D)
   thisD2  = np.interp(thisa,cosmo.a,cosmo.D2)
   thisD31 = np.interp(thisa,cosmo.a,cosmo.D31)
   thisD32 = np.interp(thisa,cosmo.a,cosmo.D32)

   xx, yy, zz = np.transpose(qPos[filter] +  Cell * (thisD * V1[filter] + thisD2 * V2[filter] + thisD31 * V31[filter] + thisD32 * V32[filter]))

   if zcentered:

      xx = ( wrapPositions(xx/Lbox +0.5 ) -0.5 )*Lbox; yy = ( wrapPositions(yy/Lbox +0.5 ) -0.5 )*Lbox; zz = wrapPositions(zz/Lbox)*Lbox;

   else:

      xx = ( wrapPositions(xx/Lbox +0.5 ) -0.5 )*Lbox; yy = ( wrapPositions(yy/Lbox +0.5 ) -0.5 )*Lbox; zz = (wrapPositions(zz/Lbox + 0.5) -0.5 )*Lbox;

   return xx, yy, zz

def snapPosPart ( q, v1, v2, v31, v32, z, zcentered=True ):
   '''
   Returns the particles Position at z
   '''

   thisa   = 1.0/(1.0+z)
   thisD   = np.interp(thisa,cosmo.a,cosmo.D)
   thisD2  = np.interp(thisa,cosmo.a,cosmo.D2)
   thisD31 = np.interp(thisa,cosmo.a,cosmo.D31)
   thisD32 = np.interp(thisa,cosmo.a,cosmo.D32)

   xx, yy, zz = q +  Cell * (thisD * v1 + thisD2 * v2 + thisD31 * v31 + thisD32 * v32)

   if zcentered:

      xx = ( wrapPositionsPart(xx/Lbox +0.5 ) -0.5 )*Lbox; yy = ( wrapPositionsPart(yy/Lbox +0.5 ) -0.5 )*Lbox; zz = wrapPositionsPart(zz/Lbox)*Lbox;

   else:

     xx = ( wrapPositionsPart(xx/Lbox +0.5 ) -0.5 )*Lbox; yy = ( wrapPositionsPart(yy/Lbox +0.5 ) -0.5 )*Lbox; zz = (wrapPositionsPart(zz/Lbox + 0.5) -0.5 )*Lbox;

   return xx, yy, zz
