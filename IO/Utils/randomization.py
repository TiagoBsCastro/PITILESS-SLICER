import numpy as np
from IO.Utils.wrapPositions import wrapPositions

def randomizePositions (center, face, sgn, pos):
   '''Randomize the positions acording to the SLICER
      random variables center, face, and sgn'''

   temp = np.ascontiguousarray(sgn*pos, dtype=np.float32)
   wrapPositions(temp)

   if face == 1:
      xx = temp[:, 0]
      yy = temp[:, 1]
      zz = temp[:, 2]
   elif face == 2:
      xx = temp[:, 0]
      yy = temp[:, 2]
      zz = temp[:, 1]
   elif face == 3:
      xx = temp[:, 1]
      yy = temp[:, 2]
      zz = temp[:, 0]
   elif face == 4:
      xx = temp[:, 1]
      yy = temp[:, 0]
      zz = temp[:, 2]
   elif face == 5:
      xx = temp[:, 2]
      yy = temp[:, 0]
      zz = temp[:, 1]
   elif face == 6:
      xx = temp[:, 2]
      yy = temp[:, 1]
      zz = temp[:, 0]

   xx -= center[0]; yy -= center[1]; zz -= center[2];
   wrapPositions(temp)
   temp = np.asfortranarray([xx-0.5, yy-0.5, zz-0.5], dtype=np.float32).T
   wrapPositions(temp)

   return temp

def randomizeVelocities (face, sgn, vel):

   '''Randomize velocities acording to the SLICER
      random variables center, face, and sgn'''

   xb, yb, zb = (vel*sgn).T

   if face == 1:
      xx = xb
      yy = yb
      zz = zb
   elif face == 2:
      xx = xb
      yy = zb
      zz = yb
   elif face == 3:
      xx = yb
      yy = zb
      zz = xb
   elif face == 4:
      xx = yb
      yy = xb
      zz = zb
   elif face == 5:
      xx = zb
      yy = xb
      zz = yb
   elif face == 6:
      xx = zb
      yy = yb
      zz = xb

   return np.asfortranarray([xx, yy, zz], dtype=np.float32).T
