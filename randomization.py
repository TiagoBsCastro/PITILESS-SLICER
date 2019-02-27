def randomizePositions (center, face, sgn, pos):

   '''Randomize the positions acording to the SLICER
      random variables center, face, and sgn'''

   xb, yb, zb = (sgn*pos).T
   xb = wrapPositions(xb); yb = wrapPositions(yb); zb = wrapPositions(zb);

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

   del xb, yb, zb

   xx -= center[0]; yy -= center[1]; zz -= center[2];
   xx = wrapPositions(xx); yy = wrapPositions(yy); zz = wrapPositions(zz);

   return np.transpose([xx - 0.5, yy - 0.5, zz])

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

   return np.transpose([xx, yy, zz])
