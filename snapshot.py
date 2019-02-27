def wrapPositions (xx):

   xxoutofbox = (xx < 0.0) | (xx > 1.0)
   xx[xxoutofbox] = np.abs(1.0 - np.abs( xx[xxoutofbox] ))
   del xxoutofbox

   return xx

def wrapPositionsPart (xx):

    return np.abs(1.0 - np.abs( xx )) if ( (xx < 0.0) | (xx > 1.0) ) else xx

def snapPos (z, zcentered=True, filter=None):
   '''
   Returns the particles Position at z
   '''
   if filter is None:
      filter=np.ones(Npart).astype(bool)

   thisa   = 1.0/(1.0+z)
   thisD   = np.interp(thisa,a,D)
   thisD2  = np.interp(thisa,a,D2)
   thisD31 = np.interp(thisa,a,D31)
   thisD32 = np.interp(thisa,a,D32)

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
   thisD   = np.interp(thisa,a,D)
   thisD2  = np.interp(thisa,a,D2)
   thisD31 = np.interp(thisa,a,D31)
   thisD32 = np.interp(thisa,a,D32)

   xx, yy, zz = q +  Cell * (thisD * v1 + thisD2 * v2 + thisD31 * v31 + thisD32 * v32)

   if zcentered:

      xx = ( wrapPositionsPart(xx/Lbox +0.5 ) -0.5 )*Lbox; yy = ( wrapPositionsPart(yy/Lbox +0.5 ) -0.5 )*Lbox; zz = wrapPositionsPart(zz/Lbox)*Lbox;

   else:

     xx = ( wrapPositionsPart(xx/Lbox +0.5 ) -0.5 )*Lbox; yy = ( wrapPositionsPart(yy/Lbox +0.5 ) -0.5 )*Lbox; zz = (wrapPositionsPart(zz/Lbox + 0.5) -0.5 )*Lbox;

   return xx, yy, zz
