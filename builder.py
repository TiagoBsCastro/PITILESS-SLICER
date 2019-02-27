def getCrossingZ(q, v1, v2, v31, v32, zcentered=True):

   zcross = lambda zz: np.interp( np.sqrt( sum( xi*xi for xi in snapPosPart(q, v1, v2, v31, v32, zz, zcentered=zcentered) ) ), Dinterp, zinterp ) - zz
   return opt.root_scalar(zcross, bracket=[0.0,2*zsource], method='bisect', xtol=deltazplc/2, rtol=deltazplc/2, maxiter = 6).root

def buildPLCFiner (zmin, zmax, zcentered=True):
   '''
   Builds the Past Light Cone from z in [zmin,zmax]
   Returns zplc, and the comoving [r, theta, phi]
   '''
   zplc = np.array([ getCrossingZ(*params, zcentered = False) for params in zip(qPos, V1, V2, V31, V32) ])
   sphericalcoord = np.empty((Npart,3))
   xx, yy, zz = np.array([ snapPosPart(*params, zcentered = False) for params in zip(qPos, V1, V2, V31, V32, zplc)]).T
   comovdistance = np.interp( zplc, zinterp, Dinterp)
   crossed = (zplc > zmin) & (zplc <= zmax)

   if not zcentered:
      theta = -np.arccos(zz[crossed]/comovdistance[crossed]) + np.pi/2.0;
      phi   = np.arctan2(yy[crossed],xx[crossed]);
      sphericalcoord[crossed] = np.transpose((comovdistance[crossed],theta,phi))
   else:
      sphericalcoord[crossed] = np.transpose( cartesian_to_spherical(yy[crossed], zz[crossed], xx[crossed]) )

   sphericalcoord[~crossed] = [np.nan, np.nan, np.nan]
   zplc[~crossed] = np.inf

   return zplc, sphericalcoord

def buildPLC (zmin, zmax, repetition, zcentered=True):
   '''
   Builds the Past Light Cone from z in [zmin,zmax]
   Returns zplc, and the comoving [r, theta, phi]
   '''

   zinterp = np.linspace(zmin,1.1*zmax, 100)
   Dinterp = cosmo.comoving_distance(zinterp).value

   crossedinthepast = np.zeros(Npart).astype(bool)
   sphericalcoord = np.empty((Npart,3))
   zplc = np.empty(Npart)

   zettab = np.linspace(zmin, zmax, int( (zmax-zmin)/deltazplc ))
   for zinf, zsup in zip(zettab[:-1],zettab[1:]):

      xx, yy, zz = snapPos(zsup, zcentered=zcentered, filter = ~crossedinthepast)
      xx += repetition[0]*Lbox; yy += repetition[1]*Lbox; zz += repetition[2]*Lbox;
      comovdistance = np.sqrt(xx**2 + yy**2 +zz**2)
      ztrial = np.interp(comovdistance, Dinterp, zinterp)
      crossednow = (ztrial > zinf) & (ztrial <= zsup)
      crossed = np.nonzero(~crossedinthepast)[0][np.nonzero(crossednow)[0]]
      if not zcentered:
         theta = -np.arccos(zz[crossednow]/comovdistance[crossednow]) + np.pi/2.0;
         phi   = np.arctan2(yy[crossednow],xx[crossednow]);
         sphericalcoord[crossed] = np.transpose((comovdistance[crossednow],theta,phi))
      else:
         sphericalcoord[crossed] = np.transpose( cartesian_to_spherical(yy[crossednow], zz[crossednow], xx[crossednow]) )

      zplc[crossed] = ztrial[crossednow]
      crossedinthepast[ crossed ] = True

   zplc[~crossedinthepast] = np.inf
   sphericalcoord[~crossedinthepast] = np.nan

   return zplc, sphericalcoord
