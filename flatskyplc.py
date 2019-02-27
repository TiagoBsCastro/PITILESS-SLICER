   # comoving Mpc
   qPos = np.array([ (ID-1)%NG,((ID-1)//NG)%NG,((ID-1)//NG**2)%NG ]).transpose().astype(float) * Cell + Cell/2.
   qPos = randomizePositions(center, face, sgn, qPos/Lbox)*Lbox
   V1 = randomizeVelocities(face, sgn, V1)
   V2 = randomizeVelocities(face, sgn, V2)
   V31 = randomizeVelocities(face, sgn, V31)
   V32 = randomizeVelocities(face, sgn, V32)

   zplc, sphericalcoord = buildPLC(0.0,zsource)
   cut = ~np.isnan(sphericalcoord[:,1]) & ( np.abs(sphericalcoord[:,1]) <= fovinradians/2.0 ) & (np.abs(sphericalcoord[:,2] - np.pi/2.0) <= fovinradians/2.0) & (zplc < Zacc)

   theta, phi, zplc = sphericalcoord[:,1][cut], sphericalcoord[:,2][cut], zplc[cut]

   for i,(z1,z2) in enumerate( zip(zlinf, zlsup) ):

      lensi = (zplc <= z2) & (zplc > z1)
      thetai, phii, zplci = theta[lensi], phi[lensi], zplc[lensi]
      deltai = np.histogram2d(thetai, phii, bins = npixels, range = [[-fovinradians/2.0,fovinradians/2.0],[np.pi/2.0-fovinradians/2.0, np.pi/2.0+fovinradians/2.0]])[0]
      deltai = deltai/deltai.mean() - 1.0
      zl = (z1+z2)/2.0
      kappai = (1+zl) * ( ( 1.0 - cosmo.comoving_distance(zl)/cosmo.comoving_distance(zsource) ) *\
                              cosmo.comoving_distance(zl) * ( cosmo.comoving_distance(z2) - cosmo.comoving_distance(z1) ) ).to_value() * deltai
      kappai *= (3.0 * cosmo.Om0*cosmo.H0**2/2.0/cspeed).to_value()
      hdu = fits.PrimaryHDU(kappai)
      hdu.writeto('kappa_field_{}.fits'.format(str(round(zl,2))))
