import numpy as np
from NFW.NFW import randomr
from NFW import NFWx
import params
import cosmology
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology as colossus
from colossus.halo import profile_nfw
from colossus.halo import concentration

# Setting cosmology for concentrations
pdict = {'flat': True, 'H0': params.h0true, 'Om0': params.omega0, 'Ob0': params.omegabaryon,\
          'sigma8': cosmology.sigma8, 'ns': params.ns}
colossus.addCosmology('myCosmo', pdict)
colossus.setCosmology('myCosmo')
rhoc = cosmology.lcdm.critical_density(0.0).to("M_sun/Mpc^3").value

mp = 1e8

prec_dict = {}

# Test 1
if False:

    for nparti in [100000]:

        prec_dict[nparti] = []
        npart  = np.array([nparti], dtype=np.int32)
        mpart  = npart * mp
        rdelta = ((3*mpart/4/np.pi/200/rhoc)**(1.0/3)).astype(np.float32)
        conc   = 4.0
        print("Mass: {}, r_Delta: {}, c:{}".format(mpart, rdelta, conc))
        conc   = np.array([conc], dtype=np.float32)
        r      = np.empty(npart, dtype=np.float32)
        theta  = np.empty(npart, dtype=np.float32)
        phi    = np.empty(npart, dtype=np.float32)

        for i in range(300):
            print(conc.dtype, rdelta.dtype, r.dtype, theta.dtype, phi.dtype)
            NFWx.random_nfw(npart, conc, rdelta, r, theta, phi)
            #r = randomr(conc[0], n=npart[0])

            # Getting Radius
            R = r
            rhist = np.linspace(R.min(), R.max())
            mhist, aux = np.histogram(R, bins=rhist)
            mhist = mp * mhist
            mhist = np.cumsum(mhist)
            mhist = np.append([0], mhist) #\+ np.sum(r<=rhist.min()) * 1e10

            profile = profile_nfw.NFWProfile(M = mpart, mdef = '200c', z = 0.0, c = conc[0])
            mfid = profile.enclosedMass(rhist*1e3)
            print(profile.enclosedMass(rdelta*1e3)/mpart)
            #plt.plot(rhist, mhist)
            #plt.plot(rhist, mfid)
            #plt.show()
            fit = profile.fit(rhist*1e3, mhist, 'M')

            prec_dict[nparti] += [conc/(rdelta*1e3/fit['x'][1])]

        plt.plot(rhist, mhist)
        plt.plot(rhist, fit['q_fit'])
        plt.show()
        plt.hist( np.array(prec_dict[nparti]).flatten(), bins = 30 )
        plt.show()


# Test 2
if True:

    for nparti in [100, 1000, 10000, 10000]:

        for i in range(1000):

            npart  = np.array([nparti], dtype=np.int32)
            mpart  = npart * 1e10
            rdelta = (3*mpart/4/np.pi/200/rhoc)**(1.0/3)
            conc   = 3 + 5 * np.random.random()
            conc   = np.array([conc], dtype=np.float32)
            r      = np.empty(npart, dtype=np.float32)
            theta  = np.empty(npart, dtype=np.float32)
            phi    = np.empty(npart, dtype=np.float32)

            NFWx.random_nfw(npart, conc, rdelta.astype(np.float32), r, theta, phi)

            # Getting Radius
            r = r
            rhist = np.linspace(r.min(), r.max())
            mhist, aux = np.histogram(r, bins=rhist)
            mhist = 1e10 * mhist
            mhist = np.cumsum(mhist)
            mhist = np.append([0], mhist)

            profile = profile_nfw.NFWProfile(M = mpart, mdef = '200c', z = 0.0, c = 4.0)
            fit = profile.fit(rhist*1e3, mhist, 'M')

            if i == 0:

                data = fit['q_fit']/mhist

            else:

                data += fit['q_fit']/mhist

        plt.plot(rhist/rdelta, data/(i+1), label="Npart: {}".format(nparti))

    plt.show()
