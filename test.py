import numpy as np
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

prec_dict = {}

# Test 1
if False:
    for nparti in [100]:

        prec_dict[nparti] = []

        for i in range(100):

            npart  = np.array([nparti], dtype=np.int32)
            mpart  = npart * 1e10
            rdelta = (3*mpart/4/np.pi/200/rhoc)**(1.0/3)
            conc   = 3 + 5 * np.random.random()
            conc   = np.array([conc], dtype=np.float32)
            r      = np.empty(npart, dtype=np.float32)
            theta  = np.empty(npart, dtype=np.float32)
            phi    = np.empty(npart, dtype=np.float32)

            NFWx.random_nfw(npart, conc, r, theta, phi)

            # Getting Radius
            r = r * rdelta
            #rhist = np.linspace(r.min(), r.max(), 100)
            rhist = np.linspace(0.2 * r.max(), r.max(), 100)
            mhist, aux = np.histogram(r, bins=rhist)
            mhist = 1e10 * mhist
            mhist = np.cumsum(mhist)
            mhist = np.append([0], mhist) + np.sum(r<=rhist.min()) * 1e10

            profile = profile_nfw.NFWProfile(M = mpart, mdef = '200c', z = 0.0, c = 4.0)

            fit = profile.fit(rhist*1e3, mhist, 'M')

            prec_dict[nparti] += [conc/(rdelta*1e3/fit['x'][1])]

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

            NFWx.random_nfw(npart, conc, r, theta, phi)

            # Getting Radius
            r = r * rdelta
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
