import sys
from cosmology import colossus
from cosmology import concentration
import NFW.NFWx as NFW
import time
import datetime
import numpy as np
import resource
import params
import cosmology
from IO.Pinocchio.ReadPinocchio import catalog
import IO.Pinocchio.TimelessSnapshot as snapshot
from IO.Utils.wrapPositions import wrapPositions
import MAS_library as MASL
import Pk_library as PKL
import lmfit
import pickle

######################### OPTIONS #########################

minimizer = "brute"

################### AUXILIARY FUNCTIONS ###################

rhoc = lambda z: cosmology.lcdm.critical_density(z).to("M_sun/Mpc^3").value/(1+z)**3

def draw_particles (pos, mass, npart, r, theta, phi, A, B, C, z):
    '''
    Returns the position of the particles inside halos assuming a generic 
    CM: c = A * (mass/1e12) ** B + C

    pos: ndarray
        Halo position
    mass: ndarray
        Halo Mass
    npart: ndarray
        Halo number of particles
    r: ndarray
        empty array for particle position
    theta: ndarray
        empty array for particle angular position
    phi: ndarray
        empty array for particle angular position
    A, B, C: float
        CM generic parameters
    '''
    conc = A * (mass/2e12) ** B + C
    rDelta = (3*mass/4/np.pi/200/rhoc(z))**(1.0/3)
    NFW.random_nfw(npart, conc, rDelta.astype(np.float32), r, theta, phi)

def res (pars, **kwargs):

    A        = pars['A'].value
    B        = pars['B'].value
    C        = pars['C'].value

    # displacing particles
    draw_particles(kwargs['pos'], kwargs['mass'], kwargs['npart'],kwargs['r'], kwargs['theta'], kwargs['phi'], A, B, C, kwargs['z'])
    pos_collapsed = np.repeat(kwargs['pos'], kwargs['npart'], axis=0) \
                  + np.transpose([kwargs['r']*np.sin(kwargs['theta'])*np.cos(kwargs['phi']), \
                                  kwargs['r']*np.sin(kwargs['theta'])*np.sin(kwargs['phi']), \
                                  kwargs['r']*np.cos(kwargs['theta'])])
    pos_collapsed /= params.boxsize
    wrapPositions(pos_collapsed)
    pos_collapsed *= params.boxsize
    # Computing delta
    delta = np.copy(delta_uncollapsed)
    MASL.MA(pos_collapsed, delta, params.boxsize, 'CIC', verbose=False)
    Pk = PKL.Pk(delta, params.boxsize, 0, 'CIC', 1, verbose=False)
    # Getting only half of the values
    size = Pk.Nmodes3D.size//2

    sigma = Pk.Pk[:size,0] * np.sqrt( (2.0/Pk.Nmodes3D[:size]) + kwargs['sigma_target']**2 )

    return np.sum( ((Pk.Pk[:size,0] - kwargs['target']) / sigma)**2 )

############################################################

# Looping on the outputlist redshifts
#for z in params.redshifts:
for z in [0.0]:

    print("[{}] # Working on redshift: {}".format(datetime.datetime.now(), z))
    delta_uncollapsed = np.zeros((params.ngrid//2, params.ngrid//2, params.ngrid//2), dtype=np.float32)
    pos   = []
    mass  = []
    npart = []

    # Looping on the number of files
    for i in range(params.numfiles):

        print("[{}] ## Working on file: {}/{}".format(datetime.datetime.now(), i, params.numfiles-1))  
        #  Reading Timeless Snapshot
        start = time.time()
        print("[{}] ### Reading Timeless Snapshot:".format(datetime.datetime.now()))
        snap = snapshot.timeless_snapshot(params.pintlessfile + ".{}".format(i))
        print("[{}] ### Time spent: {} s".format(datetime.datetime.now(), time.time() - start))

        start = time.time()
        print("[{}] ### Displacing Particles outside Halos".format(datetime.datetime.now()))
        filter = (snap.Zacc <= z)
        pos_uncollapsed = snap.snapPos(z, zcentered=False, filter=filter) + params.boxsize/2
        print("[{}] ### Time spent: {} s".format(datetime.datetime.now(), time.time() - start))
        MASL.MA(pos_uncollapsed.astype(np.float32), delta_uncollapsed, params.boxsize, 'CIC', verbose=False)

        start = time.time()
        print("[{}] ### Reading catalog: {}".format(datetime.datetime.now(), params.pincatfile.format(z)))
        cat = catalog(params.pincatfile.format(z)+".{}".format(i))
        pos   += cat.pos.tolist()
        mass  += cat.Mass.tolist(); 
        npart += cat.Npart.tolist(); del cat
 
        print("[{}] ### Time spent: {} s".format(datetime.datetime.now(), time.time() - start))

    # Particle arrays
    r = np.empty( np.sum(npart), dtype=np.float32 )
    theta = np.empty( np.sum(npart), dtype=np.float32 )
    phi = np.empty( np.sum(npart), dtype=np.float32 )
    # Casting into arrays halo arrays
    pos   = np.array(pos, dtype=np.float32)
    mass  = np.array(mass, dtype=np.float32)
    npart = np.array(npart, dtype=np.int32)

    # Parameters
    p = lmfit.Parameters()
    p.add_many(('A', 5.74, True, 2.0, 10.0), ('B', -0.097, True, -1.0, 0.0), ('C', 3.0, True, 2.0, 6.0))

    # Target Pk
    Pk = np.loadtxt("/beegfs/tcastro/Alice/nbody/uhr/Pk_matter_z={0:4.3f}.dat".format(z))[:110, 1]
    # Minimizing
    kws = {'pos':pos, 'mass':mass, 'npart':npart, 'r':r, 'theta':theta, 'phi':phi, 'target': Pk, 'sigma_target': 0.5, 'z':z}

    if minimizer == "brute":

        brute = lmfit.minimize(res, method='brute', nan_policy='omit', Ns=16, keep='all', params=p, kws=kws, workers=-1)
        with open('brute_solution_z={0:4.3f}.pickle'.format(z), 'wb') as handle:
            pickle.dump(brute, handle)

    elif minimizer == "emcee":

        sol = lmfit.minimize(res, method='emcee', nan_policy='omit', burn=0, steps=10, thin=20, params=p, is_weighted=False, progress=False, kws=kws)
        with open('emcee_solution_z={0:4.3f}.pickle'.format(z), 'wb') as handle:
            pickle.dump(sol, handle)

    else:

        print("[{}] ## Minimizer option {} not understood!".format(datetime.datetime.now(), minimizer))
        sys.exit(-1)


    print("[{}] ## High water mark Memory Consumption: {} Gb\n".format(datetime.datetime.now(), \
                                     resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2))

print("[{}] All Done!".format(datetime.datetime.now()))
