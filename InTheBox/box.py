import sys
from copy import deepcopy
from colossus.cosmology import cosmology as colossus
from colossus.halo import concentration
import NFW.NFWx as NFW
from mpi4py import MPI
import time
import datetime
import numpy as np
import contextlib

# Defining not stdout enviroment
class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

with nostdout():
    import params
    import cosmology
    from IO import Gadget
    from IO.Gadget.Blocks import line
    from IO.Pinocchio.ReadPinocchio import catalog
    import IO.Pinocchio.TimelessSnapshot as snapshot
    from IO.Utils.wrapPositions import wrapPositions

############################### Setting MPI4PY #############################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != params.numfiles:

    print("[{}] PITILESS currently only works with number of threads ({}) equal to number of Pinocchio files ({}).".format(datetime.datetime.now(), size, params.numfiles))
    comm.Abort()

else:

    if size != 1:

        params.pintlessfile += ".{}".format(rank)
        params.pincatfile += ".{}".format(rank)
        params.pinplcfile += ".{}".format(rank)

comm.Barrier()

if not rank:

    print("[{}] STDOUT will be redirected to box_log_{{rank}}.txt.".format(datetime.datetime.now()))
    print("[{}] STDERR will be redirected to box_err_{{rank}}.txt.".format(datetime.datetime.now()))
    print("[{}] Check the files for more information on the run.".format(datetime.datetime.now()))

sys.stdout = open('box_log_{}.txt'.format(rank), 'w')
sys.stderr = open('box_err_{}.txt'.format(rank), 'w')

#######################  Reading Timeless Snapshot  #######################
start = time.time()
print("[{}] # Reading Timeless Snapshot:".format(datetime.datetime.now()))
snap = snapshot.timeless_snapshot(params.pintlessfile)
dummy_head = deepcopy(snap.snap.Header)
print("[{}] # Time spent: {} s".format(datetime.datetime.now(), time.time() - start))
###########################################################################

# Setting cosmology for concentrations
pdict = {'flat': True, 'H0': params.h0true, 'Om0': params.omega0, 'Ob0': params.omegabaryon,\
          'sigma8': cosmology.sigma8, 'ns': params.ns}
colossus.addCosmology('myCosmo', pdict)
colossus.setCosmology('myCosmo')

# Looping on the outputlist redshifts
for z in params.redshifts:

    # Working on the particles inside Halos
    start = time.time()
    print("[{}] # Reading catalog: {}".format(datetime.datetime.now(), params.pincatfile.format(z)))
    with nostdout():
        cat = catalog(params.pincatfile.format(z))
    print("[{}] # Time spent: {} s".format(datetime.datetime.now(), time.time() - start))

    if cat.Mass.size != 0:

        pos1 = np.repeat(cat.pos, cat.Npart, axis=0)
        vel1 = np.repeat(cat.vel, cat.Npart, axis=0)\
              + np.random.randn( np.sum(cat.Npart), 3 ) * 150.0 # Maxwell distribution of velocities with 150 km/s dispersion

        rhoc = cosmology.lcdm.critical_density(z).to("M_sun/Mpc^3").value
        rDelta = (3*cat.Mass/4/np.pi/200/rhoc)**(1.0/3)

        # Getting concentration
        start = time.time()
        print("[{}] ## Computing concentrations".format(datetime.datetime.now()))
        minterp = np.geomspace(cat.Mass.min(), cat.Mass.max())
        cinterp = concentration.concentration(minterp, '200c', z, model = 'bhattacharya13')
        conc    = np.array([np.interp(m, minterp, cinterp) for m in cat.Mass], dtype=np.float32)
        print("[{}] ## Time spent: {} s".format(datetime.datetime.now(), time.time() - start))

        start = time.time()
        print("[{}] ## Sampling Particles in Halos".format(datetime.datetime.now()))
        r = np.empty( np.sum(cat.Npart), dtype=np.float32 )
        theta = np.empty( np.sum(cat.Npart), dtype=np.float32 )
        phi = np.empty( np.sum(cat.Npart), dtype=np.float32 )
        NFW.random_nfw(cat.Npart.astype(np.int32), conc, rDelta.astype(np.float32), r, theta, phi)
        pos1 = pos1 + np.transpose([r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)])
        print("[{}] ## Time spent: {} s".format(datetime.datetime.now(), time.time() - start))

    start = time.time()
    print("[{}] ## Displacing Particles outside Halos".format(datetime.datetime.now()))
    for xxx in range(10):
        filter = (snap.Zacc <= z)
        pos2 = snap.snapPos(z, zcentered=False, filter=filter) + params.boxsize/2
        vel2 = snap.snapVel(z, filter=filter)
    print("[{}] ## Time spent: {} s".format(datetime.datetime.now(), time.time() - start))

    if cat.Mass.size != 0:

        pos = np.vstack([pos1, pos2])/params.boxsize
        vel = np.vstack([vel1, vel2])

    else:

        pos = pos2/params.boxsize
        vel = vel2

    # Wrapping positions
    wrapPositions(pos.astype(np.float32))
    pos *= params.boxsize
    npart = pos.shape[0]

    start = time.time()
    print("[{}] ## Starting comunication with threads:".format(datetime.datetime.now()))
    # Getting the total number of particle for each thread
    if rank:

        totpart = None
        print("[{}] ## Number of Particles in this thread: {}".format(datetime.datetime.now(), npart))
        comm.send(npart, dest=0)

    else:

        totpart = [npart]
        print("[{}] ## Number of Particles in this thread: {}".format(datetime.datetime.now(), npart))

        for i in range(1, size):

            npart = comm.recv(source=i)
            totpart += [npart]

    comm.barrier()
    totpart = comm.bcast(totpart, root=0)

    if rank:
        exclusions = np.sign( np.sum(totpart) - params.nparticles ) * \
                     ( np.abs(np.sum(totpart) - params.nparticles)//size )
    else:
        exclusions = np.sign( np.sum(totpart) - params.nparticles ) * \
                     ( np.abs(np.sum(totpart) - params.nparticles)//size + np.abs(np.sum(totpart) - params.nparticles)%size )

    # Boolean variable in case only the master have to readjust number of particles
    only_master = (np.sum(totpart) - params.nparticles) and not np.bool( (np.sum(totpart) - params.nparticles)//size )
    print("[{}] ## Total number of particles:    {}".format(datetime.datetime.now(), np.sum(totpart) ))
    print("[{}] ## Expected number of particles: {}".format(datetime.datetime.now(), params.nparticles))

    if exclusions:

        if exclusions>0:

            print("[{}] ## I will randomly exclude:      {}".format(datetime.datetime.now(), exclusions))
            idxs = np.random.randint(0, pos.shape[0], exclusions)
            pos = np.delete(pos, idxs, 0)
            vel = np.delete(vel, idxs, 0)

        else:

            print("[{}] ## I will randomly duplicate:      {}".format(datetime.datetime.now(), -exclusions))
            idxs = np.random.randint(0, pos.shape[0], -exclusions)
            pos = np.vstack([pos, pos[idxs]])
            vel = np.vstack([vel, vel[idxs]])

        npart = pos.shape[0]
        # Updating totpart in case all threads have to adjust the number of particles
        if not only_master:

            if rank:

                totpart = None
                print("[{}] ## Number of Particles in this rank: {}".format(datetime.datetime.now(), npart))
                comm.send(npart, dest=0)

            else:

                totpart = [npart]
                print("[{}] ## Number of Particles in this rank: {}".format(datetime.datetime.now(), npart))

                for i in range(1, size):

                    npart = comm.recv(source=i)
                    totpart += [npart]

        # Updating only the master in this case
        else:

            print("[{}] ## Number of Particles in this rank: {}".format(datetime.datetime.now(), npart))
            totpart[0] = npart

    comm.barrier()
    totpart = comm.bcast(totpart, root=0)
    # Updating structures
    if rank:
        ids = np.arange(np.cumsum(totpart)[rank-1], np.cumsum(totpart)[rank])
    else:
        ids = np.arange(0, totpart[rank])

    print("[{}] ## Time spent: {} s".format(datetime.datetime.now(), time.time() - start))

    dummy_head.redshift = z
    dummy_head.time     = 1.0/(1.0+z)
    dummy_head.filenum  = np.array([size], dtype=np.int32)
    dummy_head.npart    = np.array([0, totpart[rank], 0, 0, 0, 0], dtype=np.uint32)

    print("[{}] ## Saving snapshot: {}\n"\
       .format( datetime.datetime.now(), params.pintlessfile.replace("t_snapshot", "{0:5.4f}".format(z))))

    zsnap = Gadget.Init(params.pintlessfile.replace("t_snapshot", "{0:5.4f}".format(z)), -1, ToWrite=True, override=True)
    zsnap.write_header(dummy_head)
    zsnap.write_block(b"ID  ", np.dtype('uint32'),  ids.size, ids.astype(np.uint32))
    zsnap.write_block(b"POS ", np.dtype('float32'), pos.size, pos.astype(np.float32))
    zsnap.write_block(b"VEL ", np.dtype('float32'), vel.size, vel.astype(np.float32))

    del pos, pos1, pos2, vel1, vel2, vel, ids, zsnap

print("[{}] All Done!".format(datetime.datetime.now()))
