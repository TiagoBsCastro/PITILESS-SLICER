import sys
from copy import deepcopy
from cosmology import colossus
from cosmology import concentration
import NFW.NFWx as NFW
from mpi4py import MPI
import time
import datetime
import numpy as np
import contextlib
import resource
from IO.Utils.print import print
import os

############################### Setting MPI4PY #############################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

############################### Setting STDOUT #############################

print("STDOUT will be redirected to box_log_{{rank}}.txt.", rank=rank)
print("STDERR will be redirected to box_err_{{rank}}.txt.", rank=rank)
print("Check the files for more information on the run.", rank=rank)

sys.stdout = open('box_log_{}.txt'.format(rank), 'w')
sys.stderr = open('box_err_{}.txt'.format(rank), 'w')

# Defining not stdout enviroment
class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

import params
import cosmology
import g3read
from IO.Pinocchio.ReadPinocchio import catalog
from IO.Pinocchio.ReadPinocchio5 import catalog as catalog5
import IO.Pinocchio.TimelessSnapshot as snapshot
from IO.Utils.wrapPositions import wrapPositions

if size != params.numfiles:

    print("PITILESS currently only works with number of threads ({}) equal to number of Pinocchio files ({}).".format(size, params.numfiles))
    comm.Abort()

else:

    if size != 1:

        params.pintlessfile += ".{}".format(rank)
        params.pincatfile += ".{}".format(rank)
        params.pinplcfile += ".{}".format(rank)

comm.barrier()

#######################  Reading Timeless Snapshot  #######################
start = time.time()
print("# Reading Timeless Snapshot:")
snap = snapshot.timeless_snapshot(params.pintlessfile, randomize=False, changebasis=False)
dummy_head = deepcopy(snap.Header)
print("# Time spent: {0:.3f} s".format(time.time() - start))

###########################################################################

# Looping on the outputlist redshifts
for z in params.redshifts:

    # Working on the particles inside Halos
    start = time.time()
    print("# Reading catalog: {}".format(params.pincatfile.format(z)))
    cat_file = params.pincatfile.format(z)

    if os.path.getsize(cat_file) !=0 :
        with nostdout():
            try:
                cat = catalog(cat_file)
            except IndexError:
                cat = catalog5(cat_file)
        print("# Time spent: {0:.3f} s".format( time.time() - start))

        if cat.Mass.size != 0:

            pos1 = np.repeat(cat.pos, cat.Npart, axis=0)
            vel1 = np.repeat(cat.vel, cat.Npart, axis=0)\
              + np.random.randn( np.sum(cat.Npart), 3 ) * 150.0 # Maxwell distribution of velocities with 150 km/s dispersion

            rhoc = cosmology.lcdm.critical_density(z).to("M_sun/Mpc^3").value/(1+z)**3
            rDelta = (3*cat.Mass/4/np.pi/200/rhoc)**(1.0/3)

            # Getting concentration
            start = time.time()
            print("## Computing concentrations".format())
            if params.cmmodel == 'mybhattacharya':

                # Bahattacharya fits for nu and c(M) - Table-2/200c-Full
                Da   = np.interp(1.0/(1.0+z), cosmology.a, cosmology.D)
                nu   = 1.0/Da * (1.12*(cat.Mass/5.0/1e13)**0.3 + 0.53 )
                conc = (Da**0.54 * 5.9 * nu**(-0.35)).astype(np.float32)

            else:

                minterp = np.geomspace(cat.Mass.min(), cat.Mass.max())
                cinterp = concentration.concentration(minterp, '200c', z, model = params.cmmodel)
                conc    = np.array([np.interp(m, minterp, cinterp) for m in cat.Mass], dtype=np.float32)

            print("## Time spent: {0:.3f} s".format( time.time() - start))

            start = time.time()
            print("## Sampling Particles in Halos")
            r = np.empty( np.sum(cat.Npart), dtype=np.float32 )
            theta = np.empty( np.sum(cat.Npart), dtype=np.float32 )
            phi = np.empty( np.sum(cat.Npart), dtype=np.float32 )
            NFW.random_nfw(cat.Npart.astype(np.int32), conc, rDelta.astype(np.float32), r, theta, phi)
            pos1 = pos1 + np.transpose([r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)])
            print("## Time spent: {0:.3f} s".format( time.time() - start))

    start = time.time()
    print("## Displacing Particles outside Halos")
    filter = (snap.Zacc <= z)
    pos2 = snap.snapPos(z, zcentered=False) + params.boxsize/2
    pos2 = pos2[filter]
    vel2 = snap.snapVel(z, filter=filter)
    ids1, ids2 = snap.ID[~filter], snap.ID[filter]
    print("## Time spent: {0:.3f} s".format( time.time() - start))

    if os.path.getsize(cat_file) !=0 :
        with nostdout():
            try:
                cat = catalog(cat_file)
            except IndexError:
                cat = catalog5(cat_file)
        
        if cat.Mass.size != 0 :

            pos = np.vstack([pos1, pos2])/params.boxsize
            vel = np.vstack([vel1, vel2])
            ids = np.vstack([ids1, ids2])

            del pos1, pos2, vel1, vel2, ids1, ids2

    else:

        pos = pos2/params.boxsize
        vel = vel2
        ids = ids2

        del pos2, vel2, ids2

    # Wrapping positions
    pos = pos.astype(np.float32)
    wrapPositions(pos)
    pos  *= params.boxsize
    npart = pos.shape[0]

    start = time.time()
    print("## Starting comunication with threads:")
    # Getting the total number of particle for each thread
    if rank:

        totpart = None
        print("## Number of Particles in this rank: {}".format( npart))
        comm.send(npart, dest=0)

    else:

        totpart = [npart]
        print("## Number of Particles in this rank: {}".format( npart))

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
    only_master = (np.sum(totpart) - params.nparticles) and not np.bool_( np.abs(np.sum(totpart) - params.nparticles)//size )
    print("## Total number of particles:    {}".format( np.sum(totpart) ))
    print("## Expected number of particles: {}".format( params.nparticles))

    if (not exclusions) and only_master and rank:

        print("## Only the master will readjust the number of particles.")

    if exclusions:

        if exclusions>0:

            print("## I will randomly exclude:      {}".format( exclusions))
            idxs = np.random.permutation(pos.shape[0])[ : exclusions]
            pos = np.delete(pos, idxs, 0)
            vel = np.delete(vel, idxs, 0)

        else:

            print("## I will randomly duplicate:      {}".format( -exclusions))
            idxs = np.random.randint(0, pos.shape[0], -exclusions)
            pos = np.vstack([pos, pos[idxs]])
            vel = np.vstack([vel, vel[idxs]])

        npart = pos.shape[0]
        # Updating totpart in case all threads have to adjust the number of particles
        if not only_master:

            if rank:

                totpart = None
                print("## Number of Particles in this rank: {}".format( npart))
                comm.send(npart, dest=0)

            else:

                totpart = [npart]
                print("## Number of Particles in this rank: {}".format( npart))

                for i in range(1, size):

                    npart = comm.recv(source=i)
                    totpart += [npart]

        # Updating only the master in this case
        else:

            print("## Number of Particles in this rank: {}".format( npart))
            totpart[0] = npart

    comm.barrier()
    totpart = comm.bcast(totpart, root=0)
    # Updating structures
    print("## Time spent: {0:.3f} s".format( time.time() - start))

    dummy_head.redshift = z
    dummy_head.time     = 1.0/(1.0+z)
    dummy_head.npart    = np.array([0, totpart[rank], 0, 0, 0, 0], dtype=np.uint32)

    print("## Saving snapshot: {}"\
       .format( params.pintlessfile.replace("t_snapshot", "{0:5.4f}".format(z))))

    filename = params.pintlessfile.replace("t_snapshot", "{0:5.4f}".format(z))
    with open(filename, 'a') as f: #create file if doesn't exists
        pass

    # write header to file
    # generate header
    header = g3read.GadgetHeader(dummy_head.npart, dummy_head.mass, dummy_head.time, dummy_head.redshift, dummy_head.BoxSize, dummy_head.Omega0, dummy_head.OmegaLambda, dummy_head.HubbleParam, num_files=dummy_head.num_files)
    f = g3read.GadgetWriteFile(filename, dummy_head.npart, {}, header) #write header file
    f.write_header(f.header)

    #allocate blocks data
    f.add_file_block('ID  ', ids.size*8, partlen=8, dtype=np.int64)
    f.add_file_block('POS ', pos.size*4, partlen=4*3)
    f.add_file_block('VEL ', vel.size*4, partlen=4*3)

    #write blocks data to disk
    f.write_block( 'ID  ', -1, ids.astype(np.int64))
    f.write_block( 'POS ', -1, pos)
    f.write_block( 'VEL ', -1, vel)

    del pos, vel, ids

    print("## High water mark Memory Consumption: {0:.3f} Gb".format( \
                                     resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2))

print("All Done!")
