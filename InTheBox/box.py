import params
from IO import snapshot, Snapshot3
from IO.Snapshot3.Blocks import line
from IO.ReadPinocchio import catalog
from IO.randomization import wrapPositions
import numpy as np
import cosmology
from copy import deepcopy
from colossus.cosmology import cosmology as colossus
from colossus.halo import concentration
from NFW import NFW
from mpi4py import MPI
import sys

############################### Setting MPI4PY #############################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != params.numfiles:

    print("PITILESS currently only works with number of threads ({}) equal to number of Pinocchio files ({}).".format(size, params.numfiles))
    comm.Abort()

else:

    if size != 1:

        params.pintlessfile += ".{}".format(rank)
        params.pincatfile += ".{}".format(rank)
        params.pinplcfile += ".{}".format(rank)

comm.Barrier()

if not rank:

    print("STDOUT will be redirected to box_log_{rank}.txt.")
    print("STDERR will be redirected to box_err_{rank}.txt.")
    print("Check the files for more information on the run.")

sys.stdout = open('box_log_{}.txt'.format(rank), 'w')
sys.stderr = open('box_err_{}.txt'.format(rank), 'w')

#######################  Reading Timeless Snapshot  #######################
snap = snapshot.Timeless_Snapshot(params.pintlessfile)

#getting the ordered Map:
dummy_head = deepcopy(snap.snap.Header)
#dummy_map  = deepcopy(snap.snap.Map)
#omap       = np.argsort([x.offset[0] for x in dummy_map])[:5]
#dummy_map  = [dummy_map[i] for i in omap]
#dummy_map[-2].name = b"POS "; dummy_map[-1].name = b"VEL "
###########################################################################

# Setting cosmology for concentrations
pdict = {'flat': True, 'H0': params.h0true, 'Om0': params.omega0, 'Ob0': params.omegabaryon,\
          'sigma8': cosmology.sigma8, 'ns': params.ns}
colossus.addCosmology('myCosmo', pdict)
colossus.setCosmology('myCosmo')

# Looping on the outputlist redshifts
for z in params.redshifts:

    # Working on the particles inside Halos
    print("# Reading catalog: {}".format(params.pincatfile.format(z)))
    cat = catalog(params.pincatfile.format(z))

    if cat.Mass.size != 0:

        pos1 = np.zeros( (np.sum(cat.Npart), 3) )
        vel1 = np.random.randn( np.sum(cat.Npart), 3 ) * 150.0 # Maxwell distribution of velocities with 150 km/s dispersion

        rhoc = cosmology.lcdm.critical_density(z).to("M_sun/Mpc^3").value
        rDelta = (3*cat.Mass/4/np.pi/200/rhoc)**(1.0/3)

        # Getting concentration
        print("## Computing concentrations")
        minterp = np.geomspace(cat.Mass.min(), cat.Mass.max())
        cinterp = concentration.concentration(minterp, '200c', z, model = 'bhattacharya13')
        conc    = np.array([np.interp(m, minterp, cinterp) for m in cat.Mass])

        print("## Sampling Particles in Halos")
        ipart = 0
        for m, n, c, r, pos in zip(cat.Mass, cat.Npart, conc, rDelta, cat.pos):

            rr     = NFW.randomr(c, n=n) * r
            rphi   = np.random.uniform(0 ,2*np.pi, n)
            rtheta = np.arccos(np.random.uniform(-1, 1, n))

            pos1[ipart:ipart+n] = pos + \
                np.transpose([rr*np.sin(rtheta)*np.cos(rphi), rr*np.sin(rtheta)*np.sin(rphi), rr*np.cos(rtheta)])

            ipart += n

    filter = (snap.Zacc <= z)
    pos2 = np.transpose(snap.snapPos(z, zcentered=False, filter=filter)) + params.boxsize/2
    vel2 = np.transpose(snap.snapVel(z, filter=filter))

    dummy_head.redshift = z
    dummy_head.time     = 1.0/(1.0+z)
    # Copying the Map from timeless snapshot
    zsnap = Snapshot3.Init(params.pintlessfile.replace("t_snapshot", "{0:5.4f}".format(z)), -1, ToWrite=True, override=True)

    if cat.Mass.size != 0:

        pos = np.vstack([pos1, pos2])
        vel = np.vstack([vel1, vel2])

    else:

        pos = pos2
        vel = vel2

    # Wrapping positions
    pos   = wrapPositions(pos/params.boxsize) * params.boxsize
    npart = pos.shape[0]

    # Getting the total number of particle for each thread
    if rank:

        totpart = None
        print("## Total number of Particles: {}".format(npart))
        comm.send(npart, dest=0)

    else:

        totpart = npart
        print("## Total number of Particles: {}".format(totpart))

        for i in range(1, size):

            npart = comm.recv(source=i)
            print("Received: {} from {}".format(npart, i))
            totpart += npart

    comm.barrier()
    totpart = comm.bcast(totpart, root=0)

    if rank:
        exclusions = (totpart - params.nparticles)//size
    else:
        exclusions = (totpart - params.nparticles)//size + (totpart - params.nparticles)%size
    print("## Total number of particles:    {}".format(totpart))
    print("## Expected number of particles: {}".format(params.nparticles))

    if exclusions:

        print("## I will randomly exclude:      {}".format(exclusions))
        idxs = np.random.randint(0, pos.shape[0], exclusions)
        pos = np.delete(pos, idxs, 0)
        vel = np.delete(vel, idxs, 0)

    ids = np.empty( pos.shape[0], dtype=np.int32 )

    print("## Saving snapshot: {}\n\n"\
       .format( params.pintlessfile.replace("t_snapshot", "{0:5.4f}".format(z))))

    map = []
    map += [line(b'HEAD', 256, 'FLOAT   ', 1, np.array([1,0,0,0,0,0], dtype=np.int32), 20, 3)]
    map += [line(b'INFO', 200, 'FLOAT   ', 1, np.array([1,0,0,0,0,0], dtype=np.int32), 300, 3)]
    map += [line(b'ID  ', ids.size, 'LONG    ', 1, np.array([0,1,0,0,0,0], dtype=np.int32), 524, 1)]
    map += [line(b'POS ', pos.size, 'FLOATN  ', 3, np.array([0,1,0,0,0,0], dtype=np.int32), ids.size + 548, 1)]
    map += [line(b'VEL ', vel.size, 'FLOATN  ', 3, np.array([0,1,0,0,0,0], dtype=np.int32), pos.size +ids.size + 572, 1)]

    zsnap.write_header(dummy_head)
    zsnap.write_info_block(map)
    print("## ID  size: {}".format(ids.size))
    print("## POS size: {}".format(pos.size))
    print("## VEL size: {}".format(vel.size))
    zsnap.write_block(b"ID  ", np.dtype('uint32'),  ids.size, ids.astype(np.uint32))
    zsnap.write_block(b"POS ", np.dtype('float32'), pos.size, pos.astype(np.float32))
    zsnap.write_block(b"VEL ", np.dtype('float32'), vel.size, vel.astype(np.float32))

print("All Done!")
