import params
from IO import snapshot, Snapshot3
from IO.ReadPinocchio import catalog
from IO.randomization import wrapPositions
import numpy as np
import cosmology
from copy import deepcopy
from colossus.cosmology import cosmology as colossus
from colossus.halo import concentration
from NFW import NFW

#######################  Reading Timeless Snapshot  #######################
snap = snapshot.Timeless_Snapshot(params.pintlessfile)

#getting the ordered Map:
dummy_head = deepcopy(snap.snap.Header)
dummy_map  = deepcopy(snap.snap.Map)
omap       = np.argsort([x.offset[0] for x in dummy_map])[:5]
dummy_map  = [dummy_map[i] for i in omap]
dummy_map[-2].name = b"POS "; dummy_map[-1].name = b"VEL "
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

        ipart = 0
        for m, n, c, r, pos in zip(cat.Mass, cat.Npart, conc, rDelta, cat.pos):

            rr     = NFW.randomr(c, n=n) * r
            rphi   = np.random.uniform(0 ,2*np.pi, n)
            rtheta = np.arccos(np.random.uniform(-1, 1, n))

            pos1[ipart:ipart+n] = pos + \
                np.transpose([rr*np.sin(rtheta)*np.cos(rphi), rr*np.sin(rtheta)*np.sin(rphi), rr*np.cos(rtheta)])

            ipart += n

    # Getting only particles that have not yet being accreated
    print("## Displacing particles")
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
    pos = wrapPositions(pos/params.boxsize) * params.boxsize

    print("## Saving snapshot: {}\n\n"\
       .format( params.pintlessfile.replace("t_snapshot", "{0:5.4f}".format(z))))
    zsnap.write_header(dummy_head)
    zsnap.write_info_block(dummy_map)
    zsnap.write_block(b"ID  ", np.dtype('uint32'), params.nparticles, snap.ID.astype(np.uint32))
    zsnap.write_block(b"POS ", np.dtype('float32'), 3*params.nparticles, pos.astype(np.float32))
    zsnap.write_block(b"VEL ", np.dtype('float32'), 3*params.nparticles, vel.astype(np.float32))

print("All Done!")
