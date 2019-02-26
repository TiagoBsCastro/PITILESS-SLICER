import numpy as np
import matplotlib.pyplot as plt
import Snapshot as S
import copy
import os

snap=S.Init("pinocchio.cdm.t_snapshot.out",-1)
ID=snap.read_block('ID')
V1=snap.read_block('VZEL')
V2=snap.read_block('V2')
V31=snap.read_block('V3_1')
V32=snap.read_block('V3_2')
F=snap.read_block('FMAX')
R=snap.read_block('RMAX')
Zacc=snap.read_block('ZACC')

Npart=len(ID)
NG=np.int(np.float(Npart)**(1./3.)+0.5)
Lbox=snap.Header.boxsize
Cell=Lbox/float(NG)

# comoving Mpc
qPos = np.array([ (ID-1)%NG,((ID-1)//NG)%NG,((ID-1)//NG**2)%NG ]).transpose().astype(float) * Cell + Cell/2.0

(a, t, D, D2, D31, D32) = np.loadtxt('pinocchio.cdm.cosmology.out',usecols=(0,1,2,3,4,5),unpack=True)

outputs=1.0/(1.0+np.loadtxt("outputs"))

header = copy.deepcopy(snap.Header)
i=0
for scalef in outputs:
    fname="snap_%03d"%i
    thisD=np.interp(scalef,a,D)
    thisD2=np.interp(scalef,a,D2)
    thisD31=np.interp(scalef,a,D31)
    thisD32=np.interp(scalef,a,D32)

    header.time=scalef
    header.redshift=1./scalef-1.

    print(( "a=%f, D=%f, D2=%f, D31=%f, D32=%f"%(scalef,thisD,thisD2,thisD31,thisD32) ))
    Pos = qPos +  Cell * (thisD * V1 + thisD2 * V2 + thisD31 * V31 + thisD32 * V32)
    Pos[Pos>=snap.Header.boxsize]-=snap.Header.boxsize
    Pos[Pos<0]+=snap.Header.boxsize

    print(Pos)

    f=open(fname,"wb")
    # header
    np.array([8],dtype=np.int32).tofile(f)
    f.write(b'HEAD')
    np.array([264,8],dtype=np.int32).tofile(f)
    np.array([256],dtype=np.int32).tofile(f)
    header.npart.tofile(f)
    header.massarr.tofile(f)
    header.time.tofile(f)
    header.redshift.tofile(f)
    header.sfr.tofile(f)
    header.feedback.tofile(f)
    header.nall.tofile(f)
    header.cooling.tofile(f)
    header.filenum.tofile(f)
    header.boxsize.tofile(f)
    header.omega_m.tofile(f)
    header.omega_l.tofile(f)
    header.hubble.tofile(f)
    header.stellarage.tofile(f)
    header.metals.tofile(f)
    header.nallHigh.tofile(f)
    header.entropy.tofile(f)
    header.metalcool.tofile(f)
    header.stellarev.tofile(f)
    np.zeros(13,dtype=np.int32).tofile(f)
    np.array([256],dtype=np.int32).tofile(f)

    # positions
    Nbytes = Npart * 12
    np.array([8],dtype=np.int32).tofile(f)
    f.write(b"POS ")
    np.array([Nbytes+8,8],dtype=np.int32).tofile(f)
    np.array([Nbytes],dtype=np.int32).tofile(f)
    Pos.astype(np.float32).tofile(f)
    np.array([Nbytes],dtype=np.int32).tofile(f)

    # velocities
    np.array([8],dtype=np.int32).tofile(f)
    f.write(b"VEL ")
    np.array([Nbytes+8,8],dtype=np.int32).tofile(f)
    np.array([Nbytes],dtype=np.int32).tofile(f)
    V1.astype(np.float32).tofile(f)
    np.array([Nbytes],dtype=np.int32).tofile(f)

    # IDs
    Nbytes = Npart * 4
    np.array([8],dtype=np.int32).tofile(f)
    f.write(b"ID  ")
    np.array([Nbytes+8,8],dtype=np.int32).tofile(f)
    np.array([Nbytes],dtype=np.int32).tofile(f)
    ID.astype(np.int32).tofile(f)
    np.array([Nbytes],dtype=np.int32).tofile(f)

    f.close()

    print(( fname+" done"))
    i+=1


# consistency check

for i in range(len(outputs)-1,0,-1):
    redshift=1./outputs[i]-1.
    pinsn=S.Init("pinocchio.%6.4f.cdm.snapshot.out"%redshift,-1)
    pPos=pinsn.read_block("POS")

    tless=S.Init("snap",i)
    tPos=tless.read_block("POS")

    diff=(tPos-pPos)/Cell
    diff[diff>NG/2]-=NG
    diff[diff<-NG/2]+=NG

    print(( "%d, redshift %6.4f"%(i,redshift)))
    print(( "average difference: ",np.mean(diff)))
    print(( "rms difference: ",np.sqrt(np.var(diff))))
    print(( "max difference: ",np.max(np.abs(diff))))

plt.figure("TLess")
cut = tPos[::,0] < 10
plt.scatter(tPos[::,1][cut], tPos[::,2][cut], s=0.01)

plt.figure("Original")
cut = pPos[::,0] < 10
plt.scatter(pPos[::,1][cut], pPos[::,2][cut], s=0.01)

plt.show()
