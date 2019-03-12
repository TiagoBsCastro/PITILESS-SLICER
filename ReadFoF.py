"""

Routines to read FoF catalogs from GADGET

This library recognizes two formats for the FoF catalogs: old format and GADGET3 format.
It checks the endianness of the files and automatically swaps the file if needed

Basic use:

  import ReadFoF
  fof = ReadFoF.catalog([base directory],[snapshot number], SFR=[False(def),True],verbose=[0,1])

base directory: the directory that contains the output files of the simulation 
(or of the postprocessing if FoF has not been run on the fly)
snapshot number: can be an integer or a string
SFR (optional): read the GroupSFR field
verbose (optional): 0 to keep it quiet

To read IDs for particles in halos:

  fof.read_IDs(long_IDs=[False(def),True],verbose=[0,1])

To know what the object fof contains:

  fof.help()

Example:
  >>> fof=ReadFoF.catalog("",100)
  >>> fof.read_IDs(True)
  >>> fof.help()


2016, written by Pierluigi Monaco (on the basis of older code)

"""

import numpy as np
import os



def myswap(a,flag):
    if flag:
        return a.byteswap()
    else:
        return a



def guess_format(fname,verbose=0):

    # checks how many files are found for the FoF catalog
    nfiles=-1
    exists=True
    while exists:
        nfiles+=1
        exists=os.path.exists(fname+".%d"%nfiles)

    if nfiles==0:
        if verbose>0:
            print("ERROR: FoF file "+fname+" not found")
        return None

    if verbose>0:
        print("FoF catalog found in %d files"%nfiles)

    # reads the header from file N. 0
    f=open(fname+".0","rb")
    f.seek(12,os.SEEK_SET)
    number=np.fromfile(f,dtype=np.uint32,count=1)[0]

    # in the old format this should be =nfiles
    if number==nfiles:
        if verbose>0:
            print("This is old format with native endianness")

        f.close()
        return (nfiles, 0, False)

    elif number.byteswap()==nfiles:
        if verbose>0:
            print("This is old format with inverted endianness")

        f.close()
        return (nfiles, 0, True)

    f.seek(20,os.SEEK_SET)
    number=(np.fromfile(f,dtype=np.uint32,count=1))[0]

    # in the G3 format this should be =nfiles
    if number==nfiles:
        if verbose>0:
            print("This is G3 format with native endianness")

        f.close()
        return (nfiles, 1, False)

    elif number.byteswap()==nfiles:
        if verbose>0:
            print("This is G3 format with inverted endianness")

        f.close()
        return (nfiles, 1, True)

    if verbose>0:
        print("I do not recognise this format")
    return None


            

class catalog:

    """

    This class defines a FoF catalog read from a simulation.
    For more details, use the help() method applied to the object
    
    Example:
    >>> fof=ReadFoF.catalog("",100)
    >>> fof.read_IDs(True)
    >>> fof.help()

    """

    def __init__(self,basedir,snapnum,SFR=False,verbose=0):
        # first it guesses which is the format
        if type(snapnum) is int:
            snapnum="%03d"%snapnum
        if (basedir!="" and basedir[-1]!="/"):
            basedir+="/"
        fname=basedir+"groups_"+snapnum+"/group_tab_"+snapnum

        gform=guess_format(fname,verbose)

        self.snapnum=snapnum
        self.basedir=basedir

        if gform==None:
            print("Error in reading FoF catalogs")
            return

        (nfiles,myformat,self.swap) = gform
        self.Nfiles=nfiles
        self.myformat=myformat


        #################  READ TAB FILES ################# 
        fnb,skip=0,0
        Final=False
        if myformat==0:
            self.TotNids=long(0)
        while not(Final):
            f=open(fname+".%d"%fnb,'rb')

            if myformat==1:
                (Ngroups,TotNgroups,Nids)=myswap(np.fromfile(f,dtype=np.int32,count=3),self.swap)
                TotNids=myswap(np.fromfile(f,dtype=np.uint64,count=1),self.swap)[0]
                Nfiles=myswap(np.fromfile(f,dtype=np.uint32,count=1),self.swap)[0]
            else:
                (Ngroups,Nids,TotNgroups,Nfiles)=myswap(np.fromfile(f,dtype=np.int32,count=4),self.swap)

            if myformat==0:
                self.TotNids+=Nids

            if fnb==0:
                self.TotNgroups=TotNgroups
                if myformat==1:
                    self.TotNids=TotNids

            if Nfiles != nfiles:
                print("WARNING: inconsistency, ",nfiles," files found but the header gives",self.Nfiles)

            if verbose>0:
                print()
                print("File N. ",fnb,":")
                print("Ngroups = ",Ngroups)
                print("TotNgroups = ",TotNgroups)
                print("Nids = ",Nids)
                print("TotNids = ",self.TotNids)
                print("Nfiles = ",Nfiles)


            if myformat==1:  # G3 format
                if fnb == 0:
                    self.GroupLen=np.empty(TotNgroups,dtype=np.int32)
                    self.GroupOffset=np.empty(TotNgroups,dtype=np.int32)
                    self.GroupMass=np.empty(TotNgroups,dtype=np.float32)
                    self.GroupPos=np.empty(TotNgroups,dtype=np.dtype((np.float32,3)))
                    self.GroupVel=np.empty(TotNgroups,dtype=np.dtype((np.float32,3)))
                    self.GroupTLen=np.empty(TotNgroups,dtype=np.dtype((np.int32,6)))
                    self.GroupTMass=np.empty(TotNgroups,dtype=np.dtype((np.float32,6)))
                    if SFR:
                        self.GroupSFR=np.empty(TotNgroups,dtype=np.float32)
                if Ngroups>0:
                    locs=slice(skip,skip+Ngroups)
                    self.GroupLen[locs]    = myswap(np.fromfile(f,dtype=np.int32,count=Ngroups),self.swap)
                    self.GroupOffset[locs] = myswap(np.fromfile(f,dtype=np.int32,count=Ngroups),self.swap)
                    self.GroupMass[locs]   = myswap(np.fromfile(f,dtype=np.float32,count=Ngroups),self.swap)
                    self.GroupPos[locs]    = myswap(np.fromfile(f,dtype=np.dtype((np.float32,3)),count=Ngroups),self.swap)
                    self.GroupVel[locs]    = myswap(np.fromfile(f,dtype=np.dtype((np.float32,3)),count=Ngroups),self.swap)
                    self.GroupTLen[locs]   = myswap(np.fromfile(f,dtype=np.dtype((np.float32,6)),count=Ngroups),self.swap)
                    self.GroupTMass[locs]  = myswap(np.fromfile(f,dtype=np.dtype((np.float32,6)),count=Ngroups),self.swap)
                    if SFR:
                        self.GroupSFR[locs]= myswap(np.fromfile(f,dtype=np.float32,count=Ngroups),self.swap)

                    skip+=Ngroups

            else:   # old format
                if fnb == 0:
                    self.GroupLen=np.empty(TotNgroups,dtype=np.int32)
                    self.GroupOffset=np.empty(TotNgroups,dtype=np.int32)
                    self.GroupMass=np.empty(TotNgroups,dtype=np.float32)
                    self.GroupPos=np.empty(TotNgroups,dtype=np.dtype((np.float32,3)))
                    self.GroupTLen=np.empty(TotNgroups,dtype=np.dtype((np.int32,6)))
                    self.GroupTMass=np.empty(TotNgroups,dtype=np.dtype((np.float32,6)))
                    if SFR:
                        self.GroupSFR=np.empty(TotNgroups,dtype=np.float32)
                if Ngroups>0:
                    locs=slice(skip,skip+Ngroups)
                    self.GroupLen[locs]    = myswap(np.fromfile(f,dtype=np.int32,count=Ngroups),self.swap)
                    self.GroupOffset[locs] = myswap(np.fromfile(f,dtype=np.int32,count=Ngroups),self.swap)
                    self.GroupTLen[locs]   = myswap(np.fromfile(f,dtype=np.dtype((np.int32,6)),count=Ngroups),self.swap)
                    tmp                    = myswap(np.fromfile(f,dtype=np.dtype((np.float64,6)),count=Ngroups),self.swap)
                    self.GroupPos[locs]    = myswap(np.fromfile(f,dtype=np.dtype((np.float32,3)),count=Ngroups),self.swap)
                    if SFR:
                        self.GroupSFR[locs]= myswap(np.fromfile(f,dtype=np.float32,count=Ngroups),self.swap)

                    self.GroupTMass[locs]=tmp.astype(np.float32)
                    self.GroupMass[locs]=np.sum(tmp.astype(np.float32),1)

                    skip+=Ngroups

            curpos = f.tell()
            f.seek(0,os.SEEK_END)
            if curpos != f.tell():
                print("Warning: the file is not finished",fnb)
            f.close()
            fnb+=1
            if fnb==self.Nfiles: Final=True


    def read_IDs(self,long_IDs=False,verbose=0):

        if long_IDs:
            self.id_format=np.uint64
        else: 
            self.id_format=np.uint32

        #################  READ IDS FILES ################# 
        fname=self.basedir+"groups_"+self.snapnum+"/group_ids_"+self.snapnum+"."
        fnb,skip=0,0
        Final=False
        while not(Final):
            f=open(fname+str(fnb),'rb')

            if self.myformat==1:
                (Ngroups,TotNgroups,Nids)=myswap(np.fromfile(f,dtype=np.int32,count=3),self.swap)
                TotNids=myswap(np.fromfile(f,dtype=np.uint64,count=1),self.swap)[0]
                (Nfiles,IdOffset)=myswap(np.fromfile(f,dtype=np.uint32,count=2),self.swap)
            else:
                (Ngroups,Nids,TotNgroups,Nfiles)=myswap(np.fromfile(f,dtype=np.int32,count=4),self.swap)
                IdOffset=-99

            if Nfiles != self.Nfiles:
                print("WARNING: inconsistency, ",nfiles," files found but the header gives",self.Nfiles)

            if verbose>0:
                print()
                print("File N. ",fnb,":")
                print("Ngroups = ",Ngroups)
                print("TotNgroups = ",TotNgroups)
                print("Nids = ",Nids)
                print("TotNids = ",self.TotNids)
                print("Nfiles = ",Nfiles)
                print("IdOffset = ",IdOffset," (skip=",skip,")")

            if fnb==0:
                self.GroupIDs=np.zeros(dtype=self.id_format,shape=self.TotNids)

            if Ngroups>0:
                if long_IDs:
                    IDs=myswap(np.fromfile(f,dtype=np.uint64,count=Nids),self.swap)
                else:
                    IDs=myswap(np.fromfile(f,dtype=np.uint32,count=Nids),self.swap)

                self.GroupIDs[skip:skip+Nids]=IDs[:]
                skip+=Nids
                del IDs

            curpos = f.tell()
            f.seek(0,os.SEEK_END)
            if curpos != f.tell():
                print("Warning: finished reading before EOF for IDs file",fnb)
            f.close()
            fnb+=1
            if fnb==Nfiles:
                Final=True

    def compute_vel(self,snap,verbose=0):


        if hasattr(self,'Vel'):
            return

        from . import Blocks

        if not hasattr(snap,'Map'):
            snap.map()

        if (not hasattr(snap, 'names')):
            snap.names=[]
            for i in range(len(snap.Map)):
                snap.names.append(snap.Map[i].name)

        place=Blocks.find_element_in_list('ID  ',snap.names)
        long_IDs=(snap.Map[place].size==8)
        
        if not hasattr(self,'GroupIDs'):
            self.read_IDs(long_IDs,verbose)

        Vel=snap.read_block('VEL')
        ID=snap.read_block('ID')
        Mass=snap.read_block('MASS',True)

        self.GroupVel=np.empty(self.TotNgroups,dtype=np.dtype((np.float32,3)))

        for group in range(self.TotNgroups):

            ind=np.in1d(ID,self.GroupIDs[self.GroupOffset[group]:self.GroupOffset[group]+self.GroupLen[group]])
            for i in range(3):
                self.GroupVel[group,i] = np.sum(Vel[ind,i]*Mass[ind])/np.sum(Mass[ind])

            if verbose>0:
                print("Group: ",group,", Vel: ",self.GroupVel[group])

        del Vel
        del ID
        del Mass


    def help(self):

        """
        Writes the content of the object
        """
        
        print("Quantities contained in the ReadFoF catalog structure:")
        print("Control parameters")
        print("  snapnum:    ",self.snapnum)
        print("  basedir:    ",self.basedir)
        print("  swap:       ",self.swap)
        print("  myformat:   ",self.myformat)
        print("  Nfiles:     ",self.Nfiles)
        print("From the header")
        print("  TotNgroups: ",self.TotNgroups)
        print("  TotNids:    ",self.TotNids)
        print("Vectors (length TotNgroups)")
        print("  GroupLen    (dtype=np.int32)")
        print("  GroupOffset (dtype=np.int32)")
        print("  GroupMass   (dtype=np.float32)")
        print("  GroupPos    (dtype=dtype((np.float32,3)))")
        print("  GroupVel    (dtype=np.dtype((np.float32,3)))")
        print("  GroupTLen   (dtype=np.dtype((np.int32,6)))")
        print("  GroupTMass  (dtype=np.dtype((np.float32,6)))")
        if hasattr(self,"GroupSFR"):
            print("  GroupSFR    (dtype=np.float32)")

        if hasattr(self,"GroupIDs"):
            print("Particle IDs of FoF groups (length TotNids)")
            print("  GroupIDs    (dtype=",self.id_format,")")
        else:
            print("Particle IDs have not been read")
            print("To read them: catalog.read_IDs(long_IDs=[True,False(default)],verbose)")

