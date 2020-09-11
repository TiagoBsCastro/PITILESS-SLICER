"""

Routines to read GADGET snapshots

TODO: writing of blocks/snapshots

Basic use:

  import Snapshot
  snap = Snapshot.Init([run name],[snapshot number],[basedir]="",ToWrite=False, override=False))

run name: base filename for the snapshot
snapshot number: can be an integer or a string
basedir, optional: directory where the run is placed
ToWrite, optional: if True, the file is opened for writing
override, optional: it allows to write on an existing file


The code will look (in the directory basedir) for one of the following files:
1) [run name]_[snapshot string]
2) [run name]_[snapshot string].0
3) snapdir_[snapshot string]/[run name]_[snapshot string].0

To read the snapshot header(s) (this is done automatically when necessary):

  snap.read_header()

To read the map of blocks (this is done automatically when necessary):

  snap.map()

To show the map of blocks:

  snap.map_content()

To know details of the snapshot:

  snap.help()

To read a block:

  MyBlock = Snapshot.read_block([Block name])

Example:
  >>> import Snapshot
  >>> snap=Snapshot.Init("snap_le","134")
  >>> myPOS=Snapshot.read_block(snap,'POS')
  >>> snap.help()

Please refer to the documentation of the Blocks library for further information

To write a snapshot:

  snap.write_header(Header)


2016, written by Pierluigi Monaco (on the basis of older code)

"""


import numpy as np
import os
import sys
from . import Blocks

class Init:


    """

    To initialize a snapshot object:
    
      snap = Snapshot.Init([run name],[snapshot number],[basedir]="",ToWrite=False, override=False)

    run name: base filename for the snapshot
    snapshot number: can be an integer or a string - if negative, the snapshot number is omitted
    basedir, optional: directory where the run is placed
    ToWrite, optional: if True, the file is opened for writing
    override, optional: it allows to write on an existing file

    To know details of the snapshot:

      snap.help()

    """


    filename="none"
    format=-99
    swap=-99

    def __init__(self,runname,snapnum,basedir="",ToWrite=False,override=False):

        if type(snapnum) is int:
            self.snapnum="%03d"%snapnum
        else:
            self.snapnum = snapnum
        self.runname = runname
        if (basedir!="" and basedir[-1]!="/"):
            basedir+="/"
        self.basedir = basedir

        # if the snapshot is opened to be written, the snapshot number is not added to the file name
        self.ToWrite=ToWrite
        self.override=override
        if ToWrite:
            self.filename=basedir+runname
            if (not override) and os.path.exists(self.filename):
                print("ERROR: file %s exists, cannot write the snapshot"%self.filename)
                print("If this is what you want, initialize the snapshot with override=True")

                return None            

            self.swap=1
            self.format=2

            return

        # determines the file name

        # a negative snapshot number indicates that the snapshot filename is in runname
        if ((type(snapnum) is int) and snapnum<0):
            self.snapnum=""
            # checks if the snapshot is in multiple files
            if os.path.exists(basedir+runname):
                self.filename=basedir+runname
                print("Snapshot filename is ",self.filename)
            elif os.path.exists(basedir+runname+".0"):
                self.filename=basedir+runname+".0"
                self.filename_nonum=basedir+runname+"."
                print("Name of first snapshot file is ",self.filename)
            else:
                print("ERROR: file "+basedir+self.filename+" or file "+basedir+runname+".0"+" not found")
                return None

        elif os.path.exists(basedir+runname+"_"+self.snapnum):
            self.filename = basedir+runname+"_"+self.snapnum
            print("Snapshot filename is ",self.filename)
        elif os.path.exists(basedir+runname+"_"+self.snapnum+".0"):
            self.filename = basedir+runname+"_"+self.snapnum+".0"
            self.filename_nonum = basedir+runname+"_"+self.snapnum+"."
            print("Name of first snapshot file is ",self.filename)
        elif os.path.exists(basedir+"snapdir_"+self.snapnum+"/"+runname+"_"+self.snapnum+".0"):
            self.filename = basedir+"snapdir_"+self.snapnum+"/"+runname+"_"+self.snapnum+".0"
            self.filename_nonum = basedir+"snapdir_"+self.snapnum+"/"+runname+"_"+self.snapnum+"."
            print("Name of first snapshot file is ",self.filename)
        else:
            print("file not found: dir "+basedir+", run "+runname+", snapshot N. ",self.snapnum)
            return None


        # determines format and endianness

        f = open(self.filename,'rb')
        blocksize = np.fromfile(f,dtype=np.int32,count=1)
        if blocksize[0] == 8:
            self.swap = 0
            self.format = 2
        elif blocksize[0] == 256:
            self.swap = 0
            self.format = 1
        else:
            blocksize.byteswap(True)
            if blocksize[0] == 8:
                self.swap = 1
                self.format = 2
            elif blocksize[0] == 256:
                self.swap = 1
                self.format = 1
            else:
                print("incorrect file format encountered when reading header of", self.filename)
                self.format = -99
                return None

        f.close()

        print("Snapshot format is ",self.format)
        if self.swap:
            print("Snapshot is not in the native endianness, data will be swapped")
        else:
            print("Snapshot is in the native endianness")


    def read_header(self):

        """
        To read the snapshot header(s) (this is done automatically when necessary):

          snap.read_header()

        """


        self.Header = Header(self.filename,self.format,self.swap)

        if self.Header.filenum>1:
            for ifile in range(self.Header.filenum):
                if ifile>0 and hasattr(self,"filename_nonum"):
                    self.Header.addfile(self.filename_nonum+str(ifile),self.format,self.swap)

            print("Header read from",self.Header.filenum,"files")
        else:
            print("Header read from a single file")


    def map(self,verbose=0):

        """

        To read the map of blocks (this is done automatically when necessary):

          snap.map()

        PROCEDURE TO DEFINE THE MAP:

        Type-1 snapshot:
          1) read it from file snap.INFOfile, if that field exists
          2) otherwise, assume a standard structure: HEAD, POS, VEL, ID, MASS (if not in masstable)

        Type-2 snapshot:
          1) read the INFO block if present
          2) otherwise make a guess for the length of blocks
          3) if snap.INFOfile if present, read information from that file and use it to force
             redefinition of block maps for all the blocks contained in the file - the others
             will not be changed

        Map building is subject to a number of consistency checks,
        use a high verbose value to have complete information

        IF THE MAP OF BLOCKS CANNOT BE CONSTRUCTED AT FIRST ATTEMPT, 
        PLEASE PROVIDE AN INFOfile

        1) produce a file with this format:

        POS  	FLOATN     3   1 1 1 1 1 0
        VEL  	FLOATN     3   1 1 1 1 1 0
        ID   	LONG       1   1 1 1 1 1 0
        MASS 	FLOAT      1   1 1 1 1 1 0
        [...]

        2) set the INFOfile attribute of the Snapshot object:

        >>> snap.INFOfile="my_INFOfile.txt"

        3) repeat map contruction

        >>> snap.map()

        """


        if not hasattr(self,"Header"):
            self.read_header();
        self.Map = Blocks.map(self,verbose)

        self.names=[]
        for i in range(len(self.Map)):
            self.names.append(self.Map[i].name)


    def map_content(self):

        """

        To show the map of blocks:

          snap.map_content()

        """
        
        if not hasattr(self,"Map"):
            self.map();
        Blocks.content(self.Map)

    def read_block(self,name,AllMasses=False,parttype=-1,verbose=0,onlythissnap=False):

        """

        Reads a block from a snapshot and returns its values in an array.
        
        usage: myBlock=snap.read_block([block name], AllMasses=[False(def),True], parttype=PT, verbose=0)

          block name: a string with the block name (no need to add extra spaces)
          AllMasses: see below
          PT: optional, if it is >=0 and <6 only that particle type will be output
          verbose: default to 0

        Block reading is based on the map, please be sure to have a correct
        and consistent map before reading the blocks. In case, provide an INFOfile
        (see documentation of Snapshot.map).

        If the optional AllMasses flag is set to True, it will add to the block 'MASS' 
        the masses for particles with mass given in the massarray

        """

        if verbose>0:
            print("Reading block "+name+" from snapshot")

        if onlythissnap:
            self.read_header()
            self.Header.filenum = 1  

        if not hasattr(self,'Map'):
            self.map()

        place=Blocks.find_element_in_list(name.ljust(4), self.names)
        if place is None:
            print("I cannot find block %s in the snapshot"%name)
            return None

        if (name=='MASS' and AllMasses):

            if place==None:
                BB2=np.array([],dtype=np.float32)
                skip=0
                for i in range(6):
                    if self.Header.nall[i]>0:
                        add=np.ones(self.Header.nall[i],dtype=np.float32)*self.Header.massarr[i].astype(np.float32)
                        BB2=np.append(BB2,add)

            else:
                BB=Blocks.read(self,name,verbose)
                BB2=np.array([],dtype=self.Map[place].dt)
                skip=0
                for i in range(6):
                
                    if self.Header.massarr[i]>0.0:
                        add=np.ones(self.Header.nall[i],dtype=self.Map[place].dt)*self.Header.massarr[i].astype(self.Map[place].dt)
                        BB2=np.concatenate((BB2,add))
                    else:
                        if self.Header.nall[i]>0:
                            BB2=np.concatenate((BB2,BB[skip:skip+self.Header.nall[i]]))
                            skip+=self.Header.nall[i]

        else:

            BB2=Blocks.read(self,name, onlythissnap = onlythissnap)

        #selects requested particle type
        if type(parttype) is int and parttype>=0 and parttype<6:

            if self.Map[place].active[parttype]==0:
                return None

            return BB2[np.sum(self.Header.npart[0:parttype]):np.sum(self.Header.npart[0:parttype+1])]

        else:

            return BB2


    def write_header(self,myHeader):

        """

        To write the snapshot header:

          snap.write_header(myHeader)

        myHeader must be a Header object already initialized

        """

        if not self.ToWrite:
            return None

        if (not self.override) and os.path.exists(self.filename):
            print("ERROR: trying to write Header on an existent file")
            print("If this is what you want, initialize the snapshot with override=True")
            return None

        self.Header = myHeader

        f=open(self.filename,'wb')

        # header block name
        np.array([8],dtype=np.int32).tofile(f)
        f.write('HEAD')
        np.array([264,8],dtype=np.int32).tofile(f)

        # header
        np.array([256],dtype=np.int32).tofile(f)
        myHeader.npart.tofile(f)
        myHeader.massarr.tofile(f)
        myHeader.time.tofile(f)
        myHeader.redshift.tofile(f)
        myHeader.sfr.tofile(f)
        myHeader.feedback.tofile(f)
        myHeader.nall.tofile(f)
        myHeader.cooling.tofile(f)
        myHeader.filenum.tofile(f)
        myHeader.boxsize.tofile(f)
        myHeader.omega_m.tofile(f)
        myHeader.omega_l.tofile(f)
        myHeader.hubble.tofile(f)
        myHeader.stellarage.tofile(f)
        myHeader.metals.tofile(f)
        myHeader.nallHigh.tofile(f)
        myHeader.entropy.tofile(f)
        myHeader.metalcool.tofile(f)
        myHeader.stellarev.tofile(f)
        np.zeros(13,dtype=np.int32).tofile(f)
        np.array([256],dtype=np.int32).tofile(f)

        f.close()


    def write_block(self,blockname,mytype,Npart,values,verbose=0):

        """

        To write a block:

          snap.write_header(blockname, type, Npart, values, verbose=0)

        blockname: string with block name (max 4 char)
        type: numpy dtype of the field
        Npart: number of particles to be written
        values: array with values to be written

        """

        if not self.ToWrite:
            return None

        if (blockname=="HEAD"):
            print("HEAD block should be written with read_header()")
            return None

        if (blockname=="INFO"):
            print("INFO block should be read with write_map()")
            return None
    
        f=open(self.filename,'ab')

        Nbytes = Npart * mytype.itemsize

        # block name
        np.array([8],dtype=np.int32).tofile(f)
        f.write(blockname.ljust(4))
        np.array([Nbytes+8,8],dtype=np.int32).tofile(f)

        # data
        np.array([Nbytes],dtype=np.int32).tofile(f)
        values.tofile(f)
        np.array([Nbytes],dtype=np.int32).tofile(f)

        f.close()

        if verbose:
            print("Written %d particles to file %s, data length: %d bytes"%(Npart,self.filename,Nbytes))

    def write_info_block(self, mymap, verbose=0):

        """

        To write the info block:

          snap.write_info_block(map, verbose=0)

        The map object must already be initialized

        """

        if not self.ToWrite:
            return None

        Nblocks=len(mymap)
        Nbytes=Nblocks*40

        if verbose:
            print("I will write %d lines in the INFO block"%Nblocks)
            print("Nbytes: %d"%Nbytes)
    
        f=open(self.filename,'ab')

        # block name
        np.array([8],dtype=np.int32).tofile(f)
        f.write('INFO')
        np.array([Nbytes+8,8],dtype=np.int32).tofile(f)

        # data
        np.array([Nbytes],dtype=np.int32).tofile(f)
        for i in range(Nblocks):
            f.write(mymap[i].name.ljust(4))
            f.write(mymap[i].type.ljust(8))
            np.array([mymap[i].ndim],dtype=np.int32).tofile(f)
            mymap[i].active.tofile(f)
        np.array([Nbytes],dtype=np.int32).tofile(f)
        
        f.close()



    def view(self, sparse=1000, center=None, scale=None, Pos=None, groups=None, 
             ViewStars=True, ViewGas=True, ViewDM=True, verbose=0):
        
        """
        
        To have a quick 3d view of the snapshot:

          snap.view(sparse=1000, center=None, scale=None, Pos=None, groups=None, verbose=0)

        sparse gives the total number of particles to be shown - default: 1000
        center requires a numpy array with dim=3, centers the snapshot on that position
        scale requires a float, limits the plot to +/- scale on all three dimensions
        In case the POS block has already been loaded, please provide it wit Pos
        groups allows to overplot a set of points, like FoF groups
        ViewStars, ViewGas and ViewDM cause those particles to be plotted and are initializes as True

        This method is meant to give a quick way to visualize a snapshot and check 
        that things are sensible, it is not a sophisticated visualization tool

        """

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        if not hasattr(self,"Map"):
            self.map()

        fig=plt.figure()
        panel=fig.gca(projection='3d')

        if Pos is None:

            Pgas=self.read_block('POS',parttype=0)
            Pdm =self.read_block('POS',parttype=1)
            Pstr=self.read_block('POS',parttype=4)

            if verbose>0:
                print("Read %d gas, %d dark matter, %d star particles"%(len(Pgas),len(Pdm),len(Pstr)))

        else:

            Pgas = np.copy(Pos[0:self.Header.npart[0]])
            Pdm  = np.copy(Pos[self.Header.npart[0]:np.sum(self.Header.npart[0:2])])
            Pstr = np.copy(Pos[np.sum(self.Header.npart[0:4]):np.sum(self.Header.npart[0:5])])

            if verbose>0:
                print("Found %d gas, %d dark matter, %d star particles"%(len(Pgas),len(Pdm),len(Pstr)))

        if center is not None:
            Pgas -= center
            Pdm  -= center
            Pstr -= center

        if scale is not None:

            d2gas=Pgas[:,0]**2+Pgas[:,1]**2+Pgas[:,2]**2
            d2dm =Pdm [:,0]**2+Pdm [:,1]**2+Pdm [:,2]**2
            d2str=Pstr[:,0]**2+Pstr[:,1]**2+Pstr[:,2]**2

            igas=d2gas<scale**2
            idm =d2dm <scale**2
            istr=d2str<scale**2

            Ngas=np.count_nonzero(igas)
            Ndm =np.count_nonzero(idm)
            Nstr=np.count_nonzero(istr)

            if verbose>0:
                print("Within a distance of %f:"%scale)
                print("Number of gas particles: ",Ngas)
                print("Number of dark matter particles: :",Ndm)
                print("Number of star particles: ",Nstr)

            factor = np.float(sparse)/np.float(Ngas+Ndm+Nstr)

            igas &= np.random.random(self.Header.npart[0]) < factor
            idm  &= np.random.random(self.Header.npart[1]) < factor
            istr &= np.random.random(self.Header.npart[4]) < factor

        else:

            factor = np.float(sparse)/np.float(self.Header.npart[0]+self.Header.npart[1]+self.Header.npart[4])

            igas = np.random.random(self.Header.npart[0]) < factor
            idm  = np.random.random(self.Header.npart[1]) < factor
            istr = np.random.random(self.Header.npart[4]) < factor


        if verbose>0:
            print("Showing %d gas, %d dark matter, %d star particles"%(
                np.count_nonzero(igas),np.count_nonzero(idm),np.count_nonzero(istr)))

        if ViewGas:
            panel.scatter(Pgas[igas,0],Pgas[igas,1],Pgas[igas,2],label='gas',c='b',marker='.')
        if ViewDM:
            panel.scatter(Pdm [idm ,0],Pdm [idm ,1],Pdm [idm ,2],label='dark matter',c='k',marker='.')
        if ViewStars:
            panel.scatter(Pstr[istr,0],Pstr[istr,1],Pstr[istr,2],label='stars',c='y',marker='.')

        if groups is not None:
            if center is None:
                panel.scatter(groups[:,0],groups[:,1],groups[:,2],label='groups',c='g',marker='o')
            else:
                panel.scatter(groups[:,0]-center[0],groups[:,1]-center[1],groups[:,2]-center[2],label='groups',c='g',marker='o')

        if scale is not None:
           panel.set_xlim([-scale,scale]) 
           panel.set_ylim([-scale,scale]) 
           panel.set_zlim([-scale,scale]) 

        panel.legend()

        fig.show()


    def help(self):

        """

        To know details of the snapshot:

          snap.help()

        """

        if self.ToWrite:

            print("This snapshot is initialized for writing")
            print("Snapshot file name: "+self.filename)
            print("Over-ride mode: "+self.override)

        else:

            print("Snapshot run name: "+self.runname)
            print("Snapshot number: "+self.snapnum)
            print("Snapshot basedir: "+self.basedir)
            print("Snapshot filename (without file number): "+self.filename)
            print("Snapshot format is ",self.format)

            if self.swap:
                print("Snapshot is not in the native endianness, data will be swapped")
            else:
                print("Snapshot is in the native endianness")

            if hasattr(self,"Header"):
                print("Header has been loaded.")
                self.Header.help()
            else:
                print("Header has not been loaded - snapshot.read_header() to read it")

            if hasattr(self,"Map"):
                print("Map has been loaded - snashot.map_content() for details")
            else:
                print("Map has not been loaded - snapshot.map() to read it")



class Header:

    """
    
    This class is defined to deal with snapshot headers.
    With multiple files the array npart will have a length equal to 6 x filenum
        
    To know the content of a header (without using a Snapshot object):

      myHeader=Snapshot.Header([snapshot filename],format=2,swap=0)
      myHeader.help()

    """

    def __init__(self,filename,gform=2,swap=0):

        f = open(filename,'rb')
        if gform==2:
            f.seek(20, os.SEEK_SET)
        else:
            f.seek(4, os.SEEK_SET)

        self.npart      =  np.fromfile(f,dtype=np.uint32 ,count=6)
        self.massarr    =  np.fromfile(f,dtype=np.float64,count=6)
        self.time       = (np.fromfile(f,dtype=np.float64,count=1))[0]
        self.redshift   = (np.fromfile(f,dtype=np.float64,count=1))[0]
        self.sfr        = (np.fromfile(f,dtype=np.int32  ,count=1))[0]
        self.feedback   = (np.fromfile(f,dtype=np.int32  ,count=1))[0]
        self.nall       =  np.fromfile(f,dtype=np.uint32 ,count=6)
        self.cooling    = (np.fromfile(f,dtype=np.int32  ,count=1))[0]
        self.filenum    = (np.fromfile(f,dtype=np.int32  ,count=1))[0]
        self.boxsize    = (np.fromfile(f,dtype=np.float64,count=1))[0]
        self.omega_m    = (np.fromfile(f,dtype=np.float64,count=1))[0]
        self.omega_l    = (np.fromfile(f,dtype=np.float64,count=1))[0]
        self.hubble     = (np.fromfile(f,dtype=np.float64,count=1))[0]
        self.stellarage = (np.fromfile(f,dtype=np.int32  ,count=1))[0]
        self.metals     = (np.fromfile(f,dtype=np.int32  ,count=1))[0]
        self.nallHigh   =  np.fromfile(f,dtype=np.uint32 ,count=6)
        self.entropy    = (np.fromfile(f,dtype=np.int32  ,count=1))[0]
        self.metalcool  = (np.fromfile(f,dtype=np.int32  ,count=1))[0]
        self.stellarev  = (np.fromfile(f,dtype=np.int32  ,count=1))[0]

        if swap:
            self.npart.byteswap(True)
            self.massarr.byteswap(True)
            self.nall.byteswap(True)
            self.nallHigh.byteswap(True)
            self.time       = self.time.byteswap()
            self.redshift   = self.redshift.byteswap()
            self.sfr        = self.sfr.byteswap()
            self.feedback   = self.feedback.byteswap()
            self.cooling    = self.cooling.byteswap()
            self.filenum    = self.filenum.byteswap()
            self.boxsize    = self.boxsize.byteswap()
            self.omega_m    = self.omega_m.byteswap()
            self.omega_l    = self.omega_l.byteswap()
            self.hubble     = self.hubble.byteswap()
            self.stellarage = self.stellarage.byteswap()
            self.metals     = self.metals.byteswap()
            self.entropy    = self.entropy.byteswap()
            self.metalcool  = self.metalcool.byteswap()
            self.stellarev  = self.stellarev.byteswap()     

        # fixes npart in case of large number of particles
        if (np.sum(self.nallHigh)>0):
            low_nall=np.copy(self.nall)
            self.nall=np.zeros(6,dtype=np.uint64)
            for i in range(6):
                self.nall[i]=low_nall[i]+self.nallHigh[i]<<32

        f.close()

    def help(self):

        print("Quantities contained in Header:")
        print("npart[6]    - int32   - ",self.npart)
        print("massarr[6]  - float64 - ",self.massarr)
        print("time        - float64 - ",self.time)
        print("redshift    - float64 - ",self.redshift)
        print("sfr         - int32   - ",self.sfr)
        print("feedback    - int32   - ",self.feedback)
        print("nall[6]     - int32   - ",self.nall)
        print("cooling     - int32   - ",self.cooling)
        print("filenum     - int32   - ",self.filenum)
        print("boxsize     - float64 - ",self.boxsize)
        print("omega_m     - float64 - ",self.omega_m)
        print("omega_l     - float64 - ",self.omega_l)
        print("hubble      - float64 - ",self.hubble)
        print("stellarage  - int32   - ",self.stellarage)
        print("metals      - int32   - ",self.metals)
        print("nallHigh[6] - uint32  - ",self.nallHigh)
        print("entropy     - int32   - ",self.entropy)
        print("metalcool   - int32   - ",self.metalcool)
        print("stellarev   - int32   - ",self.stellarev)


    def addfile(self, filename, gform=2, swap=0):
        # add the npart information of another file into the Header

        f = open(filename,'rb')
        if gform==2:
            f.seek(20, os.SEEK_SET)
        else:
            f.seek(4, os.SEEK_SET)

        npart = np.fromfile(f,dtype=np.uint32 ,count=6)
        if swap:
            npart.byteswap(True)

        self.npart = np.append(self.npart,npart)

        f.close()
