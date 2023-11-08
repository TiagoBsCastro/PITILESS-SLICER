"""

Routines to deal with blocks in GADGET snapshots

TODO: writing of blocks/snapshots

These libraries deal with Gadget blocks and are used by the Snapshot.py library.
Most likely, you will not access these libraries directly.
However, there are some features that might be of interest.

NB: multiple files are supported ONLY for format 2 snapshots

ReadWhatBlocks(filename,format=2,swap=0,verbose=0):
scans a snapshot and gives a list of all the blocks found

map([Snapshot object],verbose=0):
constructs the map of blocks for a Snapshot object.
It returns a Map object, an array of objects of the line class.
See documentation of Blocks.map for more details.

content([Map object]):
outputs the content of a map of blocks

read([Snapshot object],[block name]):
reads a block from a snapshot, defined by the Snapshot object.
This operation can be performed using the method read_block([block name]) applied to the Snapshot object.

2016, written by Pierluigi Monaco (on the basis of older code)

"""

import numpy as np
import os
import sys
import copy

# TODO:
#        multiple files  - DONE
#        endianness      - DONE
#        format 1-2
#        INFO block      - DONE
#        INFO file       - DONE
#        writing


def find_element_in_list(element, list_element):
    
    try:
        index_element = list_element.index(element.encode("ASCII"))
        return index_element
    except AttributeError:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return None


class line:

    """
    This class defines one line of the map of blocks of a snapshot
    For more information:
      line.help()
      line.content()
    """


    def __init__(self,block,length,datatype,ndim,active,offset,info):
        self.name=block.ljust(4)
        self.type=datatype.ljust(8)
        self.ndim=int(ndim)
        self.active=active
        if hasattr(datatype,"encode"):
            datatype = datatype.encode("ASCII")
        if   (datatype == b'FLOAT   '):
            self.dt = np.dtype(np.float32)
            self.size=4
        elif (datatype == b'FLOATN  '):
            self.dt = np.dtype((np.float32,ndim))
            self.size=4
        elif (datatype == b'DOUBLE  '):
            self.dt = np.dtype(np.float64)    
            self.size=8
        elif (datatype == b'DOUBLEN '):
            self.dt = np.dtype((np.float64,ndim))    
            self.size=8
        elif (datatype == b'LONG    '):
            self.dt = np.dtype(np.uint32)    
            self.size=4
        elif (datatype == b'LONGN   '):
            self.dt = np.dtype((np.uint32,ndim))    
            self.size=4
        elif (datatype == b'LLONG   '):
            self.dt = np.dtype(np.uint64)    
            self.size=8
        elif (datatype == b'LLONGN  '):
            self.dt = np.dtype((np.uint64,ndim))    
            self.size=8
        elif (datatype == b'INFO    ' or datatype == b'HEAD    '):
            self.dt = None
            self.size = 0
        else:
            print("ERROR: data type not understood - "+datatype.decode('utf-8'))
            return None

        self.info=int(info)

        # these depend on the file
        self.length_in_bytes=[int(length)]
        self.Nparticles=[int(length)/self.size/int(ndim)]
        self.offset=[int(offset)]

    def header(self):
        # a header of what a line contains
        print('NAME - TYPE     - dim  - active types    - info - Length in b.   - N. part.     - offsset')

    def help(self):
        # list what is contained in the object
        print("Quantities contained in list class for the map of blocks (nf=number of files):")
        print("name                - string (4 char)")
        print("type                - string (8 char)")
        print("ndim                - int")
        print("active[6]           - int")
        print("dt                  - python data type")
        print("size                - int")
        print("info                - int")
        print("length_in_bytes[nf] - int")
        print("Nparticles[nf]      - int")
        print("offset[nf]          - int")        


    def content(self):
        # writes object content
        print(('{0} - {1} - d={2:2d} - a=[{3}] -   {4:1d}  - L=[{5}] - Np=[{6}] - offs=[{7}]'.format(self.name, self.type, self.ndim, ','.join(str(p) for p in self.active), self.info, ','.join(str(p) for p in self.length_in_bytes), ','.join(str(p) for p in self.Nparticles), ','.join(str(p) for p in self.offset))))


    def addfile(self,length,offset):
        self.length_in_bytes.append(length)
        self.Nparticles.append(length/self.size/self.ndim)
        self.offset.append(offset)


def ReadWhatBlocks(filename,format=2,swap=0,verbose=0):

    """
    This procedure scans a snapshot and gives a list of the blocks found
    Usage: Blocks.ReadWhatBlocks(filename,format=2,swap=0,verbose=0)
    """

    if format==1:
        return ReadWhatBlocks_1(filename,swap,verbose)

    elif format==2:
        return ReadWhatBlocks_2(filename,swap,verbose)

    else:
        print("Unknown snapshot format in ReadWhatBlocks")
        return None


def ReadWhatBlocks_1(filename,swap,verbose=0):

    # scan the snapshot for all the blocks present

    # open the file and measure its size
    f = open (filename,'rb')
    f.seek(0, os.SEEK_END)
    filesize = f.tell()
    f.seek(0, os.SEEK_SET)

    # start scanning
    scanned=0
    iblock=0
    expected_blocks=['HEAD','POS ','VEL ','ID  ','MASS']
    blocks=[]
    lengths=[]
    offset=[]
    if verbose>0:
        print("Scan of the snapshot ",filename)

    while scanned < filesize:

        # reads block name and field length from the block header
        if swap:
            thislength = (np.fromfile(f,dtype=np.uint32,count=1))[0].byteswap()
        else:
            thislength = (np.fromfile(f,dtype=np.uint32,count=1))[0]
            
        if iblock==0:
            if swap:
                npart = np.fromfile(f,dtype=np.uint32 ,count=6).byteswap
            else:
                npart = np.fromfile(f,dtype=np.uint32 ,count=6)
            f.seek(4,os.SEEK_SET)
            Ntot=np.sum(npart)
            if verbose>0:
                print("Number of particles per type: ",npart)
                print("Total number of particles: ",Ntot)
            expected_size=np.array([256, Ntot*4*3, Ntot*4*3, Ntot*4, Ntot*4],dtype=np.uint32)

        if swap:
            thislength = thislength.byteswap()

        if iblock<5:
            thisblock = expected_blocks[iblock]
        else:
            thisblock = "UNKN"
            if (verbose>1):
                print("  *** Warning: there are more blocks than expected")

        blocks.append(thisblock)
        lengths.append(thislength)  # this is the true length of the data block, without tags
        offset.append(scanned)

        if verbose>1:
            if iblock<5:
                expected=expected_size[iblock]
            else:
                expected=0
            print("Block: ",thisblock, " - length: ", thislength, " (expected:",expected,") - offset: ", scanned)
            if iblock<5:
                if expected_size[iblock]!=thislength:
                    print("  *** Warning: this block size is not standard")

        f.seek(thislength+4, os.SEEK_CUR)
        scanned+=thislength+8
        iblock+=1

    if verbose>0:
        print("Total file size: ",scanned,filesize)

    f.close()

    return (blocks,lengths,offset)
    



def ReadWhatBlocks_2(filename,swap,verbose=0):

    # scan the snapshot for all the blocks present

    # open the file and measure its size
    f = open (filename,'rb')
    f.seek(0, os.SEEK_END)
    filesize = f.tell()
    f.seek(0, os.SEEK_SET)

    # start scanning
    scanned=0
    blocks=[]
    lengths=[]
    offset=[]
    if verbose>0:
        print("Scan of the snapshot ",filename)
    while scanned < filesize:

        # reads block name and field length from the block header
        tag = (np.fromfile(f,dtype=np.uint32,count=1))[0]
        if swap:
            tag = tag.byteswap()
        if tag != 8:
            print("ERROR IN SCANNING THE FILE: starting tag should be 8 but is ",tag)
            return None
        thisblock = f.read(4)
        thislength = (np.fromfile(f,dtype=np.uint32,count=1))[0]
        if swap:
            thislength = thislength.byteswap()
        blocks.append(thisblock)
        lengths.append(thislength-8)  # this is the true length of the data block, without tags

        # yet another check
        tag = (np.fromfile(f,dtype=np.uint32,count=1))[0]
        if swap:
            tag = tag.byteswap()
        if tag != 8:
            print("ERROR IN SCANNING THE FILE: closing tag should be 8 but is ",tag)
            return None

        offset.append(scanned)

        if verbose>1:
            print("Block: ",thisblock, " - length: ", thislength, " - offset: ", scanned)

        f.seek(thislength, os.SEEK_CUR)
        scanned+=thislength+16

    f.close()

    return (blocks,lengths,offset)
    

def map(snap,verbose=0):

    """

    This procedure constructs the map of blocks for a Snapshot object
    usage: Blocks.map(snap,verbose=0)

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

    if snap.format==1:
        return map_of_blocks_1(snap,verbose)

    elif snap.format==2:
        return map_of_blocks_2(snap,verbose)

    else:
        print("Unknown snapshot format")
        return (0,0)


def map_of_blocks_1(snap,verbose):

    if verbose>0:
        print("This is a format 1 snapshot")

    mymap=[]

    if snap.Header.filenum>1:
        print("Type-1 snapshots in multiple files require a map file")
        if  not hasattr(snap, 'INFOfile'):
            print("Please provide a valid map filename")
            return 1
        if not os.path.exists(snap.INFOfile):
            print("Please provide a valid map filename")
            return 1


    if snap.Header.filenum==1:
        this_fn=snap.filename
    else:
        this_fn=snap.filename_nonum+str(0)

    # lengths and offsets are read for the first file
    (blocks,lengths,offset)=ReadWhatBlocks_1(this_fn,snap.swap)

    # if the map configuration file is provided, then the map will be read from the file
    if (hasattr(snap, 'INFOfile') and os.path.exists(snap.INFOfile)):

        if verbose>0:
            print()
            print("Scan of INFOfile ",snap.INFOfile)

        # reads the file and prints what it finds
        infofile=open(snap.INFOfile,'r')
        i=0
        for sline in infofile.readlines():
            if verbose>1:
                print(sline)
            file_map=sline.split()
            types=np.array([],dtype=np.int32)
            for s in file_map[3:]:
                types=np.append(types,int(s))

            myline = line(file_map[0].ljust(4),lengths[i+1],file_map[1].ljust(8),int(file_map[2]),types,offset[i+1],2)
            i+=1

            mymap.append(myline)

    else:

        types=np.array([],dtype=np.uint32)
        for i in range(6):
            if snap.Header.npart[i]==0:
                types=np.append(types,0)
            else:
                types=np.append(types,1)

        mtypes=np.array([],dtype=np.uint32)
        for i in range(6):
            if (snap.Header.npart[i]==0 or snap.Header.mass[i]>0):
                mtypes=np.append(mtypes,0)
            else:
                mtypes=np.append(mtypes,1)

        # a guess is done for the blocks
        for i in range(len(blocks)):

            if (blocks[i]=="HEAD"):
                myline=None
            elif (blocks[i]=="MASS"):
                myline = line("MASS",lengths[i],"FLOAT   ",1,mtypes,offset[i],3)
            elif (blocks[i]=="POS "):
                myline = line("POS ",lengths[i],"FLOATN  ",3,types,offset[i],3)
            elif (blocks[i]=="VEL "):
                myline = line("VEL ",lengths[i],"FLOATN  ",3,types,offset[i],3)
            elif (blocks[i]=="ID  "):
                myline = line("ID  ",lengths[i],"LONG    ",1,types,offset[i],3)
            elif (blocks[i]=="POT "):
                myline = line("POT ",lengths[i],"FLOAT   ",1,types,offset[i],3)
            elif (blocks[i]=="AGE "):
                myline = line("AGE ",lengths[i],"FLOAT   ",1,np.array([0,0,0,0,1,0],dtype=np.uint32),offset[i],3)
            else:
                myline = line(blocks[i],lengths[i],"FLOAT   ",1,np.array([1,0,0,0,0,0],dtype=np.uint32),offset[i],3)

            if myline!=None:
                mymap.append(myline)


#    return (mymap)

    # Sanity check
    # it checks if all blocks present in the snapshot (excluding HEAD and INFO) have been found
    # and if the declared size matches with that inferred from the INFO and number of particles

    if verbose>0:
        print()
        print("Sanity check...")

    for i in range(len(mymap)):
        NN=0
        for j in range(6):
            NN+=snap.Header.npart[j]*mymap[i].active[j]
        if mymap[i].Nparticles[0]!=NN:
            print("WARNING: inconsistent size for block "+mymap[i].name+":",mymap[i].Nparticles[0],NN)
        else:
            if verbose>1:
                print("Block",blocks[i]," matches: ",mymap[i].Nparticles[0],NN)


    # Reads the lengths and offsets for other files
    #
    if snap.Header.filenum>1:
        for this_file in range(1,snap.Header.filenum):
            
            this_fn=snap.filename_nonum+str(this_file)

            # lengths and offsets are read for this_file
            (blocks,lengths,offset)=ReadWhatBlocks_1(this_fn,snap.swap)

            for i in range(len(mymap)):
                mymap[i].addfile(lengths[i+1],offset[i+1])

            if verbose>0:
                print()
                print("Sanity check for file ",this_file,"...")

            for i in range(len(mymap)):
                NN=0
                for j in range(6):
                    NN+=snap.Header.npart[j+this_file*6]*mymap[i].active[j]
                if mymap[i].Nparticles[this_file]!=NN:
                    print("WARNING: inconsistent size for block "+mymap[i].name+":",mymap[i].Nparticles[this_file],NN)
                else:
                    if verbose>1:
                        print("Block",blocks[i]," matches: ",mymap[i].Nparticles[this_file],NN)




    if verbose>0:
        print("Sanity check completed")
        print()
        print()

    del blocks
    del lengths
    del offset

    # now it returns the resulting map

    return (mymap)



def map_of_blocks_2(snap,verbose):

    if verbose>0:
        print("This is a format 2 snapshot")

    mymap=[]

    for file in range(snap.Header.filenum):

        if file==0:
            this_fn=snap.filename
        else:
            this_fn=snap.filename_nonum+str(file)
        # lengths and offsets are read for all the files
        (blocks,lengths,offset)=ReadWhatBlocks_2(this_fn,snap.swap)

        # the construction of the map is done only for the first file
        if file==0:

            # this vector will track if blocks have been described in the map file
            ininfo=np.zeros(len(blocks))

            # if the INFO block is present, then read it
            place=find_element_in_list('INFO',blocks)
            if place!=None:

                # open the file to read the INFO block
                f = open (this_fn,'rb')

                # skip to the INFO block
                f.seek(offset[place]+8, os.SEEK_SET)
                blcksz = np.fromfile(f,dtype=np.int32,count=1)
                if snap.swap:
                    blcksz = blcksz.byteswap()
                blcksz-=8
                nfields = blcksz/(12+4*7)
                f.seek(8, os.SEEK_CUR)

                # loop on fields in the INFO block
                if verbose>1:
                    print()
                    print("INFO block contains: ")
                for cnt in range(int(nfields)):
                    myname = f.read(4)
                    mytype = f.read(8)
                    myinfo = np.fromfile(f,dtype=np.uint32,count=7)
                    if snap.swap:
                        myinfo = myinfo.byteswap()

                    if verbose>1:
                        print(myname,'\t',mytype,' ',myinfo[0],' ',myinfo[1:])
                    cnt += 1

                    splace=find_element_in_list(myname,blocks)
                    if splace==None:
                        if verbose>0:
                            print("WARNING: block "+myname+" is NOT present in the snapshot")
                    else:
                        ininfo[splace]=1
                        myline = line(myname,lengths[splace],mytype,myinfo[0],myinfo[1:],offset[splace],1)
                        mymap.append(myline)

                f.close()

            # in this case there is no INFO, a guess will be made
            else:
                if verbose>0:
                    print("INFO block is not present in the snapshot")

                types=np.array([],dtype=np.uint32)
                for i in range(6):
                    if snap.Header.npart[i]==0:
                        types=np.append(types,0)
                    else:
                        types=np.append(types,1)

                mtypes=np.array([],dtype=np.uint32)
                for i in range(6):
                    if (snap.Header.npart[i]==0 or snap.Header.mass[i]>0):
                        mtypes=np.append(mtypes,0)
                    else:
                        mtypes=np.append(mtypes,1)

                for i in range(len(blocks)):

                    if (blocks[i]==b"HEAD"):
                        myline=None
                    elif (blocks[i]==b"MASS"):
                        myline = line(b"MASS",lengths[i],"FLOAT   ",1,mtypes,offset[i],3)
                    elif (blocks[i]==b"POS "):
                        myline = line(b"POS ",lengths[i],"FLOATN  ",3,types,offset[i],3)
                    elif (blocks[i]==b"VEL "):
                        myline = line(b"VEL ",lengths[i],"FLOATN  ",3,types,offset[i],3)
                    elif (blocks[i]==b"ID  "):
                        myline = line(b"ID  ",lengths[i],"LONG    ",1,types,offset[i],3)
                    elif (blocks[i]==b"POT "):
                        myline = line(b"POT ",lengths[i],"FLOAT   ",1,types,offset[i],3)
                    elif (blocks[i]==b"AGE "):
                        myline = line(b"AGE ",lengths[i],"FLOAT   ",1,np.array([0,0,0,0,1,0],dtype=np.uint32),offset[i],3)
                    else:
                        myline = line(blocks[i],lengths[i],"FLOAT   ",1,np.array([1,0,0,0,0,0],dtype=np.uint32),offset[i],3)

                    if myline!=None:
                        ininfo[i]=1
                        mymap.append(myline)


            # if the map configuration file is provided, then read it and override previous choices
            if (hasattr(snap, 'INFOfile') and os.path.exists(snap.INFOfile)):

                if verbose>0:
                    print()
                    print("Scan of INFOfile ",snap.INFOfile)

                # reads the file and prints what it finds
                # myformat = np.dtype('S4,S8,i,i,i,i,i,i,i')
                file_map=[]
                infofile=open(snap.INFOfile,'r')
                for sline in infofile.readlines():
                    if verbose>1:
                        print(sline)
                    file_map=sline.split()
                    splace=find_element_in_list(file_map[0].ljust(4),blocks)
                    if splace==None:
                        if verbose>0:
                            print("WARNING: block "+file_map[0]+"is NOT present in the snapshot")
                    else:
                        if verbose>0:
                            print("block "+file_map[0]+" will be redefined")
                        ininfo[splace]=1
                        types=np.array([],dtype=np.int32)
                        for s in file_map[3:]:
                            types=np.append(types,int(s))

                        myline = line(file_map[0].ljust(4),lengths[splace],file_map[1].ljust(8),int(file_map[2]),types,offset[splace],2)
                        names=[mymap[0].name]
                        for u in range(1,len(mymap)):
                            names.append(mymap[u].name)
                        ssplace=find_element_in_list(file_map[0].ljust(4),names)
                        if ssplace==None:
                            mymap.append(myline)
                        else:
                            mymap[ssplace]=copy.deepcopy(myline)


            # checks that everything has been found
            for i in range(len(ininfo)):
                if ininfo[i]==0 and blocks[i]!="HEAD" and blocks[i]!="INFO":
                    if verbose>0:
                        print("WARNING: BLOCK "+blocks[i]+" NOT FOUND IN MAP FILE")
                        print("Please add it to the file")
                        print("I will make a guess, but no guarantee...")
                    myline = line(blocks[i],lengths[i],"FLOAT   ",1,np.array([1,0,0,0,0,0],dtype=np.uint32),offset[i],3)
                    mymap.append(myline)

                       
        # if we are reading a further file of this snapshot:
        else:

            names=[mymap[0].name]
            for i in range(1,len(mymap)):
                names.append(mymap[i].name)

            # add the information for this file to the map
            for i in range(len(blocks)):
                if (blocks[i]!="HEAD" and blocks[i]!="INFO"):
                    place=find_element_in_list(blocks[i],names)
                    if place==None:
                        print("SEVERE ERROR: block "+blocks[i]+" not found in snapshot "+this_fn)
                        return None
                    mymap[place].addfile(lengths[i],offset[i])

            del names

        # Sanity check
        # it checks if all blocks present in the snapshot (excluding HEAD and INFO) have been found
        # and if the declared size matches with that inferred from the INFO and number of particles

        if verbose>0:
            print()
            print("Sanity check for file "+str(file)+"...")

        names=[]
        for i in range(len(mymap)):
            names.append(mymap[i].name)

        for i in range(len(blocks)):
            if ininfo[i]==1:
                NN=0
                place=find_element_in_list(blocks[i],names)
                for j in range(6):
                    NN+=snap.Header.npart[j+file*6]*mymap[place].active[j]
                if mymap[place].Nparticles[file]!=NN:
                    print("WARNING: inconsistent size for block ",mymap[place].name)
                else:
                    if verbose>1:
                        print("Block",blocks[i],"(",place,") matches: ",mymap[place].Nparticles[file],NN)

            elif (blocks[i]!=b"HEAD" and blocks[i]!=b"INFO"):
                print("ERROR: block "+blocks[i].decode('utf-8')+" is not present in the map")
                if verbose>0:
                    print("I will make a guess, but no guarantee...")
                myline = line(blocks[i],lengths[i],"FLOAT   ",1,np.array([1,0,0,0,0,0],dtype=np.uint32),offset[i],3)


        if verbose>0:
            print("Sanity check completed")


        if verbose>0:
            print()
            print()
        del names
        del blocks
        del lengths
        del offset
        

    # now it returns the resulting map

    return (mymap)


def content(map):

    """
    This procedure outputs the content of a map of blocks
    usage: Blocks.content(map)
    """

    map[0].header()
    for line in map:
        line.content()



def read(snap,blockname,verbose=0,onlythissnap=False):

    """
    This procedure reads a block from a snapshot
    usage: Blocks.read(snap,blockname)
    """

    if (blockname=="HEAD"):
        print("HEAD block should be read with read_header()")
        return None

    if (blockname=="INFO"):
        print("INFO block should be read with map()")
        return None

    # reads the header if it has not been done yet
    if (not hasattr(snap, 'Header')):
        snap.read_header()

    # builds the map of blocks if it has not been done yet
    if (not hasattr(snap, 'Map')):
        snap.map()

    # builds the name list if it has not been done yet
    if (not hasattr(snap, 'names')):
        snap.names=[]
        for i in range(len(snap.Map)):
            snap.names.append(snap.Map[i].name)

    # find block in the list of known blocks
    myblock=find_element_in_list(blockname.ljust(4),snap.names)

    if myblock==None:
        print("Error: Block ",blockname," not found in snapshot map")
        return None

    NtoLoad=np.sum(snap.Header.nall*snap.Map[myblock].active)
    content=np.zeros(NtoLoad,dtype=snap.Map[myblock].dt)
    off=np.zeros(6,dtype=np.int64)
    for i in range(5):
        off[i+1]=off[i]+snap.Header.nall[i]*snap.Map[myblock].active[i]

    if verbose>1:
        print("I expect to load",NtoLoad,"particles")

    if onlythissnap:

        filename=snap.filename
        
        if verbose>0:
            print("opening file "+filename+"...")
        f=open(filename,"rb")

        if snap.format==2:

            # this is valid for a format 2 snapshot
            f.seek(snap.Map[myblock].offset[0]+4,os.SEEK_SET)

            # check on block name
            found=f.read(4)
            if found != blockname.ljust(4).encode("ASCII"):
                print("ERROR: I expected block "+blockname.ljust(4)+", I found "+found.decode("utf-8"))
                return None

            # check on block size
            if snap.swap:
                length = (np.fromfile(f,dtype=np.uint32,count=1))[0].byteswap() - 8
            else:
                length = (np.fromfile(f,dtype=np.uint32,count=1))[0] - 8

            if length != snap.Map[myblock].length_in_bytes[0]:
                print("ERROR: I stored a length ",snap.Map[myblock].length_in_bytes[0],", I found ",length)
                return None
            if length != snap.Map[myblock].Nparticles[0] * snap.Map[myblock].size * snap.Map[myblock].ndim:
                print("ERROR: I expected a length ",snap.Map[myblock].Nparticles[0] * snap.Map[myblock].size * snap.Map[myblock].ndim,", I found ",length)
                return None

            f.seek(8, os.SEEK_CUR)

        else:

            # this is valid for a format 1 snapshot
            f.seek(snap.Map[myblock].offset,os.SEEK_SET)

            length = (np.fromfile(f,dtype=np.uint32,count=1))[0]
            if snap.swap:
                length=length.byteswap()

            if length != snap.Map[myblock].length_in_bytes:
                print("ERROR: I stored a length ",snap.Map[myblock].length_in_bytes,", I found ",length)
                return None
            if length != snap.Map[myblock].Nparticles * snap.Map[myblock].size * snap.Map[myblock].ndim:
                print("ERROR: I expected a length ",snap.Map[myblock].Nparticles * snap.Map[myblock].size * snap.Map[myblock].ndim,", I found ",length)
                return None
            
        # read the data
        if snap.swap:
            bunch = np.fromfile(f,dtype=snap.Map[myblock].dt,count=int(snap.Map[myblock].Nparticles[0])).byteswap()
        else:
            bunch = np.fromfile(f,dtype=snap.Map[myblock].dt,count=int(snap.Map[myblock].Nparticles[0]))

        check=(np.fromfile(f,dtype=np.uint32,count=1))[0]
        if snap.swap:
            check=check.byteswap()
        if check != length:
            print("ERROR: the final length %d does not match that of the block header %d"%(check,length))
            return None

        f.close()

        return bunch

    for ifile in range(snap.Header.filenum): # main loop over files

        if snap.Header.filenum==1:
            filename=snap.filename
        else:
            filename=snap.filename_nonum+"%d"%(ifile)

        if verbose>0:
            print("opening file "+filename+"...")
        f=open(filename,"rb")

        if snap.format==2:

            # this is valid for a format 2 snapshot
            f.seek(snap.Map[myblock].offset[ifile]+4,os.SEEK_SET)

            # check on block name
            found=f.read(4)
            if found != blockname.ljust(4).encode("ASCII"):
                print("ERROR: I expected block "+blockname.ljust(4)+", I found "+found.decode("utf-8"))
                return None

            # check on block size
            if snap.swap:
                length = (np.fromfile(f,dtype=np.uint32,count=1))[0].byteswap() - 8
            else:
                length = (np.fromfile(f,dtype=np.uint32,count=1))[0] - 8

            if length != snap.Map[myblock].length_in_bytes[ifile]:
                print("ERROR: I stored a length ",snap.Map[myblock].length_in_bytes[ifile],", I found ",length)
                return None
            if length != snap.Map[myblock].Nparticles[ifile] * snap.Map[myblock].size * snap.Map[myblock].ndim:
                print("ERROR: I expected a length ",snap.Map[myblock].Nparticles[ifile] * snap.Map[myblock].size * snap.Map[myblock].ndim,", I found ",length)
                return None

            f.seek(8, os.SEEK_CUR)

        else:

            # this is valid for a format 1 snapshot
            f.seek(snap.Map[myblock].offset[ifile],os.SEEK_SET)

            length = (np.fromfile(f,dtype=np.uint32,count=1))[0]
            if snap.swap:
                length=length.byteswap()

            if length != snap.Map[myblock].length_in_bytes[ifile]:
                print("ERROR: I stored a length ",snap.Map[myblock].length_in_bytes[ifile],", I found ",length)
                return None
            if length != snap.Map[myblock].Nparticles[ifile] * snap.Map[myblock].size * snap.Map[myblock].ndim:
                print("ERROR: I expected a length ",snap.Map[myblock].Nparticles[ifile] * snap.Map[myblock].size * snap.Map[myblock].ndim,", I found ",length)
                return None

            
        # read the data
        if snap.swap:
            bunch = np.fromfile(f,dtype=snap.Map[myblock].dt,count=int(snap.Map[myblock].Nparticles[ifile])).byteswap()
        else:
            bunch = np.fromfile(f,dtype=snap.Map[myblock].dt,count=int(snap.Map[myblock].Nparticles[ifile]))

        check=(np.fromfile(f,dtype=np.uint32,count=1))[0]
        if snap.swap:
            check=check.byteswap()
        if check != length:
            print("ERROR: the final length %d does not match that of the block header %d"%(check,length))
            return None


        # append them to the content array
        myoff=0
        for i in range(6):
            j=i+6*ifile
            if snap.Map[myblock].active[i] and snap.Header.npart[j]>0:
                content[off[i]:off[i]+snap.Header.npart[j]]=bunch[myoff:myoff+snap.Header.npart[j]]
                off[i]+=snap.Header.npart[j]
                myoff +=snap.Header.npart[j]

        del bunch        

        f.close()

    return content
