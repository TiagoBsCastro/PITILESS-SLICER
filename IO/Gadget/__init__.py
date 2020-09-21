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
  >>> snap=ReadSubFind.Init("snap_le","134")
  >>> myPOS=Snapshot.read_block(snap,'POS')
  >>> snap.help()

Please refer to the documentation of the Blocks library for further information

To write a snapshot:

  snap.write_header(Header)


2016, written by Pierluigi Monaco (on the basis of older code)

"""

from .Snapshot import *
from . import ReadFoF
from . import ReadSubFind

