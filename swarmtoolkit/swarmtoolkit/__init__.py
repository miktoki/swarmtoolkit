#!/usr/bin/python


"""
Documentation
=============

`swarmtoolkit` is a toolbox intended to provide simple and quick access to
the Swarm L1 and L2 products for Python3. 

Main features:
    - reading and introspection of CDF (Common Data Format) files and 
      the containing parameters to `numpy.ndarray`'s, whether the CDF's
      are stored locally, within zip-files or on an ftp-server. 
    - shift parameters with respect to each other in time, both using a
      user-defined time shift and by finding a best fit within a range 
      using a minimizer. Parameters can also be aligned through 
      interpolation to be evaluated at the same time values.
    - Compute the magnetic field and its derivative in the NEC frame 
      from SHC ascii files.
    - Convenience functions to visualize plots using `matplotlib` and 
      `mpl_toolkits.basemap`.

        
The demo notebook, found under the demo directory gives some instructional 
examples on how this package may be used, and is a good way to quickly get
started. The documentation provides a more comprehensive look at the features.


Documentation shown below may also be found for each function:
    -  by using the ``help`` function:

        >>> help(swarmtoolkit.getCDFparams)

    -  in `ipython` shell by typing ``?`` after the object:
    
        >>> swarmtoolkit.getCDFparams?
        
For ease of use, the abbreviation ``st`` is suggested for `swarmtoolkit`:

    >>> import swarmtoolkit as st
    >>> st.plot([0,1,2],[2,1,0])
"""

__version__ = '1.2.1'

import sys as _sys
if _sys.version_info[0]<3:
  print('You are importing swarmtoolkit made for python3 using'+\
  ' Python version {}.{}.{} which might not be supported by this module\n'
  .format(*_sys.version_info[:3]))
del _sys

from . import aux
from .aux import *
from . import sw_io
from .sw_io import *
from . import ops
from .ops import *
from . import sph
from .sph import *
from . import vis
from .vis import *



__all__=[]
__all__.extend(aux.__all__)
__all__.extend(sw_io.__all__)
__all__.extend(ops.__all__)
__all__.extend(sph.__all__)
__all__.extend(vis.__all__)

aux._set_sw_logger()




if __name__ == '__main__':
    pass

