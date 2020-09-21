from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name='wrapPositions',
      ext_modules=cythonize("wrapPositions.pyx"),
      include_dirs=[np.get_include()])
