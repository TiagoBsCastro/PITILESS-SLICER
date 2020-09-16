from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name='Interp 2D',
      ext_modules=cythonize("interp.pyx"),
      include_dirs=[np.get_include()])
