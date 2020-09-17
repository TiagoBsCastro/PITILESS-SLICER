from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name='NFWx',
      ext_modules=cythonize("NFWx.pyx"),
      include_dirs=[np.get_include()])
