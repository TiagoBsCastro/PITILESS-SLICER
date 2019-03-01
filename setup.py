from distutils.core import setup
from Cython.Build import cythonize

setup(name='PLC Solver App',
      ext_modules=cythonize("builder.pyx"))
