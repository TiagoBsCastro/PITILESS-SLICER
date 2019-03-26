from distutils.core import setup
from Cython.Build import cythonize

setup(name='PLC Builder',
      ext_modules=cythonize("builder.pyx"))
