from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

ext_modules = [
    Extension("builder", ["builder.pyx"],
        extra_compile_args=['-O3','-ffast-math','-march=native'],
        libraries=['m'])]

setup(name='PLC Builder',
      ext_modules=cythonize(ext_modules, compiler_directives={'language_level' : "3"}),
)
