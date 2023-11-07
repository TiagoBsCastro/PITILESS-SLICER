from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

# Define extensions
extensions = [
    Extension("IO.Utils.wrapPositions", ["IO/Utils/wrapPositions.pyx"], include_dirs=[np.get_include()]),
    Extension("NFW.NFWx", ["NFW/NFWx.pyx"], include_dirs=[np.get_include()])
]

# Use cythonize on the extensions object.
setup(
    name='PITILESS',
    version='1.0.0',  # Replace with your own project version
    author='Tiago Castro',
    url='https://github.com/TiagoBsCastro/PITILESS-SLICER/',
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    install_requires=[
        'numpy',
        'Cython',  # If you want Cython to be installed automatically when your package is installed
        # Add other dependencies here
    ],
    python_requires='>=3.6',
)

