from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

#python setup.py build_ext --inplace
setup(
     ext_modules=cythonize("special.pyx"),
     include_dirs=[numpy.get_include()]
)