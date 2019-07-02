"""Histogram Computation using Cython

"""


from distutils.core import setup, Extension

import numpy

from Cython.Distutils import build_ext


setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("compute_hist",
                 sources=["_compute_hist.pyx", "compute_hist.c"],
                 include_dirs=[numpy.get_include()])],
)