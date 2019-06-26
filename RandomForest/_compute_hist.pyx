"""Histogram Computation using Cython

"""


cimport numpy as np

np.import_array()

# cdefine signature of c function
cdef extern from "compute_hist.h":
    void compute_hist (float * mag_arr, float * angle_arr, float * hist, int num_pixels, int num_bins)

# create wrapper code with numpy type annotations
def compute_hist_func(np.ndarray[float, ndim=1, mode="c"] mag_arr not None,
                     np.ndarray[float, ndim=1, mode="c"] angle_arr not None,
					 np.ndarray[float, ndim=1, mode="c"] hist not None,
					 int num_pixels,
					 int num_bins):
    compute_hist(<float*> np.PyArray_DATA(mag_arr),
                <float*> np.PyArray_DATA(angle_arr),
				<float*> np.PyArray_DATA(hist),
				num_pixels,
				num_bins)