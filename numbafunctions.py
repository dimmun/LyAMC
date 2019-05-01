import numpy as np
from numba import jit
from numba import double
from numba.extending import get_cython_function_address
import ctypes


# numba optimized linear algebra functions based on https://gist.github.com/ufechner7/98bcd6d9915ff4660a1
@jit(nopython=True)
def numba_cross(vec1, vec2):
    """calculates the cross product of two 3d vectors"""
    result = np.zeros(3)
    return numba_cross_(vec1, vec2, result)


@jit(nopython=True)
def numba_cross_(vec1, vec2, result):
    """calculates the cross product of two 3d vectors"""
    a1, a2, a3 = double(vec1[0]), double(vec1[1]), double(vec1[2])
    b1, b2, b3 = double(vec2[0]), double(vec2[1]), double(vec2[2])
    result[0] = a2 * b3 - a3 * b2
    result[1] = a3 * b1 - a1 * b3
    result[2] = a1 * b2 - a2 * b1
    return result


@jit(nopython=True)
def numba_dot(vec1, vec2):
    """calculates the dot product of two 3d vectors"""
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]


@jit(nopython=True)
def numba_norm(vec):
    """calculates the norm of a 3d vector"""
    return np.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])


@jit(nopython=True)
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    # axis = np.asarray(axis)
    axis = axis / np.sqrt(numba_dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


@jit(nopython=False)
def rotate_by_theta(n, theta):
    """rotates vector n by theta in random direction"""
    axis = np.random.randn(3)
    axis -= axis.dot(n) * n / numba_norm(n) ** 2
    axis /= numba_norm(axis)
    return numba_dot(rotation_matrix(axis, theta), n)


# numba optimized Voigt function using numba wrapper around scipy.wofz (Faddeeva function)
# see https://github.com/numba/numba/issues/3086
_PTR = ctypes.POINTER
_dble = ctypes.c_double
_ptr_dble = _PTR(_dble)

addr = get_cython_function_address("special", "wofz")
functype = ctypes.CFUNCTYPE(None, _dble, _dble, _ptr_dble, _ptr_dble)
wofz_fn = functype(addr)


@jit(nopython=True)
def numba_Voigt(a, x):
    out_real = np.empty(1, dtype=np.float64)
    out_imag = np.empty(1, dtype=np.float64)
    wofz_fn(x, a, out_real.ctypes, out_imag.ctypes)   
    return out_real[0]