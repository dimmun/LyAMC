'''
Routines for determining the trajectory of the photon
'''

from lyamc.atomic import *
from lyamc.geometry import *


def get_trajectory(x, n, d):
    ''' For given initial position, direction and displacements returns the trajectory.
    :param x: initial position
    :param n: direction
    :param d: coordinates along the chosen direction in pc
    :return: coordinates
    '''
    l = np.array([x[0] + n[0] * d, x[1] + n[1] * d, x[2] + n[2] * d]).T
    return l, d


def get_shift(x, n, dd):
    l = np.array([x[0] + n[0] * dd, x[1] + n[1] * dd, x[2] + n[2] * dd])
    return l


def get_survival_function(nu, l, d, k, geom):
    '''
    Generates survival function
    :param nu: frequency of the photon in rest frame
    :param l: coordinates [N,3]
    :param d: coordinates along the trajectory [N]
    :param k:
    :param grad_V:
    :return:
    '''
    tau_d = DtauDl(k=k, nu=nu, v=geom.velocity(l), T=geom.temperature(l), ndens=geom.density(l))
    tau = np.cumsum(tau_d * np.gradient(d))  # replace with cumtrapz
    return np.exp(-tau)


def optimize_trajectory(d, sf):
    '''

    :param d:
    :param sf:
    :return:
    '''
    good = True
    d = np.interp(np.linspace(1, sf.min(), 100), sf[::-1], d[::-1])
    return good, d

def random_d(d, sf):
    ''' Given the survival function returns the distance traveled before scattering.
    '''
    return np.interp(np.random.rand(), sf[::-1], d[::-1])


def interp_d(d, sf, q):
    ''' Given the survival function returns the distance traveled before scattering.
    '''
    return np.interp(q, sf[::-1], d[::-1])
