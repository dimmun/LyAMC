'''
Routines for determining the trajectory of the photon
'''

from lyamc.atomic import *
from lyamc.geometry import *


def get_trajectory(x, n, d):
    '''
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


def get_survival_function(nu, l, d, k, grad_V):
    '''
    Generates survival function
    :param nu: frequency of the photon in rest frame
    :param l: coordinates [N,3]
    :param d: coordinates along the trajectory [N]
    :param k:
    :param grad_V:
    :return:
    '''
    tau_d = DtauDl(k=k, nu=nu, v=velocity(l, grad_V=grad_V), T=temperature(l), ndens=density(l))
    tau = np.cumsum(tau_d * np.gradient(d))  # replace with cumtrapz
    return np.exp(-tau)


def random_d(d, sf):
    ''' Returns the distance traveled before scattering.
    '''
    return np.interp(np.random.rand(1)[0], sf[::-1], d[::-1])
