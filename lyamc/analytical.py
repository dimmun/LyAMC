import numpy as np


def Dijkstra_sphere_test(x, a, tau0):
    '''
    Return an analytical solution for a homogeneous sphere (without recoil) from Dijkstra et al.
(2006)
    :param x:
    :param a:
    :param tau0:
    :return:
    '''
    return np.sqrt(np.pi) / np.sqrt(24) / a / tau0 * x ** 2 / (
    1. + np.cosh(np.sqrt(2. * np.pi / 27) * np.abs(x) ** 3) / a / tau0)
