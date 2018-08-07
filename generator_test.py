import numpy as np


def p_zm(u0, x, a):
    theta0 = np.atan((u0 - x) / a)
    return (theta0 + np.pi / 2) * \
           ((1. - np.exp(-u0 ** 2)) * theta0 + \
            (1. + np.exp(-u0 ** 2)) * np.pi / 2.) ** -1


def g_zm(u0, u, x, a):
