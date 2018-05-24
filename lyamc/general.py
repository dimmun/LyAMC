import glob

import numpy as np

import lyamc.cons as cons

NU0 = cons.NULYA / (cons.MHK * cons.K2HZ)

### Consts
km_in_pc = 3.086e+13
cm_in_pc = 3.086e+18
nua = cons.NULYA  # Hz
c = 2.99792e5  # km/s
DeltanuL = 99471839.  # natural line width in Hz
sigmat = 6.65e-25  # cm^2

# https://www.wolframalpha.com/input/?i=(proton+mass+%2F+2)+%2F+(Bolzmann+constant)+in+s%5E2*K%2Fkm%5E2
mp_over_2kB = 60.57  # s^2 K / km^2


def read_last(geom, params):
    s = glob.glob('output/' + decodename(geom, params, sep='_') + '*last*npz')
    k = []
    x = []
    for si in s:
        temp = np.load(si)
        k.append(temp['k'])
        x.append(temp['x'])
    x = np.array(x)
    k = np.array(k)
    direction = npsumdot(k, [0, 0, 1])
    print(len(s))
    return x, k, direction


def decodename(geom, params, sep='_'):
    if len(params) == 6:
        s = '%s %0.2f %.1e %0.2f %0.2f %0.2f %0.2f' % (
        geom, params[0], params[1], params[2], params[3], params[4], params[5])
    else:
        s = '%s %0.2f %.1e %0.2f' % (geom, params[0], params[1], params[2])
    s = s.replace(' ', sep)
    return s


def npsumdot(x, y):
    '''Dot product for two arrays'''
    if len(x.shape) > 1:
        return np.sum(x * y, axis=1)
    else:
        return np.dot(x, y)


def get_x(nu, T):
    '''returns dimensionless frequency for nu in Hz and T in K'''
    vth = get_vth(T)
    Deltanua = nua * vth / c
    return (nu - nua) / Deltanua


def get_nu(x, T):
    '''returns nu in Hz given dimensionless frequency and T in K'''
    vth = get_vth(T)
    Deltanua = nua * vth / c
    return x * Deltanua + nua


def get_vth(T):
    '''return vth in km/s for T in K'''
    return 0.1285 * np.sqrt(T)  # in km/s

