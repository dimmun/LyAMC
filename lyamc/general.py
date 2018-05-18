import numpy as np

### Consts
km_in_pc = 3.086e+13
cm_in_pc = 3.086e+18
nua = 2.47e15  # Hz
c = 2.99792e5  # km/s
DeltanuL = 99471839.  # natural line width in Hz
sigmat = 6.65e-25  # cm^2

# https://www.wolframalpha.com/input/?i=(proton+mass+%2F+2)+%2F+(Bolzmann+constant)+in+s%5E2*K%2Fkm%5E2
mp_over_2kB = 60.57  # s^2 K / km^2


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

