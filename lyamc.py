import matplotlib.pyplot as plt
import numpy as np
from numbafunctions import *
from numba import jit, jitclass
from numba import float64
import ctypes
import time



# all defintions and constants from Dijkstra (2017) arxiv.org/abs/1704.03416
SPEEDC = 2.9979e5 # in km/s
NUA = 2.47e15 # in Hz
TAU0 = 1.e5
SIGMA0 = 5.898e-14
TEMP = 1.e4 # in K
RSIZE = 1.0
VMAX = 0.0
COEF = 1.e-5 # effective cell 'size'


spec = [
    ('position', float64[:]),
    ('temperature', float64),              
    ('density', float64),
    ('vbulk', float64[:])
]


@jitclass(spec)
class Cell(object):
    def __init__(self, position):
        self.position = position
        self.temperature = TEMP
        self.density = TAU0 / SIGMA0
        self.vbulk = (VMAX / RSIZE) * position.copy()
        #self.luminosity


@jit(nopython=True)
def get_vth(T):
    """returns vth in km/s for T in K 
    vth = (2 * kB * T / mp) ** 0.5
    """
    return 0.1285 * np.sqrt(T)


@jit(nopython=True)
def get_x(nu, T):
    """returns dimensionless frequency x for frequency nu in Hz and T in K"""
    vth = get_vth(T)
    deltanua = NUA * vth / SPEEDC
    return (nu - NUA) / deltanua


@jit(nopython=True)
def get_nu(x, T):
    """returns dimensionless frequency for nu in Hz and T in K"""
    vth = get_vth(T)
    deltanua = NUA * vth / SPEEDC
    return x * deltanua + NUA


@jit(nopython=True)
def get_a(T):
    """returns Voigt parameter a for T in K"""
    return 4.7 * 1.e-4 / np.sqrt(T / 1.e4)


@jit(nopython=True)
def get_g(T):
    """returns recoil coefficient g"""
    return 2.6e-4 / np.sqrt(T / 1.e4)


@jit(nopython=True)
def get_xcw(a):
    """returns xcw that separates core from wing
    from Laursen (2010) arxiv.org/pdf/1012.3175
    """
    #return np.abs(np.sqrt(lambertw(-a / np.sqrt(np.pi), k=-1)))
    return 1.59 - 0.60 * np.log10(a) - 0.03 * np.log10(a) ** 2


@jit(nopython=True)
def get_u0(xabs, a):
    """returns parameter u0 that optimizes the acceptance rate for get_upar
    see data/acceptance_rate.pdf for comparison with altenatives from 
    Laursen (2010) arxiv.org/pdf/1012.3175 and Semelin (2007) arxiv.org/abs/0707.2483
    """
    if (xabs >= 5):
        u0 = 4.5
    elif (xabs >= 0.2):
        u0 = xabs - 0.01 * np.exp(1.2 * xabs) * (a ** (1.0 / 6.0))
    else:
        u0 = 0.0
    return u0


@jit(nopython=True)
def sigma_atom(x, T):
    """returns scattering cross section on atom"""
    a = get_a(T)
    return (5.9e-14 / np.sqrt(T / 1.e4)) * numba_Voigt(a, x)


@jit(nopython=True)
def get_upar(x, a):
    """
    generates parallel velocity of atom in units of vth using Monte-Carlo rejection method
    """
    xabs = np.abs(x)
    u0 = get_u0(xabs, a)
    N = 10
    theta0 = np.arctan((u0 - xabs) / a)
    p = (theta0 + np.pi/2) / ((1. - np.exp(-u0 ** 2)) * theta0 + (1. + np.exp(-u0 ** 2)) * np.pi/2)
    res = np.zeros(0)
    while (len(res) == 0):
        R = np.random.rand(N)
        theta = np.random.rand(N)
        theta = np.where(R <= p, theta * (theta0 + np.pi/2) - np.pi/2, theta * (np.pi/2 - theta0) + theta0)
        uu = a * np.tan(theta) + xabs
        acc = np.random.rand(N)
        res = uu[((acc < np.exp(-uu ** 2)) & (uu <= u0)) | ((acc < (np.exp(-uu ** 2) / np.exp(-u0 ** 2))) & (uu > u0))]
        if (N <= 1e8):
            N *= 10
        else:
            print("too many rejections!")
    if x < 0:
        return -res[0]
    else:
        return res[0]


@jit(nopython=True)
def get_uperp():
    """
    generates perpedicular velocity of atom in units of vth
    """
    return np.random.normal(loc=0, scale=(1/np.sqrt(2)), size=2)


@jit(nopython=True)
def get_kout_projections(x, a):
    """ 
    generates projections of kout on vperp[0], vperp[1], and vpar
    uses redistribution procedure from Laursen (2010) arxiv.org/pdf/1012.3175 section 7.3.2
    """
    R = np.random.rand()
    # xabs = np.abs(x)
    # xcw = get_xcw(a)
    # phi = np.where(xabs < xcw, (14. - 24 * R + np.sqrt(245 - 672 * R + 576 * R ** 2)) / 7.0, 
    #               2. - 4 * R + np.sqrt(5. - 16 * R + 16 * R ** 2))
    # Rayleigh:
    phi = 2. - 4 * R + np.sqrt(5. - 16 * R + 16 * R ** 2)
    mu = phi ** (-1.0 / 3.0) - phi ** (1.0 / 3.0)
    s = np.inf
    while (s > 1):
        r1 = np.random.uniform(-1.0, 1.0)
        r2 = np.random.uniform(-1.0, 1.0)
        s = r1 ** 2 + r2 ** 2
    r1 *= np.sqrt((1.0 - mu**2) / s)
    r2 *= np.sqrt((1.0 - mu**2) / s)
    # isotropic:
    # r1, r2, mu = get_random_k() 
    return r1, r2, mu


@jit(nopython=True)
def atom_scattering(x, k, T, vbulk):
    """
    returns normalized photon frequency xout and its direction kout after scattering on atom 
    :param k:  unit vector in the direction of photon propagation
    :param T:  local gas temperature in K
    :param vbulk:  bulk velocity of the gas in km/s
    """
    a = get_a(T)
    upar = get_upar(x, a)
    uperp = get_uperp()
    r1, r2, mu = get_kout_projections(x, a)
    direction0 = rotate_by_theta(k, np.pi/2)
    direction1 = numba_cross(k, direction0)
    kout = r1 * direction0 + r2 * direction1 + mu * k
    kout /= numba_norm(kout)
    vth = get_vth(T)
    g = get_g(T)
    xout = x + (mu - 1.) * upar + r1 * uperp[0] + r2 * uperp[1] + numba_dot(vbulk / vth, kout - k) + g * (mu - 1.)
    return xout, kout


@jit(nopython=True)
def get_tau():
    """generates optical depth
    """
    R = np.random.rand()
    return -np.log(R)


@jit(nopython=True)
def get_random_k():
    """returns unit vector in random direction"""
    k = np.random.randn(3)
    return k / numba_norm(k)


@jit(nopython=True)
def get_emission_cell():
    """returns emission cell at the center of the coordinates
    TO DO: return cell based on relative luminosity of the cell
    """
    r = np.array([0.0, 0.0, 0.0])
    cell = Cell(r)
    return cell


@jit(nopython=True)
def get_scattering_cell(x, tau, k, cell):
    """returns cell where scattering happens
    TO DO: generalize for general geometry
    """
    s = 0.0
    ds = COEF * RSIZE #cell size
    position0 = cell.position.copy()
    density, T, vbulk = cell.density, cell.temperature, cell.vbulk
    vth = get_vth(T)
    xlocal = x - numba_dot(vbulk / vth, k)
    f = density * sigma_atom(xlocal, T)
    # Ntau = 0
    while (tau > 0):
        fprevious = f
        s += ds
        position = position0 + s * k
        cell = Cell(position)
        if (numba_norm(position) > RSIZE):
            escaped = True
            return cell, escaped
        density, T, vbulk = cell.density, cell.temperature, cell.vbulk
        vth = get_vth(T)
        xlocal = x - numba_dot(vbulk / vth, k)
        f = density * sigma_atom(xlocal, T)
        tau -= 0.5 * (f + fprevious) * ds
        # Ntau += 1
        #print("Ntau =", Ntau, "s =", s, "tau =", tau, "xlocal =",  xlocal)
        #print(numba_dot(vbulk / vth, k), np.dot(vbulk / vth, k))
    escaped = False
    cell = Cell(position0 + (s - 0.5 * ds) * k)
    return cell, escaped


@jit(nopython=True)
def runner(Nphotons=1000):
    """
    emission at x = 0 from center of a sphere of radius RSIZE 
    expanding with outer edge velocity VMAX 
    """
    xs = np.zeros(Nphotons)
    Ns = np.zeros(Nphotons)
    for photon in range(Nphotons):
        cell = get_emission_cell()
        a = get_a(TEMP)
        x = 0.0
        k = get_random_k()
        escaped = False
        Nscattering = 0
        while ((escaped == False) and (Nscattering < 1.e5 * TAU0)):
            tau = get_tau()
            cell, escaped = get_scattering_cell(x, tau, k, cell)
            Nscattering += 1
            T, vbulk = cell.temperature, cell.vbulk
            x, k = atom_scattering(x, k, T, vbulk)
        if (photon % 100) == 0:
            print("photon #", photon, "Nscattering =", Nscattering)
        xs[photon] = x
        Ns[photon] = Nscattering
    return xs, Ns









