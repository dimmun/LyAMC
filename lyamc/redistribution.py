from numba import jit
from scipy import integrate

from lyamc.cons import *
from lyamc.general import *


# ALYA = 6.2648e+8


@jit(nopython=False)
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    # axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


@jit(nopython=False)
def rotate_by_theta(n, theta):
    '''rotates vector n by theta in random direction'''
    axis = np.random.randn(3)
    axis -= axis.dot(n) * n / np.linalg.norm(n) ** 2
    axis /= np.linalg.norm(axis)
    return np.dot(rotation_matrix(axis, theta), n)


@jit(nopython=False)
def random_n(n, mode='Rayleigh'):
    ''' Returns a new direction for the photon
    '''
    if mode == 'uniform':
        x = np.random.normal(size=(3))
        x /= np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        return x, 1
    elif mode == 'Rayleigh':
        r = np.random.rand()
        q = ((16. * r * r - 16 * r + 5.) ** 0.5 - 4. * r + 2.) ** (1. / 3.)
        nu = 1. / q - q
        theta = np.arccos(nu)
        return rotate_by_theta(np.array(n).copy(), theta), nu
    else:
        print('error')


@jit(nopython=False)
def get_g(T):
    '''returns g for T in K'''
    return 2.6e-4 * (T / 1e4) ** -0.5


@jit(nopython=False)
def get_xout(xin, v, kin, kout, mu, T):
    '''Equation 65'''
    g = get_g(T)
    vth = get_vth(T)
    return xin - np.dot(v, kin) / vth + np.dot(v, kout) / vth + g * (mu - 1)


@jit(nopython=False)
def get_parallel_PDF(v, u, n, nu, T):
    v_par = npsumdot(v, n)
    u_par = npsumdot(u, n)
    x = get_x(nu, T)
    DeltanuD = nua * np.sqrt(T / c ** 2 / mp_over_2kB)
    a = DeltanuL / 2. / DeltanuD
    # return 1. / ((x - v_par/c)**2 + a**2)
    return 1. / np.sqrt(np.pi * mp_over_2kB / T) * np.exp(- mp_over_2kB / T * (v_par - u_par) ** 2), a ** 2 / (
    (x - v_par / c) ** 2 + a ** 2)
    # return np.exp( - mp_over_2kB / T * (v_par-u_par)**2) / ((x - v_par/c)**2 + a**2)


def get_lookup_table_for_par_velocity(T):
    '''
    Precalculates lookup table for parallel velocities
    :param T:
    :return:
    '''


def integrand_for_par_vel(v, args):
    '''

    :return:
    '''
    vth, nu, umod = args
    return np.exp(-(v) ** 2 / vth ** 2) / ((nu * (1 - (v + umod) / c) - nua) ** 2 + (ALYA / 4 / np.pi) ** 2)


def integrand_for_par_vel_old(v, args):
    '''

    :return:
    '''
    vth, nu, umod = args
    return np.exp(-(v) ** 2 / vth ** 2) / ((nu * (1 - (v + umod) / c) - nua) ** 2 + (ALYA / 16 / np.pi ** 2) ** 2)


@jit(nopython=False)
def p_zm(u0, x, a):
    theta0 = np.arctan((u0 - x) / a)
    return (theta0 + np.pi / 2) * \
           ((1. - np.exp(-u0 ** 2)) * theta0 + \
            (1. + np.exp(-u0 ** 2)) * np.pi / 2.) ** -1



@jit(nopython=False)
def get_par_velocity_of_atom(nu, T, u, n, f_ltab, mode='integral'):
    '''
    Generates a parallel component for the velocity of the atom.

    :param nu: frequency
    :param T:  local gas temperature in K
    :param u:  bulk gas evlocity
    :param n:  photon direction
    :return:   vector parallel to
    '''
    if mode == 'integral':
        x = get_x(nu, T)
        vth = get_vth(T)
        a = 4.7e-4 * (T / 1e4) ** -0.5
        umod = np.dot(u, n)
        # I = lambda w: integrate.quad(q, w[0], w[1], )[0]
        # w_list = np.linspace(-10 * vth, 10 * vth, 1024)
        w_list = np.sort(
            np.concatenate([np.linspace(-7 * vth, 7 * vth, 1000), -umod / vth + np.linspace(-0.2, 0.2, 100)]))
        res = np.zeros(len(w_list))
        for i in range(len(w_list) - 1):
            res[i + 1] = \
                integrate.quad(integrand_for_par_vel, a=w_list[i], b=w_list[i + 1], args=[vth, nu, umod], limit=10)[0]
        res = np.cumsum(res)
        res /= res[-1]
        r = np.random.rand()
        print(r)
        return n * np.interp(r, res, w_list)
    elif mode == 'lookup':
        r = np.random.rand()
        # print(r)
        vth = get_vth(T)
        umod = np.dot(u, n)
        x = get_x(nu * (1 + umod / c), T)[0]
        # print(r, x)
        return n * f_ltab(r, x)
        # if x > 0:
        #     print(f_ltab(r, x))
        #     return n * f_ltab(r, x) * vth / np.sqrt(2)
        # else:
        #     print(-f_ltab(r, -x))
        #     return n * -1. * f_ltab(r, -x) * vth
    elif mode == 'zm':
        N = 1000
        x = get_x(nu, T)
        vth = get_vth(T)
        u0 = np.dot(u, n) / vth
        a = 4.7e-4 * (T / 1e4) ** -0.5
        if x > 0:
            p = p_zm(u0, x, a)
            theta0 = np.arctan((u0 - x) / a)
            res = []
            while len(res) == 0:
                print('more')
                R = np.random.rand(N)
                theta = np.random.rand(N)
                theta[R < p] = theta[R < p] * (theta0 + np.pi / 2.) - np.pi / 2
                theta[R >= p] = theta[R >= p] * (np.pi / 2. - theta0) + theta0
                # theta = np.random.rand(N)*np.pi - np.pi/2.
                u = a * np.tan(theta) + x
                # g = g_zm(u0, u, x, a)
                acc = np.random.rand(N)
                res = u[((acc < np.exp(-u ** 2)) & (u <= u0)) | ((acc < np.exp(-u ** 2) / np.exp(u0 ** 2)) & (u > u0))]
            return n * res[0] * vth
        else:
            x = -x
            u0 = -u0
            vth = get_vth(T)
            u0 = np.dot(u, n) / vth
            p = p_zm(u0, x, a)
            theta0 = np.arctan((u0 - x) / a)
            res = []
            while len(res) == 0:
                print('more1', x, u0)
                R = np.random.rand(N)
                theta = np.random.rand(N)
                theta[R < p] = theta[R < p] * (theta0 + np.pi / 2.) - np.pi / 2
                theta[R >= p] = theta[R >= p] * (np.pi / 2. - theta0) + theta0
                # theta = np.random.rand(N)*np.pi - np.pi/2.
                u = a * np.tan(theta) + x
                # g = g_zm(u0, u, x, a)
                acc = np.random.rand(N)
                res = u[((acc < np.exp(-u ** 2)) & (u <= u0)) | ((acc < np.exp(-u ** 2) / np.exp(u0 ** 2)) & (u > u0))]
            return -n * res[0] * vth




    # elif mode == 'fast':
    #     return 0
    # elif mode == 'direct_old':
    #     umod = np.dot(u, n)
    #     q = lambda v: a ** 2 * np.exp(-(v) ** 2 / vth ** 2) / (a ** 2 + (x - (v + umod) / c) ** 2)
    #     I = lambda w: integrate.quad(q, w[0], w[1], limit=1000)[0]
    #     w_list = np.linspace(-5 * vth, 5 * vth, 1000)
    #     res = np.zeros(len(w_list))
    #     for i in range(len(w_list) - 1):
    #         res[i + 1] = I([w_list[i], w_list[i + 1]])
    #     res = np.cumsum(res)
    #     res /= res[-1]
    #     r = np.random.rand()
    #     return n * np.interp(r, res, w_list)
    # elif mode == 'fast_old':
    #     # DeltanuD = nua * np.sqrt(T / c ** 2 / mp_over_2kB)
    #     # a = DeltanuL / 2. / DeltanuD
    #     v_par = np.random.normal(loc=0, scale=1., size=N) * get_vth(T) / np.sqrt(2)
    #     r = np.random.rand(N)
    #     temp = a ** 2 / ((x - (np.dot(u, n) + v_par) / c) ** 2 + a ** 2)
    #     if temp.max() < 10. / N:
    #         temp /= temp.max()
    #     r = r < temp
    #     return n * v_par[r][0]


def get_perp_velocity_of_atom(nu, T, u, n):
    '''

    Drawing a random velocity of an atom in a direction perpendicular to the current LOS.

    :param nu: frequency TODO: unused!
    :param T:  temperature
    :param u:  velocity  TODO: unused!
    :param n:  direction of the LOS
    :return:   3-vector of the velocity component perpendicular to the LOS
    '''
    vth = get_vth(T)
    v_perp = np.random.normal(loc=0, scale=1., size=2) * vth / np.sqrt(2)
    direction0 = rotate_by_theta(n, np.pi / 2)
    direction1 = np.cross(direction0, n)
    # TODO: Replace with a simpler approach without using cross
    return v_perp[0] * direction0 + v_perp[1] * direction1
