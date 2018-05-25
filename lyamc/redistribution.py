from scipy import integrate

from lyamc.general import *


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def rotate_by_theta(n, theta):
    '''rotates vector n by theta in random direction'''
    axis = np.random.randn(3)
    axis -= axis.dot(n) * n / np.linalg.norm(n) ** 2
    axis /= np.linalg.norm(axis)
    return np.dot(rotation_matrix(axis, theta), n)


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


def get_g(T):
    '''returns g for T in K'''
    return 2.6e-4 * (T / 1e4) ** -0.5


def get_xout(xin, v, kin, kout, mu, T):
    '''Equation 65'''
    g = get_g(T)
    vth = get_vth(T)
    return xin - np.dot(v, kin) / vth + np.dot(v, kout) / vth + g * (mu - 1)


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


def get_par_velocity_of_atom(nu, T, u, n, mode='direct', N=10000):
    '''
    Generates a parallel component for the velocity of the atom.

    :param nu: frequency
    :param T:  local gas temperature in K
    :param u:  bulk gas evlocity
    :param n:  photon direction
    :return:   vector parallel to
    '''
    if mode == 'direct':
        x = get_x(nu, T)
        vth = get_vth(T)
        a = 4.7e-4 * (T / 1e4) ** -0.5
        umod = np.dot(u, n)
        q = lambda v: a ** 2 * np.exp(-(v) ** 2 / vth ** 2) / (a ** 2 + (x - (v + umod) / c) ** 2)
        I = lambda w: integrate.quad(q, w[0], w[1], limit=1000)[0]
        w_list = np.linspace(-5 * vth, 5 * vth, 1000)
        res = np.zeros(len(w_list))
        for i in range(len(w_list) - 1):
            res[i + 1] = I([w_list[i], w_list[i + 1]])
        res = np.cumsum(res)
        res /= res[-1]
        r = np.random.rand()
        return n * np.interp(r, res, w_list)
    else:
        x = get_x(nu, T)
        # DeltanuD = nua * np.sqrt(T / c ** 2 / mp_over_2kB)
        # a = DeltanuL / 2. / DeltanuD
        a = 4.7e-4 * (T / 1e4) ** -0.5
        v_par = np.random.normal(loc=0, scale=1., size=N) * get_vth(T)
        r = np.random.rand(N)
        temp = a ** 2 / ((x - (np.dot(u, n) + v_par) / c) ** 2 + a ** 2)
        if temp.max() < 10. / N:
            temp /= temp.max()
        r = r < temp
        return n * v_par[r][0]


def get_perp_velocity_of_atom(nu, T, u, n):
    '''

    :param nu:
    :param T:
    :param u:
    :param n:
    :return:
    '''
    vth = get_vth(T)
    v_perp = np.random.normal(loc=0, scale=1., size=2) * vth / np.sqrt(2)
    direction0 = rotate_by_theta(n, np.pi / 2)
    direction1 = np.cross(direction0, n)
    return v_perp[0] * direction0 + v_perp[1] * direction1
