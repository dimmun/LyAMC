from general import *

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
    ''' Returns the new direction for the photon
    '''
    if mode == 'uniform':
        x = np.random.normal(size=(3))
        x /= np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        return x
    elif mode == 'Rayleigh':
        r = np.random.rand()
        q = ((16. * r * r - 16 * r + 5.) ** 0.5 - 4. * r + 2.) ** (1. / 3.)
        nu = 1. / q - q
        theta = np.arccos(nu)
        return rotate_by_theta(np.array(n), theta), nu
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
