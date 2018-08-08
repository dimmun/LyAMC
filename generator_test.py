import numpy as np
from scipy import integrate
from lyamc.redistribution import *
from multiprocessing import Pool


nua = 2466071805789080.5  # Hz
c = 2.99792e5  # km/s

def p_zm(u0, x, a):
    theta0 = np.arctan((u0 - x) / a)
    return (theta0 + np.pi / 2) * \
           ((1. - np.exp(-u0 ** 2)) * theta0 + \
            (1. + np.exp(-u0 ** 2)) * np.pi / 2.) ** -1


def g_zm(u0, u, x, a):
    t = ((x-u)**2 + a**2)**-1
    return (u <= u0) * t + \
           (u >  u0) * t * np.exp(-u0**2)


def ZM_approach(u0, x, a, N):
    p = p_zm(u0, x, a)
    theta0 = np.arctan((u0 - x) / a)
    R = np.random.rand(N)
    theta = np.random.rand(N)
    theta[R < p] = theta[R < p] * (theta0 + np.pi/2.) - np.pi/2
    theta[R >= p] = theta[R >= p] * (np.pi/2. - theta0) + theta0
    # theta = np.random.rand(N)*np.pi - np.pi/2.
    u = a*np.tan(theta) + x
    # g = g_zm(u0, u, x, a)
    acc = np.random.rand(N)
    return u[((acc < np.exp(-u**2)) & (u<=u0)) | ((acc < np.exp(-u**2)/np.exp(u0**2)) & (u>u0))]




# def integrand_for_par_vel(v, args):
#     vth, nu, umod = args
#     return np.exp(-(v) ** 2 / vth ** 2) / ((nu * (1 - (v + umod) / c) - nua) ** 2 + (ALYA / 4 / np.pi) ** 2)

def integrand(u, args):
    x, a = args
    return np.exp(-u ** 2) / ((x - u)**2 + a**2)


def direct_approach(u0, x, a, N):
    w_list = np.sort(
            np.concatenate([np.linspace(-15, 15, 1000),
                            x + np.linspace(-0.2, 0.2, 100)]))
    res = np.zeros(len(w_list))
    for i in range(len(w_list) - 1):
        res[i + 1] = \
            integrate.quad(integrand, a=w_list[i], b=w_list[i + 1], args=[x, a], limit=10)[0]
    res = np.cumsum(res)
    res /= res[-1]
    r = np.random.rand(N)
    return np.interp(r, res, w_list)

# Testing

import matplotlib.pyplot as plt
N = 1000000
u0 = 0
x = -5
T = 2e4
vth = get_vth(T)
nu = get_nu(x, T)
a = 4.7e-4 * (T / 1e4) ** -0.5


t1 = [ZM_approach(u0=u0, x=x, a=a, N=N*1) for i in range(20)]
t1 = np.concatenate(t1)
t2 = direct_approach(u0=u0, x=x,  a=a, N=N)

# def pool_f(i):
#     return get_par_velocity_of_atom(nu, T, np.array([u0, 0, 0]), np.array([1, 0, 0]), [], mode='integral')/vth
#
# p = Pool(10)
# # t3 = p.map(pool_f, range(10000))
# t3=[pool_f(1)[0] for i in range(10000)]

plt.hist([t1, t2], 100, normed=True, histtype='step')

# u = np.linspace(-10, 10, 100000)
# plt.plot(u, integrand(u, (x, a)))
plt.yscale('log')
plt.show()