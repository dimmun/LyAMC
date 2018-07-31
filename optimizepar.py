from multiprocessing import Pool

import matplotlib.pyplot as plt
from scipy import integrate
from  scipy.special import erf


def f(u, a, x, norm):
    return norm * np.exp(-u ** 2) / ((x - u) ** 2 + a ** 2)


from lyamc.cons import *
from lyamc.general import *

a = ALYA / 4 / np.pi / (NULYA * get_vth(2e4) / c)

factr = 1

x_list = np.linspace(-8., 8.0, 201)
s_list = np.linspace(-5, 5, 200)
p_list = (erf(s_list) + 1.) / 2.
temp_list = np.linspace(0, 1, 200*factr)
ltab = np.zeros([len(x_list), len(p_list)])

# for xi, x in enumerate(x_list):
#     print(xi)
#     u = np.concatenate([np.linspace(-15, 15, 2000*factr),
#                         np.tan((temp_list - 0.5) * np.pi) * a + x])
#     u = np.sort(u)
#     res = np.zeros(len(u))
#     for i in range(len(u) - 1):
#         norm = f(u[i], a, x, 1.0)
#         if norm>1e-20:
#             res[i + 1] = \
#                 integrate.quad(f, a=u[i], b=u[i + 1], args=(a, x, 1./norm), epsabs=1e-15)[0] * norm
#         else:
#             res[i + 1] = \
#                 np.trapz([u[i], u[i+1]], [f(u[i], a, x, 1./norm), f(u[i+1], a, x, 1./norm)]) * norm
#     res[np.isnan(res)] = 0.
#     res = np.cumsum(res)
#     res /= res[-1]
#     ltab[xi, :] = np.interp(p_list, res, u)
#     # plt.plot(u, res)
#     # res = f(u, a, x)
#     # res /= res.max()

####
# def ddf(x):
#     u = np.concatenate([np.linspace(-15, 15, 2000 * factr),
#                         np.tan((temp_list - 0.5) * np.pi) * a + x])
#     u = np.sort(u)
#     res = np.zeros(len(u))
#     for i in range(len(u) - 1):
#         norm = f(u[i], a, x, 1.0)
#         if norm > 1e-20:
#             res[i + 1] = \
#                 integrate.quad(f, a=u[i], b=u[i + 1], args=(a, x, 1. / norm), epsabs=1e-15)[0] * norm
#         else:
#             res[i + 1] = \
#                 np.trapz([u[i], u[i + 1]], [f(u[i], a, x, 1. / norm), f(u[i + 1], a, x, 1. / norm)]) * norm
#     res[np.isnan(res)] = 0.
#     res = np.cumsum(res)
#     res /= res[-1]
#     return np.interp(p_list, res, u)

def integrand_for_par_vel(v, args):
    '''

    :return:
    '''
    vth, nu, umod = args
    return np.exp(-(v) ** 2 / vth ** 2) / ((nu * (1 - (v + umod) / c) - nua) ** 2 + (ALYA / 4 / np.pi) ** 2)


def ddf(x):
    vth = get_vth(T=2e4)
    # I = lambda w: integrate.quad(q, w[0], w[1], )[0]
    # w_list = np.linspace(-10 * vth, 10 * vth, 1024)
    u = 0.0
    nu = get_nu(x, T=2e4)
    w_list = np.sort(
        np.concatenate([np.linspace(-7 * vth, 7 * vth, 1000), -u / vth + np.linspace(-0.2, 0.2, 100)]))
    res = np.zeros(len(w_list))
    for i in range(len(w_list) - 1):
        temp = integrate.quad(integrand_for_par_vel, a=w_list[i], b=w_list[i + 1], args=[vth, nu, u], limit=1000)
        # print(temp)
        res[i + 1] = \
            temp[0]
    res = np.cumsum(res)
    res /= res[-1]
    # print(p_list, res, u)
    return np.interp(p_list, res, w_list)


p = Pool(4)
res = p.map(ddf, x_list)

ltab = np.array(res)

np.savez('a_%0.10f.npz' % a, x_list=x_list, s_list=s_list, p_list=p_list, ltab=ltab)

plt.pcolor(s_list, x_list, ltab, vmin=-20, vmax=20)
plt.show()

plt.pcolor(s_list, x_list, ltab)
plt.colorbar()
plt.show()

#
# x = 6.5
# u = np.concatenate([np.linspace(-15, 15, 200*factr),
#                     np.tan((temp_list - 0.5) * np.pi) * a + x])
# u = np.sort(u)
# res = np.zeros(len(u))
# for i in range(len(u) - 1):
#     norm = f(u[i], a, x, 1.0)
#     # res[i + 1] = \
#     #     integrate.quad(f, a=u[i], b=u[i + 1], args=(a, x, 1. / norm), epsabs=1e-10)[0] * norm
#     res[i + 1] = \
#             np.trapz([f(u[i], a, x, 1./norm), f(u[i+1], a, x, 1./norm)], [u[i], u[i+1]]) * norm
#
# temp = f(u, a, x, 1.)
# res = np.concatenate([0, np.diff(u) * (temp[1:] - temp[:-1]) / 2.])
#
# ax = plt.subplot(3,1,1)
# plt.plot(u[1:],  (u[1:] - u[:-1]) , '.')
# plt.yscale('log')
# plt.subplot(3,1,2, sharex=ax )
# plt.plot(u[1:],   (temp[1:]  + temp[:-1]) / 2., '.')
# plt.yscale('log')
# plt.subplot(3,1,3, sharex=ax )
# t=np.cumsum(res[1:])
# t /= t[-1]
# plt.plot(u[1:],  1-t, '.')
# plt.yscale('log')
# plt.xlim([-10,10])
# plt.show()
#
#
#
# res = np.cumsum(res)
# res /= res[-1]
# ltab[xi, :] = np.interp(p_list, res, u)
#
#
# plt.plot(x_list, ltab[:, 1])
# plt.plot(x_list, ltab[:, 4999])
# plt.plot(x_list, ltab[:, -2])
# plt.show()
#
#
# dat = np.load('a_%0.10f.npz' % a)
# x_list=dat['x_list']
# s_list=dat['s_list']
# p_list=dat['p_list']
# ltab=dat['ltab']
# # np.savez('a_%0.10f.npz' % a, x_list=x_list, s_list=s_list, p_list=p_list, ltab=ltab)
#
# # plt.plot(u, res)
# #
# # plt.plot(u, (erf((u-3.5)*4.)+1.)/2., 'k', lw=2)
# # plt.show()
#
#
# plt.contour(ltab)
# plt.show()
#
# # # plt.plot(u, erf(u))
# # # plt.show()
# #
# # plt.pcolor(ltab[500:750, 500:750])
# # plt.show()
# #
# plt.plot(p_list, ltab[210, :], '.')
# plt.show()
# #
# #
# # ############################
# #
# #
# # a = 0.0001
# #
# # x_list = np.linspace(0,6,200)
# # s_list = np.linspace(-5,5,200)
# # p_temp_list = (erf(s_list)+1.)/2.
# # temp_list = np.linspace(0,1,200)
# # ltab   = np.zeros([len(x_list), len(p_temp_list)+len(temp_list)])
# #
# #
# # for xi, x in enumerate(x_list):
# #     print(xi)
# #     u = np.concatenate([np.linspace(-10, 10, 200),
# #                         np.tan((temp_list - 0.5) * np.pi) * a + x])
# #     u = np.sort(u)
# #     p_list = np.sort(np.concatenate([np.tan((temp_list - 0.5) * np.pi) * a + x,
# #                                      p_temp_list]))
# #     res = np.zeros(len(u))
# #     for i in range(len(u) - 1):
# #         res[i + 1] = \
# #             integrate.quad(f, a=u[i], b=u[i + 1], args=(a, x), limit=1000)[0]
# #     res = np.cumsum(res)
# #     res /= res[-1]
# #     ltab[xi, :] = np.interp(p_list, res, u)
# #
# #
# # plt.pcolor(ltab)
# # plt.show()
# # ###############
# #
# # plt.plot(u, np.arctan((u - x_list[100])/a)/np.pi +.5)
# # plt.plot(np.tan((p_list-0.5)*np.pi)*a+x_list[100], p_list, '.')
# # plt.xlim([-5,5])
# plt.show()
