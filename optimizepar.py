import matplotlib.pyplot as plt
from scipy import integrate
from  scipy.special import erf


def f(u, a, x):
    return np.exp(-u ** 2) / ((x - u) ** 2 + a ** 2)


from lyamc.cons import *
from lyamc.general import *

a = ALYA / 4 / np.pi / (NULYA * get_vth(1e4) / c)

x_list = np.linspace(0, 12, 2000)
s_list = np.linspace(-5, 5, 2000)
p_list = (erf(s_list) + 1.) / 2.
temp_list = np.linspace(0, 1, 2000)
ltab = np.zeros([len(x_list), len(p_list)])

for xi, x in enumerate(x_list):
    print(xi)
    u = np.concatenate([np.linspace(-10, 10, 20000),
                        np.tan((temp_list - 0.5) * np.pi) * a + x])
    u = np.sort(u)
    res = np.zeros(len(u))
    for i in range(len(u) - 1):
        res[i + 1] = \
            integrate.quad(f, a=u[i], b=u[i + 1], args=(a, x), limit=1000)[0]
    res = np.cumsum(res)
    res /= res[-1]
    ltab[xi, :] = np.interp(p_list, res, u)
    # plt.plot(u, res)
    # res = f(u, a, x)
    # res /= res.max()

np.savez('a_%0.10f.npz' % a, x_list=x_list, s_list=s_list, p_list=p_list, ltab=ltab)

# plt.plot(u, res)
#
# plt.plot(u, (erf((u-3.5)*4.)+1.)/2., 'k', lw=2)
# plt.show()

# plt.plot(x_list, ltab[:, 1])
# plt.plot(x_list, ltab[:, 4999])
# plt.plot(x_list, ltab[:, -2])
# plt.show()


# plt.contour(ltab)
# plt.show()
# # plt.plot(u, erf(u))
# # plt.show()
#
# plt.pcolor(ltab[500:750, 500:750])
# plt.show()
#
plt.plot(p_list, ltab[210, :], '.')
plt.show()
#
#
# ############################
#
#
# a = 0.0001
#
# x_list = np.linspace(0,6,200)
# s_list = np.linspace(-5,5,200)
# p_temp_list = (erf(s_list)+1.)/2.
# temp_list = np.linspace(0,1,200)
# ltab   = np.zeros([len(x_list), len(p_temp_list)+len(temp_list)])
#
#
# for xi, x in enumerate(x_list):
#     print(xi)
#     u = np.concatenate([np.linspace(-10, 10, 200),
#                         np.tan((temp_list - 0.5) * np.pi) * a + x])
#     u = np.sort(u)
#     p_list = np.sort(np.concatenate([np.tan((temp_list - 0.5) * np.pi) * a + x,
#                                      p_temp_list]))
#     res = np.zeros(len(u))
#     for i in range(len(u) - 1):
#         res[i + 1] = \
#             integrate.quad(f, a=u[i], b=u[i + 1], args=(a, x), limit=1000)[0]
#     res = np.cumsum(res)
#     res /= res[-1]
#     ltab[xi, :] = np.interp(p_list, res, u)
#
#
# plt.pcolor(ltab)
# plt.show()
# ###############
#
# plt.plot(u, np.arctan((u - x_list[100])/a)/np.pi +.5)
# plt.plot(np.tan((p_list-0.5)*np.pi)*a+x_list[100], p_list, '.')
# plt.xlim([-5,5])
plt.show()
