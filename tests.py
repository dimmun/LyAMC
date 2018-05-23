import matplotlib.pyplot as plt

from lyamc.redistribution import *
from lyamc.trajectory import *

print('Redistribution tests: ')

print('Rotate [0,0,1] by pi/2 in a random direction few times: ')
print(rotate_by_theta(np.array([0, 0, 1]), np.pi / 2.))
print(rotate_by_theta(np.array([0, 0, 1]), np.pi / 2.))
print(rotate_by_theta(np.array([0, 0, 1]), np.pi / 2.))

print('_________________________')

# print('Voight test:')
# alpha, gamma = 0.1, 0.1
# x = np.linspace(-0.8,0.8,1000)
# plt.plot(x, G(x, alpha), ls=':', c='k', label='Gaussian')
# plt.plot(x, L(x, gamma), ls='--', c='k', label='Lorentzian')
# plt.plot(x, V(x, alpha, gamma), c='k', label='Voigt')
# plt.xlim(-0.8,0.8)
# plt.legend()
# plt.show()


print('Cross section at line center, T=1e4K, no relative velocity')

print(sigma(nua, 1e4, np.array([0, 0, 0]), np.array([1, 0, 0])))

print('Cross section for dimensionless frequency x, T=1e4K, velocity along the direction of propagation 10 km/s')
# x = np.linspace(-10.,10.,1000)
# nu = get_nu(x, 1e4)
# # print(nu-nua)
# plt.plot(x, sigma(nu, 1e4, np.array([10, 0, 0]), np.array([1, 0, 0])))
# plt.yscale('log')
# plt.show()


print('________________________')
print('Single step test')

### Photon parameters:
# p = [0, 0, 0]  # position in pc
# x = 0.0  # dimensionless frequency
# k = random_n([], mode='uniform')  # normal vector
#
# local_temperature = temperature(p)
# nu = get_nu(x=x, T=local_temperature)
# # nu = 1.0
#
# d = np.linspace(0, 1, 1000)
#
# l, d = get_trajectory(p, k, d)
# sf = get_survival_function(nu, l, d, k, grad_V=1e2)
# plt.plot(d, sf)
#
# for i in range(10):
#     d_absorbed = random_d(d, sf)
#     print(d_absorbed)
#     plt.scatter(d_absorbed, 0)
#
# plt.xlabel('Coordinate in [pc]')
# plt.ylabel('Velocity in [km/s]')
# plt.show()


print('________________________')
print('Quads')

from lyamc.coordinates import *
import lyamc.cons as cons

m_hz = cons.MHK * cons.K2HZ
NU0 = cons.NULYA / (cons.MHK * cons.K2HZ)

N = 100000
nu = np.ones(N) * NU0 / m_hz
ns = np.zeros([N, 3])
ns[:, 0] = 1
vs = np.zeros([N, 3])
vs[:, 0] = 1e-4
res = scattering_lab_frame(nu, ns, vs)

plt.hist(res[0] * m_hz / NU0 - 1., bins=100)
plt.show()

####

N = 1000
T = 1e4
vth = get_vth(T)

nu = np.ones(N) * nua / m_hz
ns = np.zeros([N, 3])
ns[:, 0] = 1
# vs = np.zeros([N, 3])
vs = np.random.normal(loc=0., scale=np.sqrt(T / mp_over_2kB * 2), size=[N, 3]) / c
res = scattering_lab_frame(nu, ns, vs)

x = get_x(res[0] * m_hz, T)
np.median(x)

t1 = npsumdot(ns, vs) / vth * c
t2 = npsumdot(res[1], vs) / vth * c

delta = (res[0] - nu) / nu

print(np.mean(delta - np.sum(vs * (res[1] - ns), axis=-1)))

print(np.mean(x + t1 - t2), np.std(x + t1 - t2))

plt.hist(x + t1 - t2, bins=100)
plt.show()

####

nu = np.ones(1) * nua / m_hz

ns = np.zeros([1, 3])
ns[0] = 1
ns = ns.reshape(1, -1)

vs = np.zeros([1, 3])
vs[0] = 100 / c
vs = vs.reshape(1, -1)

res = scattering_lab_frame(nu, ns, vs)


# print('________________________')
# print('Picking an atom')
#
# N = 10000
# v = np.zeros([N, 3])
# v[:, 0] = np.linspace(-1000, 1000, N)
# u = np.zeros([N, 3])
# u[:, 0] = -200.
# n = np.zeros([N, 3])
# n[:, 0] = 1
# nu = 1. * nua
# T = 1e4
#
# u = np.array([-2000, 0, 0])
# n = np.array([1., 0., 0])
#
# vx = np.zeros([N, 3])
#
# for i in range(N):
#     vx[i, :] = u + get_par_velocity_of_atom(nu, T, u, n, N=100) + get_perp_velocity_of_atom(nu, T, u, n)
#
# print(vx.mean(0))
# print(vx.std(0))
