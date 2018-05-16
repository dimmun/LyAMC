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
p = [0, 0, 0]  # position in pc
x = 0.0  # dimensionless frequency
k = random_n([], mode='uniform')  # normal vector

local_temperature = temperature(p)
nu = get_nu(x=x, T=local_temperature)
# nu = 1.0

d = np.linspace(0, 1, 1000)

l, d = get_trajectory(p, k, d)
sf = get_survival_function(nu, l, d, k, grad_V=1e2)
plt.plot(d, sf)

for i in range(10):
    d_absorbed = random_d(d, sf)
    print(d_absorbed)
    plt.scatter(d_absorbed, 0)

plt.xlabel('Coordinate in [pc]')
plt.ylabel('Velocity in [km/s]')
plt.show()
