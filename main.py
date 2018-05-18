import matplotlib.pyplot as plt

from lyamc.redistribution import *
from lyamc.trajectory import *

### Photon parameters:

p = [0, 0, 0]  # position in pc
k = random_n([], mode='uniform')  # normal vector
x = 0.

N = 10000

p_history = np.zeros([N, 3]) * np.nan
p_history[0, :] = p

k_history = np.zeros([N, 3])
k_history[0, :] = k

x_history = np.zeros(N)
x_history[0] = x

d_absorbed = 0
d = np.linspace(0, 10, 1000)

local_temperature = temperature(p)

i = -1
while (d_absorbed < d.max()) & (i < N - 2):
    i += 1
    p = p_history[i, :].copy()
    k = k_history[i, :].copy()
    x = x_history[i].copy()
    nu = get_nu(x=x, T=local_temperature)
    # Find the position of new scattering
    l, d = get_trajectory(p, k, d)
    sf = get_survival_function(nu, l, d, k, grad_V=0.)
    d_absorbed = random_d(d, sf)
    p_new = get_shift(p, k, d_absorbed)
    # The enronment of new scattering
    local_velocity_new = velocity(p_new)
    local_temperature_new = temperature(p_new)
    k_new, mu = random_n(k)
    v_atom = local_velocity_new + \
             get_par_velocity_of_atom(nu, local_temperature_new, local_velocity_new, k, N=100) + \
             get_perp_velocity_of_atom(nu, local_temperature_new, local_velocity_new, k)
    nu = get_nu(x, local_temperature)
    x_new_in = get_x(nu, local_temperature_new)
    x_new = get_xout(xin=x_new_in,
                     v=local_velocity_new,
                     kin=k,
                     kout=k_new,
                     mu=mu,
                     T=local_temperature)

    p_history[i + 1, :] = p_new
    k_history[i + 1, :] = k_new
    x_history[i + 1] = x_new

plt.subplot(221)
plt.plot(p_history[:, 0], p_history[:, 2], lw=0.5)
plt.scatter(p_history[:, 0], p_history[:, 2], c=x_history, cmap='spectral')
# plt.scatter(p_history[:,0], p_history[:,2], c=np.arange(len(p_history)), cmap='spectral')
plt.colorbar(label='Dimensionless frequency x')

plt.subplot(222)
plt.plot(p_history[:, 0], p_history[:, 2], lw=0.5)
# plt.scatter(p_history[:, 0], p_history[:, 2], c=x_history, cmap='spectral')
plt.scatter(p_history[:, 0], p_history[:, 2], c=np.arange(len(p_history)), cmap='spectral')
plt.colorbar(label='Number of scattering')

plt.subplot(223)
plt.plot(x_history)
# plt.scatter(p_history[:, 0], p_history[:, 2], c=x_history, cmap='spectral')
plt.show()



# plt.scatter(x_history[1:], np.diff(x_history))
# plt.show()
