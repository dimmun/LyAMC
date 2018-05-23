import glob
import os
from multiprocessing import Pool

import matplotlib.pyplot as plt

from lyamc.general import *

os.system('rm output/* -f')

geom = 'Zheng_sphere'
params = [1., 2e4, 3.3, 0.5, 0.0, 0.0]

def f(x):
    os.system('python runner.py ' + decodename(geom, params, sep=' '))


p = Pool(28)
print(p.map(f, np.arange(320)))


s = glob.glob('output/last_*')

k = []
x = []
for si in s:
    temp = np.load(si)
    k.append(temp['k'])
    x.append(temp['x'])

x = np.array(x)
k = np.array(k)
direction = npsumdot(k, [0, 0, 1])
filt = np.abs(direction) < 0.1
# filt = (direction) < -0.9

# filt = direction>-2

plt.hist(direction, 32, normed=True, histtype='step')

plt.show()

filt = (direction) > -2
plt.hist(x[filt], bins=np.linspace(-20, 20, 801), normed=True, histtype='step', cumulative=True)
filt = (direction) < -0.5
plt.hist(x[filt], bins=np.linspace(-20, 20, 801), normed=True, histtype='step', cumulative=True)
filt = (direction) > 0.5
plt.hist(x[filt], bins=np.linspace(-20, 20, 801), normed=True, histtype='step', cumulative=True)
# plt.yscale('log')
plt.show()



# # Making plots
# plt.figure()
# plt.subplot(221)
# plt.plot(p_history[:, 0], p_history[:, 2], lw=0.5)
# plt.scatter(p_history[:, 0], p_history[:, 2], c=x_history, cmap='magma')
# plt.axis('equal')
# # plt.scatter(p_history[:,0], p_history[:,2], c=np.arange(len(p_history)), cmap='spectral')
# plt.colorbar(label='Dimensionless frequency x')
#
# plt.subplot(222)
# plt.plot(p_history[:, 0], p_history[:, 2], lw=0.5)
# # plt.scatter(p_history[:, 0], p_history[:, 2], c=x_history, cmap='spectral')
# plt.scatter(p_history[:, 0], p_history[:, 2], c=np.arange(len(p_history)), cmap='magma')
# plt.axis('equal')
# plt.colorbar(label='Number of scattering')
#
# plt.subplot(223)
# plt.plot(x_history[:i])
# # plt.scatter(p_history[:, 0], p_history[:, 2], c=x_history, cmap='spectral')
# plt.show()
