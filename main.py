import glob
import os
from multiprocessing import Pool

import matplotlib.pyplot as plt

from lyamc.general import *

# os.system('rm output/* -f')

p = Pool(28)

geom = 'Zheng_sphere'
params = [1., 2e4, 3.3, 0.0, 0.0, 100.0]
N_per_node = 28 * 100
N_nodes = 10

s = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=trash/%%j.out
#SBATCH --error=trash/%%j.err
#SBATCH --ntasks-per-node=28
#SBATCH --time=1:00:00
#SBATCH --export=all

conda activate cfastpm

python node_runner.py %i """ % N_per_node

file = open("temp.sh", "w")
file.write(s + decodename(geom, params, sep=' '))
file.close()

for i in range(N_nodes):
    os.system("sbatch temp.sh")


def read_last(geom, params):
    s = glob.glob('output/' + decodename(geom, params, sep='_') + '*last*')
    k = []
    x = []
    for si in s:
        temp = np.load(si)
        k.append(temp['k'])
        x.append(temp['x'])
    x = np.array(x)
    k = np.array(k)
    direction = npsumdot(k, [0, 0, 1])
    print(len(s))
    return x, k, direction


###

geom = 'Zheng_sphere'
params = [1., 2e4, 3.3, 0.25, 0.0, 0.0]
x, k, direction = read_last(geom, params=params)
t = plt.hist(direction, 16, normed=True, histtype='step')

geom2 = 'Zheng_sphere'
params2 = [1., 2e4, 3.3, 0.5, 0.0, 0.0]
x, k, direction = read_last(geom2, params=params2)
t = plt.hist(direction, 16, normed=True, histtype='step')

plt.show()

###

geom = 'Zheng_sphere'
params = [1., 2e4, 3.3, 0.0, 0.0, 100.0]
x, k, direction = read_last(geom, params=params)
t = plt.hist(direction, 16, normed=True, histtype='step')

plt.show()


filt = np.abs(direction) < 0.1
# filt = (direction) < -0.9

# filt = direction>-2
plt.show()

# filt = (direction) > -2
# plt.hist(x[filt], bins=np.linspace(-20, 20, 81), normed=True, histtype='step', cumulative=False)
filt = (direction) < -0.8
plt.hist(x[filt], bins=np.linspace(-20, 20, 81), normed=True, histtype='step', cumulative=False)
filt = (direction) > 0.8
plt.hist(x[filt], bins=np.linspace(-20, 20, 81), normed=True, histtype='step', cumulative=False)
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
