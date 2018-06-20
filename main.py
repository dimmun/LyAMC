import matplotlib.pyplot as plt

from lyamc.analytical import *
from lyamc.general import *

# os.system('rm output/* -f')

# p = Pool(28)

geom = 'Zheng_sphere'
params = [10., 1e4, 0.33, 0.0, 0.0, 200.0]

# geom = 'Neufeld_test'
# params = [1e4, 10.]

N_per_node = 28
N_per_proc = 100
N_nodes = 50

s = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=trash/%%j.out
#SBATCH --error=trash/%%j.err
#SBATCH --ntasks-per-node=28
#SBATCH --time=1:00:00
#SBATCH --export=all

conda activate cfastpm

"""

for i in range(N_per_node):
    s = s + """python runner.py %i """ % N_per_proc + decodename(geom, params, sep=' ') + """ &
"""

s += """wait

"""

file = open("temp.sh", "w")
file.write(s)
file.close()

for i in range(N_nodes):
    os.system("sbatch temp.sh")



###

# Zheng Zheng

bins = np.linspace(-25, 25, 100)

geom = 'Zheng_sphere'
params = [10., 2e4, 0.324, 0.25, 0.0, 0.0]
x, k, direction = read_last(geom, params=params)
filt = np.abs(direction) > 0
t = plt.hist(direction[filt], 32, normed=True, histtype='step', label='200')

geom = 'Zheng_sphere'
params = [10., 2e4, 0.324, 0.5, 0.0, 0.0]
x, k, direction = read_last(geom, params=params)
filt = np.abs(direction) > 0
t = plt.hist(direction[filt], 32, normed=True, histtype='step', label='100')

geom = 'Zheng_sphere'
params = [10., 2e4, 0.324, 0.0, 0.0, 0.0]
x, k, direction = read_last(geom, params=params)
filt = np.abs(direction) > 0.
t = plt.hist(direction[filt], 32, normed=True, histtype='step', label='50')

plt.show()

###

# Zheng Zheng

dat = np.genfromtxt('R19_V200.dat', skip_header=2)

bins = np.linspace(-1, 1, 100)

# geom = 'Zheng_sphere'
# params = [10., 2e4, 0.324, 0.0, 0.0, 200.0]
# x, k, direction = read_last(geom, params=params)
# filt = np.abs(direction) > 0
# t = plt.hist(direction[filt], 64, normed=True, histtype='step', label='200')

geom = 'Zheng_sphere'
params = [10., 1e4, 0.33, 0.0, 0.0, 200.0]
x, k, direction = read_last(geom, params=params)
filt = np.abs(direction) > 0
t = plt.hist(direction[filt], bins, normed=True, histtype='step', label='100')

plt.hist(dat[:, 3], bins, histtype='step', normed=True);

# geom = 'Zheng_sphere'
# params =  [10., 2e4, 0.324, 0.0, 0.0, 0.0]
# x, k, direction = read_last(geom, params=params)
# filt = np.abs(direction) > 0.
# t = plt.hist(x[filt], 32, normed=True, histtype='step', label='50')

plt.show()

###

# Zheng Zheng

bins = np.linspace(-25, 25, 100)

geom = 'Zheng_sphere'
params = [10., 2e4, 0.324, 0.0, 0.0, 200.0]
x, k, direction = read_last(geom, params=params)
filt = np.abs(direction) < 0.1
t = plt.hist(x[filt], 32, normed=True, histtype='step', label='1')

geom = 'Zheng_sphere'
params = [10., 1e4, 0.324, 0.0, 0.0, 200.0]
x, k, direction = read_last(geom, params=params)
filt = np.abs(direction) < 0.1
t = plt.hist(x[filt], 32, normed=True, histtype='step', label='2')

# geom = 'Zheng_sphere'
# params =  [10., 2e4, 0.324, 0.0, 0.0, 0.0]
# x, k, direction = read_last(geom, params=params)
# filt = np.abs(direction) > 0.
# t = plt.hist(x[filt], 32, normed=True, histtype='step', label='50')

plt.legend()

plt.show()

####


bins = np.arange(-60, 60, .5)

geom = 'Neufeld_test'
params = [1e4, 10]
x, k, direction = read_last(geom, params=params)
t = plt.hist(x, bins=bins, normed=True, histtype='step', label='200')

geom = 'Neufeld_test'
params = [1e5, 10]
x, k, direction = read_last(geom, params=params)
t = plt.hist(x, bins=bins, normed=True, histtype='step', label='200')

geom = 'Neufeld_test2'
params = [1e4, 10]
x, k, direction = read_last(geom, params=params)
t = plt.hist(x, bins=bins, normed=True, histtype='step', label='200')
plt.show()

####
bins = np.linspace(-1, 1, 64)

geom = 'Zheng_sphere'
params = [2., 2e4, 3.24, 0.0, 0.0, 50.0]
x, k, direction = read_last(geom, params=params)
t = plt.hist(direction, bins=bins, normed=True, histtype='step', label='50')

geom = 'Zheng_sphere'
params = [2., 2e4, 3.24, 0.0, 0.0, 100.0]
x, k, direction = read_last(geom, params=params)
t = plt.hist(direction, bins=bins, normed=True, histtype='step', label='100')

geom = 'Zheng_sphere'
params = [1., 2e4, 3.24, 0.0, 0.0, 100.0]
x, k, direction = read_last(geom, params=params)
t = plt.hist(direction, bins=bins, normed=True, histtype='step', label='200')

geom = 'Zheng_sphere'
params = [2., 2e4, 3.24, 0.0, 0.0, 100.0]
x, k, direction = read_last(geom, params=params)
t = plt.hist(direction, bins=bins, normed=True, histtype='step', label='200')

plt.show()

####
bins = np.linspace(-20, 20, 2000)
dd = 10

geom = 'Zheng_sphere'
params = [dd, 1e4, .324, 0.0, 200.0, 0.0]
x, k, direction = read_last(geom, params=params)
filt = np.abs(direction) > -1
t = plt.hist(x[filt], bins=bins, normed=True, histtype='step', label='200')

# geom = 'Zheng_sphere'
# params = [1., 1e4, .324, 0.0, -200.0, 0.0]
# x, k, direction = read_last(geom, params=params)
# filt = np.abs(direction) > -1
# t = plt.hist(x[filt], bins=bins, normed=True, histtype='step', label='200')

geom = 'Zheng_sphere'
params = [dd, 1e4, .324, 0.0, 0.0, 0.0]
x, k, direction = read_last(geom, params=params)
filt = np.abs(direction) > -1
t = plt.hist(x[filt], bins=bins, normed=True, histtype='step', label='200')

plt.xticks(np.arange(-10, 10.1, 1))

plt.show()

####
bins = np.linspace(-100, 40, 200)
dd = 200

geom = 'Zheng_sphere'
params = [dd, 1e4, .324, 0.0, 20.0, 0.0]
x, k, direction = read_last(geom, params=params)
filt = np.abs(direction) > -1
t = plt.hist(x[filt], bins=bins, normed=True, histtype='step', label='200')


geom = 'Zheng_sphere'
params = [dd, 1e4, .324, 0.0, 200.0, 0.0]
x, k, direction = read_last(geom, params=params)
filt = np.abs(direction) > -1
t = plt.hist(x[filt], bins=bins, normed=True, histtype='step', label='200')

geom = 'Zheng_sphere'
params = [dd, 1e4, .324, 0.0, 2000.0, 0.0]
x, k, direction = read_last(geom, params=params)
filt = np.abs(direction) > -1
t = plt.hist(x[filt], bins=bins, normed=True, histtype='step', label='200')

# geom = 'Zheng_sphere'
# params = [1., 1e4, .324, 0.0, -200.0, 0.0]
# x, k, direction = read_last(geom, params=params)
# filt = np.abs(direction) > -1
# t = plt.hist(x[filt], bins=bins, normed=True, histtype='step', label='200')

geom = 'Zheng_sphere'
params = [dd, 1e4, .324, 0.0, 0.0, 0.0]
x, k, direction = read_last(geom, params=params)
filt = np.abs(direction) > -1
t = plt.hist(x[filt], bins=bins, normed=True, histtype='step', label='200')

q = Dijkstra_sphere_test(t[1], get_a(1e4), 1.2e7 / 2) * 2.3
plt.plot(t[1], q, 'k')

plt.xticks(np.arange(-100, 40.1, 10))

plt.show()

####

geom = 'Zheng_sphere'
params = [.1, 2e4, 3.24, 0.0, 0.0, 0.0]
x, k, direction = read_last(geom, params=params)
# t = plt.hist(direction, 64, normed=True, histtype='step', label='200')
t = plt.hist(x, bins=np.linspace(-20, 20, 100), normed=True, histtype='step', label='0')

params = [.1, 2e4, 3.24, 0.0, 200.0, 0.0]
x, k, direction = read_last(geom, params=params)
# t = plt.hist(direction, 64, normed=True, histtype='step', label='200')
t = plt.hist(x, bins=np.linspace(-20, 20, 100), normed=True, histtype='step', label='200')

# params = [.1, 2e4, 3.24, 0.0, -200.0, 0.0]
# x, k, direction = read_last(geom, params=params)
# # t = plt.hist(direction, 64, normed=True, histtype='step', label='200')
# t = plt.hist(x, bins=np.linspace(-20, 20, 100), normed=True, histtype='step', label='-200')

plt.xticks(np.arange(-20, 20.1, 10))

plt.legend()
plt.show()

###

geom = 'Zheng_sphere'
params = [1., 2e4, 3.24, 0.0, 0.0, 0.0]
x, k, direction = read_last(geom, params=params)
t = plt.hist(direction, 32, normed=True, histtype='step', label='0.00')

geom2 = 'Zheng_sphere'
params2 = [1., 2e4, 3.24, 0.25, 0.0, 0.0]
x, k, direction = read_last(geom2, params=params2)
t = plt.hist(direction, 32, normed=True, histtype='step', label='0.25')

geom2 = 'Zheng_sphere'
params2 = [1., 2e4, 3.24, 0., 0.0, 0.0]
x, k, direction = read_last(geom2, params=params2)
t = plt.hist(direction, 64, normed=True, histtype='step', label='0.50')
geom2 = 'Zheng_sphere'
params2 = [2., 2e4, 3.24, 0., 0.0, 0.0]
x, k, direction = read_last(geom2, params=params2)
t = plt.hist(direction, 64, normed=True, histtype='step', label='0.50')

plt.grid('on')
plt.legend(loc=3)
plt.show()

###

geom2 = 'Zheng_sphere'
params2 = [1., 2e4, 3.24, 0., 0.0, 0.0]
x, k, direction = read_last(geom2, params=params2)
t = plt.hist(x, 64, normed=True, histtype='step', label='0.50')
geom2 = 'Zheng_sphere'
params2 = [2., 2e4, 3.24, 0., 0.0, 0.0]
x, k, direction = read_last(geom2, params=params2)
t = plt.hist(x, 64, normed=True, histtype='step', label='0.50')

plt.grid('on')
plt.legend(loc=3)
plt.show()

###

geom = 'Zheng_sphere'
params = [1., 2e4, 3.3, 0.0, 0.0, 100.0]
x, k, direction = read_last(geom, params=params)
t = plt.hist(direction, 32, normed=True, histtype='step')

geom = 'Zheng_sphere'
params = [1., 2e4, 3.24, 0.0, 0.0, 200.0]
x, k, direction = read_last(geom, params=params)
# t = plt.hist(direction, 32, normed=True, histtype='step')

# plt.show()


filt = np.abs(direction) > 0.8
# filt = (direction) < -0.9

# filt = direction>-2
# plt.show()

# filt = (direction) > -2
plt.hist(x[filt], bins=np.linspace(-20, 20, 81), normed=True, histtype='step', cumulative=False)
# filt = (direction) < -0.8
# plt.hist(x[filt], bins=np.linspace(-20, 20, 81), normed=True, histtype='step', cumulative=False)
# filt = (direction) > 0.8
# plt.hist(x[filt], bins=np.linspace(-20, 20, 81), normed=True, histtype='step', cumulative=False)
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
