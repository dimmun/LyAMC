from multiprocessing import Pool
from lyamc import *
import numpy as np



def f(i):
    print(i, Nphotons)
    xs, ks, rs, Ns = runner(Nphotons)
    return xs, ks, rs, Ns


Nphotons = 400
Npool = 25
Ntotal = Nphotons * Npool
print("Nphotons =", Nphotons, "Npool =", Npool, "Ntotal =", Ntotal)
p = Pool(Npool)
xs = []
kx, ky, kz = [], [], []
rx, ry, rz = [], [], []
Ns = []
for x, k, r, N in p.map(f, range(Npool, Nphotons)):
    xs.append(x)
    kx.append(k[:, 0])
    ky.append(k[:, 1])
    kz.append(k[:, 2])
    rx.append(r[:, 0])
    ry.append(r[:, 1])
    rz.append(r[:, 2])
    Ns.append(N)
xs = np.array(xs).ravel()
Ns = np.array(Ns).ravel()
kx = np.array(kx).ravel()
ky = np.array(ky).ravel()
kz = np.array(kz).ravel()
rx = np.array(rx).ravel()
ry = np.array(ry).ravel()
rz = np.array(rz).ravel()
p.close()
p.join()
np.savetxt('xs_data.csv', xs)
np.savetxt('ks_data.csv', np.c_[kx, ky, kz])
np.savetxt('rs_data.csv', np.c_[rx, ry, rz])
np.savetxt('Ns_data.csv', Ns)



