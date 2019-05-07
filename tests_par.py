from multiprocessing import Pool
from lyamc import *
import numpy as np
import time


# test_upar()
# test_redistribution()
# test_Neufeld()
# test_expansion()




#run
t = time.process_time()


def f(x):
    print(x)
    xs, Ns = runner(Nphotons=10)
    np.savetxt("xs_data_%04i.csv"%x, xs)
    np.savetxt("Ns_data_%04i.csv"%x, Ns)
    return

p = Pool(5)
temp_list = range(20)
print(p.map(f, temp_list))


elapsed_time = time.process_time() - t
# print("elapsed_time =", elapsed_time)
# np.savetxt("xs_data.csv", xs)
# np.savetxt("Ns_data.csv", Ns)
# print("Nscattering_avr =", np.average(Ns))
