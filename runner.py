import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('nsim', metavar='nsim', type=int, nargs=1,
                    help='geometry name')
parser.add_argument('geometry', metavar='geom', type=str, nargs=1,
                    help='geometry name')
parser.add_argument('params', metavar='params', type=float, nargs='+',
                    help='geometry name')

args = parser.parse_args()

nsim = args.nsim[0]
print(args.geometry)

from lyamc.redistribution import *
from lyamc.trajectory import *
from lyamc.coordinates import *

m_hz = 2.2687318181383202e+23

if args.geometry[0] == 'Neufeld_test':
    geom = Neufeld_test(tau=args.params[0],
                        T=args.params[1])
elif args.geometry[0] == 'plane_gradient':
    geom = plane_gradient(n=args.params[0],
                          T=args.params[1],
                          gradV=args.params[2])
elif args.geometry[0] == 'Zheng_sphere':
    geom = Zheng_sphere(nbar=args.params[0],
                        T=args.params[1],
                        R=args.params[2],
                        A=args.params[3],
                        V=args.params[4],
                        DeltaV=args.params[5],
                        IC='uniform')
else:
    print('define a proper geometry')
### Photon parameters:

p_last = []
k_last = []
x_last = []

for iii in range(nsim):
    # p = [0, 0, 0]  # position in pc
    p = geom.get_IC()

    local_temperature = geom.temperature(p)

    k, temp = random_n([], mode='uniform')  # normal vector

    x = np.random.normal(0, 1)  # * get_vth(local_temperature) / c

    N = 1000

    p_history = np.zeros([N, 3]) * np.nan
    p_history[0, :] = p

    k_history = np.zeros([N, 3])
    k_history[0, :] = k

    x_history = np.zeros(N)
    x_history[0] = x

    d_absorbed = 0
    d = np.linspace(0, 10, 100000)

    proper_redistribution = False  # True

    i = -1

    while (d_absorbed < d.max()) & (i < N - 2):
        print(i, x, p)
        i += 1
        # define initial parameters
        p = p_history[i, :].copy()  # position
        k = k_history[i, :].copy()  # direction
        x = x_history[i].copy()  # dimensionless frequency
        nu = get_nu(x=x, T=local_temperature)  # frequency
        # Find the position of new scattering
        l, d = get_trajectory(p, k, d)  # searching for a trajectory
        sf = get_survival_function(nu, l, d, k, geom)  # getting surfvival function
        q = np.random.rand()
        d_absorbed = interp_d(d, sf, q)  # randomly selecting absorption point
        while (d_absorbed == d.max()) & (d.max() < 1e3):
            # print('Expanding!')
            d *= 2
            l, d = get_trajectory(p, k, d)  # searching for a trajectory
            sf = get_survival_function(nu, l, d, k, geom)  # getting surfvival function
            d_absorbed = interp_d(d, sf, q)
        while (d_absorbed < d[10]):
            # print('Refining!')
            d /= 2
            l, d = get_trajectory(p, k, d)  # searching for a trajectory
            sf = get_survival_function(nu, l, d, k, geom)  # getting surfvival function
            d_absorbed = interp_d(d, sf, q)
        if d_absorbed < 1e3:
            p_new = get_shift(p, k, d_absorbed)  # extracting new position
            # The environment of new scattering
            local_velocity_new = geom.velocity(p_new.reshape(1, -1))  # new local velocity
            local_temperature_new = geom.temperature(p_new.reshape(1, -1))  # new local temperature
            # selecting a random atom
            v_atom = local_velocity_new + \
                     get_par_velocity_of_atom(nu, local_temperature_new, local_velocity_new, k, N=1000) + \
                     get_perp_velocity_of_atom(nu, local_temperature_new, local_velocity_new, k)
            # generating new direction and new frequency
            if proper_redistribution:
                nu_i = np.array([nu / m_hz])
                ns = k.reshape(1, -1)
                vs = v_atom.reshape(1, -1) / c
                res = scattering_lab_frame(nu_i, ns, vs)
                x_new = get_x(res[0] * m_hz, local_temperature_new)
                k_new = res[1]
                vth = get_vth(local_temperature_new)
                # print(x_new - x - np.sum(vs * (res[1] - ns), axis=-1)/vth*c)
            else:
                k_new, mu = random_n(k)  # , mode='uniform')  # new direction
                # print(k, k_new)
                x_new_in = get_x(nu, local_temperature_new)
                x_new = get_xout(xin=x_new_in,
                                 v=v_atom,
                                 kin=k,
                                 kout=k_new,
                                 mu=mu,
                                 T=local_temperature)
            # recording data into arrays
            p_history[i + 1, :] = p_new
            k_history[i + 1, :] = k_new
            x_history[i + 1] = x_new
        else:
            i = i - 1

    print(i)
    filename = str(np.random.rand())[2:]
    # np.savez('output/' + decodename(args.geometry[0], args.params) + '_%s.npz' % filename, p=p_history[:i + 2],
    #          k=k_history[:i + 2], x=x_history[:i + 2])
    p_last.append(p_history[i + 1, :])
    k_last.append(k_history[i + 1, :])
    x_last.append(x_history[i + 1])

np.savez('output/' + decodename(args.geometry[0], args.params) + '_%s_last.npz' % filename,
         p=p_last,
         k=k_last,
         x=x_last)
