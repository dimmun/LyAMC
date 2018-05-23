import argparse
import os
from multiprocessing import Pool

from lyamc.general import *

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('N', metavar='geom', type=int, nargs=1,
                    help='geometry name')
parser.add_argument('geometry', metavar='geom', type=str, nargs=1,
                    help='geometry name')
parser.add_argument('params', metavar='params', type=float, nargs='+',
                    help='geometry name')

args = parser.parse_args()

p = Pool(28)

geom = args.geometry[0]
params = args.params


def f(x):
    os.system('python runner.py ' + decodename(geom, params, sep=' '))
    return 0


print(p.map(f, np.arange(args.N[0])))
