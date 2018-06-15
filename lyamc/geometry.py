import numpy as np

from lyamc.general import sigmaa0, cm_in_pc
from lyamc.redistribution import random_n


class Neufeld_test:
    def __init__(self, tau=1e4, T=10.):
        self.T = T
        self.tau = tau
        self.R = 1
        s = sigmaa0(T)
        self.N = self.tau / s
        self.n = self.N / (self.R * cm_in_pc)
        self.IC = 'center'

    def get_IC(self):
        p = [0, 0, 0]
        return p

    def velocity(self, x):
        return 0 * x

    def temperature(self, x):
        return self.T

    def density(self, x):
        d = x[:, 0] * 0. + self.n
        d[np.abs(x[:, 0]) > self.R] = 0
        return d

class plane_gradient:
    def __init__(self, gradV=0, T=1e4, n=10.):
        self.gradV = gradV
        self.T = T
        self.n = n
        s = sigmaa0(T)
        self.R = 1e6 / s / n / cm_in_pc

    def get_IC(self):
        p = [0, 0, 0]
        return p

    def velocity(self, x):
        '''Return velocity in km/s.
        Coordinates are in pc.
        Gradient is in km/s/pc'''
        v = x.copy()
        v *= 0
        if x.shape[0] == 3:
            v[0] = x[2] * self.gradV
        else:
            v[:, 0] = x[:, 2] * self.gradV
        return v

    def temperature(self, x):
        '''Temperature in K as a function of position.'''
        return self.T

    def density(self, x):
        '''Density in 1/cm^3 as a function of position.'''
        return self.n

    def stop_condition(self, x):
        return False


class Zheng_sphere:
    '''
    Examples from http://iopscience.iop.org/article/10.1088/0004-637X/794/2/116/pdf
    '''

    def __init__(self, nbar=1., T=2e4, R=10., A=0, V=0, DeltaV=0, IC='center'):
        self.nbar = nbar
        self.T = T
        self.R = R
        self.A = A
        self.V = V
        self.DeltaV = DeltaV
        self.IC = IC

    def get_IC(self):
        if self.IC == 'center':
            p = [0, 0, 0]
        elif self.IC == 'uniform':
            k, temp = random_n([], mode='uniform')
            p = np.random.rand() ** 0.3333 * k * self.R
        return p

    def temperature(self, x):
        return self.T

    def density(self, x):
        '''
        Equation 1
        :param x:
        :return:
        '''
        r = np.sqrt((x ** 2).sum(axis=-1))
        temp = self.nbar * (1.0 - 2.0 * self.A * x[:, 2] / self.R)
        temp[r > self.R] = 0.
        return temp

    def velocity(self, x):
        '''
        Equation 2
        :param x:
        :return:
        '''
        # r = np.sqrt((x**2).sum(axis=-1))
        # rhat = np.array(x) / r
        temp = x.copy()
        temp[:, 0] = 0
        temp[:, 1] = 0
        temp[:, 2] *= 1. / self.R * self.DeltaV
        return x / self.R * self.V + \
               temp

    def stop_condition(self, x):
        r = np.sqrt((x ** 2).sum(axis=-1))
        if r > self.R:
            return True
        else:
            return False
