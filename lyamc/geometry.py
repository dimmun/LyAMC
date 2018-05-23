import numpy as np


class plane_gradient:
    def __init__(self, gradV=0, T=1e4, n=10.):
        self.gradV = gradV
        self.T = T
        self.n = n

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

    def __init__(self, nbar=1., T=2e4, R=10., A=0, V=0, DeltaV=0):
        self.nbar = nbar
        self.T = T
        self.R = R
        self.A = A
        self.V = V
        self.DeltaV = DeltaV

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
