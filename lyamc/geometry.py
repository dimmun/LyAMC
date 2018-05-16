def velocity(x, grad_V=1):
    '''Return velocity in km/s.
    Coordinates are in pc.
    Gradient is in km/s/pc'''
    v = x.copy()
    v *= 0
    if x.shape[0] == 3:
        v[0] = x[2] * grad_V
    else:
        v[:, 0] = x[:, 2] * grad_V
    return v


def temperature(x):
    '''Temperature in K as a function of position.'''
    return 1e4


def density(x):
    '''Density in 1/cm^3 as a function of position.'''
    return 1e-4
