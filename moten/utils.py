'''Help functions.
'''
# Anwar O. Nunez-Elizalde (Jan, 2016)
#
# Updates:
#  Anwar O. Nunez-Elizalde (Apr, 2020)
#
#
import numpy as np

with_tqdm = True
try:
    from tqdm import tqdm
except ImportError:
    with_tqdm = False
    pass


##############################
# helper functions
##############################

def log_compress(x, offset=1e-05):
    '''Computes :math:`log(x + \epsilon)`
    '''
    return np.log(x + offset)


def sqrt_sum_squares(x,y):
    '''Computes :math:`(x^2 + y^2)^{1/2}`
    '''
    return np.sqrt(x**2 + y**2)

def pointwise_square(data):
    '''Computes :math:`x^2`
    '''
    return data**2


class DotDict(dict):
    '''A dictionary with dot syntax.
    '''
    # Copied from pykilosort (written by C. Rossant).
    def __init__(self, *args, **kwargs):
        super(type(self), self).__init__(*args, **kwargs)
        self.__dict__ = self

    def copy(self):
        """
        """
        return DotDict(super(type(self), self).copy())


def iterator_func(*args, **kwargs):
    '''If available, show iteration progress with `tqdm`.
    '''
    try:
        from tqdm import tqdm
        return tqdm(*args, **kwargs)
    except ImportError:
        return args[0]
    raise ValueError('Unknown')
