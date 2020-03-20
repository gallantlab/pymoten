'''
'''


class DotDict(dict):
    """A subclass of dictionary with dot syntax.

    Notes
    -----
    Copied from pykilosort (written by C. Rossant).
    """
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
