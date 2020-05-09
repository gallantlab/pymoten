'''
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
    return np.log(x + offset)


def sqrt_sum_squares(x,y):
    return np.sqrt(x**2 + y**2)

def pointwise_square(data):
    return data**2


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


def pixbypix_covariance_from_frames_generator(data_generator,
                                              batch_size=1000,
                                              output_nonlinearity=pointwise_square,
                                              dtype='float32'):
    '''
    Parameters
    ----------
    data_generator : generator object
        Yields an frame of shape (vdim, hdim)

    Examples
    --------
    >>> import moten
    >>> video_file = 'http://anwarnunez.github.io/downloads/avsnr150s24fps_tiny.mp4'
    >>> small_size = (36, 64)  # downsample to (vdim, hdim) 16:9 aspect ratio
    >>> fdiffgen = moten.io.generate_frame_difference_from_greyvideo(video_file, size=small_size, nimages=333)
    >>>
    >>> nimages, XTX = moten.utils.pixbypix_covariance_from_frames_generator(fdiffgen) # doctest: +SKIP
    '''
    first_frame = data_generator.__next__()
    vdim, hdim = first_frame.shape
    npixels = vdim*hdim

    framediff_buffer = np.zeros((batch_size, npixels), dtype=dtype)
    XTX = np.zeros((npixels, npixels), dtype=np.float64)
    nframes = 0

    if with_tqdm:
        pbar = tqdm(desc='Computing pixel-by-pixel covariance (p=%i)'%npixels,
                    total=0,
                    unit='[frames]')
    else:
        print('Computing pixel-by-pixel covariance (p=%i)...'%npixels, end='')

    RUN = True
    while RUN:
        framediff_buffer *= 0.0             # clear buffer
        try:
            for batch_frame_idx in range(batch_size):
                frame_difference = data_generator.__next__().reshape(1, -1)
                framediff_buffer[batch_frame_idx] = output_nonlinearity(frame_difference)
        except StopIteration:
            RUN = False
        finally:
            if with_tqdm:
                pbar.update(batch_frame_idx + 1)
            else:
                print('.', end='')

            nframes += batch_frame_idx + 1
            XTX += framediff_buffer.T @ framediff_buffer

    if with_tqdm:
        pbar.close()
    else:
        print('Finished #%i frames'%nframes)

    return nframes, XTX
