'''
'''
#
# Adapted from MATLAB code written by S. Nishimoto (see Nishimoto, et al., 2011).
# Anwar O. Nunez-Elizalde (Jan, 2016)
#
# Updates:
#  Anwar O. Nunez-Elizalde (Apr, 2020)


##############################
# internal imports
##############################
from moten import (pyramids,
                   utils,
                   core,
                   viz,
                   io,
                   )


# some default pyramids
default_pyramids = pyramids.DefaultPyramids()


def get_default_pyramid(vhsize=(144, 256), fps=24, **kwargs):
    '''Construct a motion energy pyramid

    A motion energy pyramid consists of a set of
    spatio-temporal Gabor filters that tile the screen.
    motion energy features are extracted by convolving the
    spatio-temporal Gabor filters with the stimulus movie.

    Parameters
    ----------
    vhsize : tuple of ints
        Horizontal and vertical size of the stimulus in [pixels]
    fps : int
        Stimulus frame rate

    Returns
    -------
    pyramid : :class:`moten.pyramids.MotionEnergyPyramid`

    Examples
    --------
    >>> import moten
    >>> video_file = 'http://anwarnunez.github.io/downloads/avsnr150s24fps_tiny.mp4'
    >>> luminance_images = moten.io.video2luminance(video_file, nimages=100)
    >>> nimages, vdim, hdim = luminance_images.shape
    >>> pyramid = moten.get_default_pyramid(vhsize=(vdim, hdim), fps=24)
    >>> moten_features = pyramid.project_stimulus(luminance_images)
    '''
    return pyramids.MotionEnergyPyramid(stimulus_vhsize=vhsize,
                                        stimulus_fps=fps,
                                        **kwargs)


__all__ = ['utils', 'core', 'viz', 'io']

__version__ = '0.0.4'

if __name__ == '__main__':
    pass
