'''Motion-energy filters (after Nishimoto, 2011)

Adapted from MATLAB code written by S. Nishimoto.

Anwar O. Nunez-Elizalde (Jan, 2016)
'''
import numpy as np

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


def get_default_pyramid(hvsize=(256, 144), fps=14):
    '''Construct a motion-energy pyramid

    Parameters
    ----------
    hvsize : tuple of ints
        Horizontal and vertical size of the stimulus in [pixels]
    fps : int
        Stimulus frame rate

    Return
    '''
    return pyramids.MotionEnergyPyramid(stimulus_hvsize=hvsize,
                                        stimulus_fps=fps)


__all__ = ['utils', 'core', 'viz', 'io']

if __name__ == '__main__':
    pass
