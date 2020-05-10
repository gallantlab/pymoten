'''
'''
import numpy as np

import moten
from moten.utils import (iterator_func,
                         pointwise_square,
                         )

with_tqdm = False

##############################
# total motion energy
##############################
#
# Temporal PCs of squared pixel-wise frame differences
#

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
    >>> nimages, XTX = moten.extras.pixbypix_covariance_from_frames_generator(fdiffgen) # doctest: +SKIP
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


class StimulusTotalMotionEnergy(object):
    '''
    '''
    def __init__(self,
                 video_file,
                 size=None,
                 nimages=np.inf,
                 batch_size=100,
                 output_nonlinearity=pointwise_square,
                 dtype='float32'):
        '''
        '''
        self.size = size
        self.dtype = dtype
        self.nimages = nimages
        self.video_file = video_file
        self.batch_size = batch_size
        self.output_nonlinearity = output_nonlinearity

    def get_frame_difference_generator(self):
        '''
        '''
        import moten.io
        generator = moten.io.generate_frame_difference_from_greyvideo(
            self.video_file, size=self.size, nimages=self.nimages, dtype=self.dtype)

        return generator

    def compute_pixel_by_pixel_covariance(self,
                                          generator=None,
                                          ):
        '''
        '''
        import moten.utils

        if generator is None:
            # allow the user to provide their own frame difference generator
            generator = self.get_frame_difference_generator()

        nframes, xtx = pixbypix_covariance_from_frames_generator(
            generator, batch_size=self.batch_size,
            dtype=self.dtype, output_nonlinearity=self.output_nonlinearity)
        self.covariance_pixbypix = xtx
        self.covariance_nframes = nframes
        self.npixels = xtx.shape[0]

    def compute_spatial_pcs(self, npcs=None):
        '''
        '''
        from scipy import linalg

        if npcs is None:
            npcs = min(self.npixels, self.covariance_nframes) + 1

        # Recall the eigendecomposition
        # Q L QT = XTX
        # U,S,Vt = X
        # Q = V
        L, Q = linalg.eigh(self.covariance_pixbypix)

        # increasing order
        L = L[::-1]             # eigenvals (npcs)
        Q = Q[:, ::-1]          # eigenvecs (npixels, npcs)

        # store: (npixels, npcs)
        self.decomposition_spatial_pcs = np.asarray(Q[:, :npcs], dtype=self.dtype)
        self.decomposition_eigenvalues = np.asarray(L, dtype=self.dtype)

    def compute_temporal_pcs(self, generator=None, skip_first=False):
        '''
        '''
        import moten.utils

        if generator is None:
            # allow the user to provide their own frame difference generator
            generator = self.get_frame_difference_generator()

        if skip_first:
            # drop the first frame b/c the difference is with 0's
            # and so projection is with itself
            generator.__next__()

        self.decomposition_temporal_pcs = []
        ## TODO: batch for faster performance
        for frame_diff in iterator_func(generator,
                                             'projecting stimuli',
                                             unit='[frames]'):
            # flatten and square
            frame_diff = self.output_nonlinearity(frame_diff.reshape(1, -1))
            projection = frame_diff @ self.decomposition_spatial_pcs
            self.decomposition_temporal_pcs.append(projection.squeeze())

        if skip_first is False:
            # if not skipped, set the first frame to 0s
            # b/c its difference is not really defined
            self.decomposition_temporal_pcs[0][:] *= 0
