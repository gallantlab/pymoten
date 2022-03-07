'''Compute total motion energy from greyscale videos.
'''
import numpy as np

import moten
from moten.utils import (iterator_func,
                         pointwise_square,
                         )

# Make tqdm optional
with_tqdm = True
try:
    from tqdm import tqdm
except ImportError:
    with_tqdm = False

def process_motion_energy_from_files(filenames,
                                     size=None,
                                     nimages=np.inf,
                                     batch_size=1000,
                                     dtype='float32',
                                     mask=None,
                                     ):
    '''
    '''
    import moten.io
    if not isinstance(filenames, (list, tuple)):
        filenames = [filenames]

    XTX = 0
    NFRAMES = 0
    for fl in filenames:
        generator = moten.io.generate_frame_difference_from_greyvideo(
            fl, size=size, nimages=nimages, dtype=dtype)

        if mask is not None:
            generator = moten.io.apply_mask(mask, generator)

        nframes, xtx = pixbypix_covariance_from_frames_generator(generator,
                                                                 batch_size=batch_size,
                                                                 mask=mask)
        XTX += xtx
        NFRAMES = nframes
    return NFRAMES, XTX


def pixbypix_covariance_from_frames_generator(data_generator,
                                              batch_size=1000,
                                              output_nonlinearity=pointwise_square,
                                              dtype='float32'):
    '''Compute the pixel-by-pixel covariance from a video frame generator in batches.

    Parameters
    ----------
    data_generator : generator object
        Yields a video frame of shape (vdim, hdim)
    batch_size : optional, int
        Number of frames to process simultaneously while
    output_nonlinearity : optiona, function
        A pointwise function applied to the pixels of each frame.

    Examples
    --------
    >>> import moten
    >>> video_file = 'http://anwarnunez.github.io/downloads/avsnr150s24fps_tiny.mp4'
    >>> small_size = (36, 64)  # downsample to (vdim, hdim) 16:9 aspect ratio
    >>> fdiffgen = moten.io.generate_frame_difference_from_greyvideo(video_file, size=small_size, nimages=333)
    >>> nimages, XTX = moten.extras.pixbypix_covariance_from_frames_generator(fdiffgen) # doctest: +SKIP
    '''
    first_frame = next(data_generator)

    vdim, hdim = first_frame.shape
    npixels = vdim*hdim

    framediff_buffer = np.zeros((batch_size, npixels), dtype=dtype)
    XTX = np.zeros((npixels, npixels), dtype=dtype)
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
                frame_difference = next(data_generator).reshape(1, -1)

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
    '''Compute the principal components of the total motion energy.

    Total motion energy is defined as the squared difference
    between the previous and current frame. The pixel-by-pixel
    covariance of the total energy is computed frame-by-frame. Then,
    the spatial principal components are estimated. In a second pass,
    the temporal principal components are computed by projecting
    the total energy onto the spatial components.


    Parameters
    ----------
    video_file : str
        Full path to the video file. The video must be greyscale.
    size : optional, tuple (vdim, hdim)
        The desired output image size.
        If specified, the image is scaled or shrunk to this size.
        If not specified, the original size is kept.
    nimages : optional, int
        If specified, only `nimages` frames are loaded.
    batch_size : optional, int
        Number of frames to process simultaneously while
        computing the pixel covariances.
    output_nonlinearity : optional, function
        Defaults to point-wise square.


    Notes
    -----
    The time-by-pixel total motion energy matrix is defined as :math:`T`.
    Its singular value decomposition is :math:`U S V^{\intercal} = T`.
    The spatial components are :math:`V` and the temporal components are :math:`U`.

    As implemented in this class,

    * The spatial components computed are as above (:math:`V`).
    * The temporal compoonents are scaled by their singular values (:math:`US`).
    * The eigenvalues are the squared singular values (:math:`S^2`).


    Attributes
    ----------
    decomposition_spatial_pcs  : np.ndarray, (npixels, npcs)
    decomposition_temporal_pcs : list, (ntimepoints, npcs)
    decomposition_eigenvalues  : np.ndarray, (min(npixels, nframes),)

    Examples
    --------
    >>> import moten.extras
    >>> video_file = 'http://anwarnunez.github.io/downloads/avsnr150s24fps_tiny.mp4'
    >>> small_size = (36, 64) # (vdim, hdim)
    >>> totalmoten = moten.extras.StimulusTotalMotionEnergy(video_file, small_size, nimages=300)
    >>> totalmoten.compute_pixel_by_pixel_covariance()
    >>> totalmoten.compute_spatial_pcs(npcs=10)
    >>> totalmoten.compute_temporal_pcs()
    '''
    def __init__(self,
                 video_file,
                 size=None,
                 nimages=np.inf,
                 batch_size=1000,
                 output_nonlinearity=pointwise_square,
                 dtype='float32',
                 mask=None,
                 ):
        '''
        '''
        self.mask = mask
        self.size = size
        self.dtype = dtype
        self.nimages = nimages
        self.video_file = video_file
        self.batch_size = batch_size
        self.output_nonlinearity = output_nonlinearity

    def get_frame_difference_generator(self):
        '''Return a video buffer that generates the difference between
        the previous and the current frame.
        '''
        import moten.io
        generator = moten.io.generate_frame_difference_from_greyvideo(
            self.video_file, size=self.size, nimages=self.nimages, dtype=self.dtype)

        if self.mask is not None:
            generator = moten.io.apply_mask(self.mask, generator)

        return generator

    def compute_pixel_by_pixel_covariance(self,
                                          generator=None,
                                          ):
        '''Compute the pixel-by-pixel covariance of the total energy video.

        Notes
        -----
        Covariance is estimated in batches.

        Parameters
        ----------
        generator : optional
            The video frame difference generator. Defaults to the ``video_file``
            used to instantiate the class.

        Attributes
        ----------
        covariance_pixbypix : np.ndarray, (npixels, npixels)
            The full covarianc matrix
        covariance_nframes : int
            Number of frames used in estimating the covariance matrix.
            Defaults to the total number of frames in the video.
        npixels : int
            Total number of pixels in the video (after downsampling).
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
        '''Compute the principal components from the pixel-by-pixel total energy covariance matrix.

        Parameters
        ----------
        npcs : optional, int
            Number of principal components to keep

        Attributes
        ----------
        decomposition_spatial_pcs : np.ndarray, (npixels, npcs)
        decomposition_eigenvalues : np.ndarray
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
        '''Extract the temporal principal components of the total motion energy.

        Parameters
        ----------
        generator : optional
            The video frame difference generator. Defaults to the ``video_file``
            used to instantiate the class.
        skip_first : optional, bool
            By default, set the first timepoint of all the PCs is set to zeros because
            the first timepoint corresponds to the difference between the first frame and nothing.
            If `skip_first=True`, then the first frame is removed from the timecourse.

        Attributes
        ----------
        decomposition_temporal_pcs : list, (ntimepoints, npcs)
            The temporal compoonents are scaled by their singular values (:math:`US`).
        '''
        import moten.utils

        if generator is None:
            # allow the user to provide their own frame difference generator
            generator = self.get_frame_difference_generator()

        if skip_first:
            # drop the first frame b/c the difference is with 0's
            # and so projection is with itself
            next(generator)

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
