'''Motion-energy filters (after Nishimoto, 2011)

Adapted from MATLAB code written by S. Nishimoto.

Anwar O. Nunez-Elizalde (Jan, 2016)
'''
import itertools
from PIL import Image

import numpy as np

from moten.utils import (DotDict,
                         iterator_func,
                         log_compress,
                         sqrt_sum_squares,
                         )





##############################
# core functionality
##############################

def mk_3d_gabor(hvt,
                centerh=0.5,
                centerv=0.5,
                direction=45.0,
                spatial_freq=16.0,
                spatial_env=0.3,
                temporal_freq=2.0,
                temporal_env=0.3,
                phase_offset=0.0,
                aspect_ratio=1.0,
                ):
    '''Make a motion-energy filter.

    A motion-energy filter is a 3D gabor with
    two spatial and one temporal dimension.
    Each dimension is defined by two sine waves which
    differ in phase by 90 degrees. The sine waves are
    then multiplied by a gaussian.

    Parameters
    ----------
    hvt : array-like, (hdim, vdim, tdim)
        Defines the 3D field-of-view of the filter
        `hdim` : horizontal dimension size
        `vdim` : vertical dimension size
        `tdim` : temporal dimension size
    centerh, centerv : float
        Horizontal and vertical position in space, respectively.
        The image center is (0.5,0.5) for square aspect ratios
    direction : float, degrees
        Direction of spatial motion
    spatial_freq : float
        Spatial frequency
    spatial_env : float
        Spatial envelope (s.d. of the gaussian)
    temporal_freq : float
        Temporal frequency
    temporal_env : float
        Temporal envelope (s.d. of gaussian)
    phase_offset : float, degrees
        Phase offset for the spatial sinusoid
    aspect_ratio : float-like,
        Useful for preserving the spatial gabors circular even
        when images have non-square aspect ratios. For example,
        a 16:9 image would have `aspect_ratio`=16/9.

    Returns
    -------
    spatial_gabor_sin, spatial_gabor_cos : np.array, (vdim,hdim)
        Spatial gabor quadrature pair. `spatial_gabor_cos` has
        a 90 degree phase offset relative to `spatial_gabor_sin`

    temporal_gabor_sin, temporal_gabor_cos : np.array, (tdim)
        Temporal gabor quadrature pair. `temporal_gabor_cos` has
        a 90 degree phase offset relative to `temporal_gabor_sin`

    Notes
    -----
    Same method as Nishimoto, et al., 2011.
    '''
    szx, szy, szt = hvt

    dx = np.linspace(0,aspect_ratio,szx, endpoint=True)
    dy = np.linspace(0,1,szy, endpoint=True)
    dt = np.linspace(0,1,szt, endpoint=False)
    ixs, iys = np.meshgrid(dx,dy)

    fx = -spatial_freq*np.cos(direction/180.*np.pi)*2*np.pi
    fy = spatial_freq*np.sin(direction/180.*np.pi)*2*np.pi
    ft = np.real(temporal_freq)*2*np.pi

    # spatial filters
    spatial_gaussian = np.exp(-((ixs - centerh)**2 + (iys - centerv)**2)/(2*spatial_env**2))

    spatial_grating_sin = np.sin((ixs - centerh)*fx + (iys - centerv)*fy + phase_offset)
    spatial_grating_cos = np.cos((ixs - centerh)*fx + (iys - centerv)*fy + phase_offset)

    spatial_gabor_sin = spatial_gaussian * spatial_grating_sin
    spatial_gabor_cos = spatial_gaussian * spatial_grating_cos

    ##############################
    temporal_gaussian = np.exp(-(dt - 0.5)**2/(2*temporal_env**2))
    temporal_grating_sin = np.sin((dt - 0.5)*ft)
    temporal_grating_cos = np.cos((dt - 0.5)*ft)

    temporal_gabor_sin = temporal_gaussian*temporal_grating_sin
    temporal_gabor_cos = temporal_gaussian*temporal_grating_cos

    return spatial_gabor_sin, spatial_gabor_cos, temporal_gabor_sin, temporal_gabor_cos


def generate_3dgabor_array(gabor_hvt_size=(576, 1024, 24),
                           aspect_ratio='auto',
                           centerh=0.5,
                           centerv=0.5,
                           direction=45.0,
                           spatial_freq=16.0,
                           spatial_env=0.3,
                           temporal_freq=2.0,
                           temporal_env=0.3,
                           phase_offset=0.0):
    '''
    gabor_hvt_size : (vdim, hdim, tdim),
    '''
    if aspect_ratio == 'auto':
        aspect_ratio = hdim/float(vdim)

    gabor_components = mk_3d_gabor(gabor_hvt_size,
                                   aspect_ratio=aspect_ratio,
                                   centerh=centerh,
                                   centerv=centerv,
                                   direction=direction,
                                   spatial_freq=spatial_freq,
                                   spatial_env=spatial_env,
                                   temporal_freq=temporal_freq,
                                   temporal_env=temporal_env,
                                   phase_offset=phase_offset,
                                   )

    gabor_video = mk_spatiotemporal_gabor(*gabor_components)
    return gabor_video


def dotspatial_frames(spatial_gabor_sin, spatial_gabor_cos,
                      stimuli,
                      masklimit=0.001):
    '''Dot the spatial gabor filters filter with the stimuli

    Parameters
    ----------
    spatial_gabor_sin, spatial_gabor_cos : np.array, (vdim,hdim)
        Spatial gabor quadrature pair
    stimuli : 2D np.array (n, vdim*hdim)
        The movie frames with the spatial dimension collapsed.
    masklimit : float-like
        Threshold to find the non-zero filter region

    Returns
    -------
    channel_sin, channel_cos : np.ndarray, (n, )
        The filter response to each stimulus
        The quadrature pair can be combined: (x^2 + y^2)^0.5
    '''
    gabors = np.asarray([spatial_gabor_sin.ravel(),
                         spatial_gabor_cos.ravel()])

    # dot the gabors with the stimuli
    mask = np.abs(gabors).sum(0) > masklimit
    gabor_prod = np.dot(gabors[:,mask].squeeze(), stimuli.T[mask].squeeze()).T
    gabor_sin, gabor_cos = gabor_prod[:,0], gabor_prod[:,1]
    return gabor_sin, gabor_cos


def dotdelay_frames(spatial_gabor_sin, spatial_gabor_cos,
                    temporal_gabor_sin, temporal_gabor_cos,
                    stimulus,
                    masklimit=0.001):
    '''Convolve the motion-energy filter with a stimulus

    Parameters
    ----------
    spatial_gabor_sin, spatial_gabor_cos : np.array, (vdim,hdim)
        Spatial gabor quadrature pair

    temporal_gabor_sin, temporal_gabor_cos : np.array, (tdim)
        Temporal gabor quadrature pair

    stimulus : 2D np.array (n, vdim*hdim)
        The movie frames with the spatial dimension collapsed.

    Returns
    -------
    channel_sin, channel_cos : np.ndarray, (n, )
        The filter response to the stimulus at each time point
        The quadrature pair can be combined: (x^2 + y^2)^0.5
    '''

    gabor_sin, gabor_cos = dotspatial_frames(spatial_gabor_sin, spatial_gabor_cos,
                                             stimulus, masklimit=masklimit)
    gabor_prod = np.c_[gabor_sin, gabor_cos]


    temporal_gabors = np.asarray([temporal_gabor_sin,
                                  temporal_gabor_cos])

    # dot the product with the temporal gabors
    outs = np.dot(gabor_prod[:, [0]], temporal_gabors[[1]]) + np.dot(gabor_prod[:, [1]], temporal_gabors[[0]])
    outc = np.dot(-gabor_prod[:, [0]], temporal_gabors[[0]]) + np.dot(gabor_prod[:, [1]], temporal_gabors[[1]])

    # sum across delays
    nouts = np.zeros_like(outs)
    noutc = np.zeros_like(outc)
    tdxc = int(np.ceil(outs.shape[1]/2.0))
    delays = np.arange(outs.shape[1])-tdxc +1
    for ddx, num in enumerate(delays):
        if num == 0:
            nouts[:, ddx] = outs[:,ddx]
            noutc[:, ddx] = outc[:,ddx]
        elif num > 0:
            nouts[num:, ddx] = outs[:-num,ddx]
            noutc[num:, ddx] = outc[:-num,ddx]
        elif num < 0:
            nouts[:num, ddx] = outs[abs(num):,ddx]
            noutc[:num, ddx] = outc[abs(num):,ddx]

    channel_sin = nouts.sum(-1)
    channel_cos = noutc.sum(-1)
    return channel_sin, channel_cos


def mk_spatiotemporal_gabor(spatial_gabor_sin, spatial_gabor_cos,
                            temporal_gabor_sin, temporal_gabor_cos):
    '''Make 3D motion-energy filter defined by the spatial and temporal gabors.

    Takes the output of :func:`mk_3d_gabor` and constructs the 3D filter.
    This is useful for visualization.

    Parameters
    ----------
    spatial_gabor_sin, spatial_gabor_cos : np.array, (vdim,hdim)
        Spatial gabor quadrature pair
    temporal_gabor_sin, temporal_gabor_cos : np.array, (tdim)
        Temporal gabor quadrature pair

    Returns
    -------
    motion_energy_filter : np.array, (vdim, hdim, tdim)
        The 3D motion-energy filter

    '''
    a = np.dot(-spatial_gabor_sin.ravel()[...,None], temporal_gabor_sin[...,None].T)
    b = np.dot(spatial_gabor_cos.ravel()[...,None], temporal_gabor_cos[...,None].T)
    x,y = spatial_gabor_sin.shape
    t = temporal_gabor_sin.shape[0]
    return (a+b).reshape(x,y,t)



def compute_spatial_gabor_responses(stimulus,
                                    aspect_ratio='auto',
                                    spatial_frequencies=[0,2,4,8,16,32],
                                    quadrature_combination=sqrt_sum_squares,
                                    output_nonlinearity=log_compress,
                                    dtype=np.float64,
                                    dozscore=True):
    """Compute the spatial gabor filters' response to each stimulus.

    Parameters
    ----------
    stimulus : 3D np.array (n, vdim, hdim)
        The stimulus frames.
    spatial_frequencies : array-like
        The spatial frequencies to compute. The spatial envelope is determined by this.
    quadrature_combination : function, optional
        Specifies how to combine the channel reponses quadratures.
        The function must take the sin and cos as arguments in order.
        Defaults to: (sin^2 + cos^2)^1/2
    output_nonlinearity : function, optional
        Passes the channels (after `quadrature_combination`) through a
        non-linearity. The function input is the (`n`,`nfilters`) array.
        Defaults to: ln(x + 1e-05)
    dozscore : bool, optional
        Whether to z-score the channel responses in time

    dtype : np.dtype
        Defaults to np.float64

    Returns
    -------
    filter_responses : np.array, (n, nfilters)
    """
    nimages, vdim, hdim = stimulus.shape

    if aspect_ratio == 'auto':
        aspect_ratio = hdim/float(vdim)

    stimulus = stimulus.reshape(stimulus.shape[0], -1)
    parameter_names, gabor_parameters = mk_moten_pyramid_params(
        1.,
        1.,
        aspect_ratio=aspect_ratio,
        temporal_frequencies=[0.],
        spatial_directions=[0.],
        spatial_frequencies=spatial_frequencies,
        )


    ngabors = gabor_parameters.shape[0]
    filters = [{name : gabor_parameters[idx, pdx] for pdx, name \
                in enumerate(parameter_names)} \
               for idx in range(ngabors)]

    info = 'Computing responses for #%i filters across #%i images (aspect_ratio=%0.03f)'
    print(info%(len(gabor_parameters), nimages, aspect_ratio))

    channels = np.zeros((nimages, len(gabor_parameters)), dtype=dtype)
    for idx, gabor_param_dict in iterator_func(enumerate(filters),
                                          '%s.compute_spatial_gabor_responses'%__name__,
                                          total=len(gabor_parameters)):
        sgabor_sin, sgabor_cos, _, _ = mk_3d_gabor((hdim,vdim, 1),
                                                   aspect_ratio=aspect_ratio,
                                                   **gabor_param_dict)

        channel_sin, channel_cos = dotspatial_frames(sgabor_sin, sgabor_cos, stimulus)
        channel = quadrature_combination(channel_sin, channel_cos)
        channels[:, idx] = channel
    channels = output_nonlinearity(channels)
    if dozscore:
        from scipy.stats import zscore
        channels = zscore(channels)
    return channels


def compute_filter_responses(stimulus,
                             stimulus_fps,
                             gabor_nframes='auto',
                             quadrature_combination=sqrt_sum_squares,
                             output_nonlinearity=log_compress,
                             dozscore=True,
                             aspect_ratio='auto',
                             dtype=np.float64,
                             pyramid_parameters={}):
    """Compute the motion-energy filters' response to the stimuli.

    Parameters
    ----------
    stimulus : 3D np.array (n, vdim, hdim)
        The movie frames.
    stimulus_fps : scalar
        The temporal frequency of the stimulus
    aspect_ratio : bool, or scalar
        Defaults to hdim/vdim. Otherwise, pass as scalar

    gabor_nframes : scalar, None
        The number of frames in one filter.
        If None, it defaults to `floor(stimulus_fps*(2/3))`.
        Similar to Nishimoto, 2011.

    quadrature_combination : function, optional
        Specifies how to combine the channel reponses quadratures.
        The function must take the sin and cos as arguments in order.
        Defaults to: (sin^2 + cos^2)^1/2
    output_nonlinearity : function, optional
        Passes the channels (after `quadrature_combination`) through a
        non-linearity. The function input is the (`n`,`nfilters`) array.
        Defaults to: ln(x + 1e-05)
    dozscore : bool, optional
        Whether to z-score the channel responses in time

    dtype : np.dtype
        Defaults to np.float64

    pyramid_parameters: dict
        See :func:`mk_moten_pyramid_params` for details on parameters
        specifiying a motion-energy pyramid.

    Returns
    -------
    filter_responses : np.array, (n, nfilters)
    """
    nimages, vdim, hdim = stimulus.shape
    stimulus = stimulus.reshape(stimulus.shape[0], -1)

    if aspect_ratio == 'auto':
        aspect_ratio = hdim/float(vdim)

    if gabor_nframes == 'auto':
        gabor_nframes = int(stimulus_fps*(2./3.))

    parameter_names, gabor_parameters = mk_moten_pyramid_params(stimulus_fps,
                                                                gabor_nframes,
                                                                aspect_ratio=aspect_ratio,
                                                                **pyramid_parameters)
    ngabors = gabor_parameters.shape[0]
    filters = [{name : gabor_parameters[idx, pdx] for pdx, name \
                in enumerate(parameter_names)} \
               for idx in range(ngabors)]

    info = 'Computing responses for #%i filters across #%i images (aspect_ratio=%0.03f)'
    print(info%(len(gabor_parameters), nimages, aspect_ratio))

    channels = np.zeros((nimages, len(gabor_parameters)), dtype=dtype)
    for idx, gabor_param_dict in iterator_func(enumerate(filters),
                                          '%s.compute_filter_responses'%__name__,
                                          total=len(filters)):

        gabor = mk_3d_gabor((hdim,vdim,gabor_nframes),
                            aspect_ratio=aspect_ratio,
                            **gabor_param_dict)

        gabor0, gabor90, tgabor0, tgabor90 = gabor

        channel_sin, channel_cos = dotdelay_frames(gabor0, gabor90,
                                                   tgabor0, tgabor90,
                                                   stimulus,
                                                   )
        channel = quadrature_combination(channel_sin, channel_cos)
        channels[:,idx] = channel
    channels = output_nonlinearity(channels)
    if dozscore:
        from scipy.stats import zscore
        channels = zscore(channels)
    return channels


def mk_moten_pyramid_params(movie_fps,
                            gabor_nframes,
                            aspect_ratio=1.0,
                            temporal_frequencies=[0,2,4],
                            spatial_frequencies=[0,2,4,8,16,32],
                            spatial_directions=[0,45,90,135,180,225,270,315],
                            sf_gauss_ratio=0.6,
                            max_spatial_env=0.3,
                            gabor_spacing=3.5,
                            tf_gauss_ratio=10.,
                            max_temp_env=0.3,
                            include_edges=False,
                            ):
    """Parametrize a motion-energy pyramid that tiles the stimulus.

    Parameters
    ----------
    movie_fps : scalar, Hz
        Temporal resolution of the stimulus (e..g. 15)
    gabor_nframes : scalar, Hz
        Temporal window of the motion-energy filter (e.g. 10)
    temporal_frequencies : array-like, Hz
        Temporal frequencies of the filters for use on the stimulus
    spatial_frequencies : array-like, degrees
        Spatial frequencies for the filters
    spatial_directions : array-like, degrees
        Direction of filter motion. Degree position corresponds
        to standard unit-circle coordinates.


    sf_gauss_ratio : scalar
        The ratio of spatial frequency to gaussian s.d.
        This controls the number of cycles in a filter
    max_spatial_env : scalar
        Defines the maximum s.d. of the gaussian
    gabor_spacing : scalar
        Defines the spacing between spatial gabors
        (in s.d. units)
    tf_gauss_ratio : scalar
        The ratio of temporal frequency to gaussian s.d.
        This controls the number of temporal cycles
    max_temp_env : scalar
        Defines the maximum s.d. of the temporal gaussian
    aspect_ratio : scalar, horizontal/vertical
        The image aspect ratio. This ensures full image
        coverage for non-square images (e.g. 16:9)
    include_edges : bool
        Determines whether to include filters at the edge
        of the image which might be partially outside the
        stimulus field-of-view

    Returns
    -------
    gabor_parameters : np.array, (nfilters, 7)
        Parameters that defined the motion-energy filter
        Each of the `nfilters` has the following parameters:
            * centerh,centerv : x:horizontal and y:vertical position ('0,0' is top left)
            * direction       : direction of motion
            * spatial_freq    : spatial frequency
            * spatial_env     : spatial envelope (gaussian s.d.)
            * temporal_freq   : temporal frequency (scaled `tfq*(gabor_nframes/movie_fps)`)
            * temporal_env    : temporal envelope (gaussian s.d.)

    Notes
    -----
    Same method as Nishimoto, et al., 2011.
    """
    assert isinstance(aspect_ratio, (int, float, np.ndarray))

    def compute_envelope(freq, ratio):
        return np.inf if freq == 0 else (1.0/freq)*ratio

    spatial_frequencies = np.asarray(spatial_frequencies).astype(np.float)
    spatial_directions = np.asarray(spatial_directions).astype(np.float)
    temporal_frequencies = np.asarray(temporal_frequencies).astype(np.float)
    include_edges = int(include_edges)

    # normalize temporal frequency to wavelet size
    temporal_frequencies = temporal_frequencies*(gabor_nframes/float(movie_fps))

    # We have to deal with zero frequency spatial filters differently
    include_local_dc = True if 0 in spatial_frequencies else False
    spatial_frequencies = np.asarray([t for t in spatial_frequencies if t != 0])

    # add temporal envelope max
    params = list(itertools.product(spatial_frequencies, spatial_directions))

    gabor_parameters = []

    for spatial_freq, spatial_direction in params:
        spatial_env = min(compute_envelope(spatial_freq, sf_gauss_ratio), max_spatial_env)

        # compute the number of gaussians that will fit in the FOV
        vertical_space = np.floor(((1.0 - spatial_env*gabor_spacing)/(gabor_spacing*spatial_env))/2.0)
        horizontal_space = np.floor(((aspect_ratio - spatial_env*gabor_spacing)/(gabor_spacing*spatial_env))/2.0)

        # include the edges of screen?
        vertical_space = max(vertical_space, 0) + include_edges
        horizontal_space = max(horizontal_space, 0) + include_edges

        # get the spatial gabor locations
        ycenters = spatial_env*gabor_spacing*np.arange(-vertical_space, vertical_space+1) + 0.5
        xcenters = spatial_env*gabor_spacing*np.arange(-horizontal_space, horizontal_space+1) + aspect_ratio/2.

        for ii, (cx, cy) in enumerate(itertools.product(xcenters,ycenters)):
            for temp_freq in temporal_frequencies:
                temp_env = min(compute_envelope(temp_freq, tf_gauss_ratio), max_temp_env)

                if temp_freq == 0 and spatial_direction >= 180:
                    # 0Hz temporal filter doesn't have motion, so
                    # 0 and 180 degrees orientations are the same filters
                    continue

                gabor_parameters.append([cx,
                                         cy,
                                         spatial_direction,
                                         spatial_freq,
                                         spatial_env,
                                         temp_freq,
                                         temp_env,
                                         ])

                if spatial_direction == 0 and include_local_dc:
                    # add local 0 spatial frequency non-directional temporal filter
                    gabor_parameters.append([cx,
                                             cy,
                                             spatial_direction,
                                             0., # zero spatial freq
                                             spatial_env,
                                             temp_freq,
                                             temp_env,
                                             ])

    parameter_names = ('centerh',
                       'centerv',
                       'direction',
                       'spatial_freq',
                       'spatial_env',
                       'temporal_freq',
                       'temporal_env')
    gabor_parameters = np.asarray(gabor_parameters)
    return parameter_names, gabor_parameters
