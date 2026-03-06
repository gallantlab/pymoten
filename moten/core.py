'''
'''
#
# Adapted from MATLAB code written by S. Nishimoto (see Nishimoto, et al., 2011).
# Anwar O. Nunez-Elizalde (Jan, 2016)
#
# Updates:
#  Anwar O. Nunez-Elizalde (Apr, 2020)
#
import itertools
import math
from PIL import Image

import numpy as np

from moten.backend import get_backend
from moten.utils import (DotDict,
                         iterator_func,
                         log_compress,
                         sqrt_sum_squares,
                         pointwise_square,
                         )


##############################
#
##############################

def raw_project_stimulus(stimulus,
                         filters,
                         vhsize=(),
                         dtype='float32'):
    '''Obtain responses to the stimuli from all filter quadrature-pairs.

    Parameters
    ----------
    stimulus : np.ndarray, (nimages, vdim, hdim) or (nimages, npixels)
        The movie frames.
        If `stimulus` is two-dimensional with shape (nimages, npixels), then
        `vhsize=(vdim,hdim)` is required and `npixels == vdim*hdim`.

    Returns
    -------
    output_sin : np.ndarray, (nimages, nfilters)
    output_cos : np.ndarray, (nimages, nfilters)
    '''
    # parameters
    if stimulus.ndim == 3:
        nimages, vdim, hdim = stimulus.shape
        stimulus = stimulus.reshape(stimulus.shape[0], -1)
        vhsize = (vdim, hdim)

    backend = get_backend()

    # checks for 2D stimuli
    assert stimulus.ndim == 2                             # (nimages, pixels)
    assert isinstance(vhsize, tuple) and len(vhsize) == 2 # (hdim, vdim)
    assert vhsize[0] * vhsize[1] == stimulus.shape[1]    # hdim*vdim == pixels

    # Compute responses
    nfilters = len(filters)
    nimages = stimulus.shape[0]
    sin_responses = backend.zeros((nimages, nfilters), dtype=dtype)
    cos_responses = backend.zeros((nimages, nfilters), dtype=dtype)

    for gaborid, gabor_parameters in iterator_func(enumerate(filters),
                                                   'project_stimulus',
                                                   total=len(filters)):

        sgabor0, sgabor90, tgabor0, tgabor90 = mk_3d_gabor(vhsize, **gabor_parameters)

        channel_sin, channel_cos = dotdelay_frames(sgabor0, sgabor90,
                                                   tgabor0, tgabor90,
                                                   stimulus)

        sin_responses[:, gaborid] = channel_sin
        cos_responses[:, gaborid] = channel_cos

    return sin_responses, cos_responses


def project_stimulus(stimulus,
                     filters,
                     quadrature_combination=sqrt_sum_squares,
                     output_nonlinearity=log_compress,
                     vhsize=(),
                     dtype='float32'):
    '''Compute the motion energy filter responses to the stimuli.

    Parameters
    ----------
    stimulus : np.ndarray, (nimages, vdim, hdim) or (nimages, npixels)
        The movie frames.
        If `stimulus` is two-dimensional with shape (nimages, npixels), then
        `vhsize=(vdim,hdim)` is required and `npixels == vdim*hdim`.

    Returns
    -------
    filter_responses : np.ndarray, (nimages, nfilters)
    '''
    # parameters
    if stimulus.ndim == 3:
        nimages, vdim, hdim = stimulus.shape
        stimulus = stimulus.reshape(stimulus.shape[0], -1)
        vhsize = (vdim, hdim)

    backend = get_backend()

    # checks for 2D stimuli
    assert stimulus.ndim == 2                             # (nimages, pixels)
    assert isinstance(vhsize, tuple) and len(vhsize) == 2 # (hdim, vdim)
    assert vhsize[0] * vhsize[1] == stimulus.shape[1]    # hdim*vdim == pixels

    # Compute responses
    nfilters = len(filters)
    nimages = stimulus.shape[0]
    filter_responses = backend.zeros((nimages, nfilters), dtype=dtype)
    for gaborid, gabor_parameters in iterator_func(enumerate(filters),
                                                   'project_stimulus',
                                                   total=len(filters)):

        sgabor0, sgabor90, tgabor0, tgabor90 = mk_3d_gabor(vhsize, **gabor_parameters)

        channel_sin, channel_cos = dotdelay_frames(sgabor0, sgabor90,
                                                   tgabor0, tgabor90,
                                                   stimulus)

        channel_response = quadrature_combination(channel_sin, channel_cos)
        channel_response = output_nonlinearity(channel_response)
        filter_responses[:, gaborid] = channel_response
    return filter_responses


##############################
# core functionality
##############################

def mk_3d_gabor(vhsize,
                stimulus_fps,
                aspect_ratio='auto',
                filter_temporal_width='auto',
                centerh=0.5,
                centerv=0.5,
                direction=45.0,
                spatial_freq=16.0,
                spatial_env=0.3,
                temporal_freq=2.0,
                temporal_env=0.3,
                spatial_phase_offset=0.0,
                ):
    '''Make a motion energy filter.

    A motion energy filter is a 3D gabor with
    two spatial and one temporal dimension.
    Each dimension is defined by two sine waves which
    differ in phase by 90 degrees. The sine waves are
    then multiplied by a gaussian.

    Parameters
    ----------
    vhsize : tuple of ints,  (vdim, hdim)
        Size of the stimulus in pixels (vdim, hdim)
        `vdim` : vertical dimension
        `hdim` : horizontal dimension
    stimulus_fps : scalar, [Hz]
        Stimulus playback speed in frames per second.
    centerv : scalar
        Vertical filter position from top of frame (min=0, max=1.0).
    centerh : scalar
        Horizontal filter position from left of frame (min=0, max=aspect_ratio).
    direction : scalar, [degrees]
        Direction of filter motion. Degree position corresponds
        to standard unit-circle coordinates (i.e. 0=right, 180=left).
    spatial_freq : float, [cycles-per-image]
        Spatial frequency of the filter.
    temporal_freq : float, [Hz]
        Temporal frequency of the filter
    filter_temporal_width : int
        Temporal window of the motion energy filter (e.g. 10).
        Defaults to approximately 0.666[secs] (`floor(stimulus_fps*(2/3))`).
    aspect_ratio : optional, 'auto' or float-like,
        Defaults to stimulus aspect ratio: hdim/vdim
        Useful for preserving the spatial gabors circular even
        when images have non-square aspect ratios. For example,
        a 16:9 image would have `aspect_ratio`=16/9.

    spatial_env : float
        Spatial envelope (s.d. of the gaussian)
    temporal_env : float
        Temporal envelope (s.d. of gaussian)
    spatial_phase_offset : float, [degrees
        Phase offset for the spatial sinusoid

    Returns
    -------
    spatial_gabor_sin : 2D np.ndarray, (vdim, hdim)
    spatial_gabor_cos : 2D np.ndarray, (vdim, hdim)
        Spatial gabor quadrature pair. ``spatial_gabor_cos`` has
        a 90 degree phase offset relative to ``spatial_gabor_sin``

    temporal_gabor_sin : 1D np.ndarray, (`filter_temporal_width`,)
    temporal_gabor_cos : 1D np.ndarray, (`filter_temporal_width`,)
        Temporal gabor quadrature pair. ``temporal_gabor_cos`` has
        a 90 degree phase offset relative to ``temporal_gabor_sin``

    Notes
    -----
    Same method as Nishimoto, et al., 2011.
    '''
    backend = get_backend()

    vdim, hdim = vhsize
    if aspect_ratio == 'auto':
        aspect_ratio = hdim/float(vdim)

    if filter_temporal_width == 'auto':
        filter_temporal_width = int(stimulus_fps*(2/3.))

    # cast filter width to integer frames
    assert math.isclose(filter_temporal_width, int(filter_temporal_width))
    filter_temporal_width = int(filter_temporal_width)

    dh = backend.linspace(0, aspect_ratio, hdim, endpoint=True)
    dv = backend.linspace(0, 1, vdim, endpoint=True)
    dt = backend.linspace(0, 1, filter_temporal_width, endpoint=False)
    # AN: Actually, `dt` should include endpoint.
    # Currently, the center of the filter width is +(1./fps)/2.
    # However, this would break backwards compatibility.
    # TODO: Allow for `dt_endpoint` as an argument
    # and set default to False.

    ihs, ivs = backend.meshgrid(dh, dv)

    fh = -spatial_freq*backend.cos(backend.asarray(direction/180.*backend.pi))*2*backend.pi
    fv = spatial_freq*backend.sin(backend.asarray(direction/180.*backend.pi))*2*backend.pi
    # normalize temporal frequency to wavelet size
    ft = backend.real(backend.asarray(temporal_freq*(filter_temporal_width/float(stimulus_fps))))*2*backend.pi

    # spatial filters
    spatial_gaussian = backend.exp(-((ihs - centerh)**2 + (ivs - centerv)**2)/(2*spatial_env**2))

    spatial_grating_sin = backend.sin((ihs - centerh)*fh + (ivs - centerv)*fv + spatial_phase_offset)
    spatial_grating_cos = backend.cos((ihs - centerh)*fh + (ivs - centerv)*fv + spatial_phase_offset)

    spatial_gabor_sin = spatial_gaussian * spatial_grating_sin
    spatial_gabor_cos = spatial_gaussian * spatial_grating_cos

    ##############################
    temporal_gaussian = backend.exp(-(dt - 0.5)**2/(2*temporal_env**2))
    temporal_grating_sin = backend.sin((dt - 0.5)*ft)
    temporal_grating_cos = backend.cos((dt - 0.5)*ft)

    temporal_gabor_sin = temporal_gaussian*temporal_grating_sin
    temporal_gabor_cos = temporal_gaussian*temporal_grating_cos

    return spatial_gabor_sin, spatial_gabor_cos, temporal_gabor_sin, temporal_gabor_cos


def generate_3dgabor_array(vhsize=(576,1024),
                           stimulus_fps=24,
                           aspect_ratio='auto',
                           filter_temporal_width='auto',
                           centerh=0.5,
                           centerv=0.5,
                           direction=45.0,
                           spatial_freq=16.0,
                           spatial_env=0.3,
                           temporal_freq=2.0,
                           temporal_env=0.3,
                           phase_offset=0.0):
    '''
    '''
    vdim, hdim = vhsize
    if aspect_ratio == 'auto':
        aspect_ratio = hdim/float(vdim)

    if filter_temporal_width == 'auto':
        filter_temporal_width = int(stimulus_fps*(2/3.))

    gabor_components = mk_3d_gabor(vhsize,
                                   stimulus_fps=stimulus_fps,
                                   aspect_ratio=aspect_ratio,
                                   filter_temporal_width=filter_temporal_width,
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
                      stimulus,
                      masklimit=0.001):
    '''Dot the spatial gabor filters filter with the stimulus

    Parameters
    ----------
    spatial_gabor_sin : np.array, (vdim,hdim)
    spatial_gabor_cos : np.array, (vdim,hdim)
        Spatial gabor quadrature pair
    stimulus : 2D np.array (nimages, vdim*hdim)
        The movie frames with the spatial dimension collapsed.
    masklimit : float-like
        Threshold to find the non-zero filter region

    Returns
    -------
    channel_sin : np.ndarray, (nimages, )
    channel_cos : np.ndarray, (nimages, )
        The filter response to each stimulus
        The quadrature pair can be combined: (x^2 + y^2)^0.5
    '''
    backend = get_backend()
    gabors = backend.stack([spatial_gabor_sin.reshape(-1),
                            spatial_gabor_cos.reshape(-1)])
    # dot the gabors with the stimulus
    mask = backend.abs(gabors).sum(0) > masklimit
    gabor_prod = (gabors[:,mask].squeeze() @ stimulus.T[mask].squeeze()).T
    gabor_sin, gabor_cos = gabor_prod[:,0], gabor_prod[:,1]
    return gabor_sin, gabor_cos


def dotdelay_frames(spatial_gabor_sin, spatial_gabor_cos,
                    temporal_gabor_sin, temporal_gabor_cos,
                    stimulus,
                    masklimit=0.001):
    '''Convolve the motion energy filter with a stimulus

    Parameters
    ----------
    spatial_gabor_sin : np.array, (vdim,hdim)
    spatial_gabor_cos : np.array, (vdim,hdim)
        Spatial gabor quadrature pair

    temporal_gabor_sin : np.array, (temporal_filter_width,)
    temporal_gabor_cos : np.array, (temporal_filter_width,)
        Temporal gabor quadrature pair

    stimulus : 2D np.array (nimages, vdim*hdim)
        The movie frames with the spatial dimension collapsed.

    Returns
    -------
    channel_sin : np.ndarray, (nimages, )
    channel_cos : np.ndarray, (nimages, )
        The filter response to the stimulus at each time point
        The quadrature pair can be combined: (x^2 + y^2)^0.5
    '''

    backend = get_backend()

    gabor_sin, gabor_cos = dotspatial_frames(spatial_gabor_sin, spatial_gabor_cos,
                                             stimulus, masklimit=masklimit)
    gabor_prod = backend.column_stack([gabor_sin, gabor_cos])

    temporal_gabors = backend.stack([temporal_gabor_sin,
                                     temporal_gabor_cos])

    # dot the product with the temporal gabors
    outs =  gabor_prod[:, [0]] @ temporal_gabors[[1]] + gabor_prod[:, [1]] @ temporal_gabors[[0]]
    outc = -gabor_prod[:, [0]] @ temporal_gabors[[0]] + gabor_prod[:, [1]] @ temporal_gabors[[1]]

    # sum across delays
    nouts = backend.zeros_like(outs)
    noutc = backend.zeros_like(outc)
    tdxc = int(math.ceil(outs.shape[1]/2.0))
    delays = range(outs.shape[1])
    for ddx in delays:
        num = ddx - tdxc + 1
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
    '''Make 3D motion energy filter defined by the spatial and temporal gabors.

    Takes the output of :func:`mk_3d_gabor` and constructs the 3D filter.
    This is useful for visualization.

    Parameters
    ----------
    spatial_gabor_sin : np.array, (vdim,hdim)
    spatial_gabor_cos : np.array, (vdim,hdim)
        Spatial gabor quadrature pair
    temporal_gabor_sin : np.array, (filter_temporal_width,)
    temporal_gabor_cos : np.array, (filter_temporal_width,)
        Temporal gabor quadrature pair

    Returns
    -------
    motion_energy_filter : np.array, (vdim, hdim, filter_temporal_width)
        The motion energy filter
    '''
    a = -spatial_gabor_sin.ravel()[...,None] @ temporal_gabor_sin[...,None].T
    b =  spatial_gabor_cos.ravel()[...,None] @ temporal_gabor_cos[...,None].T
    x,y = spatial_gabor_sin.shape
    t = temporal_gabor_sin.shape[0]
    return (a+b).reshape(x,y,t)



def compute_spatial_gabor_responses(stimulus,
                                    aspect_ratio='auto',
                                    spatial_frequencies=[0,2,4,8,16,32],
                                    quadrature_combination=sqrt_sum_squares,
                                    output_nonlinearity=log_compress,
                                    dtype='float64',
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

    dtype : str or dtype
        Defaults to 'float64'

    Returns
    -------
    filter_responses : np.array, (n, nfilters)
    """
    backend = get_backend()
    nimages, vdim, hdim = stimulus.shape
    vhsize = (vdim, hdim)

    if aspect_ratio == 'auto':
        aspect_ratio = hdim/float(vdim)

    stimulus = stimulus.reshape(stimulus.shape[0], -1)
    parameter_names, gabor_parameters = mk_moten_pyramid_params(
        1.,                     # fps
        filter_temporal_width=1.,
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

    channels = backend.zeros((nimages, len(gabor_parameters)), dtype=dtype)
    for idx, gabor_param_dict in iterator_func(enumerate(filters),
                                          '%s.compute_spatial_gabor_responses'%__name__,
                                          total=len(gabor_parameters)):
        sgabor_sin, sgabor_cos, _, _ = mk_3d_gabor(vhsize,
                                                   **gabor_param_dict)

        channel_sin, channel_cos = dotspatial_frames(sgabor_sin, sgabor_cos, stimulus)
        channel = quadrature_combination(channel_sin, channel_cos)
        channels[:, idx] = channel
    channels = output_nonlinearity(channels)
    if dozscore:
        from scipy.stats import zscore
        channels = backend.to_numpy(channels)
        channels = zscore(channels)
    return channels


def compute_filter_responses(stimulus,
                             stimulus_fps,
                             aspect_ratio='auto',
                             filter_temporal_width='auto',
                             quadrature_combination=sqrt_sum_squares,
                             output_nonlinearity=log_compress,
                             dozscore=True,
                             dtype='float64',
                             pyramid_parameters={}):
    """Compute the motion energy filters' response to the stimuli.

    Parameters
    ----------
    stimulus : 3D np.array (n, vdim, hdim)
        The movie frames.
    stimulus_fps : scalar
        The temporal frequency of the stimulus
    aspect_ratio : bool, or scalar
        Defaults to hdim/vdim. Otherwise, pass as scalar
    filter_temporal_width : int, None
        The number of frames in one filter.
        Defaults to approximately 0.666[secs] (floor(stimulus_fps*(2/3))).
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

    dtype : str or dtype
        Defaults to 'float64'

    pyramid_parameters: dict
        See :func:`mk_moten_pyramid_params` for details on parameters
        specifiying a motion energy pyramid.

    Returns
    -------
    filter_responses : np.array, (n, nfilters)
    """
    backend = get_backend()
    nimages, vdim, hdim = stimulus.shape
    stimulus = stimulus.reshape(stimulus.shape[0], -1)
    vhsize = (vdim, hdim)

    if aspect_ratio == 'auto':
        aspect_ratio = hdim/float(vdim)

    if filter_temporal_width == 'auto':
        filter_temporal_width = int(stimulus_fps*(2./3.))

    # pass parameters
    pkwargs = dict(aspect_ratio=aspect_ratio,
                   filter_temporal_width=filter_temporal_width)
    pkwargs.update(**pyramid_parameters)
    parameter_names, gabor_parameters = mk_moten_pyramid_params(stimulus_fps,
                                                                **pkwargs)

    ngabors = gabor_parameters.shape[0]
    filters = [{name : gabor_parameters[idx, pdx] for pdx, name \
                in enumerate(parameter_names)} \
               for idx in range(ngabors)]

    info = 'Computing responses for #%i filters across #%i images (aspect_ratio=%0.03f)'
    print(info%(len(gabor_parameters), nimages, aspect_ratio))

    channels = backend.zeros((nimages, len(gabor_parameters)), dtype=dtype)
    for idx, gabor_param_dict in iterator_func(enumerate(filters),
                                          '%s.compute_filter_responses'%__name__,
                                          total=len(filters)):

        gabor = mk_3d_gabor(vhsize,
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
        channels = backend.to_numpy(channels)
        channels = zscore(channels)
    return channels


##############################
# batched computation
##############################

def mk_3d_gabor_batched(vhsize, filters_batch):
    '''Build spatial and temporal gabor filter banks for a batch of filters.

    Vectorizes :func:`mk_3d_gabor` across multiple filters so that all
    gabor arrays are constructed with batched tensor operations instead
    of a Python for-loop.

    Parameters
    ----------
    vhsize : tuple of ints, (vdim, hdim)
    filters_batch : list of dicts
        Each dict has the same keys accepted by :func:`mk_3d_gabor`.

    Returns
    -------
    spatial_gabors_sin : array, (B, npixels)
    spatial_gabors_cos : array, (B, npixels)
    temporal_gabors_sin : array, (B, T)
    temporal_gabors_cos : array, (B, T)
    '''
    backend = get_backend()
    B = len(filters_batch)
    vdim, hdim = vhsize
    npixels = vdim * hdim

    f0 = filters_batch[0]
    aspect_ratio = f0.get('aspect_ratio', 'auto')
    if aspect_ratio == 'auto':
        aspect_ratio = hdim / float(vdim)

    stimulus_fps = f0['stimulus_fps']
    filter_temporal_width = int(f0['filter_temporal_width'])

    # Extract per-filter parameters as 1-D arrays
    centerh = backend.asarray([f['centerh'] for f in filters_batch])
    centerv = backend.asarray([f['centerv'] for f in filters_batch])
    direction = backend.asarray([f['direction'] for f in filters_batch])
    spatial_freq = backend.asarray([f['spatial_freq'] for f in filters_batch])
    spatial_env = backend.asarray([f['spatial_env'] for f in filters_batch])
    temporal_freq = backend.asarray([f['temporal_freq'] for f in filters_batch])
    temporal_env = backend.asarray([f['temporal_env'] for f in filters_batch])
    spatial_phase_offset = backend.asarray(
        [f.get('spatial_phase_offset', 0.0) for f in filters_batch])

    # Shared spatial grid -- (vdim, hdim)
    dh = backend.linspace(0, aspect_ratio, hdim, endpoint=True)
    dv = backend.linspace(0, 1, vdim, endpoint=True)
    ihs, ivs = backend.meshgrid(dh, dv)  # (vdim, hdim)

    # Reshape for broadcasting: params → (B, 1, 1), grid → (1, vdim, hdim)
    dir_rad = direction / 180.0 * backend.pi
    fh = (-spatial_freq * backend.cos(dir_rad) * 2 * backend.pi).reshape(B, 1, 1)
    fv = (spatial_freq * backend.sin(dir_rad) * 2 * backend.pi).reshape(B, 1, 1)
    ch = centerh.reshape(B, 1, 1)
    cv = centerv.reshape(B, 1, 1)
    se = spatial_env.reshape(B, 1, 1)
    spo = spatial_phase_offset.reshape(B, 1, 1)

    ihs_3d = ihs.reshape(1, vdim, hdim)  # broadcast over batch
    ivs_3d = ivs.reshape(1, vdim, hdim)

    dih = ihs_3d - ch  # (B, vdim, hdim)
    div = ivs_3d - cv

    spatial_gaussian = backend.exp(-(dih ** 2 + div ** 2) / (2 * se ** 2))
    phase = dih * fh + div * fv + spo
    spatial_gabors_sin = (spatial_gaussian * backend.sin(phase)).reshape(B, npixels)
    spatial_gabors_cos = (spatial_gaussian * backend.cos(phase)).reshape(B, npixels)

    # Temporal filters -- (B, T)
    dt = backend.linspace(0, 1, filter_temporal_width, endpoint=False)  # (T,)
    ft = (temporal_freq * (filter_temporal_width / float(stimulus_fps))
          * 2 * backend.pi).reshape(B, 1)
    te = temporal_env.reshape(B, 1)
    dt_3d = dt.reshape(1, filter_temporal_width)

    temporal_gaussian = backend.exp(-(dt_3d - 0.5) ** 2 / (2 * te ** 2))
    temporal_gabors_sin = temporal_gaussian * backend.sin((dt_3d - 0.5) * ft)
    temporal_gabors_cos = temporal_gaussian * backend.cos((dt_3d - 0.5) * ft)

    return spatial_gabors_sin, spatial_gabors_cos, temporal_gabors_sin, temporal_gabors_cos


def project_stimulus_batched(stimulus,
                             filters,
                             quadrature_combination=sqrt_sum_squares,
                             output_nonlinearity=log_compress,
                             vhsize=(),
                             dtype='float32',
                             batch_size=128):
    '''Compute motion energy responses using batched operations.

    Functionally equivalent to :func:`project_stimulus` but constructs
    spatial and temporal gabor filter banks in batches and uses a single
    large matrix multiply per batch instead of per-filter dot products.
    This is significantly faster on GPU backends and can also be faster
    on CPU for large filter sets.

    Parameters
    ----------
    stimulus : array, (nimages, vdim, hdim) or (nimages, npixels)
        The movie frames.
    filters : list of dicts
        Filter parameter dictionaries (as produced by a pyramid).
    quadrature_combination : callable, optional
        Defaults to ``sqrt_sum_squares``.
    output_nonlinearity : callable, optional
        Defaults to ``log_compress``.
    vhsize : tuple of ints
        ``(vdim, hdim)`` required when stimulus is 2-D.
    dtype : str
        Output dtype.
    batch_size : int
        Number of filters to process simultaneously.  Larger values use
        more memory but reduce Python-loop overhead.

    Returns
    -------
    filter_responses : array, (nimages, nfilters)
    '''
    if stimulus.ndim == 3:
        nimages, vdim, hdim = stimulus.shape
        stimulus = stimulus.reshape(stimulus.shape[0], -1)
        vhsize = (vdim, hdim)

    backend = get_backend()

    assert stimulus.ndim == 2
    assert isinstance(vhsize, tuple) and len(vhsize) == 2
    assert vhsize[0] * vhsize[1] == stimulus.shape[1]

    nfilters = len(filters)
    nimages = stimulus.shape[0]
    filter_responses = backend.zeros((nimages, nfilters), dtype=dtype)

    # stimulus transpose computed once -- (npixels, nimages)
    stim_T = stimulus.T

    for batch_start in range(0, nfilters, batch_size):
        batch_end = min(batch_start + batch_size, nfilters)
        batch_filters = filters[batch_start:batch_end]
        B = len(batch_filters)

        # Build gabor filter banks for this batch
        sg_sin, sg_cos, tg_sin, tg_cos = mk_3d_gabor_batched(vhsize,
                                                               batch_filters)
        # sg_sin: (B, npixels), tg_sin: (B, T)

        # Apply per-filter mask: zero out pixels where the gabor
        # amplitude is below threshold (matches dotspatial_frames
        # masklimit behaviour without breaking the batched matmul).
        gabor_mask = (backend.abs(sg_sin) + backend.abs(sg_cos)) > 0.001
        sg_sin = sg_sin * gabor_mask
        sg_cos = sg_cos * gabor_mask

        # Spatial dot product -- single matmul per batch
        # (B, npixels) @ (npixels, nimages) → (B, nimages) → transpose
        spatial_sin = (sg_sin @ stim_T).T  # (nimages, B)
        spatial_cos = (sg_cos @ stim_T).T  # (nimages, B)

        # Temporal convolution via broadcasting
        # (nimages, B, 1) * (1, B, T) → (nimages, B, T)
        T = tg_sin.shape[1]
        sin_3d = spatial_sin.reshape(nimages, B, 1)
        cos_3d = spatial_cos.reshape(nimages, B, 1)
        tg_sin_3d = tg_sin.reshape(1, B, T)
        tg_cos_3d = tg_cos.reshape(1, B, T)

        outs = sin_3d * tg_cos_3d + cos_3d * tg_sin_3d   # (nimages, B, T)
        outc = -sin_3d * tg_sin_3d + cos_3d * tg_cos_3d  # (nimages, B, T)

        # Delay shifting -- loop over T (small, typically ~16)
        nouts = backend.zeros_like(outs)
        noutc = backend.zeros_like(outc)
        tdxc = int(math.ceil(T / 2.0))
        for ddx in range(T):
            num = ddx - tdxc + 1
            if num == 0:
                nouts[:, :, ddx] = outs[:, :, ddx]
                noutc[:, :, ddx] = outc[:, :, ddx]
            elif num > 0:
                nouts[num:, :, ddx] = outs[:-num, :, ddx]
                noutc[num:, :, ddx] = outc[:-num, :, ddx]
            elif num < 0:
                nouts[:num, :, ddx] = outs[abs(num):, :, ddx]
                noutc[:num, :, ddx] = outc[abs(num):, :, ddx]

        channel_sin = nouts.sum(-1)  # (nimages, B)
        channel_cos = noutc.sum(-1)  # (nimages, B)

        channel_response = quadrature_combination(channel_sin, channel_cos)
        channel_response = output_nonlinearity(channel_response)
        filter_responses[:, batch_start:batch_end] = channel_response

    return filter_responses


def mk_moten_pyramid_params(stimulus_fps,
                            filter_temporal_width='auto',
                            aspect_ratio='auto',
                            temporal_frequencies=[0,2,4],
                            spatial_frequencies=[0,2,4,8,16,32],
                            spatial_directions=[0,45,90,135,180,225,270,315],
                            sf_gauss_ratio=0.6,
                            max_spatial_env=0.3,
                            gabor_spacing=3.5,
                            tf_gauss_ratio=10.,
                            max_temp_env=0.3,
                            spatial_phase_offset=0.0,
                            include_edges=False,
                            ):
    """Parametrize a motion energy pyramid that tiles the stimulus.

    Parameters
    ----------
    stimulus_fps : scalar, [Hz]
        Stimulus playback speed in frames per second.
    spatial_frequencies : array-like, [cycles-per-image]
        Spatial frequencies for the filters
    spatial_directions : array-like, [degrees]
        Direction of filter motion. Degree position corresponds
        to standard unit-circle coordinates (i.e. 0=right, 180=left).
    temporal_frequencies : array-like, [Hz]
        Temporal frequencies of the filters
    filter_temporal_width : int
        Temporal window of the motion energy filter (e.g. 10).
        Defaults to approximately 0.666[secs] (`floor(stimulus_fps*(2/3))`).
    aspect_ratio : optional, 'auto' or float-like,
        Defaults to stimulus aspect ratio: hdim/vdim
        Useful for preserving the spatial gabors circular even
        when images have non-square aspect ratios. For example,
        a 16:9 image would have `aspect_ratio`=16/9.

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
    include_edges : bool
        Determines whether to include filters at the edge
        of the image which might be partially outside the
        stimulus field-of-view

    Returns
    -------
    parameter_names : list of strings
        The name of the parameters
    gabor_parameters : 2D np.ndarray, (nfilters, 11)
        Parameters that define the motion energy filter
        Each of the `nfilters` has the following parameters:
            * centerv,centerh : y:vertical and x:horizontal position ('0,0' is top left)
            * direction       : direction of motion [degrees]
            * spatial_freq    : spatial frequency [cpi]
            * spatial_env     : spatial envelope (gaussian s.d.)
            * temporal_freq   : temporal frequency [Hz]
            * temporal_env    : temporal envelope (gaussian s.d.)
            * filter_temporal_width : temporal window of filter [frames]
            * aspect_ratio    : width/height
            * stimulus_fps    : stimulus playback speed in frames per second
            * spatial_phase_offset : filter phase offset in [degrees]

    Notes
    -----
    Same method as Nishimoto, et al., 2011.
    """
    assert isinstance(aspect_ratio, (int, float)) or (hasattr(aspect_ratio, 'ndim'))

    def compute_envelope(freq, ratio):
        return float('inf') if freq == 0 else (1.0/freq)*ratio

    # mk_moten_pyramid_params always uses numpy for parameter construction
    # since it produces metadata (filter parameters), not computed signals.
    spatial_frequencies = np.asarray(spatial_frequencies)
    spatial_directions = np.asarray(spatial_directions)
    temporal_frequencies = np.asarray(temporal_frequencies)
    include_edges = int(include_edges)

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
                                         filter_temporal_width,
                                         aspect_ratio,
                                         stimulus_fps,
                                         spatial_phase_offset,
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
                                             filter_temporal_width,
                                             aspect_ratio,
                                             stimulus_fps,
                                             spatial_phase_offset,
                                             ])

    parameter_names = ('centerh',
                       'centerv',
                       'direction',
                       'spatial_freq',
                       'spatial_env',
                       'temporal_freq',
                       'temporal_env',
                       'filter_temporal_width',
                       'aspect_ratio',
                       'stimulus_fps',
                       'spatial_phase_offset',
                       )

    gabor_parameters = np.asarray(gabor_parameters)
    return parameter_names, gabor_parameters
