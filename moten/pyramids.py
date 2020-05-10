'''
'''
#
# Adapted from MATLAB code written by S. Nishimoto (see Nishimoto, et al., 2011).
# Anwar O. Nunez-Elizalde (Jan, 2016)
#
# Updates:
#  Anwar O. Nunez-Elizalde (Apr, 2020)
#
#
# Implementation notes:
#
# The terminology here is "pyramid" and "filters".
# In moten.core, the termilogy is inconsistent.
#
# TODO: Fix terminology in moten.core to match API.
#
import numpy as np


from moten import (utils,
                   core,
                   viz,
                   )

__all__ = ['MotionEnergyPyramid',
           'StimulusMotionEnergy',
           'StimulusStaticGaborPyramid',
           'DefaultPyramids',
           ]

##############################
#
##############################

class MotionEnergyPyramid(object):
    '''Construct a motion energy pyramid that tiles the stimulus.

    Generates motion energy filters at the desired
    spatio-temporal frequencies and directions of motion.
    Multiple motion energy filters per spatio-temporal frequency
    are constructed and each is centered at different locations
    in the image in order to tile the stimulus.

    Parameters
    ----------
    stimulus_vhsize : tuple of ints
        Size of the stimulus in pixels (vdim, hdim)
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

    Attributes
    ----------
    nfilters : int
    filters : list of dicts
        Each item is a dict defining a motion energy filter.
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

    parameters : dict of arrays
        The individual parameter values across all filters.
        Each item is an array of length ``nfilters``
    definition : dict
        Parameters used to define the pyramid.
    parameters_matrix : np.array, (nfilters, 11)
        Parameters that defined the motion energy filter.
    parameters_names  : tuple of strings

    Notes
    -----
    See :func:`moten.core.mk_moten_pyramid_params` for more details on
    making the pyramid.
    '''
    def __init__(self,
                 stimulus_vhsize=(576, 1024),
                 stimulus_fps=24,
                 temporal_frequencies=[0,2,4],
                 spatial_frequencies=[0,2,4,8,16],
                 spatial_directions=[0,45,90,135,180,225,270,315],
                 sf_gauss_ratio=0.6,
                 max_spatial_env=0.3,
                 filter_spacing=3.5,
                 tf_gauss_ratio=10.,
                 max_temp_env=0.3,
                 include_edges=False,
                 spatial_phase_offset=0.0,
                 filter_temporal_width='auto'):
        '''...More parameters

        Parameters
        ----------
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
        '''
        vdim, hdim = stimulus_vhsize
        aspect_ratio = hdim/float(vdim) # same as stimulus

        if filter_temporal_width == 'auto':
            # default to 2/3 stimulus frame rate
            filter_temporal_width = int(stimulus_fps*(2/3.))

        stimulus_vht_fov = (vdim, hdim, stimulus_fps)
        filter_vht_fov = (vdim, hdim, filter_temporal_width)

        # store pyramid parameters
        definition = utils.DotDict()
        for local_name, local_values in sorted(locals().items()):
            if local_name in ('self', 'definition') :
                continue
            definition[local_name] = local_values

        # sanity checks
        assert isinstance(stimulus_fps, int)
        assert isinstance(filter_temporal_width, int)

        # construct the gabor pyramid
        parameter_names, filter_params_array = core.mk_moten_pyramid_params(
            stimulus_fps,
            filter_temporal_width=filter_temporal_width,
            aspect_ratio=aspect_ratio,
            temporal_frequencies=temporal_frequencies,
            spatial_frequencies=spatial_frequencies,
            spatial_directions=spatial_directions,
            sf_gauss_ratio=sf_gauss_ratio,
            max_spatial_env=max_spatial_env,
            gabor_spacing=filter_spacing,
            tf_gauss_ratio=tf_gauss_ratio,
            max_temp_env=max_temp_env,
            include_edges=include_edges,
            spatial_phase_offset=spatial_phase_offset,
            )

        nfilters = filter_params_array.shape[0]

        # make individual filters readable
        filters = [utils.DotDict({name : filter_params_array[idx, pdx] for pdx, name \
                                  in enumerate(parameter_names)}) \
                   for idx in range(nfilters)]

        # parameters across filters
        parameters = utils.DotDict({name : filter_params_array[:,idx] for idx, name \
                                    in enumerate(parameter_names)})

        # store pyramid information as attributes
        self.filters = filters
        self.nfilters = nfilters
        self.parameters = parameters
        self.definition = definition
        self.parameters_matrix = filter_params_array
        self.parameters_names = parameter_names
        self.mask_threshold = 0.001 # HARDCODED. TODO: Make parameter.

    def __repr__(self):
        info = '<%s.%s [#%i filters (ntfq=%i, nsfq=%i, ndirs=%i) aspect=%0.03f]>'
        details = (__name__, type(self).__name__, self.nfilters,
                   len(self.definition.temporal_frequencies),
                   len(self.definition.spatial_frequencies),
                   len(self.definition.spatial_directions),
                   self.definition.aspect_ratio)
        return info%details

    def filters_at_vhposition(self, centerv, centerh):
        '''Center spatio-temporal filters to requested vh-position.

        Parameters
        ----------
        centerv : scalar
            Vertical filter position from top of frame (min=0, max=1.0).
        centerh : scalar
            Horizontal filter position from left of frame (min=0, max=aspect_ratio).

        Returns
        -------
        centered_filters : list of dicts
            Spatio-temporal filter parameters at vh-position.
        '''
        unique_parameters = np.unique(self.parameters_matrix[:, 2:], axis=0)
        nunique = unique_parameters.shape[0]

        new_filters = [utils.DotDict({name : unique_parameters[idx, pdx] for pdx, name \
                                      in enumerate(self.parameters_names[2:])}) \
                       for idx in range(nunique)]
        for filt in new_filters:
            filt['centerh'] = centerh
            filt['centerv'] = centerv
        return new_filters

    def get_filter_spatiotemporal_quadratures(self, filterid=0):
        '''Generate the spatial and temporal arrays that define the motion energy filter.

        Parameters
        ----------
        filterid : int, or dict
            If int, the filter index.
            If dict, a filter dictionary definition

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
        '''
        vhsize = (self.definition.vdim, self.definition.hdim)

        # extract parameters for this filterid
        if isinstance(filterid, dict):
            gabor_parameters_dict = filterid.copy()
        else:
            gabor_parameters_dict = self.filters[filterid]

        sgabor0, sgabor90, tgabor0, tgabor90 = core.mk_3d_gabor(vhsize,
                                                                **gabor_parameters_dict)
        return sgabor0, sgabor90, tgabor0, tgabor90

    def get_filter_temporal_quadrature(self, filterid=0):
        '''Generate the temporal arrays that define the motion energy filter.

        Parameters
        ----------
        filterid : int, or dict
            If int, the filter index.
            If dict, a filter dictionary definition

        Returns
        -------
        temporal_gabor_sin : 1D np.ndarray, (`filter_temporal_width`,)
        temporal_gabor_cos : 1D np.ndarray, (`filter_temporal_width`,)
            Temporal gabor quadrature pair. ``temporal_gabor_cos`` has
            a 90 degree phase offset relative to ``temporal_gabor_sin``
        '''
        _,_, tgabor0, tgabor90 = self.get_filter_spatiotemporal_quadratures(filterid=filterid)
        return tgabor0, tgabor90

    def get_filter_spatial_quadrature(self, filterid=0):
        '''Generate the spatial arrays that define the motion energy filter.

        Parameters
        ----------
        filterid : int, or dict
            If int, the filter index.
            If dict, a filter dictionary definition

        Returns
        -------
        spatial_gabor_sin : 2D np.ndarray, (vdim, hdim)
        spatial_gabor_cos : 2D np.ndarray, (vdim, hdim)
            Spatial gabor quadrature pair. ``spatial_gabor_cos`` has
            a 90 degree phase offset relative to ``spatial_gabor_sin``
        '''
        sgabor0, sgabor90, _, _ = self.get_filter_spatiotemporal_quadratures(filterid=filterid)
        return sgabor0, sgabor90

    def get_filter_mask(self, filterid=0):
        '''Generate a mask of the filter in the image

        Parameters
        ----------
        filterid : int, or dict
            If int, the filter index.
            If dict, a filter dictionary definition

        Returns
        -------
        mask : 2D np.ndarray, (vdim, hdim)
            Filter spatial mask
        '''
        abs = np.abs
        threshold = self.mask_threshold

        spsin, spcos = self.get_filter_spatial_quadrature(filterid)
        mask = (abs(spsin) + abs(spcos)) > threshold
        return mask

    def get_filter_pixel_sizes(self):
        '''Measure the size of the each filter in the pyramid in pixels.

        Returns
        -------
        filter_npixels : 2D np.ndarray, (nfilters,)
            Array containing the size of each filter in pixels.
        '''
        npixels = []
        for filter_idx in range(self.nfilters):
            mask = self.get_filter_mask(filter_idx)
            npixels.append(mask.sum())
        return npixels


    def show_filter(self, filterid=0, speed=1.0, background=None):
        '''Display the motion energy filter as an animation.

        Parameters
        ----------
        filterid : int, or dict
            If int, it's the index of the filter to display.
            If dict, it's a motion energy filter definition.

        Returns
        -------
        animation : matplotlib.animation

        Examples
        --------
        >>> import moten
        >>> pyramid = moten.get_default_pyramid()
        >>> _ = pyramid.show_filter(12)
        >>> custom_filter = {'centerh': 0.8888888888888888,
                             'centerv': 0.5,
                             'direction': 45.0,
                             'spatial_freq': 2.0,
                             'spatial_env': 0.3,
                             'temporal_freq': 2.0,
                             'temporal_env': 0.3,
                             'filter_temporal_width': 16.0,
                             'aspect_ratio': 1.7777777777777777,
                             'stimulus_fps': 24.0,
                             'spatial_phase_offset': 0.0}
        >>> _ = pyramid.show_filter(custom_filter)
        '''
        # Get dimensions of movie
        vdim, hdim, stimulus_fps = self.definition.stimulus_vht_fov
        aspect_ratio = self.definition.aspect_ratio
        filter_width = self.definition.filter_temporal_width
        vhsize = (vdim, hdim)

        # extract parameters for this filterid
        if isinstance(filterid, dict):
            gabor_params_dict = filterid.copy()
            filterid = 0
        else:
            gabor_params_dict = self.filters[filterid]

        # construct title from parameters
        title = ''
        for pdx, (pname, pval) in enumerate(sorted(gabor_params_dict.items())):
            title += '%s=%0.02f, '%(pname, pval)
            if np.mod(pdx, 3) == 0 and pdx > 0:
                title += '\n'

        return viz.plot_3dgabor(vhsize,
                                gabor_params_dict,
                                background=background,
                                title=title, speed=speed)

    def project_stimulus(self,
                         stimulus,
                         filters='all',
                         quadrature_combination=utils.sqrt_sum_squares,
                         output_nonlinearity=utils.log_compress,
                         dtype='float32',
                         use_cuda=False):
        '''Compute the motion energy filter responses to the stimulus.

        Parameters
        ----------
        stimulus : np.ndarray, (nimages, vdim, hdim) or (nimages, npixels)
            The movie frames.
            If ``stimulus`` is two-dimensional with shape (nimages, npixels), then
            ``vhsize=(vdim,hdim)`` is required and `npixels == vdim*hdim`.
        filters : optional, 'all' or list of dicts
            By default compute the responses for all filters.
            Otherwise, provide a list of filter definitions to use.
        quadrature_combination : function, optional
            Specifies how to combine the channel reponse quadratures.
            The function must take the sin and cos as arguments in that order.
            Defaults to: :math:`(sin^2 + cos^2)^{1/2}`
        output_nonlinearity : function, optional
            Passes the channels (after ``quadrature_combination``) through a
            non-linearity. The function input is the (`nimages, nfilters`) array.
            Defaults to: :math:`log(x + 1e-05)`

        Returns
        -------
        filter_responses : np.ndarray, (nimages, nfilters)
        '''
        if filters == 'all':
            filters = self.filters

        vdim, hdim = self.definition.stimulus_vhsize

        output = core.project_stimulus(stimulus,
                                       filters,
                                       quadrature_combination=quadrature_combination,
                                       output_nonlinearity=output_nonlinearity,
                                       vhsize=(vdim, hdim),
                                       dtype=dtype)

        return output

    def raw_project_stimulus(self, stimulus, filters='all', dtype='float32'):
        '''Obtain responses to the stimulus from all filter quadrature-pairs.

        Parameters
        ----------
        stimulus : np.array, (nimages, vdim, hdim)
            The movie frames.
        filters : optional, 'all' or list of dicts
            By default compute the responses for all filters.
            Otherwise, provide a list of filter definitions to use.

        Returns
        -------
        output_sin : np.ndarray, (nimages, nfilters)
        output_cos : np.ndarray, (nimages, nfilters)
        '''
        if filters == 'all':
            filters = self.filters

        vdim, hdim = self.definition.stimulus_vhsize

        output_sin, output_cos = core.raw_project_stimulus(stimulus,
                                                           filters,
                                                           vhsize=(vdim, hdim),
                                                           dtype=dtype)

        return output_sin, output_cos


class StimulusMotionEnergy(object):
    '''Parametrize a motion energy pyramid that tiles the stimulus.

    Generates motion energy filters at the desired
    spatio-temporal frequencies and directions of motion.
    Multiple motion energy filters per spatio-temporal frequency
    are constructed and each is centered at different locations
    in the image in order to tile the stimulus.

    Parameters
    ----------
    stimulus : np.array, (nimages, vdim, hdim)
        The movie frames.
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

    Attributes
    ----------
    stimulus : 2D np.ndarray, (nimages, vdim*hdim)
        Time-space representation of stimulus
    view : :class:`MotionEnergyPyramid`
        Full description of the motion energy pyramid.
    nimages : int
        Number of video frames
    nfilters : int
    aspect_ratio : scalar, (hdim/vdim)
    stimulus_fps : int (fps)
    stimulus_vhsize : tuple of ints, (vdim, hdim)
    original_stimulus : 3D np.ndarray, (nimages, vdim, hdim)
    '''
    def __init__(self,
                 stimulus,
                 stimulus_fps,
                 temporal_frequencies=[0,2,4],
                 spatial_frequencies=[0,2,4,8,16],
                 spatial_directions=[0,45,90,135,180,225,270,315],
                 sf_gauss_ratio=0.6,
                 max_spatial_env=0.3,
                 filter_spacing=3.5,
                 tf_gauss_ratio=10.,
                 max_temp_env=0.3,
                 include_edges=False,
                 spatial_phase_offset=0.0,
                 filter_temporal_width='auto'):
        '''
        Notes
        -----
        See :class:`MotionEnergyPyramid` for more detail on
        keyword arguments used to construct the motion energy pyramid.
        '''
        nimages, vdim, hdim = stimulus.shape
        stimulus_vhsize = (vdim, hdim)
        original_stimulus = stimulus
        stimulus = stimulus.reshape(stimulus.shape[0], -1)
        aspect_ratio = hdim/vdim

        pyramid = MotionEnergyPyramid(stimulus_vhsize=(vdim, hdim),
                                      stimulus_fps=stimulus_fps,
                                      filter_temporal_width=filter_temporal_width,
                                      temporal_frequencies=temporal_frequencies,
                                      spatial_frequencies=spatial_frequencies,
                                      spatial_directions=spatial_directions,
                                      sf_gauss_ratio=sf_gauss_ratio,
                                      max_spatial_env=max_spatial_env,
                                      filter_spacing=filter_spacing,
                                      tf_gauss_ratio=tf_gauss_ratio,
                                      max_temp_env=max_temp_env,
                                      spatial_phase_offset=spatial_phase_offset,
                                      include_edges=include_edges)

        self.nimages = nimages
        self.stimulus = stimulus
        self.stimulus_fps = stimulus_fps
        self.stimulus_vhsize = stimulus_vhsize
        self.original_stimulus = original_stimulus
        self.aspect_ratio = aspect_ratio

        self.view = pyramid
        self.nfilters = pyramid.nfilters

    def __repr__(self):
        info = '<%s.%s [%i nframes @ %i fps. #%i filters. aspect=%0.03f]>'
        details = (__name__, type(self).__name__,
                   self.nimages,
                   self.stimulus_fps,
                   self.nfilters,
                   self.aspect_ratio)
        return info%details

    def project_stimulus(self, *args, **kwargs):
        '''See :meth:`MotionEnergyPyramid.project_stimulus`
        '''
        filter_responses  = self.view.project_stimulus(*args, **kwargs)
        return filter_responses

    def project(self, filters='all',
                quadrature_combination=utils.sqrt_sum_squares,
                output_nonlinearity=utils.log_compress,
                dtype='float32'):
        '''Compute the motion energy filter responses to the stimulus.

        Parameters
        ----------
        filters : optional, 'all' or list of dicts
            By default compute the responses for all filters.
            Otherwise, provide a list of filter definitions to use.
        quadrature_combination : function, optional
            Specifies how to combine the channel reponse quadratures.
            The function must take the sin and cos as arguments in that order.
            Defaults to: :math:`(sin^2 + cos^2)^{1/2}`
        output_nonlinearity : function, optional
            Passes the channels (after ``quadrature_combination``) through a
            non-linearity. The function input is the (`nimages, nfilters`) array.
            Defaults to: :math:`log(x + 1e-05)`

        Returns
        -------
        filter_responses : np.ndarray, (nimages, nfilters)
        '''
        if filters == 'all':
            filters = self.view.filters
        stimulus = self.original_stimulus

        filter_responses = self.view.project_stimulus(
            stimulus,
            filters=filters,
            quadrature_combination=quadrature_combination,
            output_nonlinearity=output_nonlinearity,
            dtype=dtype)

        return filter_responses

    def project_at_vhposition(self, centerv, centerh,
                              quadrature_combination=utils.sqrt_sum_squares,
                              output_nonlinearity=utils.log_compress,
                              dtype='float32'):
        '''Center filters at vh-position and compute their response to the stimulus.

        Parameters
        ----------
        centerv : scalar
            Vertical filter from top of frame (min=0, max=1.0).
        centerh : scalar
            Horizontal filter position from left of frame (min=0, max=aspect_ratio).

        Returns
        -------
        centered_filters : list of dicts
            Spatio-temporal filter parameters at vh-position.
        filter_responses : np.ndarray, (nimages, len(vh_centered_filters))
        '''
        # center all spatial and temporal filters at desired vh-position
        filters = self.view.filters_at_vhposition(centerv, centerh)
        stimulus = self.original_stimulus

        filter_responses = self.view.project_stimulus(
            stimulus,
            filters=filters,
            quadrature_combination=quadrature_combination,
            output_nonlinearity=output_nonlinearity,
            dtype=dtype)

        return filters, filter_responses

    def raw_projection(self, filters='all', dtype='float32'):
        '''Obtain responses to the stimulus from all filter quadrature-pairs.

        Parameters
        ----------
        filters : optional, 'all' or list of dicts
            By default compute the responses for all filters.
            Otherwise, provide a list of filter definitions to use.

        Returns
        -------
        output_sin : np.ndarray, (nimages, nfilters)
        output_cos : np.ndarray, (nimages, nfilters)
        '''
        if filters == 'all':
            filters = self.view.filters
        stimulus = self.original_stimulus

        output_sin, output_cos = self.view.raw_project_stimulus(
            stimulus, filters=filters, dtype=dtype)

        return output_sin, output_cos


class StimulusStaticGaborPyramid(StimulusMotionEnergy):
    '''
    '''
    def __init__(self,
                 stimulus,
                 spatial_frequencies=[0,2,4,8,16],
                 spatial_orientations=(0,45,90,135),
                 sf_gauss_ratio=0.6,
                 max_spatial_env=0.3,
                 filter_spacing=3.5,
                 include_edges=False,
                 spatial_phase_offset=0.0,
                 ):
        super(type(self), self).__init__(stimulus,
                                         spatial_frequencies=spatial_frequencies,
                                         spatial_directions=spatial_orientations,
                                         spatial_phase_offset=spatial_phase_offset,
                                         include_edges=include_edges,
                                         sf_gauss_ratio=sf_gauss_ratio,
                                         max_spatial_env=max_spatial_env,
                                         filter_spacing=filter_spacing,
                                         # fixed parameters for static filters:
                                         max_temp_env=np.inf,
                                         stimulus_fps=1,
                                         temporal_frequencies=[0],
                                         filter_temporal_width=1,
                                         tf_gauss_ratio=10)


class DefaultPyramids(object):
    '''
    '''
    def __init__(self):
        pass

    @property
    def pyramid15fps512x512(self):
        pyramid =  MotionEnergyPyramid(stimulus_vhsize=(512, 512),
                                       stimulus_fps=15)
        return pyramid

    @property
    def pyramid15fps96x96(self):
        pyramid =  MotionEnergyPyramid(stimulus_vhsize=(96, 96),
                                       stimulus_fps=15)
        return pyramid

    @property
    def pyramid24fps800x600(self):
        pyramid =  MotionEnergyPyramid(stimulus_vhsize=(600, 800),
                                       stimulus_fps=24)
        return pyramid

    @property
    def pyramid24fps640x480(self):
        pyramid =  MotionEnergyPyramid(stimulus_vhsize=(480, 640),
                                       stimulus_fps=24)
        return pyramid

    @property
    def pyramid24fps128x72(self):
        pyramid =  MotionEnergyPyramid(stimulus_vhsize=(72, 128),
                                       stimulus_fps=24)
        return pyramid

    @property
    def pyramid24fps256x144(self):
        pyramid =  MotionEnergyPyramid(stimulus_vhsize=(144, 256),
                                       stimulus_fps=24)
        return pyramid
