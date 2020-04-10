'''Motion-energy filters (after Nishimoto, 2011)

Adapted from MATLAB code written by S. Nishimoto.

Anwar O. Nunez-Elizalde (Jan, 2016)

Updates:
 Anwar O. Nunez-Elizalde (Apr, 2020)
'''
# Implementation notes:
#
# The terminology here is "pyramid" and "filters".
# In moten.core, the termilogy is inconsistent.
#
# TODO: Fix terminology in moten.core to match API.

import numpy as np


from moten import (utils,
                   core,
                   viz,
                   )

##############################
#
##############################

class PyramidConstructor(object):
    '''Construct a motion-energy pyramid that tiles the stimulus.

    Generates motion-energy filters at the desired
    spatio-temporal frequencies and directions of motion.
    Multiple motion-energy filters per spatio-temporal frequency
    are constructed and each is centered at different locations
    in the image in order to tile the stimulus.

    Parameters
    ----------
    stimulus_hvt_fov : tuple of ints
        Resolution of the stimulus (hdim, vdim, fps)
    temporal_frequencies : array-like, [Hz]
        Temporal frequencies of the filters for use on the stimulus
    spatial_frequencies : array-like, [cycles-per-image]
        Spatial frequencies for the filters
    spatial_directions : array-like, [degrees]
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
    include_edges : bool
        Determines whether to include filters at the edge
        of the image which might be partially outside the
        stimulus field-of-view
    filter_nframes : int
        Temporal window of the motion-energy filter (e.g. 10).
        Defaults to the stimulus `fps`.
        This is called `gabor_nframes` in `moten.core`.

    Methods
    -------
    show_filter(gaborid=0)
        Display the selected filter as a matplotlib animation.
    filters_at_hvposition(hvpos=(0.5, 0.5))
        Center spatio-temporal filters to requested hv-position.

    Attributes
    ----------
    nfilters : int
    filters : list of dicts
        Each item is a dict defining a motion-energy filter.
        Each of the `nfilters` has the following parameters:
            * centerh,centerv : x:horizontal and y:vertical position ('0,0' is top left)
            * direction       : direction of motion
            * spatial_freq    : spatial frequency
            * spatial_env     : spatial envelope (gaussian s.d.)
            * temporal_freq   : temporal frequency
            * temporal_env    : temporal envelope (gaussian s.d.)

    parameters : dict of arrays
        The individual parameter values across all filters.
        Each item is an array of length `nfilters`
    definition : dict
        Parameters used to construct pyramid
    parameters_matrix : np.array, (nfilters, 7)
        Parameters that defined the motion-energy filter
    parameters_names  : tuple of strings
    '''
    def __init__(self,
                 stimulus_hvt_fov=(1024, 576, 24),
                 temporal_frequencies=(0,2,4),
                 spatial_frequencies=(0,2,4,8,16,32),
                 spatial_directions=(0,45,90,135,180,225,270,315),
                 sf_gauss_ratio=0.6,
                 max_spatial_env=0.3,
                 filter_spacing=3.5,
                 tf_gauss_ratio=10.,
                 max_temp_env=0.3,
                 include_edges=False,
                 filter_nframes='auto'):
        '''
        '''
        hdim, vdim, stimulus_fps = stimulus_hvt_fov
        filter_aspect_ratio = hdim/float(vdim) # same as stimulus

        if filter_nframes == 'auto':
            # default to stimulus frame rate
            filter_nframes = stimulus_fps

        filter_hvt_fov = (hdim, vdim, filter_nframes)

        # store pyramid parameters
        definition = utils.DotDict()
        for local_name, local_values in sorted(locals().items()):
            if local_name in ('self', 'definition') :
                continue
            definition[local_name] = local_values

        # sanity checks
        assert isinstance(stimulus_fps, int)
        assert isinstance(filter_nframes, int)

        # construct the gabor pyramid
        parameter_names, filter_params_array = core.mk_moten_pyramid_params(
            stimulus_fps,
            filter_nframes,
            aspect_ratio=filter_aspect_ratio,
            temporal_frequencies=temporal_frequencies,
            spatial_frequencies=spatial_frequencies,
            spatial_directions=spatial_directions,
            sf_gauss_ratio=sf_gauss_ratio,
            max_spatial_env=max_spatial_env,
            gabor_spacing=filter_spacing,
            tf_gauss_ratio=tf_gauss_ratio,
            max_temp_env=max_temp_env,
            include_edges=include_edges,
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

    def __repr__(self):
        info = '<%s.%s [#%i filters (ntfq=%i, nsfq=%i, ndirs=%i) aspect=%0.03f]>'
        details = (__name__, type(self).__name__, self.nfilters,
                   len(self.definition.temporal_frequencies),
                   len(self.definition.spatial_frequencies),
                   len(self.definition.spatial_directions),
                   self.definition.filter_aspect_ratio)
        return info%details

    def filters_at_hvposition(self, centerh, centerv):
        '''Center spatio-temporal filters to requested hv-position.

        Parameters
        ----------
        centerh : scalar
            Horizontal filter position from left of frame (min=0, max=aspect_ratio).
        centerv : scalar
            Vertical filter from top of frame (min=0, max=1.0).

        Returns
        -------
        centered_filters : dict
            Spatio-temporal filter parameters at hv-position.
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

    def show_filter(self, gaborid=0, speed=1.0, background=None):
        '''Display the motion-energy filter as an animation.

        Parameters
        ----------
        gaborid : int, or dict
            If int, it's the index of the filter to display.
            If dict, it's a motion-energy filter definition. E.g.:
                {'centerh': 0.5,
                 'centerv': 0.5,
                 'direction': 0.0,
                 'spatial_freq': 2.0,
                 'spatial_env': 0.3,
                 'temporal_freq': 0.0,
                 'temporal_env': 0.3}

        Returns
        -------
        animation : matplotlib.animation

        Examples
        --------
        >>> pyramid.show_filter(12)
        '''
        # Get dimensions of movie
        hdim, vdim, stimulus_fps = self.definition.stimulus_hvt_fov
        aspect_ratio = self.definition.filter_aspect_ratio
        tdim = self.definition.filter_nframes

        # extract parameters for this gaborid
        if isinstance(gaborid, dict):
            gabor_params = gaborid.copy()
            gaborid = 0
        else:
            gabor_params = self.filters[gaborid]

        # construct title from parameters
        title = ''
        for pdx, (pname, pval) in enumerate(sorted(gabor_params.items())):
            if pname == 'temporal_freq':
                pval = pval*(stimulus_fps/tdim)
            title += '%s=%0.02f, '%(pname, pval)
            if np.mod(pdx, 3) == 0 and pdx > 0:
                title += '\n'

        return viz.plot_3dgabor(gabor_params, vdim=vdim, hdim=hdim, tdim=tdim,
                                fps=stimulus_fps, aspect_ratio=aspect_ratio, background=background,
                                title=title, speed=speed)


class StimulusMotionEnergy(object):
    '''Parametrize a motion-energy pyramid that tiles the stimulus.

    Generates motion-energy filters at the desired
    spatio-temporal frequencies and directions of motion.
    Multiple motion-energy filters per spatio-temporal frequency
    are constructed and each is centered at different locations
    in the image in order to tile the stimulus.

    Parameters
    ----------
    stimulus : np.array, (n, vdim, hdim)
        The movie frames.
    stimulus_fps : int, [Hz]
        The temporal frequency of the stimulus.
    spatial_frequencies : array-like, [cycles-per-image]
        Spatial frequencies for the filters.
    spatial_directions : array-like, [degrees]
        Direction of filter motion. Degree position corresponds
        to standard unit-circle coordinates.
    temporal_frequencies : array-like, [Hz]
        Temporal frequencies of the filters for use on the stimulus.

    Attributes
    ----------
    stimulus : 2D np.ndarray, (nimages, vdim*hdim)
        Time-space representation of stimulus
    view : :class:`PyramidConstructor`
        Full description of the motion-energy pyramid.
    nimages : int
    aspect_ratio : scalar, (hdim/vdim)
    stimulus_fps : int (fps)
    stimulus_hvt_fov : tuple of ints, (hdim, vdim, fps)
    nfilters : int

    Methods
    -------
    project()
        Compute the motion-energy filter responses to the stimulus.

    project_at_hvposition(centerh, centerv)
        Center the motion-energy filters at hv-position in the stimulus
        and compute the filter responses.

    raw_projection()
        Return the filter quadrature-pair responses to the stimuli
        at 0 and 90 degree phase offsets.
    '''
    def __init__(self,
                 stimulus,
                 stimulus_fps,
                 temporal_frequencies=[0,2,4],
                 spatial_frequencies=[0,2,4,8,16,32],
                 spatial_directions=[0,45,90,135,180,225,270,315],
                 sf_gauss_ratio=0.6,
                 max_spatial_env=0.3,
                 filter_spacing=3.5,
                 tf_gauss_ratio=10.,
                 max_temp_env=0.3,
                 include_edges=False,
                 filter_nframes='auto'):
        '''
        Notes
        -----
        See :class:`PyramidConstructor` for more detail on
        keyword arguments used to construct the motion-energy pyramid.
        '''
        nimages, vdim, hdim = stimulus.shape
        stimulus_hvt_fov = (hdim, vdim, stimulus_fps)
        stimulus = stimulus.reshape(stimulus.shape[0], -1)
        aspect_ratio = hdim/vdim

        pyramid = PyramidConstructor(stimulus_hvt_fov=stimulus_hvt_fov,
                                     filter_nframes=filter_nframes,
                                     temporal_frequencies=temporal_frequencies,
                                     spatial_frequencies=spatial_frequencies,
                                     spatial_directions=spatial_directions,
                                     sf_gauss_ratio=sf_gauss_ratio,
                                     max_spatial_env=max_spatial_env,
                                     filter_spacing=filter_spacing,
                                     tf_gauss_ratio=tf_gauss_ratio,
                                     max_temp_env=max_temp_env,
                                     include_edges=include_edges)

        self.nimages = nimages
        self.stimulus = stimulus
        self.aspect_ratio = aspect_ratio
        self.stimulus_fps = stimulus_fps
        self.stimulus_hvt_fov=stimulus_hvt_fov

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

    def project(self, filters='all',
                quadrature_combination=utils.sqrt_sum_squares,
                output_nonlinearity=utils.log_compress,
                dtype=np.float32):
        '''Compute the motion-energy filter responses to the stimuli.

        Parameters
        ----------
        quadrature_combination : function, optional
            Specifies how to combine the channel reponses quadratures.
            The function must take the sin and cos as arguments in order.
            Defaults to: (sin^2 + cos^2)^1/2
        output_nonlinearity : function, optional
            Passes the channels (after `quadrature_combination`) through a
            non-linearity. The function input is the (`nimages`,`nfilters`) array.
            Defaults to: ln(x + 1e-05)
        dtype : np.dtype
            Defaults to np.float32
        filters : optional, 'all' or list of dicts
            By default compute the responses for all filters.
            Otherwise, provide a list of filter definitions to use.

        Returns
        -------
        filter_responses : np.ndarray, (nimages, nfilters)
        '''
        if filters == 'all':
            filters = self.view.filters

        # Set parameters
        stimulus = self.stimulus
        nimages = self.nimages
        nfilters = len(filters)
        filter_hvt_fov = self.view.definition.filter_hvt_fov
        filter_aspect_ratio = self.view.definition.filter_aspect_ratio

        # Compute responses
        filter_responses = np.zeros((nimages,nfilters), dtype=dtype)
        for gaborid, gabor_parameters in utils.iterator_func(enumerate(filters),
                                                             '%s.project'%type(self).__name__,
                                                             total=len(filters)):

            sgabor0, sgabor90, tgabor0, tgabor90 = core.mk_3d_gabor(filter_hvt_fov,
                                                                    aspect_ratio=filter_aspect_ratio,
                                                                    **gabor_parameters)

            channel_sin, channel_cos = core.dotdelay_frames(sgabor0, sgabor90,
                                                            tgabor0, tgabor90,
                                                            stimulus)

            channel_response = quadrature_combination(channel_sin, channel_cos)
            channel_response = output_nonlinearity(channel_response)
            filter_responses[:, gaborid] = channel_response

        return filter_responses

    def project_at_hvposition(self, centerh, centerv,
                              quadrature_combination=utils.sqrt_sum_squares,
                              output_nonlinearity=utils.log_compress,
                              dtype=np.float32):
        '''Center filters at hv-position and compute their response to the stimulus.

        Parameters
        ----------
        centerh : scalar
            Horizontal filter position from left of frame (min=0, max=aspect_ratio).
        centerv : scalar
            Vertical filter from top of frame (min=0, max=1.0).

        quadrature_combination : function, optional
            Defaults to: (sin^2 + cos^2)^1/2
        output_nonlinearity : function, optional
            Defaults to: ln(x + 1e-05)

        Returns
        -------
        hv_centered_filters : list of dicts
            Definition of filters used. All centered at hv-position.
        filter_responses : np.ndarray, (nimages, len(hv_centered_filters))

        '''
        # center all spatial and temporal filters at desired hv-position
        filters = self.view.filters_at_hvposition(centerh, centerv)

        filter_responses = self.project(filters,
                                        quadrature_combination=quadrature_combination,
                                        output_nonlinearity=output_nonlinearity,
                                        dtype=dtype)
        return filters, filter_responses

    def raw_projection(self, filters='all', dtype=np.float32):
        '''Obtain stimulus responses from all filter quadrature-pairs.

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
        nfilters = len(filters)

        stimulus = self.stimulus
        nimages = self.nimages

        filter_hvt_fov = self.view.definition.filter_hvt_fov
        filter_aspect_ratio = self.view.definition.filter_aspect_ratio

        # Compute responses
        output_sin = np.zeros((nimages,nfilters), dtype=dtype)
        output_cos = np.zeros((nimages,nfilters), dtype=dtype)
        for gaborid, gabor_parameters in utils.iterator_func(enumerate(filters),
                                                             '%s.raw_projection'%type(self).__name__,
                                                             total=len(filters)):

            sgabor0, sgabor90, tgabor0, tgabor90 = core.mk_3d_gabor(filter_hvt_fov,
                                                                    aspect_ratio=filter_aspect_ratio,
                                                                    **gabor_parameters)

            channel_sin, channel_cos = core.dotdelay_frames(sgabor0, sgabor90,
                                                            tgabor0, tgabor90,
                                                            stimulus)
            output_sin[:, gaborid] = channel_sin
            output_cos[:, gaborid] = channel_cos

        return output_sin, output_cos
