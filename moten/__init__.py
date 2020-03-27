'''Motion-energy filters (after Nishimoto, 2011)

Adapted from MATLAB code written by S. Nishimoto.

Anwar O. Nunez-Elizalde (Jan, 2016)
'''
import numpy as np

##############################
# internal imports
##############################
from moten import (colorspace,
                   utils,
                   core,
                   viz,
                   io,
                   )


##############################
#
##############################

class GaborPyramid(object):
    '''
    '''
    def __init__(self,
                 movie_fps,
                 vdim=576,
                 hdim=1024,
                 aspect_ratio='auto',
                 gabor_duration='auto',
                 temporal_frequencies=[0,2,4],
                 spatial_frequencies=[0,2,4,8,16,32],
                 spatial_directions=[0,45,90,135,180,225,270,315],
                 sf_gauss_ratio=0.6,
                 max_spatial_env=0.3,
                 gabor_spacing=3.5,
                 tf_gauss_ratio=10.,
                 max_temp_env=0.3,
                 include_edges=False):
        '''
        '''
        # Update aspect ratio
        if aspect_ratio == 'auto':
            aspect_ratio = hdim/float(vdim)

        # update gabor temporal window
        if gabor_duration == 'auto':
            gabor_duration = int(movie_fps*(2/3.0))

        # Set parameters as object attribute
        for local_name, local_values in locals().items():
            if local_name == 'self':
                continue
            setattr(self, local_name, np.asarray(local_values, dtype=np.float))

        # construct the gabor pyramid
        parameter_names, gabor_parameters = core.mk_moten_pyramid_params(
            movie_fps,
            gabor_duration,
            aspect_ratio=aspect_ratio,
            temporal_frequencies=temporal_frequencies,
            spatial_frequencies=spatial_frequencies,
            spatial_directions=spatial_directions,
            sf_gauss_ratio=sf_gauss_ratio,
            max_spatial_env=max_spatial_env,
            gabor_spacing=gabor_spacing,
            tf_gauss_ratio=tf_gauss_ratio,
            max_temp_env=max_temp_env,
            include_edges=include_edges,
            )

        ngabors = gabor_parameters.shape[0]

        # make the parameters accessible
        parameter_values = utils.DotDict({name : gabor_parameters[:,idx] for idx, name \
                                          in enumerate(parameter_names)})

        # make the individual filters accessible
        filters = [utils.DotDict({name : gabor_parameters[idx, pdx] for pdx, name \
                                  in enumerate(parameter_names)}) \
                   for idx in range(ngabors)]

        # Store as attributes
        self.ngabors = ngabors
        self.gabor_parameters = gabor_parameters
        self.parameter_names = parameter_names
        self.parameters = parameter_values
        self.filters = filters
        self.aspect_ratio = aspect_ratio
        self.gabor_hvt_size = (hdim, vdim, gabor_duration)

    def get_gabor_components(self, gaborid):
        '''Spatial and temporal quadrature pairs for motion-energy filter.

        A motion-energy filter is a 3D gabor with two spatial and
        one temporal dimension.

        The spatial and temporal dimensions are each defined by
        two sine waves which differ in phase by 90 degrees.
        The sine waves are then multiplied by a gaussian.

        Returns
        -------
        spatial_gabor_sin, spatial_gabor_cos : np.array, (vdim,hdim)
            Spatial gabor quadrature pair. `spatial_gabor_cos` has
            a 90 degree phase offset relative to `spatial_gabor_sin`

        temporal_gabor_sin, temporal_gabor_cos : np.array, (tdim)
            Temporal gabor quadrature pair. `temporal_gabor_cos` has
            a 90 degree phase offset relative to `temporal_gabor_sin`
        '''
        aspect_ratio = self.aspect_ratio
        gabor_parameters = self.filters[gaborid]
        gabor_hvt_size = self.gabor_hvt_size
        print(gabor_parameters)
        quadratures = core.mk_3d_gabor(gabor_hvt_size,
                                       aspect_ratio=aspect_ratio,
                                       **gabor_parameters)
        return quadratures

    def project_stimuli(self, stimulus, filters='all', dtype=np.float32):
        '''Responses of the motion-energy filter to the stimuli
        '''
        # define gabor filter according to stimulus
        nimages, vdim, hdim = stimulus.shape
        gabor_hvt_size = (hdim, vdim, self.gabor_duration)
        aspect_ratio = self.aspect_ratio
        stimulus = stimulus.reshape(nimages, -1)

        output_sin = np.zeros((nimages, self.ngabors), dtype=dtype)
        output_cos = np.zeros((nimages, self.ngabors), dtype=dtype)


        info = 'Computing responses for #%i filters across #%i images (aspect_ratio=%0.03f)'
        print(info%(len(self.filters), nimages, aspect_ratio))

        for gaborid, gabor_parameters in utils.iterator_func(enumerate(self.filters),
                                                             '%s.project_stimuli'%type(self).__name__,
                                                             total=len(self.filters)):

            sgabor0, sgabor90, tgabor0, tgabor90 = core.mk_3d_gabor(gabor_hvt_size,
                                                                    aspect_ratio=aspect_ratio,
                                                                    **gabor_parameters)

            channel_sin, channel_cos = core.dotdelay_frames(sgabor0, sgabor90,
                                                            tgabor0, tgabor90,
                                                            stimulus)
            output_sin[:, gaborid] = channel_sin
            output_cos[:, gaborid] = channel_cos
        return output_sin, output_cos

    def get_3dgabor_array(self, gaborid):
        '''Array representation of motion-energy filter.
        '''
        spatiotemporal_gabor_components = self.get_gabor_components(gaborid)
        gabor_movie_array = core.mk_spatiotemporal_gabor(*spatiotemporal_gabor_components)
        return gabor_movie_array

    def view_gabor(self, gaborid, speed=1.0, background=None):
        '''Animation of motion-energy filter
        '''
        # Get dimensions of movie
        hdim, vdim, tdim = self.gabor_hvt_size
        aspect_ratio = self.aspect_ratio
        fps = self.movie_fps

        # extract parameters for this gaborid
        gabor_params = self.filters[gaborid]

        # construct title from parameters
        title = ''
        for pdx, (pname, pval) in enumerate(sorted(self.filters[gaborid].items())):
            if pname == 'temporal_freq':
                pval = pval*(fps/tdim)
            title += '%s=%0.02f, '%(pname, pval)
            if np.mod(pdx, 3) == 0 and pdx > 0:
                title += '\n'

        return viz.plot_3dgabor(gabor_params, vdim=vdim, hdim=hdim, tdim=tdim,
                                fps=fps, aspect_ratio=aspect_ratio, background=background,
                                title=title, speed=speed)


if __name__ == '__main__':
    pass
