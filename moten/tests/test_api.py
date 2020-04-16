import numpy as np
import matplotlib

from moten import (pyramids,
                   utils,
                   core,
                   io,
                   )


if 1:
    video_file = 'http://anwarnunez.github.io/downloads/avsnr150s24fps_tiny.mp4'
    nimages = 200
    small_size = (72, 128) # downsampled size (vdim, hdim) 16:9 aspect ratio
    luminance_images = io.video2luminance(video_file, size=small_size, nimages=nimages)
    #luminance_images = io.video2grey(video_file, size=small_size, nimages=nimages)
    nimages, vdim, hdim = luminance_images.shape

stimulus_fps = 24

# need high accuracy for tests
DTYPE = np.float64

##############################
# static gabor pyramid
##############################
static_pyramid = pyramids.StimulusStaticGaborPyramid(luminance_images,
                                                     spatial_frequencies=(0,1,2,4),
                                                     spatial_phase_offset=0.0)

print(static_pyramid)
static_pyramid.view.show_filter(5)


static_pyramid_90 = pyramids.StimulusStaticGaborPyramid(luminance_images,
                                                     spatial_frequencies=(0,1,2,4),
                                                     spatial_phase_offset=90.0)

print(static_pyramid_90)
static_pyramid_90.view.show_filter(5)

if 0:
    if static_pyramid.nfilters > 25:
        idxs = np.arange(static_pyramid.nfilters)
        # idx = np.sort(np.random.permutation(static_pyramid.nfilters)
        idxs = idxs[-25:]
        for idx in idxs:
            static_pyramid.view.show_filter(idx)
    else:
        for idx in range(static_pyramid.nfilters):
            static_pyramid.view.show_filter(idx)


##############################
# motion-energy pyramid
##############################
pyramid = pyramids.MotionEnergyPyramid(stimulus_vhsize=(vdim, hdim),
                                       stimulus_fps=stimulus_fps,
                                       spatial_frequencies=(0,1,2,4),
                                       filter_temporal_width=16)
print(pyramid)

# project stimulus
output = pyramid.project_stimulus(luminance_images, dtype=DTYPE)
outsin, outcos = pyramid.raw_project_stimulus(luminance_images, dtype=DTYPE)

# stand-alone function
##############################
filter_definitions = pyramid.filters
func_proj_3darray = core.project_stimulus(luminance_images,
                                          filter_definitions,
                                          dtype=DTYPE)
assert np.allclose(func_proj_3darray, output)

func_proj_2darray = core.project_stimulus(luminance_images.reshape(luminance_images.shape[0], -1),
                                          filter_definitions,
                                          vhsize=(vdim, hdim), dtype=DTYPE)

assert np.allclose(func_proj_2darray, output)


##################################################
# stimulus-specific motion-energy pyramid
##################################################
stim_pyramid = pyramids.StimulusMotionEnergy(luminance_images,
                                             stimulus_fps,
                                             spatial_frequencies=(0,1,2,4),
                                             filter_temporal_width=16)

# the pyramid is an attribute
print(stim_pyramid.view)

# default projection
filter_responses = stim_pyramid.project(dtype=DTYPE)
print(stim_pyramid.view.filters[10])
stim_pyramid.view.show_filter(10)

# test against simple pyramid
assert np.allclose(filter_responses, output)

# centered projection
hvcentered_filters, hvcentered_responses = stim_pyramid.project_at_vhposition(0.33, 1.11, dtype=DTYPE)
example_filter = hvcentered_filters[10]
stim_pyramid.view.show_filter(example_filter)

# modify filter spatial frequency to 8cpi and
# direction of motion to 60 degrees
modified_filter = example_filter.copy()
modified_filter['spatial_freq'] = 8.0
modified_filter['direction'] = 60.0
stim_pyramid.view.show_filter(modified_filter)

# raw responses from filter quadrature-pair
responses_sin, responses_cos = stim_pyramid.raw_projection(dtype=DTYPE)
responses_manual = utils.log_compress(utils.sqrt_sum_squares(responses_sin, responses_cos))
assert np.allclose(filter_responses, responses_manual)

# test against simple pyramid
assert np.allclose(outsin, responses_sin)
assert np.allclose(outcos, responses_cos)


# centered quadrature centered responses
hvresponses_sin, hvresponses_cos = stim_pyramid.raw_projection(hvcentered_filters, dtype=DTYPE)
responses_manualhv = utils.log_compress(utils.sqrt_sum_squares(hvresponses_sin, hvresponses_cos))
assert np.allclose(hvcentered_responses, responses_manualhv)

# project specific stimuli
filter_stimulus_responses = stim_pyramid.project_stimulus(luminance_images,
                                                     dtype=DTYPE)
assert np.allclose(filter_stimulus_responses, filter_responses)


##############################
# stimulus batches
##############################

# project subset of original stimuli
first_frame, last_frame = 100, 110
filter_stimulus_responses = stim_pyramid.project_stimulus(
    luminance_images[first_frame:last_frame], dtype=DTYPE)

# these differ because of convolution edge effects
assert not np.allclose(filter_stimulus_responses,
                       filter_responses[first_frame:last_frame])

# we have to include a window in order to avoid edge effects.
# This window is determined by the FOV of the motion-energy filter.
# which is stored in the pyramid definition
filter_width = stim_pyramid.view.definition.filter_temporal_width
window = int(filter_width/2)
windowed_filter_stimulus_responses = stim_pyramid.project_stimulus(
    luminance_images[first_frame - window:last_frame + window],
    dtype=DTYPE)

# Now they are exactly the same
assert np.allclose(windowed_filter_stimulus_responses[window:-window],
                   filter_responses[first_frame:last_frame])
