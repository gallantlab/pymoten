import numpy as np

import matplotlib
matplotlib.interactive(False)

from moten import (io,
                   api,
                   utils,
                   )

if 1:
    video_file = 'http://anwarnunez.github.io/downloads/avsnr150s24fps_tiny.mp4'
    nimages = 100
    small_size = (72, 128) # downsampled size (vdim, hdim) 16:9 aspect ratio
    luminance_images = io.video2luminance(video_file, size=small_size, nimages=nimages)
    #luminance_images = io.video2grey(video_file, size=small_size, nimages=nimages)
    nimages, vdim, hdim = luminance_images.shape

stimulus_fps = 24

# need high accuracy for tests
DTYPE = np.float64

# pyramid viewer
##############################
pyramid_view = api.PyramidConstructor(stimulus_hvt_fov=(hdim, vdim, stimulus_fps),
                                      spatial_frequencies=(0,1,2,4))
print(pyramid_view)

# stimulus pyramid
##############################
pyramid = api.StimulusMotionEnergy(luminance_images,
                                   stimulus_fps,
                                   spatial_frequencies=(0,1,2,4))
# pyramid constructor is an attribute
print(pyramid.view)

# default projection
filter_responses = pyramid.project(dtype=DTYPE)
print(pyramid.view.filters[10])
pyramid.view.show_filter(10)

# centered projection
hvcentered_filters, hvcentered_responses = pyramid.project_at_hvposition(1.11, 0.33, dtype=DTYPE)
example_filter = hvcentered_filters[10]
pyramid.view.show_filter(example_filter)

# modify filter spatial frequency to 8cpi and
# direction of motion to 60 degrees
modified_filter = example_filter.copy()
modified_filter['spatial_freq'] = 8.0
modified_filter['direction'] = 60.0
pyramid.view.show_filter(modified_filter)

# raw responses from filter quadrature-pair
responses_sin, responses_cos = pyramid.raw_projection(dtype=DTYPE)
responses_manual = utils.log_compress(utils.sqrt_sum_squares(responses_sin, responses_cos))
assert np.allclose(filter_responses, responses_manual)

# centered quadrature centered responses
hvresponses_sin, hvresponses_cos = pyramid.raw_projection(hvcentered_filters, dtype=DTYPE)
responses_manualhv = utils.log_compress(utils.sqrt_sum_squares(hvresponses_sin, hvresponses_cos))
assert np.allclose(hvcentered_responses, responses_manualhv)
