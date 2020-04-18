import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from moten import (pyramids,
                   utils,
                   core,
                   io,
                   )

DTYPE = np.float64

##############################
# preliminaries
##############################
video_file = 'http://anwarnunez.github.io/downloads/avsnr150s24fps_tiny.mp4'
nimages = 200
small_size = (72, 128) # downsampled size (vdim, hdim) 16:9 aspect ratio
luminance_images = io.video2luminance(video_file, size=small_size, nimages=nimages)
nimages, vdim, hdim = luminance_images.shape
stimulus_fps = 24
aspect_ratio = hdim/vdim

pyramid = pyramids.MotionEnergyPyramid(stimulus_vhsize=(vdim, hdim),
                                       stimulus_fps=stimulus_fps,
                                       spatial_frequencies=[8],   # only small filters
                                       temporal_frequencies=[12], # only one temporal freq
                                       filter_temporal_width=16)
print(pyramid)

# centered projection
# hvsize=(0,0): top left
location_filters = {'BL' : [pyramid.filters_at_hvposition(0.9, 0.1)[0]],
                    'BR' : [pyramid.filters_at_hvposition(0.9, 0.9*aspect_ratio)[0]],
                    'TL' : [pyramid.filters_at_hvposition(0.1, 0.1)[0]],
                    'TR' : [pyramid.filters_at_hvposition(0.1, 0.9*aspect_ratio)[0]]}

# store pyramid results
output = {}
controls = {}


def test_hvorientation():
    '''Check images and filters are oriented the same
    '''
    # normal output
    for location, filters in sorted(location_filters.items()):
        output[location] = pyramid.project_stimulus(luminance_images,
                                                    filters,
                                                    dtype=DTYPE,
                                                    output_nonlinearity=lambda x: x)

    vdim, hdim = small_size
    vhalf, hhalf = int(vdim/2), int(hdim/2)

    blank_squares = {'BL' : ((-vhalf, None), (0, hhalf)),
                     'BR' : ((-vhalf, None), (hhalf, None)),
                     'TL' : ((0, vhalf), (0, hhalf)),
                     'TR' : ((0, vhalf), (hhalf, None))}

    for location, filters in sorted(location_filters.items()):
        fig, ax = plt.subplots()
        movie = luminance_images.copy()
        (vstart, vend), (hstart, hend) = blank_squares[location]
        movie[:, vstart:vend, hstart:hend] = 0.0
        ax.matshow(movie[0])
        ax.set_title(location)

        filters = location_filters[location]
        controls[location] = pyramid.project_stimulus(movie,
                                                      filters,
                                                      dtype=DTYPE,
                                                      output_nonlinearity=lambda x: x)
        pyramid.show_filter(filters[0])

        # check output is zeros
        assert np.allclose(controls[location], 0)

def test_smoke_test():
    fig, ax = plt.subplots(nrows=len(location_filters))
    for idx, (location, filters) in enumerate(sorted(location_filters.items())):
        ax[idx].plot(output[location])
        ax[idx].plot(controls[location], '--')
        ax[idx].set_title(location)

        assert np.allclose(controls[location], 0)
        assert not np.allclose(output[location], 0)
