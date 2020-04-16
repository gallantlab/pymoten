from importlib import reload
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from moten import (pyramids,
                   utils,
                   core,
                   io,
                   )

reload(core)

# high accuracy for test
DTYPE = np.float64
temp_freq = 4.0

if 1:
    video_file = 'http://anwarnunez.github.io/downloads/avsnr150s24fps_tiny.mp4'
    nimages = 200
    small_size = (72, 128) # downsampled size (vdim, hdim) 16:9 aspect ratio
    luminance_images = io.video2luminance(video_file, size=small_size, nimages=nimages)
    #luminance_images = io.video2grey(video_file, size=small_size, nimages=nimages)
    nimages, vdim, hdim = luminance_images.shape

stimulus_fps = 24
aspect_ratio = hdim/vdim

pyramid = pyramids.MotionEnergyPyramid(stimulus_vhsize=(vdim, hdim),
                                       stimulus_fps=stimulus_fps,
                                       spatial_frequencies=[8],   # only small filters
                                       temporal_frequencies=[temp_freq], # only one temporal freq
                                       filter_temporal_width=16)
print(pyramid)

# centered projection
# hvsize=(0,0): top left
location_filters = {'BL' : [pyramid.filters_at_hvposition(0.9, 0.1)[0]],
                    'BR' : [pyramid.filters_at_hvposition(0.9, 0.9*aspect_ratio)[0]],
                    'TL' : [pyramid.filters_at_hvposition(0.1, 0.1)[0]],
                    'TR' : [pyramid.filters_at_hvposition(0.1, 0.9*aspect_ratio)[0]]}

# normal output
output = {}
for location, filters in sorted(location_filters.items()):
    output[location] = pyramid.raw_project_stimulus(luminance_images,
                                                    filters,
                                                    dtype=DTYPE)



vdim, hdim = small_size
vhalf, hhalf = int(vdim/2), int(hdim/2)
# vdim (vstart, vend)
# hdim (hstart, hend)

blank_squares = {'BL' : ((-vhalf, None), (0, hhalf)),
                 'BR' : ((-vhalf, None), (hhalf, None)),
                 'TL' : ((0, vhalf), (0, hhalf)),
                 'TR' : ((0, vhalf), (hhalf, None))}


tcenter = int(nimages/2)
controls = {}
for location, filters in sorted(location_filters.items()):
    fig, ax = plt.subplots()
    movie = luminance_images.copy()
    movie /= movie.max()
    (vstart, vend), (hstart, hend) = blank_squares[location]
    movie[:, vstart:vend, hstart:hend] = 0.0
    # insert spike in the middle
    movie[tcenter, vstart:vend, hstart:hend] = 100
    ax.matshow(movie[0])
    ax.set_title(location)

    filters = location_filters[location]
    controls[location] = pyramid.raw_project_stimulus(movie,
                                                      filters,
                                                      dtype=DTYPE,
                                                      )
    pyramid.show_filter(filters[0])

fig, ax = plt.subplots(nrows=len(location_filters), sharey=True, sharex=True)
time = np.arange(nimages)*(1./stimulus_fps)
for idx, (location, filters) in enumerate(sorted(location_filters.items())):

    ax[idx].plot(time, output[location][0], ':')
    ax[idx].plot(time, controls[location][0], '-')

    # ax[idx].plot(np.sqrt(output[location][0]**2), '-')
    # ax[idx].plot(np.sqrt(output[location][1]**2), '-')
    # ax[idx].plot(np.sqrt(controls[location][0]**2), '--')
    # ax[idx].plot(np.sqrt(controls[location][1]**2), '--')
    ax[idx].set_title(location)
    ax[idx].vlines(tcenter/stimulus_fps, -1000, 1000)
    ax[idx].set_ylim(-200, 200)
    ax[idx].grid(True)

info = (tcenter/stimulus_fps, tcenter, temp_freq)
fig.suptitle('spike at %0.02f [sec] (frame #%i) filter=%0.02f[Hz]'%info)
