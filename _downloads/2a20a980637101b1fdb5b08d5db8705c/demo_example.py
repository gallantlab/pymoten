'''
=======================================
 Using the motion-energy pyramid class
=======================================

This example shows how to extract motion-energy features from a video.

First, we need to define the video we want to use. In this example, we'll use a small video. The video is 2.5 minutes in duration with a frame rate of 24fps. For the purposes of this example, we'll only use the first 200 frames.
'''

nimages = 200
stimulus_fps = 24
video_file = 'http://anwarnunez.github.io/downloads/avsnr150s24fps_tiny.mp4'

# %%
# .. raw:: html
#
#    <video width=100% height=100% preload=none muted controls>
#     <source src="https://anwarnunez.github.io/downloads/avsnr150s24fps_tiny.mp4" type="video/mp4">
#    </video>
#

# %%
# The video is RGB. So, the first step is to convert it to a luminance representation. Internally, this is achieved by converting RGB pixel values to CIE-LAB pixel values and keeping only the "L" channel. The function :func:`moten.io.video2luminance` takes care of downloading the video, converting RGB to luminance, and spatial downsampling if needed.

import moten
import matplotlib.pyplot as plt
luminance_images = moten.io.video2luminance(video_file, nimages=nimages)
nimages, vdim, hdim = luminance_images.shape

# %%
# Next we need to construct the motion-energy pyramid. To achieve this, we must provide the size of the stimulus frames in pixels (``vdim`` and ``hdim``) and also the frame rate

pyramid = moten.pyramids.MotionEnergyPyramid(stimulus_vhsize=(vdim, hdim),
                                             stimulus_fps=stimulus_fps)

print(pyramid)

# %%
# Finally, we use the method ``project_stimulus`` to compute the motion-energy features (see :meth:`moten.pyramids.MotionEnergyPyramid.project_stimulus`).

features = pyramid.project_stimulus(luminance_images)
print(features.shape)
fig, ax = plt.subplots(figsize=(12, 12))
ax.matshow(features, aspect='auto')
plt.show()
