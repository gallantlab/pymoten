'''
==========================================================
 Computing motion-energy features from batches of stimuli
==========================================================

This example shows how to extract motion-energy features from batches of a video.

When the stimulus is very high-resolution (e.g. 4K) or is several hours long, it might not be possible to fit load the stimulus into memory. In such situations, it is useful to load a small number of video frames and extract motion-energy features from that subset of frames alone. In order to do this properly, one must avoid edge effects. In this example we show how to do that.
'''


# %%
# Features from stimulus
# ======================
#
# First, we specify the stimulus we want to load.

import moten
import numpy as np
import matplotlib.pyplot as plt
stimulus_fps = 24
video_file = 'http://anwarnunez.github.io/downloads/avsnr150s24fps_tiny.mp4'

# %%
# Load the first 300 images and spatially downsample the video.
small_vhsize = (72, 128)        # height x width
luminance_images = moten.io.video2luminance(video_file, size=small_vhsize, nimages=300)
nimages, vdim, hdim = luminance_images.shape
print(vdim, hdim)

fig, ax = plt.subplots()
ax.matshow(luminance_images[200], vmin=0, vmax=100, cmap='inferno')
ax.set_xticks([])
ax.set_yticks([])

# %%
# Next we construct the pyramid and extract the motion-energy features from the full stimulus.

pyramid = moten.pyramids.MotionEnergyPyramid(stimulus_vhsize=(vdim, hdim),
                                             stimulus_fps=stimulus_fps,
                                             filter_temporal_width=16)

moten_features = pyramid.project_stimulus(luminance_images)
print(moten_features.shape)


# %%
# Features from stimulus batches
# ==============================
#
# Next, instead of computing the features from the full stimulus, we compute them from separate but continous stimulus chunks. These stimulus chunks are the stimulus batches.
#
# We have to include some padding to the batches in order to avoid convolution edge effects. The padding is determined by the temporal width of the motion-energy filter. By default, the temporal width is 2/3 of the stimulus frame rate (`int(fps*(2/3))`). This parameter can be specified when instantating a pyramid by passing e.g. ``filter_temporal_width=16``. Once the pyramid is defined, the parameter can also be accessed from the ``pyramid.definition`` dictionary.

filter_temporal_width = pyramid.definition['filter_temporal_width']

# %%
# Finally, we define the padding window as half the temporal filter width.

window = int(np.ceil((filter_temporal_width/2)))
print(filter_temporal_width, window)

# %%
# Now we are ready to extract motion-energy features in batches:

nbatches = 5
batch_size = int(np.ceil(nimages/nbatches))
batched_data = []
for bdx in range(nbatches):
    start_frame, end_frame = batch_size*bdx, batch_size*(bdx + 1)
    print('Batch %i/%i [%i:%i]'%(bdx+1, nbatches, start_frame, end_frame))

    # Padding
    batch_start = max(start_frame - window, 0)
    batch_end = end_frame + window
    stimulus_batch = luminance_images[batch_start:batch_end]
    batched_responses = pyramid.project_stimulus(stimulus_batch)

    # Trim edges
    if bdx == 0:
        batched_responses = batched_responses[:-window]
    elif bdx + 1 == nbatches:
        batched_responses = batched_responses[window:]
    else:
        batched_responses = batched_responses[window:-window]
    batched_data.append(batched_responses)

batched_data = np.vstack(batched_data)

# %%
# They are exactly the same.
assert np.allclose(moten_features, batched_data)

# %%
# In this example, the stimulus (``luminance_images``) is already in memory and so batching does not provide any benefits. However, there are situations in which the stimulus cannot be loaded all at once. In such situations, batching is necessary. One can modify the code above and write a function to load a subset of frames that can fit into memory (e.g. ``stimulus_batch = load_my_video_frames_batch(`my_stimulus_video_file.avi`, batch_start, batch_end)``).
