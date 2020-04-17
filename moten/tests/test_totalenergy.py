'''
'''
import time
import numpy as np
from scipy import linalg
from importlib import reload

from moten import (pyramids,
                   utils,
                   core,
                   io,
                   )
reload(io)


##############################
##############################

##############################
# video info
##############################
video_file = 'http://anwarnunez.github.io/downloads/avsnr150s24fps_tiny.mp4'
nimages = 333
small_size = (36, 64) # downsampled size (vdim, hdim) 16:9 aspect ratio


##############################
# batched frames
##############################

frame_diff_generator = io.generate_frame_difference_from_greyvideo(video_file,
                                                                   size=small_size,
                                                                   nimages=nimages)

nframes, xtx = utils.pixbypix_covariance_from_frames_generator(frame_diff_generator,
                                                               batch_size=1000,
                                                               dtype=np.float64)


##############################
# in-memory
##############################

print('Direct computation')
frame_generator = io.generate_frames_from_greyvideo(video_file,
                                                    size=small_size,
                                                    nimages=nimages)

frames = np.asarray([t for t in frame_generator], dtype=np.float64)
frames = frames.reshape(frames.shape[0], -1)

frames_diff = utils.pointwise_square(frames[1:] - frames[:-1])
xtx_direct = frames_diff.T @ frames_diff

np.testing.assert_array_almost_equal(xtx, xtx_direct)

# compute pcs
##############################

Ufull, Sfull, Vtfull = linalg.svd(frames_diff, full_matrices=False)
USfull = np.dot(Ufull, np.diag(Sfull))  # (ntimepoints, npcs)
Vfull = Vtfull.T                # (npixels, npcs)
npcs = Vfull.shape[1]
Lfull, Qfull = linalg.eigh(xtx_direct)
Qfull = Qfull[:, ::-1][:,:npcs]        # (n, npcs)
Pfull = np.dot(frames_diff, Qfull)

# they are flipped relative to each other
np.testing.assert_array_almost_equal(np.abs(Qfull), np.abs(Vfull))



##############################
# using class
##############################
import moten.core
reload(moten.core)
totalmoten = moten.core.StimulusTotalMotionEnergy(video_file,
                                                  size=small_size,
                                                  nimages=nimages,
                                                  dtype=np.float64)

totalmoten.compute_pixelby_pixel_covariance()

# check spatial PCs
totalmoten.compute_spatial_pcs(npcs=npcs)
np.testing.assert_array_almost_equal(np.abs(Qfull),
                                     np.abs(totalmoten.decomposition_spatial_pcs))

# projection
proj = np.dot(frames_diff, totalmoten.decomposition_spatial_pcs)
np.testing.assert_array_almost_equal(np.abs(Pfull), np.abs(proj))
np.testing.assert_array_almost_equal(np.abs(USfull), np.abs(proj))

# check temporal PCs
totalmoten.compute_temporal_pcs(skip_first=True)
# ntimepoints x npcs
temporal_pcs = np.asarray(totalmoten.decomposition_temporal_pcs).squeeze()
# ntimepoints x ncps

# same as projecting all frames at once
np.testing.assert_array_almost_equal(np.abs(proj),
                                     np.abs(temporal_pcs))

# same as SVD on original data
np.testing.assert_array_almost_equal(np.abs(USfull),
                                     np.abs(temporal_pcs))


##############################
# batched testing
##############################
if 0:
    batch_size = 100
    idxframe = 0

    first_frame = frame_diff_generator.__next__()
    vdim, hdim = first_frame.shape
    npixels = vdim*hdim

    framediff_buffer = np.zeros((batch_size, npixels), dtype=np.float64)
    XTX = np.zeros((npixels, npixels), dtype=np.float64)

    import time
    start_time = time.time()

    RUN = True
    while RUN:
        framediff_buffer *= 0.0             # clear buffer
        try:
            for batch_frame_idx in range(batch_size):
                framediff_buffer[batch_frame_idx] = frame_diff_generator.__next__().reshape(1, -1)
        except StopIteration:
            print('Finished buffer')
            RUN = False
        finally:
            idxframe += batch_size
            XTX = framediff_buffer.T @ framediff_buffer

    print(time.time() - start_time)

    from scipy import linalg
    L, Q = linalg.eigh(XTX)
    # flip to descending order
    V = Q[:,::-1]
    L = L[::-1]


##############################
# slow, light memory
##############################

frame_generator = io.generate_frames_from_greyvideo(video_file,
                                                    size=small_size,
                                                    nimages=nimages)

previous_frame = frame_generator.__next__()
previous_frame = previous_frame.reshape(1, -1).astype(np.float64)
npixels = previous_frame.shape[1]
# XTX = previous_frame.T @ previous_frame # DO NOT INCLUDE FIRST FRAME
XTX = np.zeros((npixels, npixels), dtype=np.float64)

start_time = time.time()
for idx, current_frame in enumerate(frame_generator):
    current_frame = np.asarray(current_frame.reshape(1, -1), dtype=np.float64)
    frame_diff = utils.pointwise_square(current_frame - previous_frame)
    XTX += frame_diff.T @ frame_diff
    previous_frame = current_frame
print(time.time() - start_time)

# both are equal
np.testing.assert_array_almost_equal(XTX, xtx_direct)
