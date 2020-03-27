import numpy as np
from importlib import reload

from moten import io
import moten
reload(moten)

fps = 24
pyramid = moten.GaborPyramid(24, gabor_duration=16)
gaborid = 5394
sgabor0, sgabor90, tgabor0, tgabor90 = pyramid.get_gabor_components(gaborid)

if 1:
    video_file = 'http://anwarnunez.github.io/downloads/avsnr150s24fps_tiny.mp4'
    nimages = 100
    aspect_ratio = 16/9.0
    small_size = (96, int(96*aspect_ratio))
    rgb_images = np.asarray([t for t in io.video_buffer(video_file, nimages=nimages)])
    luminance_images = io.video2luminance(video_file,size=small_size, nimages=nimages)

    # images
    nimages, vdim, hdim = luminance_images.shape

csin, ccos = pyramid.project_stimuli(luminance_images)

gabor_movie = pyramid.get_3dgabor_array(gaborid)
o15 = pyramid.view_gabor(gaborid,
                         speed=5.01,
                         # background=rgb_images.astype(np.float64)/255.,
                         # background=luminance_images[70:]/100.,
                         )


if 0:
    from aone.stimuli import shorts_api as sapi
    frames = sapi.data2figures(gabor_movie, aspect_ratio=1., vmin=-1, vmax=1, cmap='coolwarm')
    sapi.arr2video('/tmp/gabor_%ifps.avi'%fps, frames, fps=fps, isrgb=True)
