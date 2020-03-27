from importlib import reload

from moten import utils as mutils, readers
from moten import moten
reload(moten)
reload(mutils)

fps = 24
pyramid = moten.GaborPyramid(24, gabor_duration=16)
gaborid = 5394
sgabor0, sgabor90, tgabor0, tgabor90 = pyramid.get_gabor_components(gaborid)

if 1:
    video_file = 'http://anwarnunez.github.io/downloads/avsnr150s24fps_tiny.mp4'
    nimages = 100
    aspect_ratio = 16/9.0
    small_size = (96, int(96*aspect_ratio))
    luminance_images = readers.video2luminance(video_file,size=small_size, nimages=nimages)

    # images
    nimages, vdim, hdim = luminance_images.shape

csin, ccos = pyramid.project_stimuli(gaborid, luminance_images)
gabor_movie = pyramid.get_3dgabor_array(gaborid)
o15 = pyramid.view_gabor(gaborid, speed=1.0)
