import numpy as np
from moten import io, colorspace
from importlib import reload
reload(io)

##############################
# Video example
##############################

# This can also be a local file or an HTTP link
video_file = 'http://anwarnunez.github.io/downloads/avsnr150s24fps_tiny.mp4'

# single image
##############################
video_buffer = io.video_buffer(video_file)
image_rgb = video_buffer.__next__() # load a single image
vdim, hdim, cdim = image_rgb.shape
aspect_ratio = hdim/vdim

# single images
image_luminance = io.imagearray2luminance(image_rgb, size=None)
image_luminance_resized = io.imagearray2luminance(image_rgb, size=(96, int(96*aspect_ratio)))

# process is reversible
resized_image = io.resize_image(image_rgb, size=(96, int(96*aspect_ratio)))
resized_image_luminance = io.imagearray2luminance(resized_image, size=None)
assert np.allclose(image_luminance_resized[0], resized_image_luminance[0])

# NB: skimage comparison.
import skimage.color
skimage_cielab = skimage.color.rgb2lab(image_rgb)
skimage_luminance = skimage_cielab[...,0]
# skimage is not the same...
assert np.allclose(skimage_luminance, image_luminance[0]) is False
# ...But it is highly correlated.
corr = np.corrcoef(skimage_luminance.ravel(), image_luminance[0].ravel())[0,1]
assert corr > 0.999
# Neither the observer nor the illuminant options account for this difference.
# TODO: Figure out the exact reason for this difference.


# multiple images
##############################

# load only 100 images
video_buffer = io.video_buffer(video_file, nimages=100)

images_rgb = np.asarray([image for image in video_buffer])
nimages, vdim, hdim, cdim = images_rgb.shape
aspect_ratio = hdim/vdim

images_luminance = io.imagearray2luminance(images_rgb,
                                               size=None)
images_luminance_resized = io.imagearray2luminance(images_rgb,
                                                        size=(96, int(96*aspect_ratio)))

assert np.allclose(images_luminance_resized[0], image_luminance_resized[0])

# test video2luminance generator
nimages = 256
video_buffer = io.video_buffer(video_file, nimages=nimages)
# load and downsample 1000 images
aspect_ratio = 16/9.0
small_size = (96, int(96*aspect_ratio))

luminance_images = np.asarray([io.imagearray2luminance(image, size=small_size).squeeze() \
                               for image in video_buffer])

lum = io.video2luminance(video_file, size=small_size, nimages=nimages)
assert np.allclose(luminance_images, lum)


##############################
# Example with PNG images
##############################
import PIL
from glob import glob

# Convert the first 100 video frames to PNGs
video_buffer = io.video_buffer(video_file, nimages=100)
for frameidx, video_frame in enumerate(video_buffer):
    image_object = PIL.Image.fromarray(video_frame)
    image_object.save('frame%08i.png'%frameidx)


image_files = sorted(glob('*.png'))
files_luminance = io.load_image_luminance(image_files)

files_luminance_resized = io.load_image_luminance(image_files,
                                                  vdim=96, hdim=int(96*aspect_ratio))

assert np.allclose(files_luminance, images_luminance)
assert np.allclose(files_luminance_resized, images_luminance_resized)
