import numpy as np
from moten import io, colorspace
from importlib import reload
reload(io)

##############################
# Video example
##############################

# This can also be a local file or an HTTP link
video_file = 'http://anwarnunez.github.io/downloads/avsnr150s24fps_tiny.mp4'
video_buffer = io.video_buffer(video_file)
image_rgb = video_buffer.__next__() # load a single image
vdim, hdim, cdim = image_rgb.shape
aspect_ratio = hdim/vdim
small_size = (96, int(96*aspect_ratio))

##############################
# single image
##############################

def test_video_buffer_image():
    video_buffer = io.video_buffer(video_file)
    image_rgb = video_buffer.__next__() # load a single image
    assert image_rgb.ndim == 3
    assert image_rgb.dtype == np.uint8


def test_imagearray2luminance_image():
    # single images
    image_luminance = io.imagearray2luminance(image_rgb, size=None)


def test_imagearray2luminance_resizing_image():
    # single images
    image_luminance_resized = io.imagearray2luminance(image_rgb, size=small_size)
    vdim, hdim = small_size
    assert (1, vdim, hdim) == image_luminance_resized.shape


def test_imagearray2luminance_reversible():
    # process is reversible
    resized_image = io.resize_image(image_rgb, size=small_size)
    assert resized_image.ndim == 3
    resized_image_luminance = io.imagearray2luminance(resized_image, size=None)
    nimages, vdim, hdim = resized_image_luminance.shape
    assert (vdim, hdim) == small_size
    assert resized_image_luminance.ndim == 3


def test_skimage_compare():
    import skimage.color
    skimage_luminance = skimage.color.rgb2lab(image_rgb)[..., 0]
    pymoten_luminance = io.imagearray2luminance(image_rgb, size=None)[0]
    np.testing.assert_array_almost_equal(skimage_luminance, pymoten_luminance,
                                         decimal=5)

# multiple images
##############################
def test_video_buffer_smk():
    # load only 100 images
    video_buffer = io.video_buffer(video_file, nimages=100)
    images_rgb = np.asarray([image for image in video_buffer])
    images_luminance = io.imagearray2luminance(images_rgb,
                                               size=None)

    images_luminance_resized = io.imagearray2luminance(images_rgb, size=small_size)


def test_video2luminance():
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
import os
import PIL
import tempfile
from glob import glob

def test_pngs():
    images_luminance = io.video2luminance(video_file, nimages=100)
    images_luminance_resized = io.video2luminance(video_file, size=small_size, nimages=100)

    # Convert the first 100 video frames to PNGs
    video_buffer = io.video_buffer(video_file, nimages=100)

    # store frames in temporary directory
    tmpdir = tempfile.mkdtemp()

    for frameidx, video_frame in enumerate(video_buffer):
        image_object = PIL.Image.fromarray(video_frame)
        image_object.save(os.path.join(tmpdir, 'frame%08i.png'%frameidx))

    image_files = sorted(glob(os.path.join(tmpdir, '*.png')))
    files_luminance = io.load_image_luminance(image_files)

    files_luminance_resized = io.load_image_luminance(image_files,
                                                      vdim=96, hdim=int(96*aspect_ratio))

    assert np.allclose(files_luminance, images_luminance)
    assert np.allclose(files_luminance_resized, images_luminance_resized)
