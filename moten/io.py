'''
'''
# Anwar O. Nunez-Elizalde (Jan, 2016)
#
# Updates:
#  Anwar O. Nunez-Elizalde (Apr, 2020)
#
#
from PIL import Image
import numpy as np

from moten.utils import iterator_func
from moten.colorspace import rgb2lab

def video_buffer(video_file, nimages=np.inf):
    '''Generator for a video file.

    Yields individual uint8 images from a video file.

    Parameters
    ----------
    video_file : str
        Full path to the video file

    Returns
    -------
    vbuff : generator
        Each ``__next()__`` call yields an RGB frame from video.

    Example
    -------
    >>> video_file = 'http://anwarnunez.github.io/downloads/avsnr150s24fps_tiny.mp4'
    >>> image_buffer = video_buffer(video_file, nimages=50)
    >>> movie = np.asarray([frame for frame in image_buffer])
    >>> print(movie.shape) # (nimages, vdim, hdim, color)
    (50, 144, 256, 3)
    '''
    import cv2
    cap = cv2.VideoCapture(video_file)
    frameidx = 0
    while True:
        if frameidx >= nimages:
            break
        flag, im = cap.read()
        frameidx += 1
        if flag:
            yield im[...,::-1] # flip to RGB
        else:
            break


def generate_frames_from_greyvideo(video_file, size=None, nimages=np.inf):
    vbuffer = video_buffer(video_file, nimages=nimages)
    grey_video = []
    for imageidx, image_rgb in enumerate(vbuffer):
        if size is not None:
            image_rgb = resize_image(image_rgb, size=size)
        grey_image = image_rgb[...,0]/255.0
        yield grey_image


def generate_frame_difference_from_greyvideo(video_file,
                                             size=None,
                                             nimages=np.inf,
                                             dtype=np.float32):
    vbuffer = video_buffer(video_file, nimages=nimages)
    previous_frame = 0
    for frame_index, image_rgb in enumerate(vbuffer):
        if size is not None:
            image_rgb = resize_image(image_rgb, size=size)
        current_image = np.asarray(image_rgb[...,0]/255.0, dtype=dtype)
        frame_difference = current_image - previous_frame
        previous_frame = current_image
        yield frame_difference


def video2luminance(video_file, size=None, nimages=np.inf):
    '''
    size (optional) : tuple, (vdim, hdim)
        The desired output image size

    '''
    vbuffer = video_buffer(video_file, nimages=nimages)
    luminance_video = []
    for imageidx, image_rgb in iterator_func(enumerate(vbuffer),
                                             '%s.video2luminance'%__name__):
        luminance_image = imagearray2luminance(image_rgb, size=size).squeeze()
        luminance_video.append(luminance_image)
    return np.asarray(luminance_video)


def video2grey(video_file, size=None, nimages=np.inf):
    '''
    size (optional) : tuple, (vdim, hdim)
        The desired output image size

    '''
    vbuffer = video_buffer(video_file, nimages=nimages)
    grey_video = []
    for imageidx, image_rgb in iterator_func(enumerate(vbuffer),
                                             '%s.video2grey'%__name__):
        if size is not None:
            image_rgb = resize_image(image_rgb, size=size)
        grey_image = image_rgb.mean(-1)/255.0
        grey_video.append(grey_image)
    return np.asarray(grey_video)


def imagearray2luminance(uint8arr, size=None, filter=Image.ANTIALIAS, dtype=np.float64):
    '''Convert an array of uint8 RGB images to a luminance image

    Parameters
    ----------
    uint8arr : 4D np.ndarray (n, vdim, hdim, rgb)
        The uint8 RGB frames.

    size (optional) : tuple, (vdim, hdim)
        The desired output image size

    filter: to be passed to PIL

    Returns
    -------
    luminance_array = 3D np.ndarray (n, vdim, hdim)
        The luminance image representation (0-100 range)
    '''
    from scipy import misc
    from moten.colorspace import rgb2lab

    if uint8arr.ndim == 3:
        # handle single image case
        uint8arr = np.asarray([uint8arr])

    nimages, vdim, hdim, cdim = uint8arr.shape
    outshape = (nimages, vdim, hdim) if size is None \
        else (nimages, size[0], size[1])

    luminance = np.zeros(outshape, dtype=dtype)
    for imdx in range(nimages):
        im = uint8arr[imdx]
        if size is not None:
            im = Image.fromarray(im)
            im = resize_image(im, size=size, filter=filter)
        im = rgb2lab(im/255.)[...,0]
        luminance[imdx] = im
    return luminance


def resize_image(im, size=(96,96), filter=Image.ANTIALIAS):
    '''Resize an image and return its array representation

    Parameters
    ----------
    im : str, np.ndarray(uint8), or PIL.Image object
        The path to the image, an image array, or a loaded PIL.Image
    size : tuple, (vdim, hdim)
        The desired output image size

    Returns
    -------
    arr : uint8 np.array, (vdim, hdim, 3)
        The resized image array
    '''
    if isinstance(im, str):
        im = Image.open(im)
    elif isinstance(im, np.ndarray):
        im = Image.fromarray(im)
    im.load()

    # flip to PIL.Image convention
    size = size[::-1]
    try:
        im = im._new(im.im.stretch(size, filter))
    except AttributeError:
        # PIL 4.0.0 The stretch function on the core image object has been removed.
        # This used to be for enlarging the image, but has been aliased to resize recently.
        im = im._new(im.im.resize(size, filter))
    im = np.asarray(im)
    return im


def load_image_luminance(image_files, hdim=None, vdim=None):
    '''Load a set of RGB images and return its luminance representation

    Parameters
    ----------
    image_files : list-like, (n,)
        A list of file names.
        The images should be in RGB uint8 format
    vdim, hdim : int, optional
        Vertical and horizontal dimensions, respectively.
        If provided the images will be scaled to this size.

    Returns
    -------
    arr : 3D np.array (n,vdim,hdim)
        The luminance representation of the images
    '''


    if (hdim and vdim):
        loader = lambda stim,sz: resize_image(stim,sz)
    else:
        loader = lambda stim,sz: np.asarray(stim)

    stimuli = []

    for fdx, fl in iterator_func(enumerate(image_files),
                                 "load_image_luminance",
                                 total=len(image_files)):
        stimulus = Image.open(fl)
        stimulus = loader(stimulus,(vdim,hdim))
        stimulus = rgb2lab(stimulus/255.)[...,0]
        stimuli.append(stimulus)
    return np.asarray(stimuli)
