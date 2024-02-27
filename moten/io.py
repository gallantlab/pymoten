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

ANTIALIAS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS


def video_buffer(video_file, nimages=np.inf):
    '''Generator for a video file.

    Yields individual uint8 images from a video file.
    The video is loaded into memory one frame at a time.

    Parameters
    ----------
    video_file : str
        Full path to the video file.
        This can be a video file on disk or from a website.
    nimages : optional, int
        If specified, only `nimages` frames are loaded.

    Yields
    ------
    video_frame : uint8 3D np.ndarray, (vdim, hdim, color)
        Each ``next()`` call yields an uint8 RGB frame from video.

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
    '''Yields one frame from the greyscale video file.

    Notes
    -----
    The video is assumed to be greyscale.

    Parameters
    ----------
    video_file : str
        Full path to the video file.
        This can be a video file on disk or from a website.
    size : optional, tuple (vdim, hdim)
        The desired output image size.
        If specified, the image is scaled or shrunk to this size.
        If not specified, the original size is kept.
    nimages : optional, int
        If specified, only `nimages` frames are loaded.

    Yields
    ------
    greyscale_image : 2D np.ndarray, (vdim, hdim)
        Pixel values are in the 0-1 range.
    '''
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
                                             dtype='float32'):
    '''Generates the difference between the current frame and the previous frame.

    Notes
    -----
    The video is assumed to be greyscale.

    Parameters
    ----------
    video_file : str
        Full path to the video file.
        This can be a video file on disk or from a website.
    size : optional, tuple (vdim, hdim)
        The desired output image size.
        If specified, the image is scaled or shrunk to this size.
        If not specified, the original size is kept.
    nimages : optional, int
        If specified, only `nimages` frames are loaded.

    Yields
    ------
    greyscale_image_difference : 2D np.ndarray, (vdim, hdim)
        The difference image: (current_frame - previous_frame).
        Pixel values are in the (-1, 1) range.
    '''
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
    '''Convert the video frames to luminance images.

    Internally, this function loads one video frame into memory at a time.
    Tt converts the RGB pixel values from one frame to CIE-LAB pixel values.
    It then keeps the luminance channel only. This process is performed
    for all frames requested or until we reach the end of the video file.

    Parameters
    ----------
    video_file : str
        Full path to the video file.
        This can be a video file on disk or from a website.
    size : optional, tuple (vdim, hdim)
        The desired output image size.
        If specified, the image is scaled or shrunk to this size.
        If not specified, the original size is kept.
    nimages : optional, int
        If specified, only `nimages` frames are converted to luminance.

    Returns
    -------
    luminance_images : 3D np.ndarray, (nimages, vdim, hdim)
        Pixel values are in the 0-100 range.
    '''
    vbuffer = video_buffer(video_file, nimages=nimages)
    luminance_video = []
    for imageidx, image_rgb in iterator_func(enumerate(vbuffer),
                                             '%s.video2luminance'%__name__):
        luminance_image = imagearray2luminance(image_rgb, size=size).squeeze()
        luminance_video.append(luminance_image)
    return np.asarray(luminance_video)


def video2grey(video_file, size=None, nimages=np.inf):
    '''Convert the video frames to greyscale images.

    This function computes the mean across RGB color channels.

    Parameters
    ----------
    video_file : str
        Full path to the video file.
        This can be a video file on disk or from a website.
    size : optional, tuple (vdim, hdim)
        The desired output image size.
        If specified, the image is scaled or shrunk to this size.
        If not specified, the original size is kept.
    nimages : optional, int
        If specified, only `nimages` frames are converted to greyscale.

    Returns
    -------
    greyscale_images : 3D np.ndarray, (nimages, vdim, hdim)
        Pixel values are in the 0-1 range.
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


def imagearray2luminance(uint8arr, size=None, filter=ANTIALIAS, dtype=np.float64):
    '''Convert an array of uint8 RGB images to a luminance image

    Parameters
    ----------
    uint8arr : 4D np.ndarray, (nimages, vdim, hdim, color)
        The uint8 RGB frames.
    size : optional, tuple (vdim, hdim)
        The desired output image size.
    filter: to be passed to PIL

    Returns
    -------
    luminance_array : 3D np.ndarray, (nimages, vdim, hdim)
        The luminance image representation.
        Pixel values are in the 0-100 range.
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


def resize_image(im, size=(96,96), filter=ANTIALIAS):
    '''Resize an image and return its array representation.

    Parameters
    ----------
    im : str, np.ndarray(uint8), or PIL.Image object
        The path to the image, an image array, or a loaded PIL.Image.
    size : tuple, (vdim, hdim)
        The desired output image size.

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
    '''Load a set of RGB images and return its luminance representation.

    Parameters
    ----------
    image_files : list-like, (nimages,)
        A list of file names.
        The images should be in RGB uint8 format.
    vdim : int, optional
    hdim : int, optional
        Vertical and horizontal dimensions, respectively.
        If provided the images will be scaled to this size.

    Returns
    -------
    arr : 3D np.array (nimages, vdim, hdim)
        The luminance representation of the images.
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


def apply_mask(mask, generator):
    '''
    Parameters
    ----------
    mask : 2D np.ndarray
    generator : generator
        Yields a video frame

    Yields
    ------
    masked_image : 2D np.ndarray
        Masked image of each frame (i.e. ``original_image[mask]``)

    Examples
    --------
    >>> import moten
    >>> video_file = 'http://anwarnunez.github.io/downloads/avsnr150s24fps_tiny.mp4'
    >>> small_size = (36, 64)  # downsample to (vdim, hdim) 16:9 aspect ratio
    >>> oim = next(moten.io.generate_frame_difference_from_greyvideo(video_file, size=small_size))
    >>> mask = np.zeros(small_size, dtype=np.bool)
    >>> mask[16:, :40] = True
    >>> nim = next(moten.io.apply_mask(mask, moten.io.generate_frame_difference_from_greyvideo(video_file, size=small_size)))
    >>> np.allclose(oim[16:, :40], nim)
    '''
    assert mask.ndim == 2
    vshape = np.unique(mask.sum(0)).max()
    hshape = np.unique(mask.sum(1)).max()
    shape = (vshape, hshape)
    print('mask size:', shape)
    for im in generator:
        yield im[mask].reshape(shape)
