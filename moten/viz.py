'''
'''
import numpy as np

from moten.core import (mk_spatiotemporal_gabor,
                        generate_3dgabor_array,
                        dotspatial_frames,
                        dotdelay_frames,
                        mk_3d_gabor,
                        )

def plot_3dgabor(gabor_params, background=None,
                 vdim=576, hdim=1024, tdim=16,
                 fps=24, aspect_ratio=1.0, title=None,
                 speed=1.0, time_padding=False):
    '''Show an animation of the 3D Gabor

    Parameters
    ----------

    '''
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    if background is not None:
        nimgs, vdim, hdim = background.shape[:3]


    # get the 3D gabor as an array
    spatiotemporal_gabor_components = mk_3d_gabor((hdim, vdim, tdim),
                                                  aspect_ratio=aspect_ratio,
                                                  **gabor_params)

    gabor_video = mk_spatiotemporal_gabor(*spatiotemporal_gabor_components)
    assert gabor_video.min() >= -1 and gabor_video.max() <= 1

    ## TODO: wrap plot_3dgabor_array
    # generate individual frames
    nframes = gabor_video.shape[-1]
    fig, ax = plt.subplots()
    images = []
    for frameidx in range(nframes):
        if background is None:
            image_view = gabor_video[..., frameidx]

        else:
            gmask = np.abs(gabor_video[...,frameidx]) > 0.01
            image_view = background[frameidx].copy()
            image_view[gmask] *= gabor_video[...,frameidx][gmask]

        im = ax.imshow(image_view, vmin=-1, vmax=1,
                       cmap='coolwarm' if background is None else 'Greys')
        framenum = ax.text(0,0,'frame #%04i/%04i (fps=%i)'%(frameidx+1, nframes, fps))
        images.append([framenum, im])


    # create animation
    repeat_delay = 1000*(fps - np.mod(nframes, fps))/fps if time_padding else 0.0
    ani = animation.ArtistAnimation(fig, images,
                                    interval=(1000*(1./fps))*speed,
                                    blit=False,
                                    repeat=True,
                                    repeat_delay=repeat_delay)
    fig.suptitle(title)
    return ani


def plot_3dgabor_array(gabor_video,
                       fps=24, aspect_ratio=1.0, title=None,
                       background=False,
                       speed=1.0, time_padding=False):
    '''
    '''
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # generate individual frames
    nframes = gabor_video.shape[-1]
    fig, ax = plt.subplots()
    images = []
    for frameidx in range(nframes):
        if background is None:
            gabor_view = gabor_video[..., frameidx]
        else:
            gabor_view = background[frameidx]*gabor_video[...,frameidx]

        im = ax.imshow(gabor_view, vmin=-1, vmax=1, cmap='coolwarm')
        framenum = ax.text(0,0,'frame #%04i/%04i (fps=%i)'%(frameidx+1, nframes, fps))
        images.append([framenum, im])


    # create animation
    repeat_delay = 1000*(fps - np.mod(nframes, fps))/fps if time_padding else 0.0
    ani = animation.ArtistAnimation(fig, images,
                                    interval=(1000*(1./fps))*speed,
                                    blit=False,
                                    repeat=True,
                                    repeat_delay=repeat_delay)
    fig.suptitle(title)
    return ani
