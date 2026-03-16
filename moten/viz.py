'''
'''
# Anwar O. Nunez-Elizalde (Apr, 2020)
#
# Updates:
#
#
#
import numpy as np

from moten.core import (mk_spatiotemporal_gabor,
                        generate_3dgabor_array,
                        dotspatial_frames,
                        dotdelay_frames,
                        mk_3d_gabor,
                        )

def plot_3dgabor(vhsize=(576, 1024),
                 gabor_params_dict={},
                 background=None,
                 title=None,
                 speed=1.0, time_padding=False):
    '''Show an animation of the 3D Gabor

    Parameters
    ----------

    '''
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # get the 3D gabor as an array
    spatiotemporal_gabor_components = mk_3d_gabor(vhsize,
                                                  **gabor_params_dict)

    gabor_video = mk_spatiotemporal_gabor(*spatiotemporal_gabor_components)
    assert gabor_video.min() >= -1 and gabor_video.max() <= 1

    ## TODO: wrap plot_3dgabor_array
    # generate individual frames
    fps = gabor_params_dict['stimulus_fps']
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


def plot_moten_values(feature_values, params, vmin=None, vmax=None, cmap=None, 
        marker_scale=2000, line_scale=1, lw_dict=None, 
        ax=None, figsize=None, is_overlay=False,
        bg_col=(.11, .11, .11), bg_alpha=0.1, 
        groups=None, combine_ori_fn=np.max, 
        tf_to_show=None, sf_to_show=None):
    '''Display short lines at location/orientation/scale of Gabor wavelet channels 

    A simplified visualization of Gabor motion energy features. Each color represents
    a different temporal frequency (originally, r = 0 hz, g = 2 hz, b = 4 hz). For
    other motion energy filters, different visualizations will be needed.

    Parameters
    ----------
    feature_values : 1D array
        One value for each Gabor wavelet channel to plot
    params : dict
        Parameters used to compute the motion energy features;
        easiest to use `pyramid.parameters` used to compute features
    vmin : scalar
        Minimum value for color mapping. If both vmin and vmax
        are None, defaults to -max(abs(feature_values))
    vmax : scalar
        Maximum value for color mapping. If both vmin and vmax
        are None, defaults to +max(abs(feature_values))
    cmap : string or matplotlib colormap
        colormap (for plots of single temporal frequencies only)
    marker_scale : scalar
        Scaling from values for size of dots
    line_scale : scalar
        Scaling from values for size of lines
    lw_dict : dict
        dict of widths for lines of each spatial frequency. 
        Defaults to thicker lines for lower spatial frequencies,
        but results of defaults are not optimal for all plot sizes.
    ax : axis
        Axis into which to plot. If None, new figure + axis
        are created
    figsize : tuple
        figure size passed to matplotlib figure creation
    is_overlay : bool
        whether plot is meant as an overlay for an image or movie 
        frame (if True, background options are applied, i.e.
        background is mostly transparent)
    bg_col : tuple
        background color
    bg_alpha : scalar
        background alpha for overlay plots
    groups : list of matplotlib groups
        used for animations. If passed in, function updates colors 
        in a list of matplotlib artists rather than creating new lines
    combine_ori_fn : function
        function to use to combine values for values that would occupy
        exactly the same line (i.e. motion in exactly opposite directions)
        There is no good value for this; having to combine these is a 
        shortcoming of this plotting method.
    tf_to_show : scalar
        value for which temporal frequency to show, if only one temporal
        frequency is desired. Must match one of the temporal frequencies
        of the filters.
    sf_to_show : scalar
        same as tf_to_show, but for spatial frequencies. Both of these
        selection criteria can be applied simultaneously to show
        e.g. only values of high spatial and temporal frequency Gabors

    Notes
    -----
    As they are currently computed, Gabor wavelets are not normalized by different scales and spatial
    frequencies. This means that large / low-frequency Gabor wavelets computed from an image generally
    have much larger values than small / low-frequency Gabor wavelets. Thus, for purposes of visualizing
    Gabor wavelets of different scales computed for the same image, it is currently a good idea to 
    normalize the values in different channels in some way. For example, you can take the Z score across 
    time and clip outliers (say, > 4.5)


    '''
    import six
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors, animation
    from matplotlib.collections import LineCollection

    # Handle inputs
    gmax = np.max(np.abs(feature_values))
    update = groups is not None
    if not update:
        groups = []
    if vmin is None:
        vmin = -gmax
    if vmax is None:
        vmax = gmax
    # Set colors of displayed lines
    cnorm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    gnorm = cnorm(feature_values)
    if isinstance(cmap, six.string_types):
        cmap = cm.get_cmap(cmap)
    # cols = cnorm(gnorm)
    # Simpler parameters
    xs = params['centerh'] # / params['aspect_ratio'] # Seems sketch
    ys = params['centerv']
    tfs = params['temporal_freq']
    sfs = params['spatial_freq']
    oris = params['direction']
    # `spatial_envelope` parameter is the std. dev. of the Gabor;
    # thus, a good radius for the lines to be drawn here.
    radii = params['spatial_env'] * line_scale
    # Define marker size for sf=0 (Gaussians)
    mksz = params['spatial_env'] * marker_scale    
    height = 1.0
    width = params['aspect_ratio'][0]
    # Optionally cull some values
    to_keep = np.ones(xs.shape) > 0
    if sf_to_show is not None:
        to_keep = to_keep & np.isclose(sfs, sf_to_show)
    if tf_to_show is not None:
        to_keep = to_keep & np.isclose(tfs, tf_to_show)
    gnorm = gnorm[to_keep]
    xs = xs[to_keep]
    ys = ys[to_keep]
    tfs = tfs[to_keep]
    sfs = sfs[to_keep]
    oris = oris[to_keep]
    radii = radii[to_keep]
    mksz = mksz[to_keep]
    # Define locations for lines
    xa = radii * np.sin(np.radians(oris))
    ya = radii * np.cos(np.radians(oris))
    # Define line segments
    X = np.array([xs + xa, xs - xa])
    Y = 1 - np.array([ys + ya, ys - ya])
    edges = np.array([[(X.T[i, 0], Y.T[i, 0]), (X.T[i, 1], Y.T[i, 1])] for i in range(np.max(X.shape))])
    # Prep plot
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.set_position((0, 0, 1, 1))
        show_fig = True
    else:
        fig = ax.get_figure()
        show_fig = False
    # Get indices to select specific temporal or spatial frequency Gabors
    u_tfs = np.unique(tfs)
    if len(u_tfs) > 3:
        raise ValueError(('Cannot plot more than 3 temporal frequencies in the same plot. \n'
                          'You can use `tf_to_show` to plot each individaully.'))
    u_sfs = np.unique(sfs)
    # Get linewidths for spatial frequencies
    if lw_dict is None:
        max_lw = 16
        min_lw = 2
        u_sfs = np.unique(sfs)
        lw_ = np.linspace(max_lw, min_lw, len(u_sfs))
        # Other options for mapping spatial freq. to line width:
        # or: lw_ = np.logspace(1, 3, len(sfs), base=2)
        # or: lw_ = np.unique(sfs)**-1 / np.max(np.unique(sfs)**-1)*6.
        lw_dict = dict((sf, lw) for sf, lw in zip(u_sfs, lw_))
    lws = np.zeros(xs.shape) # Nans?
    for sf in u_sfs:
        if sf==0:
            continue
        jj = sfs == sf
        lws[jj] = lw_dict[sf]

    oris_big = oris > 179.9

    for j, sf in enumerate(u_sfs):
        sfi = np.isclose(sfs, sf)
        # Deal with opposite orientations (directions of motion, if present)
        # These lines (maybe, implicilty?) assume they will be combined somehow 
        # rather than dealt with separately.
        tfis = [np.isclose(tfs, tf) for tf in u_tfs]
        ns = [np.sum(sfi & tfi) for tfi in tfis]
        n = np.min(ns)
        if (len(u_tfs) == 1) and (cmap is not None):
            # Only one temporal frequency, color map it
            cols = cmap(gnorm[sfi])
        else:
            # Attempt to map multiple tfs to R, G, B colors
            cols = np.zeros((n, 4))
            for itf, tf in enumerate(u_tfs):
                tfi = np.isclose(tfs, tf)
                if np.any(oris_big[sfi & tfi]):
                    frac_gt_180 = np.mean(oris_big[sfi & tfi])
                    if not frac_gt_180 == 0.5:
                        raise ValueError('Assumptions not met! half of orientations are not 180 + other half!')
                    to_plot = gnorm[sfi & tfi].copy()
                    # Test that oris match up
                    o1 = oris[sfi & tfi][~oris_big[sfi & tfi]]
                    o2 = oris[sfi & tfi][oris_big[sfi & tfi]]
                    assert np.allclose(o1 + 180, o2)
                    combined_data = np.vstack([to_plot[oris_big[sfi & tfi]],
                                               to_plot[~oris_big[sfi & tfi]]])
                    # Redefine to_plot to be some function of other orientations
                    to_plot = combine_ori_fn(combined_data, axis=0)
                else:
                    to_plot = gnorm[sfi & tfi].copy()
                cols[:, itf] = to_plot
            # Alpha channel
            cols[:, 3] = np.abs(cols[:, :2] - 0.5).max(axis=1) * 2
        tfi = tfis[np.argmin(ns)]
        jj = sfi & tfi
        if sf == 0:
            if update:
                groups[j].set_color(cols)
            else:
                DOTS = ax.scatter(xs[jj], (1 - ys[jj]), color=cols, s=mksz[jj])
                groups.append(DOTS)
        else:
            if update:
                groups[j].set_color(cols)
            else:
                LC = LineCollection(edges[jj], colors=cols, linewidth=lws[jj])
                groups.append(LC)
                ax.add_collection(LC)    
    # Final Setup
    plt.setp(ax, aspect='equal', xlim=(0, width), ylim=(0, height),
             xticks=(), yticks=())
    pdict = dict(color=bg_col, alpha=1)
    if is_overlay:
        pdict.update(alpha=bg_alpha)
        plt.setp(fig.patch, alpha=0.)
    plt.setp(ax.patch, **pdict)
    return groups

def plot_moten_value_movie(images, feature_values, params, figsize=(5, 5), **kwargs):
    """Make a colorized animation of motion energy features

    Parameters
    ----------
    images : array
        stack of images, (time, vdim, hdim, [c]), in a format that can be 
        displayed by plt.imshow()
    feature_values : array
        (time x features) array of motion energy feature values
    params : dict
        dictionary of parameters used to compute the motion energy features
    figsize : tuple
        Size of figure

    Other Parameters
    ----------------
    kwargs are passed to `plot_moten_values()`. Note it is often
    necessary to set vmin and vmax for consistent plotting of values
    across time.
    
    Notes
    -----
    Good tutorial, fancy extras: https://alexgude.com/blog/matplotlib-blitting-supernova/
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors, animation
    from functools import partial
    # First set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots(figsize=figsize)
    # Shape
    extent = [0, params['aspect_ratio'][0], 0, 1]
    # interval is milliseconds; convert fps to milliseconds per frame
    interval = 1000 / params['stimulus_fps'][0]
    # Setup
    if np.ndim(images) == 3:
        n_frames, y, x = images.shape
        im_shape = (y, x)
        imkw=dict(cmap='gray')
    else:
        n_frames, y, x, c = images.shape
        im_shape = (y, x, c) 
        imkw = {}
    im = ax.imshow(images[0], extent=extent, **imkw)
    grps = plot_moten_values(feature_values[0], params, ax=ax, 
                is_overlay=True, **kwargs)
    artists = (im, *grps)
    plt.close(fig.number)
    # initialization function: plot the background of each frame
    def init_func(fig, ax, artists):
        _ = plot_moten_values(np.zeros_like(feature_values[0]), params, 
            ax=ax, is_overlay=True, groups=artists[1:], **kwargs)
        im.set_array(np.zeros(im_shape))
        return artists 
    # animation function. This is called sequentially
    def update_func(i, artists, feature_values):
        _ = plot_moten_values(feature_values[i], params, 
            ax=ax, is_overlay=True, groups=artists[1:], **kwargs)
        artists[0].set_array(images[i])
        return artists
    init = partial(init_func, fig=fig, ax=ax, artists=artists)
    update = partial(update_func, artists=artists, feature_values=feature_values)
    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, 
                func=update, 
                init_func=init,
                frames=n_frames, 
                interval=interval, 
                blit=True)
    return anim