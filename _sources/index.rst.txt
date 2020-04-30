=====================
 Welcome to pymoten!
=====================

https://github.com/gallantlab/pymoten


What is pymoten?
================

A python package that provides a convenient way to extract motion-energy features from video using spatio-temporal Gabor filters. The 3D Gabor filters are created at multiple spatial and temporal frequencies, directions of motion, x-y positions, and sizes. Each filter quadrature-pair is convolved in with the video and their activation energy is computed for each frame. These features provide a good basis to model brain responses to natural movies (Nishimoto, et al., 2011a,b).


Installation
============


Clone the repo from GitHub and do the usual python install

::

   git clone https://github.com/gallantlab/pymoten.git
   cd pymoten
   sudo python setup.py install


Getting started
===============

Synthetic data example

::

   >>> import moten
   >>> noise_movie = np.random.randn(100, 90, 180)
   >>> nimages, vdim, hdim = noise_movie.shape
   >>> pyramid = moten.get_default_pyramid(vhsize=(vdim, hdim), fps=24)
   >>> moten_features = pyramid.project_stimulus(noise_movie)


Simple example

::

   >>> import moten
   >>> video_file = 'http://anwarnunez.github.io/downloads/avsnr150s24fps_tiny.mp4'
   >>> luminance_images = moten.io.video2luminance(video_file, nimages=100)
   >>> nimages, vdim, hdim = luminance_images.shape
   >>> pyramid = moten.get_default_pyramid(vhsize=(vdim, hdim), fps=24)
   >>> moten_features = pyramid.project_stimulus(luminance_images)


Examples
========

.. toctree::
    :maxdepth: 3

    auto_examples/index.rst


Learn more
==========

[General docs]

.. toctree::
   :maxdepth: 2
   :caption: Contents:



API Reference
=============

.. autosummary:: moten
   :toctree: modules


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
