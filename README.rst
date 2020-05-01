=====================
 Welcome to pymoten!
=====================

|Build Status| |Github| |Python|


What is pymoten?
================

A python package that provides a convenient way to extract motion-energy
features from video using spatio-temporal Gabor filters. The 3D Gabor filters
are created at multiple spatial and temporal frequencies, directions of motion,
x-y positions, and sizes. Each filter quadrature-pair is convolved in with the
video and their activation energy is computed for each frame. These features
provide a good basis to model brain responses to natural movies
[1]_ [2]_.


Installation
============


Clone the repo from GitHub and do the usual python install

.. code-block:: bash

   git clone https://github.com/gallantlab/pymoten.git
   cd pymoten
   sudo python setup.py install


Getting started
===============

Synthetic data example

.. code-block:: python

   import moten
   import numpy as np
   noise_movie = np.random.randn(100, 90, 180)
   nimages, vdim, hdim = noise_movie.shape
   pyramid = moten.get_default_pyramid(vhsize=(vdim, hdim), fps=24)
   moten_features = pyramid.project_stimulus(noise_movie)


Simple example

.. code-block:: python

   import moten
   video_file = 'http://anwarnunez.github.io/downloads/avsnr150s24fps_tiny.mp4'
   luminance_images = moten.io.video2luminance(video_file, nimages=100)
   nimages, vdim, hdim = luminance_images.shape
   pyramid = moten.get_default_pyramid(vhsize=(vdim, hdim), fps=24)
   moten_features = pyramid.project_stimulus(luminance_images)


.. |Build Status| image:: https://travis-ci.com/gallantlab/pymoten.svg?token=DG1xpt4Upohy9kdU6zzg&branch=master
   :target: https://travis-ci.com/gallantlab/pymoten

.. |Github| image:: https://img.shields.io/badge/github-pymoten-blue
   :target: https://github.com/gallantlab/pymoten

.. |Python| image:: https://img.shields.io/badge/python-3.7%2B-blue
   :target: https://www.python.org/downloads/release/python-370


References
==========

.. [1] Nishimoto, S., Vu, A. T., Naselaris, T., Benjamini, Y., Yu, B., &
   Gallant, J. L. (2011). Reconstructing visual experiences from brain activity
   evoked by natural movies. Current Biology, 21(19), 1641-1646.

.. [2] Nishimoto, S., & Gallant, J. L. (2011). A three-dimensional
   spatiotemporal receptive field model explains responses of area MT neurons
   to naturalistic movies. Journal of Neuroscience, 31(41), 14551-14564.