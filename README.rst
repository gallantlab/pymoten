=====================
 Welcome to pymoten!
=====================

|Zenodo| |Github| |codecov| |Python| 


What is pymoten?
================

``pymoten`` is a python package that provides a convenient way to extract motion energy
features from video using a pyramid of spatio-temporal Gabor filters [1]_ [2]_. The filters
are created at multiple spatial and temporal frequencies, directions of motion,
x-y positions, and sizes. Each filter quadrature-pair is convolved with the
video and their activation energy is computed for each frame. These features
provide a good basis to model brain responses to natural movies
[3]_ [4]_.


Installation
============


Clone the repo from GitHub and do the usual python install

.. code-block:: bash

   git clone https://github.com/gallantlab/pymoten.git
   cd pymoten
   sudo python setup.py install

Or with pip:

.. code-block:: bash

   pip install pymoten
   

Getting started
===============

Example using synthetic data

.. code-block:: python

   import moten
   import numpy as np

   # Generate synthetic data
   nimages, vdim, hdim = (100, 90, 180)
   noise_movie = np.random.randn(nimages, vdim, hdim)

   # Create a pyramid of spatio-temporal gabor filters
   pyramid = moten.get_default_pyramid(vhsize=(vdim, hdim), fps=24)

   # Compute motion energy features
   moten_features = pyramid.project_stimulus(noise_movie)


Simple example using a video file

.. code-block:: python

   import moten

   # Stream and convert the RGB video into a sequence of luminance images
   video_file = 'http://anwarnunez.github.io/downloads/avsnr150s24fps_tiny.mp4'
   luminance_images = moten.io.video2luminance(video_file, nimages=100)

   # Create a pyramid of spatio-temporal gabor filters
   nimages, vdim, hdim = luminance_images.shape
   pyramid = moten.get_default_pyramid(vhsize=(vdim, hdim), fps=24)

   # Compute motion energy features
   moten_features = pyramid.project_stimulus(luminance_images)


.. |Build Status| image:: https://travis-ci.org/gallantlab/pymoten.svg?branch=main
    :target: https://travis-ci.org/gallantlab/pymoten
    
.. |Github| image:: https://img.shields.io/badge/github-pymoten-blue
   :target: https://github.com/gallantlab/pymoten

.. |Python| image:: https://img.shields.io/badge/python-3.7%2B-blue
   :target: https://www.python.org/downloads/release/python-370

.. |Codecov| image:: https://codecov.io/gh/gallantlab/pymoten/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/gallantlab/pymoten

.. |Zenodo| image:: https://zenodo.org/badge/240954590.svg
   :target: https://zenodo.org/badge/latestdoi/240954590


Cite as
=======
Nunez-Elizalde AO, Deniz F, Dupré la Tour T, Visconti di Oleggio Castello M, and Gallant JL (2021). pymoten: scientific python package for computing motion energy features from video. Zenodo. https://doi.org/10.5281/zenodo.6349625

References
==========


.. [1] Adelson, E. H., & Bergen, J. R. (1985). Spatiotemporal energy models for the perception of motion. 
   Journal of the Optical Society of America A, 2(2), 284-299.

.. [2] Watson, A. B., & Ahumada, A. J. (1985). Model of human visual-motion sensing. 
   Journal of the Optical Society of America A, 2(2), 322–342. 

.. [3] Nishimoto, S., & Gallant, J. L. (2011). A three-dimensional
   spatiotemporal receptive field model explains responses of area MT neurons
   to naturalistic movies. Journal of Neuroscience, 31(41), 14551-14564.

.. [4] Nishimoto, S., Vu, A. T., Naselaris, T., Benjamini, Y., Yu, B., &
   Gallant, J. L. (2011). Reconstructing visual experiences from brain activity
   evoked by natural movies. Current Biology, 21(19), 1641-1646.

=======

A MATLAB implementation can be found `here <https://github.com/gallantlab/motion_energy_matlab/>`_.
