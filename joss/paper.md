---
title: 'pymoten: motion energy features from video using a pyramid of spatio-temporal Gabor filters'
tags:
- computer vision
- motion energy
- video processing
authors:
- name: Anwar O. Nunez-Elizalde
  affiliation: 1
- name: Fatma Deniz
  affiliation: 1
- name: Tom Dupré la Tour
  affiliation: 1
- name: Matteo Visconti di Oleggio Castello
  affiliation: 1
- name: Jack L. Gallant
  affiliation: "1,2"
affiliations:
- name: Helen Wills Neuroscience Institute, University of California, Berkeley, CA, USA
  index: 1
- name: Department of Psychology, University of California, Berkeley, CA, USA
  index: 2
date: 01 January 2021
bibliography: paper.bib

# A list of the authors of the software and their affiliations, using the correct format (see the example below).
# A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience.
# A clear Statement of Need that illustrates the research purpose of the software.
# A list of key references, including to other software addressing related needs.
# Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it.
# Acknowledgement of any financial support.
---

# Summary

[pymoten](http://gallantlab.github.io/pymoten) provides a way to measure local pixel motion in videos. To do this, motion energy features are computed from the video using a pyramid of spatio-temporal Gabor filters. This approach is inspired by the way in which neurons in the visual cortex work [@Adelson1985, @Watson1985]. The spatio-temporal Gabor filters are created at multiple spatial and temporal frequencies, directions of motion, x-y positions, and sizes. Each filter quadrature-pair is convolved with the video and their activation energy is computed for each video frame. Empirically, motion energy features provide a good basis to model brain responses to natural movies and to reconstruct natural movies from brain responses [@Nishimoto2011a, @Nishimoto2011b].

# Statement of need

[pymoten](http://gallantlab.github.io/pymoten) is a computer vision Python package for extracting motion energy features from video. The `pymoten` API allows the user to create a pyramid of spatio-temporal Gabor filters by specifying the spatial and temporal frequencies of interest along other parameters. Once created, the motion energy pyramid consists of multiple filters that tile the video frame. To extract the motion energy features from a video, the video array is passed to the pyramid in order to compute the filter activations. After processing, the pyramid returns a set of motion energy features and each captures the amount of motion at a particular speed (e.g. 10Hz), direction (e.g. rightward motion) and spatial frequency (e.g. 8 cycles-per-image) in a local patch of the video (e.g. the top left). 

`pymoten` was designed for visual neuroscience research. The original version of this package was written in MATLAB and has been used in multiple publications (REFS). This Python version provides a much simpler API and allows for the processing standard video formats (e.g. standard and wide-screen video) in addition of square images, which was a limitation of the previous version. Another related package is `skimage`, which provides an implementation of spatial Gabor filters for images but not for video. 

# Acknowledgments
This work was supported by grants from the Office of Naval Research (N00014-15-1-2861), the National Science Foundation (NSF; IIS1208203) and the National Eye Institute (EY019684 and EY022454).

# References
