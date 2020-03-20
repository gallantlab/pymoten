from moten import readers, colorspace, moten
from importlib import reload
reload(readers)
reload(moten)

# This can also be a local file or an HTTP link
video_file = 'http://anwarnunez.github.io/downloads/avsnr150s24fps_tiny.mp4'

nimages = 100
aspect_ratio = 16/9.0
small_size = (96, int(96*aspect_ratio))

luminance_images = readers.video2luminance(video_file,size=small_size, nimages=nimages)


# inferred aspect ratio
moten_features_defaults = moten.compute_filter_responses(luminance_images, 24,
                                                         dtype=np.float32)

# wrong aspect ratio
moten_features_wrong = moten.compute_filter_responses(luminance_images, 24,
                                                      aspect_ratio=1.0,
                                                      dtype=np.float32)

# exact aspect ratio
moten_features_exact = moten.compute_filter_responses(luminance_images, 24,
                                                      aspect_ratio=aspect_ratio,
                                                      dtype=np.float32)


# custom pyramid
pyramid_parameters = dict(temporal_frequencies=[0,2,4],
                          spatial_frequencies=[0,2,4,8,16],
                          spatial_directions=[0,90,180,270],
                          sf_gauss_ratio=0.6,
                          max_spatial_env=0.3,
                          gabor_spacing=3.5,
                          tf_gauss_ratio=10.,
                          max_temp_env=0.3,
                          include_edges=True)

moten_features_custom = moten.compute_filter_responses(luminance_images, 24,
                                                       aspect_ratio=aspect_ratio,
                                                       dtype=np.float32,
                                                       pyramid_parameters=pyramid_parameters)


# spatial only
##############################
reload(moten)
gabor_features_custom = moten.compute_spatial_gabor_responses(luminance_images,
                                                              dtype=np.float32,
                                                              )
