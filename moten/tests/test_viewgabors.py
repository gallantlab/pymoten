from moten import moten

vdim = 720
hdim = 1280
fps = 24
gabor_temporal_window = int(24*(2/3))
aspect_ratio = 16/9.
# custom pyramid

# custom pyramid
pyramid_parameters = dict(temporal_frequencies=[0,2,4],
                          spatial_frequencies=[0,2,4,8,16],
                          spatial_directions=[0,45,90],
                          sf_gauss_ratio=0.6,
                          max_spatial_env=0.3,
                          gabor_spacing=3.5,
                          tf_gauss_ratio=10.,
                          max_temp_env=0.3,
                          include_edges=False)


gabor_parameters = moten.mk_moten_pyramid_params(24, int(24*(2/3)),
                                                 aspect_ratio=aspect_ratio,
                                                 **pyramid_parameters)

gabor_parameters = np.asarray(gabor_parameters)
idx = (gabor_parameters[:, 3] == 0).nonzero()[0][0]
print(gabor_parameters.shape)

gabor_param = gabor_parameters[331]

gabor = moten.mk_3d_gabor((hdim,vdim,gabor_temporal_window),
                          *gabor_param,
                          aspect_ratio=aspect_ratio)

gabor0, gabor90, tgabor0, tgabor90 = gabor

o = moten.mk_spatiotemporal_gabor(*gabor)

gabor_param = gabor_parameters[334]
ani = moten.plot_3dgabor(gabor_param, aspect_ratio=aspect_ratio)


ani = moten.plot_3dgabor(gabor_param, aspect_ratio=aspect_ratio, fps=24, tdim=16)
