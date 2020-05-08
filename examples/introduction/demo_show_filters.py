'''
===========================================
 Visualizing spatio-temporal Gabor filters
===========================================

This example demonstrate how to display a motion-energy filter from the pyramid.
'''
import moten

pyramid = moten.pyramids.MotionEnergyPyramid(stimulus_vhsize=(768, 1024),
                                             stimulus_fps=24,
                                             filter_temporal_width=16)
animation = pyramid.show_filter(1337)

# %%
# The animation should look like this:
#
# .. raw:: html
#
#    <video width=100% height=100% autoplay=True loop=True controls>
#     <source src="../../_downloads/vid.mp4" type="video/mp4">
#    </video>

# %%
# These are the filter parameters:
from pprint import pprint
filter_parameters = pyramid.filters[1337]
pprint(filter_parameters)



# %%
# (*Ignore. This code is needed to convert animation to MP4 and adding it to the gallery*)
output = '../../docs/build/html/_downloads/vid.mp4'
fig = animation._fig
fig.suptitle('Example filter:\ndirection of motion=180, spatial fq=16cpi, temporal fq=4Hz')
animation.save(output)
fig.clf()
del(fig)


# sphinx_gallery_thumbnail_number = 2
import matplotlib.pyplot as plt
spatial_component = pyramid.get_filter_spatial_quadrature(1337)[1]
fig, ax = plt.subplots()
ax.matshow(spatial_component, vmin=-1, vmax=1, cmap='coolwarm')
ax.set_xticks([])
ax.set_yticks([])
