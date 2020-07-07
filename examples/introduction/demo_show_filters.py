'''
=======================================
 Visualizing the motion energy filters
=======================================

This example demonstrates how to display a motion energy filter from the pyramid.
'''
import moten

pyramid = moten.pyramids.MotionEnergyPyramid(stimulus_vhsize=(768, 1024),
                                             stimulus_fps=24)
animation = pyramid.show_filter(1337)

# %%
# The animation should look something like this:
#
# .. raw:: html
#
#    <video width=100% height=100% preload=auto autoplay loop muted controls>
#     <source src="../../_downloads/example_moten_filter.mp4" type="video/mp4">
#    </video>

# %%
# These are the filter parameters:
from pprint import pprint
pprint(pyramid.filters[1337])


# %%
# (*Ignore this code block. It is needed to display the animation as a video on this website*)
output = '../../docs/build/html/_downloads/example_moten_filter.mp4'
fig = animation._fig
title = 'Example filter:\ndirection of motion=180, spatial fq=16cpi, temporal fq=4Hz'
fig.suptitle(title)
animation.save(output)

# sphinx_gallery_thumbnail_number = 2
import matplotlib.pyplot as plt
spatial_component = pyramid.get_filter_spatial_quadrature(1337)[1]
fig, ax = plt.subplots()
ax.matshow(spatial_component, vmin=-1, vmax=1, cmap='coolwarm')
__ = ax.set_xticks([])
__ = ax.set_yticks([])
