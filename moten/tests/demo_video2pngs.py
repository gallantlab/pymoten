import PIL

from moten import readers
from importlib import reload
reload(readers)

# This can also be an HTTP link
video_buffer = readers.video_buffer('avsnr150s24fps_sd.mp4')

# Write 100 images from video as PNG
for frameidx in range(100):
    image_array = video_buffer.__next__()
    image_object = PIL.Image.fromarray(image_array)
    image_object.save('frame%04i.png'%frameidx)

video_buffer.close()
