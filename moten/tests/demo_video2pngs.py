import os
import PIL
import tempfile


from moten import readers
from importlib import reload
reload(readers)


# This can also be a local file or an HTTP link
video_file = 'http://anwarnunez.github.io/downloads/avsnr150s24fps_tiny.mp4'
video_buffer = readers.video_buffer(video_file)

# store frames in temporary directory
tmpdir = tempfile.mkdtemp()

# Write 100 images from video as PNG
for frameidx in range(100):
    image_array = video_buffer.__next__()
    image_object = PIL.Image.fromarray(image_array)
    image_object.save(os.path.join(tmpdir, 'frame%08i.png'%frameidx))

video_buffer.close()
