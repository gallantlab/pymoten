# pymoten

```python
>>> import moten
>>> video_file = 'http://anwarnunez.github.io/downloads/avsnr150s24fps_tiny.mp4'
>>> luminance_images = moten.io.video2luminance(video_file, nimages=100)
>>> nimages, vdim, hdim = luminance_images.shape
>>> pyramid = moten.get_default_pyramid(vhsize=(vdim, hdim), fps=24)
>>> moten_features = pyramid.project_stimulus(luminance_images)
```
