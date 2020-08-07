#!/usr/bin/env python3
import sys

try:
    import configparser
except ImportError:
    import ConfigParser as configparser


if len(set(('develop', 'bdist_wheel', 'bdist_egg',
            'bdist_rpm', 'bdist',
            'sdist', 'bdist_wheel', 'bdist_dumb',
            'bdist_wininst', 'install_egg_info',
            'egg_info', 'easy_install')).intersection(sys.argv)) > 0:

    from setuptools import setup
    from setuptools.command.install import install
else:
    # Use standard
    from distutils.core import setup
    from distutils.command.install import install

def set_default_options(optfile):
    config = configparser.ConfigParser()
    config.read(optfile)
    with open(optfile, 'w') as fp:
        config.write(fp)

class my_install(install):
    def run(self):
        install.run(self)
        optfile = [f for f in self.get_outputs() if 'defaults.cfg' in f]
        # set_default_options(optfile[0])


if not 'extra_setuptools_args' in globals():
    extra_setuptools_args = dict()


with open('README.rst', 'r') as fid:
    long_description = fid.read()


def main(**kwargs):
    setup(name='pymoten',
          version='0.0.2',
          description="""Extract motion energy features from video using spatio-temporal Gabors""",
          author='Anwar O. Nunez-Elizalde',
          author_email='anwarnunez@gmail.com',
          url='https://gallantlab.github.io/pymoten/',
          packages=['moten',
                    ],
          package_data={
              'moten':[
                'defaults.cfg',
                  ],
              },
          cmdclass=dict(install=my_install),
          include_package_data=True,
          long_description=long_description,
          long_description_content_type='text/x-rst',
          **kwargs)

if __name__ == "__main__":
    main(**extra_setuptools_args)
