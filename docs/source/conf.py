# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
# import sphinx_bootstrap_theme
import sphinx_rtd_theme
# import numpydoc

# -- Project information -----------------------------------------------------

project = 'pymoten'
copyright = '2020, Anwar O. Nunez-Elizalde'
author = 'Anwar O. Nunez-Elizalde'

# The full version, including alpha/beta/rc tags
import moten  # noqa
release = moten.__version__
version = moten__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.mathjax',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.coverage',
              'sphinx.ext.inheritance_diagram',
              'sphinx.ext.viewcode',
              'sphinx.ext.napoleon',
              'sphinx_gallery.gen_gallery',
              'sphinx_rtd_theme'              ]


napoleon_use_ivar = True
autosummary_generate = True
numpydoc_class_members_toctree = True
numpydoc_show_class_members = True


# # Sphinx-gallery
sphinx_gallery_conf = {
    # path to your examples scripts
    'examples_dirs' : '../../examples',
    # path where to save gallery generated examples
    'gallery_dirs'  : 'auto_examples',
    # which files to execute? only those starting with "plot_"
    'filename_pattern' : '/demo_',
    }


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
if 0:
    html_theme = 'bootstrap'
    html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()
    html_theme_options = {'bootswatch_theme' : 'cosmo',
                          'bootstrap_version' : '3',
                          # Render the next and previous page links in navbar. (Default: true)
                          'navbar_sidebarrel': True,
                          # Render the current pages TOC in the navbar. (Default: true)
                          'navbar_pagenav': True,
                          # Tab name for the current pages TOC. (Default: "Page")
                          'navbar_pagenav_name': "Page content",
                          'globaltoc_depth': 2,
                          # 'navbar_links': [
                          #     ("Examples", "examples"),
                          #     ("Link", "http://example.com", True),
                          # ],
                          }
else:
    html_theme = 'sphinx_rtd_theme'
    html_theme_options = {'canonical_url': '',
                          # # 'analytics_id': 'UA-XXXXXXX-1',
                          # 'logo_only': False,
                          'display_version': True,
                          # 'prev_next_buttons_location': 'bottom',
                          # 'style_external_links': False,
                          # 'vcs_pageview_mode': '',
                          # 'style_nav_header_background': 'white',
                          # # Toc options
                          'collapse_navigation': False,
                          # 'sticky_navigation': True,
                          'navigation_depth': 3,
                          # 'includehidden': True,
                          # 'titles_only': False,
                          }


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
