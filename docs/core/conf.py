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
import os
import sys
import datetime
sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------

year = datetime.datetime.now().year
project = 'synthesized'
copyright = f'{year}, Synthesized Ltd.'
author = 'Synthesized Ltd.'

# The full version, including alpha/beta/rc tags
import synthesized
release = str(synthesized.__version__)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.coverage',
              'sphinx.ext.napoleon',
              'sphinx_autodoc_typehints',
              'sphinx_panels',
              "IPython.sphinxext.ipython_directive",
              "IPython.sphinxext.ipython_console_highlighting"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

napoleon_attr_annotations = True
napoleon_use_param = True
autosummary_generate = True
autodoc_typehints = "signature"
always_document_param_types = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_logo = '_static/synthesized-logo-light.svg'
html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/synthesized-io/synthesized",
            "icon": "fab fa-github-square",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/synthesizedio?lang=en",
            "icon": "fab fa-twitter-square",
        },
    ],
    "external_links": [
        {
            "name": "Synthesized Cloud", "url": "https://cloud.synthesized.io"
        }
    ],
    "favicons": [
        {
            "rel": "icon",
            "sizes": "16x16",
            "href": "https://cloud.synthesized.io/favicon.ico",
        },
    ]
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
