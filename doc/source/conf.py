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
sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------

project = 'OCD modeling'
copyright = '2023, Sebastien Naze'
author = 'Sebastien Naze'

import OCD_modeling

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # include documentation from docstrings
    'sphinx.ext.autodoc',
    # generate autodoc summaries
    'sphinx.ext.autosummary',
    # use mathjax for latex formulas
    'sphinx.ext.mathjax',
    # link to code
    'sphinx.ext.viewcode',
    # link to other projects' docs
    'sphinx.ext.intersphinx',
    # support numpy and google style docstrings
    'sphinx.ext.napoleon',
    # support todo items
    'sphinx.ext.todo',
    # test snippets in the documentation
    'sphinx.ext.doctest',
    # source parser for jupyter notebook files
    'nbsphinx',
    # code highlighting in jupyter cells
    'IPython.sphinxext.ipython_console_highlighting',
    # ensure that jQuery is installed
    'sphinxcontrib.jquery',
    # generate autodoc for argument parsing in scripts
    'sphinxarg.ext',
    # video/movie
    'sphinxcontrib.video',
]

# default autodoc options
# list for special-members seems not to be possible before 1.8
autodoc_default_options = {
    'members': True,
    'special-members': '__init__, __call__',
    'show-inheritance': True,
    'autodoc_inherit_docstrings': True,
    'imported-members': True,
}


# links for intersphinx
#intersphinx_mapping = {
#    'python': ('https://docs.python.org/3', None),
#    'numpy': ('https://numpy.org/devdocs/', None),
#    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
#    'pandas': ('https://pandas.pydata.org/pandas-docs/dev', None),
#    "sklearn": ("https://scikit-learn.org/stable/", None),
#}

autodoc_mock_imports = ["sklearn", "pandas", "scipy", "numpy", "datetime", "concurrent"]

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
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
#html_theme = 'furo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
