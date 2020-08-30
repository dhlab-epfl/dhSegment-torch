# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import inspect
import os
import sys


sys.path.insert(0, os.path.abspath('..'))

from dh_segment_torch.config import Registrable



# -- Project information -----------------------------------------------------

project = 'dhSegment-torch'
copyright = '2020, Digital Humanities Lab - EPFL'
author = 'Raphaël Barman, Sofia Ares Oliveria, Benoit Seguin'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx_autodoc_typehints',
              'sphinxcontrib.katex',
              'sphinx_autosummary_dhsegment',
              "sphinx.ext.intersphinx"]

intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'numpy': ('https://docs.scipy.org/doc/numpy/', None),
                       'torch': ('https://pytorch.org/docs/stable/', None),
                       'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
                       'albumentations': ("https://albumentations.readthedocs.io/en/latest/", None)
                       }

#autodoc_default_flags = ["members", "special-members"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Prerender math
katex_prerender = True

# Auto generate summary
autosummary_generate = True


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'
html_theme_options = {
    "page_width": "1024px",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


def process_docstring(app, what, name, obj, options, lines):
    if what == 'class' and 'Loss' in name and issubclass(obj, Registrable):
        type = None
        default = None
        method_resolution_order = inspect.getmro(obj)
        # print(method_resolution_order)
        for base_class in method_resolution_order:
            if issubclass(base_class, Registrable) and base_class is not Registrable:
                try:
                    type = base_class.get_type(obj)
                    default = base_class.default_implementation
                except KeyError:
                    pass
        if type and type != 'default':
            if type == default:
                type = f"**{type}**"
            if len(lines) == 0:
                lines.append("")
            elif len(lines[0]) > 0:
                lines[0] += " "
            lines[0] += f'Config type name: "{type}"'


def setup(app):
    app.connect('autodoc-process-docstring', process_docstring)
