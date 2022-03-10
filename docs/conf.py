
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
from os import path
import sys
import mock

MOCK_MODULES = ['numba']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

# sys.path.insert(0, path.abspath('./'))
sys.path.insert(0, path.abspath('../../'))

import topo as package

pkg_name = 'topometry'
pkg_file = package.__file__
pkg_version = str(package.__version__)
pkg_location = path.dirname(path.dirname(pkg_file))

autoapi_dirs = ['../../topometry']

# -- Project information -----------------------------------------------------

project = 'TopOMetry'
copyright = '2021, Davi Sidarta-Oliveira'
author = 'Davi Sidarta-Oliveira'
copyright = f'2021, {author}'

github_user = 'davisidarta'
github_repo = 'topometry'
github_version = 'master'

# -- General configuration ---------------------------------------------------

github_url = f'https://github.com/{github_user}/{github_repo}/'
gh_page_url = f'https://{github_repo}.readthedocs.io/'

html_baseurl = gh_page_url
html_context = {
    'display_github': True,
    'github_user': github_user,
    'github_repo': github_repo,
    'github_version': github_version,
    "conf_py_path": "/docs/source/",  # Path in the checkout to the docs root
}

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    'myst_parser',
    'sphinx_rtd_theme',
    # 'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'autoapi.extension',
]
# autodoc_mock_imports = ['pandas', 'numba', 'matplotlib',
#                         'torch', 'kneed', 'nmslib', 'hnswlib', 'pymde',
#                         'pacmap', 'trimap', 'ncvis', 'multicoretsne']


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.venv',
                    'docs/run_livereload.py', 'docs/conf.py', 'base/dists.py',
                    'base/sparse.py', ]

master_doc = 'index'
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_logo = 'img/logo.png'
html_favicon = 'img/favicon.ico'
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'restructuredtext',
    '.md': 'markdown',
}

# -- General default extension configuration ------------------------------


autoapi_type = 'python'
autoapi_generate_api_docs = True

# autosectionlabel options
# autosectionlabel throws warnings if section names are duplicated.
# The following tells autosectionlabel to not throw a warning for
# duplicated section names that are in different documents.
autosectionlabel_prefix_document = False

# katex options
katex_prerender = True

# napoleon options
napoleon_use_ivar = True
napoleon_use_rtype = False

# todo options
# If true, `todo` and `todoList` produce output, else they produce nothing.
# todo_include_todos = True

