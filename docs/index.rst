Welcome to TopOMetry documentation!
=======================================================
.. raw:: html

    <a href="https://pypi.org/project/topometry/"><img src="https://img.shields.io/pypi/v/topometry" alt="Latest PyPi version"></a>

.. raw:: html

    <a href="https://github.com/davisidarta/topometry/"><img src="https://img.shields.io/github/stars/davisidarta/topometry?style=social&label=Stars" alt="GitHub stars"></a>

.. raw:: html

    <a href="https://twitter.com/davisidarta"><img src="https://img.shields.io/twitter/follow/davisidarta.svg?style=social&label=Follow" alt="Twitter"></a>

TopOMetry (Topologically Optimized geoMetry) is a high-level python library to explore data topology.
It allows learning topological metrics, dimensionality reduced basis and graphs from data, as well
to visualize them with different layout optimization algorithms. The idea is to do so by finding approximations of
the Laplace-Beltrami Operator (LBO), a natural way to describe
data geometry and its high-dimensional topology. Except for layout optimization, each independently approximates the
LBO.

.. toctree::
    :maxdepth: 2
    :glob:
    :titlesonly:
    :caption: Intro:

    about
    installation
    quickstart

.. toctree::
    :maxdepth: 3
    :caption: General Tutorials:

    MNIST_TopOMetry_Tutorial
    20Newsgroups_Tutorial

.. toctree::
    :maxdepth: 3
    :caption: Single-cell analysis:

    pbmc3k
    pbmc68k
    Non_euclidean_tutorial

.. toctree::
    :maxdepth: 2
    :caption: API:

    topograph

.. toctree::
    :maxdepth: 3


