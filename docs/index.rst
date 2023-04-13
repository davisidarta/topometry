Welcome to TopOMetry documentation!
=======================================================
.. raw:: html

    <a href="https://pypi.org/project/topometry/"><img src="https://img.shields.io/pypi/v/topometry" alt="Latest PyPi version"></a>


.. raw:: html

    <a href="https://github.com/davisidarta/topometry/"><img src="https://img.shields.io/github/stars/davisidarta/topometry?style=social&label=Stars" alt="GitHub stars"></a>


.. raw:: html

    <a href="https://pepy.tech/project/topometry"><img src="https://static.pepy.tech/personalized-badge/topometry?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads" alt="Downloads"></a>


.. raw:: html

    <a href="https://www.codefactor.io/repository/github/davisidarta/topometry"><img src="https://www.codefactor.io/repository/github/davisidarta/topometry/badge" alt="CodeFactor" /></a>


.. raw:: html

    <a href="https://twitter.com/davisidarta"><img src="https://img.shields.io/twitter/follow/davisidarta.svg?style=social&label=Follow @davisidarta" alt="Twitter"></a>



TopOMetry (Topologically Optimized geoMetry) is a comprehensive dimensional reduction framework
that dissects the steps of dimensional reduction approaches to learn latent topological representations.
It allows learning topological metrics, latent topological orthogonal basis, and topological graphs from data.
Visualization can then be performed with different layout optimization algorithms.

TopOMetry focus is on approximating the Laplace-Beltrami Operator (LBO) at each step of dimensional reduction.
The LBO is a natural way to describe data geometry and its high-dimensional topology.

TopOMetry can yield strikingly new biological insights on single-cell data, and I have developed CellTOMetry to work
as an interface between topometry and the more general python computational environment for single-cell analysis.
The orthogonal bases can be used in a way similar that PCA is used in most existing algorithms, and contributors
are welcome to test how their method perform when using topological denoised information. Check the single-cell
tutorials and TopOMetry's API for more information.

TopOMetry classes are built in a modular fashion using scikit-learn `BaseEstimator`,
meaning they can be easily pipelined.

.. toctree::
    :maxdepth: 2
    :glob:
    :titlesonly:
    :caption: Installation:

    installation
    
.. toctree::
    :maxdepth: 2
    :glob:
    :titlesonly:
    :caption: Getting started:

    about
    quickstart

.. toctree::
    :maxdepth: 2
    :caption: General tutorials:

    MNIST
    evaluations
    classes_tutorial
    id_estimatation
    non_euclidean
    20Newsgroups_Tutorial

.. toctree::
    :maxdepth: 3
    :caption: Single-cell tutorials:

    singlecell

.. toctree::
    :maxdepth: 2
    :caption: API:

    topograph

.. toctree::
    :maxdepth: 3


