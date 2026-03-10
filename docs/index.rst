Welcome to TopoMetry documentation!
=======================================================
.. raw:: html

    <a href="https://pypi.org/project/topometry/"><img src="https://img.shields.io/pypi/v/topometry" alt="Latest PyPi version"></a>


.. raw:: html

    <a href="https://github.com/davisidarta/topometry/"><img src="https://img.shields.io/github/stars/davisidarta/topometry?style=social&label=Stars" alt="GitHub stars"></a>


.. raw:: html

    <a href="https://pepy.tech/project/topometry"><img src="https://static.pepy.tech/personalized-badge/topometry?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads" alt="Downloads"></a>


.. raw:: html

    <a href="https://twitter.com/davisidarta"><img src="https://img.shields.io/twitter/follow/davisidarta.svg?style=social&label=Follow @davisidarta" alt="Twitter"></a>



TopoMetry (Topologically Optimized geoMetry) is a comprehensive toolkit to explore high-dimensional
data, with a focus on single-cell genomics. It allows users to:

* construct k-nearest-neighbors graphs with several approximate-nearest-neighbors algorithms
* compute similarity metrics and topological operators to describe the geometry of the data
* estimate intrinsic dimensionalities 
* obtain properly weighted eigenbases to represent the underlying data manifold
* combine different kernel, eigendecomposition and graph-layout-optimization methods to obtain dozens of representations of single-cell data
* evaluate the quality of the learned embeddings with quantitative metrics
* assess distortions in the learned embeddings with with the Riemannian metric 

TopoMetry was designed to be user-friendly, consistent with the scikit-learn API,
and to be easily integrated with the more general python computational environment for single-cell analysis. Users can compute and evaluate
dozens of representations with a single line of code.

TopoMetry's based on Laplacian-type topological operators, with a focus on the Laplace-Beltrami Operator (LBO) and its eigenfunctions.
The LBO is a natural way to describe data geometry and its high-dimensional topology, and is guaranteed to recover
all of the relevant geometry if the manifold hypothesis holds true. These learned representations can be used for several downstream tasks in data analysis and single-cell bioinformatics,such as clustering, visualization with graph-layout optimization, RNA velocity and pseudotime estimation.
This can yield strikingly new biological insights on single-cell data. Check the preprint for more information.


TopoMetry classes are built in a modular fashion using scikit-learn `BaseEstimator`,
meaning they can be easily pipelined.

.. toctree::
    :maxdepth: 2
    :glob:
    :titlesonly:
    :caption: Getting started:

    about
    installation
    math_details

.. toctree::
    :maxdepth: 2
    :caption: Tutorials:

    T1_introduction
    T2_step_by_step

.. toctree::
    :maxdepth: 2
    :caption: API:

    topograph

.. toctree::
    :maxdepth: 3


