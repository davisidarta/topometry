About TopoMetry
=======================================================

.. raw:: html

    <a href="https://pypi.org/project/topometry/"><img src="https://img.shields.io/pypi/v/topometry" alt="Latest PyPi version"></a>



.. raw:: html

    <a href="https://github.com/davisidarta/topometry/"><img src="https://img.shields.io/github/stars/davisidarta/topometry?style=social&label=Stars" alt="GitHub stars"></a>



.. raw:: html

    <a href="https://pepy.tech/project/topometry"><img src="https://static.pepy.tech/personalized-badge/topometry?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads" alt="Downloads"></a>



.. raw:: html

    <a href="https://twitter.com/davisidarta"><img src="https://img.shields.io/twitter/follow/davisidarta.svg?style=social&label=Follow @davisidarta" alt="Twitter"></a>



.. raw:: html

    <a href="https://readthedocs.org/projects/topometry/badge/?version=latest"><img src="https://readthedocs.org/projects/topometry/badge/?version=latest" alt="Documentation Status"></a>



.. raw:: html

    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>



**TopoMetry** is a geometry-aware Python toolkit for exploring high-dimensional data via diffusion/Laplacian operators. It learns **neighborhood graphs → Laplace–Beltrami–type operators → spectral scaffolds → refined graphs** and then finds clusters and builds low-dimensional layouts for analysis and visualization.

- **AnnData/Scanpy wrappers** for single-cell workflows
- **scikit-learn–style transformers** with a high-level orchestrator
- **Fixed-time & multiscale spectral scaffolds**, **multi-resolution clustering** and **2D layouts** that preserve geometry
- **Operator-native metrics** to quantify geometry preservation and **Riemannian diagnostics** to evaluate its distortion in visualizations
- Designed for **large, diverse datasets** (e.g., single-cell omics)

For background, see our preprint: https://doi.org/10.1101/2022.03.14.484134

Geometry-first rationale
---------------------------

We approximate the **Laplace–Beltrami operator (LBO)** by learning well-weighted similarity graphs and their Laplacian/diffusion operators. The **eigenfunctions** of these operators form an orthonormal basis—the **spectral scaffold**—that captures the dataset's intrinsic geometry across scales. This view connects to **Diffusion Maps**, **Laplacian Eigenmaps**, and related kernel eigenmaps, and enables downstream tasks such as clustering and graph-layout optimization with geometry preserved.

When to use TopoMetry
---------------------------

Use TopoMetry when you want:

- Geometry-faithful representations beyond variance maximization (e.g., PCA)
- Robust low-dimensional views and clustering from operator-grounded features
- Quantitative **operator-native** metrics to compare methods and parameter choices
- Reproducible, **non-destructive** pipelines (no mutation of ``adata.X``)
- To identify rare cell populations or subtle trajectories that are often missed by variance-based methods

Empirically, TopoMetry almost always outperforms PCA-based pipelines (e.g., Seurat, scanpy) and stand-alone layouts (i.e., "pure" UMAPs). Still, **let the data decide** — TopoMetry includes metrics and reports to support evidence-based choices.

When not to use TopoMetry
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Very small sample sizes** where the manifold hypothesis is weak
- Workflows needing **streaming/online** updates or **inverse transforms** (embedding new points without recomputing operators is not currently supported). If that's critical for your work, consider UMAP or parametric/autoencoder approaches — and you can still use TopoMetry to **audit geometry** or **estimate intrinsic dimensionality** to guide model design.

Minimal example
---------------------------

.. code-block:: python

    import scanpy as sc
    import topo as tp

    adata = sc.datasets.pbmc3k_processed()

    # Fit TopoMetry end-to-end (non-destructive; outputs are namespaced)
    # TopoMetry is highly parallelized, so we recommend n_jobs=-1 to use all cores.
    # Set verbosity=0 to suppress progress bars and info messages.
    tg = tp.sc.fit_adata(adata, n_jobs=-1, verbosity=0, random_state=7)

    # Plot some results
    sc.pl.embedding(adata, basis='spectral_scaffold', color='topo_clusters')
    sc.pl.embedding(adata, basis='TopoMAP', color='topo_clusters')
    sc.pl.embedding(adata, basis='TopoPaCMAP', color='topo_clusters')

    # Save cleanly (I/O-safe)
    adata.write_h5ad("pbmc3k_topometry.h5ad")
    tp.save_topograph(tg, "pbmc3k_topograph.pkl")

Citation
---------------------------

.. code-block:: bibtex

    @article {Oliveira2022.03.14.484134,
        author = {Oliveira, David S and Domingos, Ana I. and Velloso, Licio A},
        title = {TopoMetry systematically learns and evaluates the latent geometry of single-cell data},
        elocation-id = {2022.03.14.484134},
        year = {2025},
        doi = {10.1101/2022.03.14.484134},
        publisher = {Cold Spring Harbor Laboratory},
        URL = {https://www.biorxiv.org/content/early/2025/10/15/2022.03.14.484134},
        eprint = {https://www.biorxiv.org/content/early/2025/10/15/2022.03.14.484134.full.pdf},
        journal = {bioRxiv}
    }


.. toctree::
    :maxdepth: 2
    :glob:
    :titlesonly:
    :caption: Getting started:

    installation
    math_details

.. toctree::
    :maxdepth: 2
    :caption: Tutorials:

    T1_introduction
    T2_step_by_step
    T3_Integration

.. toctree::
    :maxdepth: 2
    :caption: API:

    topograph

.. toctree::
    :maxdepth: 3


Changelog
---------------------------

**v1.1.0** — Batch integration and data mapping

- CCA-anchor batch correction (Seurat v3-style) via ``tp.sc.run_cca_integration``
- Reference atlas persistence (``save_cca_reference`` / ``load_cca_reference``) and sequential query mapping (``map_to_cca_reference``)
- High-level preparation utilities (``prepare_for_integration``, ``prepare_for_mapping``, ``find_mapping_order``)
- Neighbourhood-based integration quality metrics (``compute_all_integration_metrics``: kNN purity, kNN mixing, iLISI, cLISI, ARI, NMI)
- Memory-efficient merge loop with systematic garbage collection

**v1.0.x** — Complete overhaul

- Redesigned user API with ``tp.sc.fit_adata`` and ``tp.sc.run_and_report`` one-liner workflows
- New utilities for single-cell analysis: intrinsic dimensionality, spectral selectivity, feature modes, graph-signal filtering, imputation
- Overhauled geometry-preservation metrics (PF1, PJS, SP) and Riemannian diagnostics (pullback metric, deformation maps)
- Full compatibility with the ``scverse`` ecosystem (scanpy, scVelo, AnnData)

