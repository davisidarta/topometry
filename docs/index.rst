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



TopoMetry is a Python library for geometric analysis of high-dimensional data, with
a focus on single-cell genomics — scRNA-seq, scATAC-seq, and multi-omics. It learns
geometry directly from data in a fully unsupervised manner, and uses that geometry as
the foundation for all downstream analysis steps. In practice, this means users get:

- More accurate identification of cell types, lineages, and rare populations — including
  subpopulations missed by standard workflows
- Better 2-D visualizations of cellular relationships, with tools to diagnose and
  interpret the distortions they introduce
- Principled identification of artifacts such as doublets, which tend to occupy
  geometrically anomalous regions of the data manifold
- Geometry-grounded trajectories, pseudotime estimates, and RNA velocity analyses
- A quantitative way to evaluate and compare representations, rather than relying on
  visual inspection alone

TopoMetry operates on AnnData and MuData objects and is fully compatible with Scanpy
and the broader Python ecosystem for single-cell analysis. A complete analysis can be
run with a single line of code, generating a report that covers clustering, embeddings,
geometry metrics, and distortion diagnostics.

For background, see our preprint: https://doi.org/10.1101/2022.03.14.484134

The problem with PCA
---------------------------

Nearly all single-cell workflows begin with PCA. This is convenient, but it introduces
a systematic error that propagates through every subsequent step: PCA is a linear method,
and it finds directions of maximum global variance. Gene expression data is not linear.
Cell states, lineages, and transitions live on curved, often branching structures in
high-dimensional space, and a linear projection discards most of that structure by
design.

In practice, PCA typically explains around 36% of total variance in scRNA-seq datasets
— sometimes as little as 20% — and adding more components does not fix this. The
eigenspectrum genuinely flattens; the remaining variance is not noise, it is nonlinear
signal that PCA cannot represent. On top of this, single-cell data is almost never
uniformly sampled: cell-type abundances vary by orders of magnitude, and PCA ignores
these differences entirely. The neighborhood graph built on PCA coordinates inherits
all of these distortions, and UMAP then optimizes a visual layout on top of an already
compromised representation.

The result is that subtle cell states, rare populations, and fine-grained transitions are
systematically collapsed or discarded long before any biological interpretation takes
place — and there is currently no standard way to detect or quantify how much has been
lost.

TopoMetry addresses this by abandoning linear projections entirely. Instead, it builds
similarity graphs with kernels that adapt to local sampling density and intrinsic
dimensionality, then decomposes the resulting operators into spectral scaffolds:
geometry-faithful coordinate systems that capture both local neighborhoods and
long-range structure across scales, sized automatically from the data rather than fixed
arbitrarily. A refined graph built on these coordinates serves as the backbone for
clustering, visualization, and all other downstream tasks. Crucially, TopoMetry also
provides operator-native metrics to measure how well any given representation preserves
the original geometry, and Riemannian diagnostics to reveal where 2-D embeddings
introduce distortion — so users can evaluate their representations rather than just
inspect them visually.

.. note::
   For a detailed discussion of these issues and how TopoMetry addresses them,
   see :doc:`why_topometry`.


When to use TopoMetry
---------------------------

TopoMetry is a good fit when the standard PCA-based workflow feels like a black box —
when you want to know whether your clusters and embeddings are trustworthy, when you
suspect rare populations are being missed, or when the biology you are studying involves
continuous transitions and gradients that linear methods tend to flatten. It is
particularly well-suited to datasets where cell-type composition is highly skewed,
where multiple lineages coexist, or where subtle transcriptional variation is expected
to carry biological meaning.

When not to use TopoMetry
~~~~~~~~~~~~~~~~~~~~~~~~~~

TopoMetry is not currently designed for workflows that require online embedding — that
is, projecting new cells into a fixed, precomputed space without recomputing operators
from scratch. This is worth pausing on: if the new cells represent cell types, states,
or conditions that were not present in the original dataset, the geometry of the full
data has changed, and any asymmetric integration approach risks biasing the result
toward the reference. This is not a limitation specific to TopoMetry; it is a general
challenge for manifold-based methods, and one that deserves more scrutiny in the field
than it currently receives. For workflows that genuinely require online mapping,
parametric or autoencoder-based approaches are more appropriate — though TopoMetry
can still be useful for auditing the geometry of the reference or estimating intrinsic
dimensionality to guide model design.

Minimal example
---------------------------

.. code-block:: python

    import scanpy as sc
    import topo as tp

    adata = sc.datasets.pbmc3k_processed()

    # Fit TopoMetry end-to-end (non-destructive; outputs are namespaced).
    # TopoMetry is highly parallelized — n_jobs=-1 uses all available cores.
    # Set verbosity=0 to suppress progress bars.
    tg = tp.sc.fit_adata(adata, n_jobs=-1, verbosity=0, random_state=7)

    # Plot results
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
    why_topometry
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


Changelog
---------------------------

**v1.2.x** — Multi-omics integration and spatial data support
- Multi-omics integration (paired and unpaired): joint geometry learning across modalities, with modality-specific and shared TopoMAPs
- Spatial data support: integration of spatial transcriptomics data with single-cell RNA-seq data, with spatially-aware geometry learning and visualization tools

**v1.1.x** — Batch integration and data mapping

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