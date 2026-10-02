[![Latest PyPI version](https://img.shields.io/pypi/v/topometry.svg)](https://pypi.org/project/topometry/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/topometry/badge/?version=latest)](https://topometry.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://static.pepy.tech/personalized-badge/topometry?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/topometry)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/DaviSidarta.svg?style=social&label=Follow%20%40davisidarta)](https://twitter.com/davisidarta)

# About TopoMetry

**TopoMetry** is a geometry-aware Python toolkit for exploring high-dimensional data via diffusion/Laplacian operators. It learns **neighborhood graphs → Laplace–Beltrami–type operators → spectral scaffolds → refined graphs** and then finds clusters and builds low-dimensional layouts for analysis and visualization.

- **AnnData/Scanpy wrappers** for single-cell workflows
- **scikit-learn–style transformers** with a high-level orchestrator
- **Fixed-time & multiscale spectral scaffolds** (no `.X` mutation; namespaced outputs)
- **Operator-native metrics** to quantify geometry preservation and **Riemannian diagnostics** to evaluate distortion in visualizations
- Designed for **large, diverse datasets** (e.g., single-cell omics)

For background, see our preprint: https://doi.org/10.1101/2022.03.14.484134

## Geometry-first rationale (short)

We approximate the **Laplace–Beltrami operator (LBO)** by learning well-weighted similarity graphs and their Laplacian/diffusion operators. The **eigenfunctions** of these operators form an orthonormal basis—the **spectral scaffold**—that captures the dataset’s intrinsic geometry across scales. This view connects to **Diffusion Maps**, **Laplacian Eigenmaps**, and related kernel eigenmaps, and enables downstream tasks such as clustering and graph-layout optimization with geometry preserved.

## When to use TopoMetry

Use TopoMetry when you want:

- Geometry-faithful representations beyond variance maximization (e.g., PCA)
- Robust low-dimensional views and clustering from operator-grounded features
- Quantitative **operator-native** metrics to compare methods and parameter choices
- Reproducible, **non-destructive** pipelines (no mutation of `adata.X`)

Empirically, TopoMetry often outperforms PCA-based pipelines and stand-alone layouts. Still, **let the data decide**—TopoMetry includes metrics and reports to support evidence-based choices.

### When not to use TopoMetry

- **Very small sample sizes** where the manifold hypothesis is weak
- Workflows needing **streaming/online** updates or **inverse transforms** (embedding new points without recomputing operators is not currently supported). If that’s critical, consider UMAP or parametric/autoencoder approaches—and you can still use TopoMetry to **audit geometry** or **estimate intrinsic dimensionality** to guide model design.

## Installation

Prior to installing TopoMetry, make sure you have [cmake](https://cmake.org/), [scikit-build](https://scikit-build.readthedocs.io/en/latest/) and [setuptools](https://setuptools.readthedocs.io/en/latest/) available in your system. If using Linux:
```
sudo apt-get install cmake
pip install scikit-build setuptools
```

Then you can install TopoMetry from PyPI:

```
pip install topometry
```

Neighbor search uses [hnswlib](https://github.com/nmslib/hnswlib) when it is installed (`pip install hnswlib`), which is recommended for all but small datasets. Without it, TopoMetry falls back to nmslib if present, and otherwise to scikit-learn's exact search. PaCMAP layouts need `pip install pacmap`.


## Tutorials and documentation

Check TopoMetry's [documentation](https://topometry.readthedocs.io/en/latest/) for tutorials, guided analyses and other documentation.



## Minimal example (current API)

```python
import scanpy as sc
import topo as tp

adata = sc.datasets.pbmc3k_processed()

# Fit TopoMetry end-to-end (non-destructive; outputs are namespaced)
tg = tp.sc.fit_adata(adata, n_jobs=1, verbosity=0, random_state=7)

# Plot some results
sc.pl.embedding(adata, basis='spectral_scaffold', color='topo_clusters')
sc.pl.embedding(adata, basis='TopoMAP', color='topo_clusters')
sc.pl.embedding(adata, basis='TopoPaCMAP', color='topo_clusters')

# Save cleanly (I/O-safe)
adata.write_h5ad("pbmc3k_topometry.h5ad")
```

## Changelog

**v1.1.2** — Notebook display

- Displaying a `TopOGraph` or `Kernel` in a Jupyter notebook, or listing its attributes with `dir()` or tab completion, failed with recent scikit-learn, which reads every attribute of an estimator: a `Kernel` property raised `ValueError`, and another started an all-pairs shortest-path computation. These objects now show their own text summary, and listing attributes evaluates nothing.
- The PDF report named two `adata.obsm` keys that do not exist; it now names `X_ms_spectral_scaffold`.

**v1.1.1** — Fixes to standing issues

⚠️ Results computed with a cosine metric change, and `base_metric='cosine'` is the default. Analyses run with any release from 0.2.0.0 to 1.1.0 on a cosine metric should be re-run. Euclidean graphs and the default kernels are unchanged on the hnswlib and nmslib backends.

- **Cosine neighbor graphs held similarities instead of distances.** Within each neighborhood the closest points got the lowest kernel weights. Neighbor sets were right; the weights were not. Cosine kernels, geodesics and intrinsic-dimension estimates are now computed on angles, and kernels have no self-loops.
- **Neighbor search.** A missing backend falls back to the next available of hnswlib, nmslib and scikit-learn with a warning; every backend returns the same number of neighbors; `backend='nmslib'` is honoured; nmslib no longer returns squared distances for dense input.
- **Kernel options that did not do what they say** now do: CkNN (`delta` is passed on and usually needs tuning), neighborhood expansion, `use_angular=False`, `pairwise=True`.
- **Projections.** Isomap and the MDE recipes are run on distances rather than on the diffusion operator; UMAP and landmarks work; a standalone `Projector` builds proper affinities.
- **Reproducibility.** With `n_jobs=1` and a `random_state`, two fits agree bit for bit. Eigenvector signs no longer flip between runs.
- **TopOGraph.** `global_id`, `local_ids()`, `global_id_mle()` and `global_id_fsa()` return the estimates (the scaffold size is `n_scaffold_components`); `pseudotime` and `spectral_selectivity` pair each eigenvector with its own eigenvalue; saving no longer strips the fitted object.
- **Riemannian diagnostics** keep the Laplacian sparse, and the plotting wrapper honours `diffusion_t` and `groupby`.
- **Single-cell wrappers.** `tp.sc.preprocess` no longer modifies its input; BBKNN integration builds kernels from distances; PaCMAP is optional; `fit_adata` works without hnswlib.
- Compatibility with current scikit-learn, SciPy and matplotlib.

**v1.1.0** — Batch integration and data mapping
- CCA-anchor batch correction (Seurat v3-style) via `tp.sc.run_cca_integration`
- Reference atlas persistence (`save_cca_reference` / `load_cca_reference`) and sequential query mapping (`map_to_cca_reference`)
- High-level preparation utilities (`prepare_for_integration`, `prepare_for_mapping`, `find_mapping_order`)
- Neighbourhood-based integration quality metrics (`compute_all_integration_metrics`: kNN purity, kNN mixing, iLISI, cLISI, ARI, NMI)

**v1.0.x** — Complete overhaul
- Redesigned user API with `tp.sc.fit_adata` and `tp.sc.run_and_report` one-liner workflows
- New utilities for single-cell analysis: intrinsic dimensionality, spectral selectivity, feature modes, graph-signal filtering, imputation
- Overhauled geometry-preservation metrics (PF1, PJS, SP) and Riemannian diagnostics (pullback metric, deformation maps)
- Full compatibility with the `scverse` ecosystem (scanpy, scVelo, AnnData)

#### Citation

---

```
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
```
