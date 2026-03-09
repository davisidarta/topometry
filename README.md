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
