
[![Latest PyPI version](https://img.shields.io/pypi/v/topometry.svg)](https://pypi.org/project/topometry/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/topometry/badge/?version=latest)](https://topometry.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://static.pepy.tech/personalized-badge/topometry?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/topometry)
[![CodeFactor](https://www.codefactor.io/repository/github/davisidarta/topometry/badge)](https://www.codefactor.io/repository/github/davisidarta/topometry)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/DaviSidarta.svg?style=social&label=Follow%20%40davisidarta)](https://twitter.com/davisidarta)

# TopOMetry - Topologically Optimized geoMetry


TopOMetry is a high-level python library to explore data topology through manifold learning. It is compatible with scikit-learn, meaning most of its operators can be easily pipelined.

Its main idea is to approximate the [Laplace-Beltrami Operator (LBO)](https://en.wikipedia.org/wiki/Laplace%E2%80%93Beltrami_operator). This is done by learning properly weighted similarity graphs and their Laplacian and Diffusion operators. By definition, the eigenfunctions of these operators describe all underlying data topology in an set of orthonormal eigenbases (classically named the spectral or diffusion components). New topological operators are then learned from such eigenbases and can be used for clustering and graph-layout optimization (visualization). 

There are many different ways to computationally approximate the LBO. TopOMetry tests a wide array of possible algorithmic combinations, combines them with existing graph-layout algorithm and scores them aftwerwards. This way, users do not have to choose a fixed method _a priori_, and can instead decide what works best for each use case. It also includes various utilities for scoring the performance of similarity kernels and dimensional reductions of high-dimensional data. It includes methods for the estimation of intrinsic dimensionalities (global and local), and implements the [Riemann metric]() to qualitatively visualize distortions in 2-D embeddings.

For more information, see the [manuscript](https://doi.org/10.1101/2022.03.14.484134).


## Single-cell analysis

TopOMetry was designed to handle large-scale data matrices containing extreme sample diversity, such as those generated from high-throughput [single-cell experiments](https://en.wikipedia.org/wiki/Single_cell_sequencing). It includes wrappers to deal with [AnnData](https://anndata.readthedocs.io/en/latest/index.html) objects using [scanpy](https://scanpy.readthedocs.io/en/stable/) and integrates well with tools in the [scverse](https://scverse.org/) python suite for single-cell analysis.

-------------------

## Documentation

Instalation can be quickly done with pip:

```
pip install topometry
```

Further installation instructions such as optional dependencies, information about the implemented methods tutorials and a detailed API are available at the available at the [documentation](https://topometry.readthedocs.io/en/latest/). 

-------------------

## Contributing

Contributions are very welcome! If you're interested in adding a new feature, just let me know in the Issues section.

-------------------

## License

[MIT License](https://github.com/davisidarta/topometry/blob/master/LICENSE)

-------------------

## Citation

If you use TopOMetry for your work, please cite the manuscript:

``` bibtex
@article {Sidarta-Oliveira2022.03.14.484134,
	author = {Sidarta-Oliveira, Davi and Velloso, Licio A},
	title = {A comprehensive dimensional reduction framework to learn single-cell phenotypic topology uncovers T cell diversity},
	elocation-id = {2022.03.14.484134},
	year = {2022},
	doi = {10.1101/2022.03.14.484134},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2022/03/17/2022.03.14.484134},
	eprint = {https://www.biorxiv.org/content/early/2022/03/17/2022.03.14.484134.full.pdf},
	journal = {bioRxiv}
}
```