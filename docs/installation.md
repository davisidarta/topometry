## Installation and dependencies

TopOMetry is implemented in python and and its models are implemented as classes that inherit from [scikit-learn](https://github.com/scikit-learn/scikit-learn) ``BaseEstimator`` and ``TransformerMixin``. This makes these classes compatible with `scikit-learn` Pipelines and thus flexible and easy to apply and/or combine with different workflows on virtually any domain. 

The hard dependencies are common building-blocks of the python machine-learning environment:

* numpy
* scipy
* pandas
* numba
* scikit-learn
* matplotlib


Prior to installing TopOMetry, make sure you have [cmake](https://cmake.org/), [scikit-build](https://scikit-build.readthedocs.io/en/latest/) and [setuptools](https://setuptools.readthedocs.io/en/latest/) available in your system. If using Linux:
```
sudo apt-get install cmake
pip install scikit-build setuptools
```

Then you can install TopOMetry from PyPI:

```
pip install topometry
```

NOTE: if your version of python is beyond 3.9, you may want to avoid an existing issue with `setuptools` by setting the flag `--use-PEP517`:

```
pip install --use-pep517 topometry
```

--------------

## Optional dependencies

Some optional packages can enhance the use of TopOMetry, but are not listed as hard-dependencies. These are libraries for approximate-nearest-neighbors search and libraries for graph-layout optimization.

### Approximate Nearest Neighbors

Included in TopOMetry there is `topo.ann.kNN()` - an utility wrapper around these methods that can learn k-nearest-neighbors graphs from data using various approximate nearest-neighbors search methods. The reason I tried to make it so flexible was to allow it to be efficiently used in multiple computational settings/environments.

The optional libraries for approximate-nearest-neighbors are:

* [NMSlib](https://github.com/nmslib/nmslib)
* [HNSWlib](https://github.com/nmslib/hnswlib)
* [ANNOY](https://github.com/spotify/annoy)
* [FAISS](https://github.com/facebookresearch/faiss)
* [PyNNDescent](https://pynndescent.readthedocs.io/en/latest/)

If your CPU supports advanced instructions, I recommend you install
nmslib separately for the best performance:

```bash
pip install --no-binary :all: nmslib
```

 If you don't have any of these installed, TopOMetry will run using `scikit-learn` neighborhood search, which can be quite slow when analysing large datasets. NMSLib and HNSWlib are my primary recommendations for large-scale data, but other methods also work well.


### Additional layout methods

From version `2.0.0 ` onwards, TopOMetry does not include any graph layout algorithm as a dependency, and includes fast versions of [Isomap](https://doi.org/10.1126/science.290.5500.2319) and of the cross-entropy minimization of [UMAP](https://umap-learn.readthedocs.io/en/latest/index.html) (MAP) for graph layout and visualization. Other layout algorithms can be used, but are not listed as hard-dependencies and the choice of installing and using them is left to the user:

* ['t-SNE'](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) - one of the first manifold learning methods  (optionally with `multicore-tsne`, otherwise uses `scikit-learn`)
* ['UMAP'](https://umap-learn.readthedocs.io/en/latest/index.html) - arguably the state-of-the-art for graph layout optimization (requires installing `umap-learn`)
* ['PaCMAP'](http://jmlr.org/papers/v22/20-1061.html) (Pairwise-controlled Manifold Approximation and Projection) - for balanced visualizations (requires installing `pacmap`)
* ['TriMAP'](https://github.com/eamid/trimap) - dimensionality reduction using triplets (requires installing `trimap`)
* 'IsomorphicMDE' - [MDE](https://github.com/cvxgrp/pymde) with preservation of nearest neighbors (requires installing `pymde`)
* 'IsometricMDE' - [MDE](https://github.com/cvxgrp/pymde) with preservation of pairwise distances (requires installing `pymde`)
* ['NCVis'](https://github.com/stat-ml/ncvis) (Noise Contrastive Visualization) - a UMAP-like method with blazing fast performance (requires installing `ncvis`)

If you want to use them all, install them with:

```
pip install multicore-tsne umap-learn pacmap trimap pymde ncvis
```

These projection methods are handled by the `topo.layout.Projector()` class, and are quite straightforward to be added into the framework, so please open an Issue if your favorite method is not listed.

--------------

Please open a note in the [Issue tracker](https://github.com/davisidarta/topometry/issues) if you have any trouble with installation!

