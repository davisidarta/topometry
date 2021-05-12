[![Latest PyPI version](https://img.shields.io/pypi/v/topometry.svg)](https://pypi.org/project/topometry/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/topometry/badge/?version=latest)](https://topometry.readthedocs.io/en/latest/?badge=latest)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/DaviSidarta.svg?label=Follow%20%40davisidarta&style=social)](https://twitter.com/davisidarta)

![](https://github.com/davisidarta/topometry/logo.png)

## TopOMetry - Topologically Optimized geoMetry

**Table of Contents**

- [What is TopOMetry and why it is useful](#topological-metrics-basis-graphs-and-layouts)
- [Installation](#installation-and-dependencies)
- [Quick-start](#quick-start)
- [Tutorials](#tutorials-and-examples)

##Topological metrics, basis, graphs and layouts

TopOMetry is a high-level python library to explore data topology.
It allows learning topological metrics, dimensionality reduced basis and graphs from data, as well
to visualize them with different layout optimization algorithms. The main objective is to achieve approximations of 
the [Laplace-Beltrami Operator](https://en.wikipedia.org/wiki/Laplace%E2%80%93Beltrami_operator), a natural way to describe
data geometry and its high-dimensional topology. 

TopOMetry is designed to handle large-scale data matrices containing 
extreme topological diversity, such as those 
generated from [single-cell omics](https://en.wikipedia.org/wiki/Single_cell_sequencing), and can be used to perform topology-preserving 
visualizations. 

TopoMetry main class is the ``TopoGraph`` object. In a ``TopoGraph``, topological metrics are recovered with diffusion
harmonics or Continuous-k-Nearest-Neighbors, and used to obtain topological basis (multiscale Diffusion Maps and/or 
diffuse or continuous versions of Laplacian Eigenmaps). 

On top of these basis, new graphs can be learned using k-nearest-neighbors
graphs or additional topological operators. The learned metrics, basis and graphs are stored as different attributes of the
``TopoGraph`` object. 

Finally, different visualizations of the learned topology can be optimized with ``pyMDE`` by solving a 
[Minimum-Distortion Embedding](https://github.com/cvxgrp/pymde) problem. TopOMetry also implements an adapted, non-uniform
version of the seminal [Uniform Manifold Approximation and Projection (UMAP)](https://github.com/lmcinnes/umap) 
for graph layout optimization (we call it MAP for short). 

Alternatively, you can use TopOMetry to add topological information to your favorite workflow
by using its dimensionality reduced basis to compute k-nearest-neighbors instead of PCA.

## Installation and dependencies

TopOMetry requires some pre-existing libraries to power its scalability and flexibility. TopOMetry is implemented in python and builds complex, high-level models 
inherited from [scikit-learn](https://github.com/scikit-learn/scikit-learn)
``BaseEstimator``, making it flexible and easy to apply and/or combine with different workflows on virtually any domain.


* [scikit-learn](https://github.com/scikit-learn/scikit-learn) - for general algorithms
* [ANNOY](https://github.com/spotify/annoy) - for optimized neighbor index search
* [nmslib](https://github.com/nmslib/nmslib) - for fast and accurate k-nearest-neighbors
* [kneed](https://github.com/arvkevi/kneed) - for finding nice cuttofs
* [pyMDE](https://github.com/cvxgrp/pymde) - for optimizing layouts

Prior to installing TopOMetry, make sure you have [cmake](https://cmake.org/), [scikit-build](https://scikit-build.readthedocs.io/en/latest/) and [setuptools](https://setuptools.readthedocs.io/en/latest/) available in your system. If using Linux:
```
sudo apt-get install cmake
pip3 install scikit-build setuptools
```
We're also going to need NMSlib for really fast approximate nearest-neighborhood search across different distance metrics.
If your CPU supports advanced instructions, we recommend you install nmslib separately for better performance:
```
pip3 install --no-binary :all: nmslib
```
Then, you can install TopOMetry and its other requirements with pip:
```
pip3 install numpy pandas annoy scipy numba torch scikit-learn kneed pymde
```
```
pip3 install topometry
```
Alternatevely, clone this repo and build from source:
```
git clone https://github.com/davisidarta/topometry
cd topometry
pip3 install .
```
## Quick-start 

From a large data matrix ``data`` (np.ndarray, pd.DataFrame or sp.csr_matrix), you can set up a ``TopoGraph`` with default parameters: 

```
import topo.models as tp
   
# Learn topological metrics and basis from data. The default is to use diffusion harmonics.
tg = tp.TopoGraph()
tg = tg.fit(data)

```

After learning a topological basis, we can access topological metrics and basis in the ``tg`` object, and build different
topological graphs.
```
# Learn a topological graph. Again, the default is to use diffusion harmonics.
tgraph = tg.transform(data) # tgraph.K is the affinity graph, and tgraph.T is the transition graph
   
# Graph layout optimization with UMAP (not so much 'uniform' in this case)
topo_umap, aux = tp.MAP(tg.MSDiffMaps, tgraph.T)

emb_t, aux = tp.MAP(tg.MSDiffMaps, traph)
```

## Tutorials and examples

Coming soon! For now, check these 
jupyter notebooks ([1](https://github.com/davisidarta/topometry/blob/master/docs/TopOMetry_Intro_pbmc.ipynb),
[2](https://github.com/davisidarta/topometry/blob/master/docs/TopOMetry_Intro.ipynb))
 showing how apply TopOMetry to single-cell data analysis.

## Contributing

Contributions are very welcome! If you're interested in adding a new feature, just let me know in the Issues section.

## License

[MIT License](https://github.com/davisidarta/topometry/blob/master/LICENSE)

Copyright (c) 2021 Davi Sidarta-Oliveira, davisidarta(at)gmail.com

