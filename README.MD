[![Latest PyPI version](https://img.shields.io/pypi/v/topometry.svg)](https://pypi.org/project/topometry/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/topometry/badge/?version=latest)](https://topometry.readthedocs.io/en/latest/?badge=latest)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/DaviSidarta.svg?label=Follow%20%40davisidarta&style=social)](https://twitter.com/davisidarta)


# TopOMetry
TopOMetry: Topologically Optimized geoMetry - 
Fast, accurate learning of data topology with self-adaptive metrics, graphs and layouts. 

TopOMetry is a modularized python library to explore data topology. 
It allows building graphs from data, learning graph-oriented metrics, and 
performing graph clustering, coarsening and layout optimization. 
TopOMetry is designed to handle large-scale data matrices containing 
extreme sample diversity and topological information, such as those 
generated from single-cell data, and can be used to perform topology-preserving 
dimensional reductions. TopOMetry is implemented in python and builds complex, high-level models 
inherited from [scikit-learn](https://github.com/scikit-learn/scikit-learn)
``BaseEstimator``, making it flexible and easy to apply and/or combine with different workflows.

### Installation and dependencies

TopOMetry is a high-level topological interface that uses some pre-existing libraries to power its scalability and flexibility:

* [scikit-learn](https://github.com/scikit-learn/scikit-learn) - for general machine-learning algorithms
* [ANNOY](https://github.com/spotify/annoy) - for optimized neighbor index search
* [nmslib](https://github.com/nmslib/nmslib) - for fast and accurate k-nearest-neighbors
* [igraph](https://github.com/igraph/python-igraph) - as a generic graph interface
* [kneed](https://github.com/arvkevi/kneed) - for finding nice cuttofs


Prior to installing TopOMetry, make sure you have [cmake](https://cmake.org/), [scikit-build](https://scikit-build.readthedocs.io/en/latest/) and [setuptools](https://setuptools.readthedocs.io/en/latest/) available in your system. If using Linux:
   ```
   sudo apt-get install cmake
   pip3 install scikit-build setuptools
   ```
We're also going to need NMSlib for really fast approximate nearest-neighborhood search. If your CPU supports
advanced instructions, we recommend you install nmslib separately for better performance:
   ```
   pip3 install --no-binary :all: nmslib
   ```
Then, you can install TopOMetry and its other requirements with pip:
   ```
   pip3 install numpy pandas annoy scipy numba scikit-learn kneed
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
### Quick-start 

TopoMetry main class is the ``TopoGraph`` object. In a ``TopoGraph``, topological basis are built 
with self-adaptive Diffusion Maps or Continuous-k-Nearest-Neighbors to achieve approximations of 
the Laplace-Beltrami Operator. The learned basis and graphs are stored as different attributes of the
object. For example, the multiscale diffusion basis is stored in ``TopoGraph.MSDiffMaps`` once they have
been computed by ``TopoGraph().fit_transform(data)``.

From a large data matrix ``mat`` (np.ndarray, pd.DataFrame or csr_matrix), you can set up a ``TopoGraph`` with default parameters: 

   ```
   import topograph.models as tp
   
   # Build graph basis from data
   tg = tp.TopoGraph()
   tg = tg.fit(data)
   ```

After fitting, we can access topological metrics and basis in the ``tg`` object.

   ```
   # Build topologicalrepresentations from basis
   dgraph, tgraph = tg.transform(data)
   
   # Graph layout optimization
   # from basis and topology
   emb_d, aux = MAP(tg.MSDiffMaps, draph)
   emb_t, aux = MAP(tg.MSDiffMaps, traph)
   
   ```

### Tutorials and examples



## License

[MIT License]()

Copyright (c) 2021 Davi Sidarta-Oliveira, davisidarta(at)gmail.com


