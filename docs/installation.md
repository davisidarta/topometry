
TopOMetry requires some pre-existing libraries to power its scalability and flexibility. TopOMetry is implemented in python and builds complex, high-level models 
inherited from [scikit-learn](https://github.com/scikit-learn/scikit-learn)
``BaseEstimator``, making it flexible and easy to apply and/or combine with different workflows on virtually any domain.

Some general machine-learning libraries:

* [numpy](https://numpy.org/) 

* [pandas](https://pandas.pydata.org/)

* [numba](http://numba.pydata.org/)

* [pytorch](https://pytorch.org/)

* [scipy](https://www.scipy.org/)

* [scikit-learn](https://github.com/scikit-learn/scikit-learn)

And: 

* [hnswlib](https://github.com/nmslib/hnswlib) OR [nmslib](https://github.com/nmslib/nmslib) - for fast and accurate k-nearest-neighbors

* [kneed](https://github.com/arvkevi/kneed) - for finding nice cuttofs

* [pyMDE](https://github.com/cvxgrp/pymde) - for optimizing layouts with Minimum Distortion Embeddings

Prior to installing TopOMetry, make sure you have [cmake](https://cmake.org/), [scikit-build](https://scikit-build.readthedocs.io/en/latest/) and [setuptools](https://setuptools.readthedocs.io/en/latest/) available in your system. If using Linux:
```
sudo apt-get install cmake
pip3 install scikit-build setuptools
```
We're also going to need either HNSWlib or NMSlib for really fast approximate nearest-neighborhood search across different distance metrics.

```
pip3 install hnswlib
```
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
Alternatevely, if you want the very last version, clone this repo and build from source:
```
git clone https://github.com/davisidarta/topometry
cd topometry
pip3 install .
```

Please open a note in the [Issue tracker](https://github.com/davisidarta/topometry/issues) if you have any trouble with installation!


