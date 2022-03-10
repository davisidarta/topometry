## Installation and dependencies

TopOMetry requires some pre-existing libraries to power its scalability and flexibility. TopOMetry is implemented in python and builds complex, high-level models
inherited from [scikit-learn](https://github.com/scikit-learn/scikit-learn)
``BaseEstimator``, making it flexible and easy to apply and/or combine with different workflows on virtually any domain.

Prior to installing TopOMetry, make sure you have [cmake](https://cmake.org/), [scikit-build](https://scikit-build.readthedocs.io/en/latest/) and [setuptools](https://setuptools.readthedocs.io/en/latest/) available in your system. If using Linux:
```
sudo apt-get install cmake
pip3 install scikit-build setuptools
```
TopOMetry uses either NMSlib or HNSWlib really fast approximate nearest-neighborhood search across different
distance metrics. By default, it uses NMSlib. If your CPU supports advanced instructions, we recommend you install
nmslib separately for better performance:
```
pip3 install --no-binary :all: nmslib
```
Alternatively, you can use HNSWlib for k-nearest-neighbor search backend:
```
pip3 install hnswlib
```

Then, you can install TopOMetry requirements:
```
pip3 install numpy pandas scipy numba torch matplotlib scikit-learn kneed pymde multicoretsne pacmap trimap ncvis
```
And finally install TopOMetry itself:
```
pip3 install topometry
```

Please open a note in the [Issue tracker](https://github.com/davisidarta/topometry/issues) if you have any trouble with installation!

