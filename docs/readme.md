[![Latest PyPI version](https://img.shields.io/pypi/v/topometry.svg)](https://pypi.org/project/topometry/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/topometry/badge/?version=latest)](https://topometry.readthedocs.io/en/latest/?badge=latest)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/DaviSidarta.svg?label=Follow%20%40davisidarta&style=social)](https://twitter.com/davisidarta)


# TopOMetry
TopOMetry: Topologically Optimized geoMetry - Fast, accurate learning of data topology with self-adaptive metrics, graphs and layouts

## Installation and dependencies

Prior to installing TopOMetry, make sure you have scikit-build and cmake available in your system. These are required for installation.
   ```
     $> sudo apt-get install cmake
     $> pip3 install scikit-build
   ```
We're also going to need NMSlib for really fast approximate nearest-neighborhood search. If your CPU supports
advanced instructions, we recommend you install nmslib separately for better performance:
   ```
    $> pip3 install --no-binary :all: nmslib
   ```
Then, you can install TopOMetry with pip:
   ```
     $> pip3 install topometry
   ```
Alternatevely, clone this repo and build from source:
   ```
     $> git clone https://github.com/davisidarta/topometry
     $> cd topometry
     $> pip3 install .
   ```
## Tutorials and guided analysis

Some jupyter notebook examples here

## Documentation

## License
MIT

