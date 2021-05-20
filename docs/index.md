## Welcome to TopOMetry documentation!

TopOMetry is a high-level python library to explore data topology.
It allows learning topological metrics, dimensionality reduced basis and graphs from data, as well
to visualize them with different layout optimization algorithms. The main objective is to achieve approximations of 
the [Laplace-Beltrami Operator](https://en.wikipedia.org/wiki/Laplace%E2%80%93Beltrami_operator), a natural way to describe
data geometry and its high-dimensional topology. 

TopOMetry is designed to handle large-scale data matrices containing 
extreme topological diversity, such as those 
generated from [single-cell omics](https://en.wikipedia.org/wiki/Single_cell_sequencing), and can be used to perform topology-preserving 
visualizations. 

TopOMetry main class is the ``TopOGraph`` object. In a ``TopOGraph``, topological metrics are recovered with diffusion
harmonics or Continuous-k-Nearest-Neighbors, and used to obtain topological basis (multiscale Diffusion Maps and/or 
diffuse or continuous versions of Laplacian Eigenmaps). 

On top of these basis, new graphs can be learned using k-nearest-neighbors
graphs or additional topological operators. The learned metrics, basis and graphs are stored as different attributes of the
``TopOGraph`` object. 

Finally, different visualizations of the learned topology can be optimized with ``pyMDE`` by solving a 
[Minimum-Distortion Embedding](https://github.com/cvxgrp/pymde) problem. TopOMetry also implements an adapted, non-uniform
version of the seminal [Uniform Manifold Approximation and Projection (UMAP)](https://github.com/lmcinnes/umap) 
for graph layout optimization (we call it MAP for short). 

Alternatively, you can use TopOMetry to add topological information to your favorite workflow
by using its dimensionality reduced basis to compute k-nearest-neighbors instead of PCA.
