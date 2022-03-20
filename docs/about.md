# About TopOMetry

TopOMetry (Topologically Optimized geoMetry) is a high-level python library to explore data topology.
It allows learning topological metrics, dimensionality reduced basis and graphs from data, as well
to visualize them with different layout optimization algorithms. The main objective is to achieve approximations of
the [Laplace-Beltrami Operator](https://en.wikipedia.org/wiki/Laplace%E2%80%93Beltrami_operator), a natural way to describe
data geometry and its high-dimensional topology. You can read more in our [manuscript](https://doi.org/10.1101/2022.03.14.484134).
I also have a Twitter [thread](https://twitter.com/davisidarta/status/1504511680373428224?s=20&t=F_sYf6rw4WVnb4EcO5C3MA) which overviews TopOMetry and it's results in 16 tweets. 

TopOMetry main class is the [TopOGraph](https://topometry.readthedocs.io/en/latest/topograph/) object. In a ``TopOGraph``, topological metrics are recovered with diffusion
harmonics, fuzzy simplicial sets or Continuous-k-Nearest-Neighbors, and used to obtain topological basis (multiscale Diffusion Maps and/or
fuzzy or continuous versions of Laplacian Eigenmaps). On top of these basis, new graphs can be learned using k-nearest-neighbors
graphs or with new topological metrics. The learned metrics, basis and graphs are stored as different attributes of the
``TopOGraph`` object. Finally, built-in adaptations of graph layout methods such as t-SNE and UMAP are used to obtain
visualizations to obtain further insight from data. You can also use TopOMetry to add topological information to your favorite workflow
by using its dimensionality reduced bases to compute k-nearest-neighbors instead of PCA, or its topological graphs as
affinity matrices for other algorithms.

The following diagram represent the different possible combinations of topological models and layouts options:

![TopOMetry in a glance](img/TopOGraph_models.png)

If you haven't already, [install](installation.md) *topometry* and start using it ([quick-start](quickstart.md)).

For users not familiar with single-cell analysis, check the tutorials with [MNIST](MNIST_TopOMetry_Tutorial.md) 
and [document embedding](20Newsgroups_Tutorial.md). 

For single-cell data
analysis, check the tutorials on [evaluating different workflows](pbmc3k.md), 
[learning T CD4 diversity (PBMC 68k)](pbmc68k.md)  and 
[embedding single-cell data to non-Euclidean spaces](Non_euclidean_tutorial.md).


## When it is best to use TopOMetry
If you are still confused on when using TopOMetry instead of UMAP, check this [issue](https://github.com/davisidarta/topometry/issues/8#issue-1173107992) (I'll improve the documentation later). 


## Citation

```
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