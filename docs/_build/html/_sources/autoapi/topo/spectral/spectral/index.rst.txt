:py:mod:`topo.spectral.spectral`
================================

.. py:module:: topo.spectral.spectral


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   topo.spectral.spectral.LapEigenmap
   topo.spectral.spectral.component_layout
   topo.spectral.spectral.multi_component_layout
   topo.spectral.spectral.spectral_layout
   topo.spectral.spectral.spectral_clustering



.. py:function:: LapEigenmap(affinity_matrix, dim, random_state)


.. py:function:: component_layout(data, n_components, component_labels, dim, random_state, metric='euclidean', metric_kwds={})

   Provide a layout relating the separate connected components. This is done
   by taking the centroid of each component and then performing a spectral embedding
   of the centroids.
   :param data: The source data -- required so we can generate centroids for each
                connected component of the graph.
   :type data: array of shape (n_samples, n_features)
   :param is_dist: Whether if X is a distance or affinity matrix.
   :type is_dist: bool
   :param n_components: The number of distinct components to be layed out.
   :type n_components: int
   :param component_labels: For each vertex in the graph the label of the component to
                            which the vertex belongs.
   :type component_labels: array of shape (n_samples)
   :param dim: The chosen embedding dimension.
   :type dim: int
   :param metric: The metric used to measure distances among the source data points.
   :type metric: string or callable (optional, default 'euclidean')
   :param metric_kwds: Keyword arguments to be passed to the metric function.
                       If metric is 'precomputed', 'linkage' keyword can be used to specify
                       'average', 'complete', or 'single' linkage. Default is 'average'
   :type metric_kwds: dict (optional, default {})

   :returns: **component_embedding** (*array of shape (n_components, dim)*) -- The ``dim``-dimensional embedding of the ``n_components``-many
             connected components.


.. py:function:: multi_component_layout(data, graph, n_components, component_labels, dim, random_state, metric='euclidean', metric_kwds={})

   Specialised layout algorithm for dealing with graphs with many connected components.
   This will first fid relative positions for the components by spectrally embedding
   their centroids, then spectrally embed each individual connected component positioning
   them according to the centroid embeddings. This provides a decent embedding of each
   component while placing the components in good relative positions to one another.
   :param data: The source data -- required so we can generate centroids for each
                connected component of the graph.
   :type data: array of shape (n_samples, n_features)
   :param graph: The adjacency matrix of the graph to be emebdded.
   :type graph: sparse matrix
   :param n_components: The number of distinct components to be layed out.
   :type n_components: int
   :param component_labels: For each vertex in the graph the label of the component to
                            which the vertex belongs.
   :type component_labels: array of shape (n_samples)
   :param dim: The chosen embedding dimension.
   :type dim: int
   :param metric: The metric used to measure distances among the source data points.
   :type metric: string or callable (optional, default 'euclidean')
   :param metric_kwds: Keyword arguments to be passed to the metric function.
   :type metric_kwds: dict (optional, default {})

   :returns: **embedding** (*array of shape (n_samples, dim)*) -- The initial embedding of ``graph``.


.. py:function:: spectral_layout(data, graph, dim, random_state, metric='euclidean', metric_kwds={})

   Given a graph compute the spectral embedding of the graph. This is
   simply the eigenvectors of the laplacian of the graph. Here we use the
   normalized laplacian.
   :param data: The source data
   :type data: array of shape (n_samples, n_features)
   :param graph: The (weighted) adjacency matrix of the graph as a sparse matrix.
   :type graph: sparse matrix
   :param dim: The dimension of the space into which to embed.
   :type dim: int
   :param random_state: A state capable being used as a numpy random state.
   :type random_state: numpy RandomState or equivalent

   :returns: **embedding** (*array of shape (n_vertices, dim)*) -- The spectral embedding of the graph.


.. py:function:: spectral_clustering(init, max_svd_restarts=30, n_iter_max=30, random_state=None, copy=True)

   Search for a partition matrix (clustering) which is closest to the
       eigenvector embedding.

   :param init: The embedding space of the samples.
   :type init: array-like of shape (n_samples, n_clusters)
   :param max_svd_restarts: Maximum number of attempts to restart SVD if convergence fails
   :type max_svd_restarts: int, default=30
   :param n_iter_max: Maximum number of iterations to attempt in rotation and partition
                      matrix search if machine precision convergence is not reached
   :type n_iter_max: int, default=30
   :param random_state: Determines random number generation for rotation matrix initialization.
                        Use an int to make the randomness deterministic.
                        See :term:`Glossary <random_state>`.
   :type random_state: int, RandomState instance, default=None
   :param copy: Whether to copy vectors, or perform in-place normalization.
   :type copy: bool, default=True

   :returns: **labels** (*array of integers, shape: n_samples*) -- The labels of the clusters.

   .. rubric:: References

   - Multiclass spectral clustering, 2003
     Stella X. Yu, Jianbo Shi
     https://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf

   .. rubric:: Notes

   The eigenvector embedding is used to iteratively search for the
   closest discrete partition.  First, the eigenvector embedding is
   normalized to the space of partition matrices. An optimal discrete
   partition matrix closest to this normalized embedding multiplied by
   an initial rotation is calculated.  Fixing this discrete partition
   matrix, an optimal rotation matrix is calculated.  These two
   calculations are performed until convergence.  The discrete partition
   matrix is returned as the clustering solution.  Used in spectral
   clustering, this method tends to be faster and more robust to random
   initialization than k-means.


