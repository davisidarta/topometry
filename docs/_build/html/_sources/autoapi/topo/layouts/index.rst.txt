:py:mod:`topo.layouts`
======================

.. py:module:: topo.layouts


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   graph_utils/index.rst
   map/index.rst
   mde/index.rst
   ncvis/index.rst
   pairwise/index.rst
   trimap/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   topo.layouts.LiteralMeta
   topo.layouts.LiteralMeta



Functions
~~~~~~~~~

.. autoapisummary::

   topo.layouts.fuzzy_simplicial_set_ann
   topo.layouts.compute_diff_connectivities
   topo.layouts.get_sparse_matrix_from_indices_distances_dbmap
   topo.layouts.approximate_n_neighbors
   topo.layouts.compute_membership_strengths
   topo.layouts.smooth_knn_dist
   topo.layouts.get_igraph_from_adjacency
   topo.layouts.make_epochs_per_sample
   topo.layouts.simplicial_set_embedding
   topo.layouts.find_ab_params
   topo.layouts.fuzzy_embedding
   topo.layouts._remove_anchor_anchor_edges
   topo.layouts.IsomorphicMDE
   topo.layouts.IsometricMDE
   topo.layouts.SpectralMDE
   topo.layouts.TriMAP
   topo.layouts.PaCMAP
   topo.layouts.NCVis



Attributes
~~~~~~~~~~

.. autoapisummary::

   topo.layouts.ts
   topo.layouts.csr_unique
   topo.layouts.fast_knn_indices
   topo.layouts.optimize_layout_euclidean
   topo.layouts.optimize_layout_generic
   topo.layouts.optimize_layout_inverse
   topo.layouts.SMOOTH_K_TOLERANCE
   topo.layouts.MIN_K_DIST_SCALE
   topo.layouts.NPY_INFINITY
   topo.layouts.INT32_MIN
   topo.layouts.INT32_MAX


.. py:data:: ts
   

   

.. py:data:: csr_unique
   

   

.. py:data:: fast_knn_indices
   

   

.. py:data:: optimize_layout_euclidean
   

   

.. py:data:: optimize_layout_generic
   

   

.. py:data:: optimize_layout_inverse
   

   

.. py:data:: SMOOTH_K_TOLERANCE
   :annotation: = 1e-05

   

.. py:data:: MIN_K_DIST_SCALE
   :annotation: = 0.001

   

.. py:data:: NPY_INFINITY
   

   

.. py:data:: INT32_MIN
   

   

.. py:data:: INT32_MAX
   

   

.. py:function:: fuzzy_simplicial_set_ann(X, n_neighbors=15, knn_indices=None, knn_dists=None, backend='hnswlib', metric='cosine', n_jobs=None, efC=50, efS=50, M=15, set_op_mix_ratio=1.0, local_connectivity=1.0, apply_set_operations=True, return_dists=False, verbose=False)

   Given a set of data X, a neighborhood size, and a measure of distance
   compute the fuzzy simplicial set (here represented as a fuzzy graph in
   the form of a sparse matrix) associated to the data. This is done by
   locally approximating geodesic distance at each point, creating a fuzzy
   simplicial set for each such point, and then combining all the local
   fuzzy simplicial sets into a global one via a fuzzy union.
   :param X: The data to be modelled as a fuzzy simplicial set.
   :type X: array of shape (n_samples, n_features).
   :param n_neighbors: The number of neighbors to use to approximate geodesic distance.
                       Larger numbers induce more global estimates of the manifold that can
                       miss finer detail, while smaller values will focus on fine manifold
                       structure to the detriment of the larger picture.
   :type n_neighbors: int.
   :param backend: Which backend to use to compute nearest-neighbors. Options for fast, approximate nearest-neighbors
                   are 'hnwslib' (default) and 'nmslib'. For exact nearest-neighbors, use 'sklearn'.
   :type backend: str (optional, default 'hnwslib').
   :param metric: Distance metric for building an approximate kNN graph. Defaults to
                  'cosine'. Users are encouraged to explore different metrics, such as 'euclidean' and 'inner_product'.
                  The 'hamming' and 'jaccard' distances are also available for string vectors.
                   Accepted metrics include NMSLib*, HNSWlib** and sklearn metrics. Some examples are:

                  -'sqeuclidean' (*, **)

                  -'euclidean' (*, **)

                  -'l1' (*)

                  -'lp' - requires setting the parameter ``p`` (*) - similar to Minkowski

                  -'cosine' (*, **)

                  -'inner_product' (**)

                  -'angular' (*)

                  -'negdotprod' (*)

                  -'levenshtein' (*)

                  -'hamming' (*)

                  -'jaccard' (*)

                  -'jansen-shan' (*).
   :type metric: str (optional, default 'cosine').
   :param n_jobs: number of threads to be used in computation. Defaults to 1. The algorithm is highly
                  scalable to multi-threading.
   :type n_jobs: int (optional, default 1).
   :param M: defines the maximum number of neighbors in the zero and above-zero layers during HSNW
             (Hierarchical Navigable Small World Graph). However, the actual default maximum number
             of neighbors for the zero layer is 2*M.  A reasonable range for this parameter
             is 5-100. For more information on HSNW, please check https://arxiv.org/abs/1603.09320.
             HSNW is implemented in python via NMSlib. Please check more about NMSlib at https://github.com/nmslib/nmslib.
   :type M: int (optional, default 30).
   :param efC: A 'hnsw' parameter. Increasing this value improves the quality of a constructed graph
               and leads to higher accuracy of search. However this also leads to longer indexing times.
               A reasonable range for this parameter is 50-2000.
   :type efC: int (optional, default 100).
   :param efS: A 'hnsw' parameter. Similarly to efC, increasing this value improves recall at the
               expense of longer retrieval time. A reasonable range for this parameter is 50-2000.
   :type efS: int (optional, default 100).
   :param knn_indices: If the k-nearest neighbors of each point has already been calculated
                       you can pass them in here to save computation time. This should be
                       an array with the indices of the k-nearest neighbors as a row for
                       each data point.
   :type knn_indices: array of shape (n_samples, n_neighbors) (optional).
   :param knn_dists: If the k-nearest neighbors of each point has already been calculated
                     you can pass them in here to save computation time. This should be
                     an array with the distances of the k-nearest neighbors as a row for
                     each data point.
   :type knn_dists: array of shape (n_samples, n_neighbors) (optional).
   :param set_op_mix_ratio: Interpolate between (fuzzy) union and intersection as the set operation
                            used to combine local fuzzy simplicial sets to obtain a global fuzzy
                            simplicial sets. Both fuzzy set operations use the product t-norm.
                            The value of this parameter should be between 0.0 and 1.0; a value of
                            1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
                            intersection.
   :type set_op_mix_ratio: float (optional, default 1.0).
   :param local_connectivity: The local connectivity required -- i.e. the number of nearest
                              neighbors that should be assumed to be connected at a local level.
                              The higher this value the more connected the manifold becomes
                              locally. In practice this should be not more than the local intrinsic
                              dimension of the manifold.
   :type local_connectivity: int (optional, default 1)
   :param verbose: Whether to report information on the current progress of the algorithm.
   :type verbose: bool (optional, default False)
   :param return_dists: Whether to return the pairwise distance associated with each edge.
   :type return_dists: bool or None (optional, default none)

   :returns: **fuzzy_simplicial_set** (*coo_matrix*) -- A fuzzy simplicial set represented as a sparse matrix. The (i,
             j) entry of the matrix represents the membership strength of the
             1-simplex between the ith and jth sample points.


.. py:function:: compute_diff_connectivities(data, n_components=100, n_neighbors=30, alpha=0.0, n_jobs=10, ann=True, ann_dist='cosine', M=30, efC=100, efS=100, knn_dist='euclidean', kernel_use='simple_adaptive', sensitivity=1, set_op_mix_ratio=1.0, local_connectivity=1.0, verbose=False)

       Sklearn estimator for using fast anisotropic diffusion with an anisotropic
       adaptive algorithm as proposed by Setty et al, 2018, and optimized by Sidarta-Oliveira, 2020.
       This procedure generates diffusion components that effectivelly carry the maximum amount of
       information regarding the data geometric structure (structure components).
       These structure components then undergo a fuzzy-union of simplicial sets. This step is
       from umap.fuzzy_simplicial_set [McInnes18]_. Given a set of data X, a neighborhood size, and a measure of distance
       compute the fuzzy simplicial set (here represented as a fuzzy graph in the form of a sparse matrix) associated to the data. This is done by
       locally approximating geodesic distance at each point, creating a fuzzy simplicial set for each such point, and then combining all the local
       fuzzy simplicial sets into a global one via a fuzzy union.

   :param n_components: analyzing more than 10,000 cells.
   :type n_components: Number of diffusion components to compute. Defaults to 100. We suggest larger values if
   :param n_neighbors: distance of its median neighbor.
   :type n_neighbors: Number of k-nearest-neighbors to compute. The adaptive kernel will normalize distances by each cell
   :param knn_dist:
   :type knn_dist: Distance metric for building kNN graph. Defaults to 'euclidean'.
   :param ann:
   :type ann: Boolean. Whether to use approximate nearest neighbors for graph construction. Defaults to True.
   :param alpha: Defaults to 1, which is suitable for normalized data.
   :type alpha: Alpha in the diffusion maps literature. Controls how much the results are biased by data distribution.
   :param n_jobs:
   :type n_jobs: Number of threads to use in calculations. Defaults to all but one.
   :param sensitivity:
   :type sensitivity: Sensitivity to select eigenvectors if diff_normalization is set to 'knee'. Useful when dealing wit
   :param : resulting components to use during Multiscaling.
   :type : returns: Diffusion components ['EigenVectors'], associated eigenvalues ['EigenValues'] and suggested number of

   .. rubric:: Example

   import numpy as np
   from sklearn.datasets import load_digits
   from scipy.sparse import csr_matrix
   from topo.tpgraph.diffusion import Diffusor

   # Load the MNIST digits data, convert to sparse for speed
   digits = load_digits()
   data = csr_matrix(digits)

   # Fit the anisotropic diffusion process
   tpgraph = Diffusor().fit(data)

   # Find multiscale diffusion components
   mds = tpgraph.transform(data)


.. py:function:: get_sparse_matrix_from_indices_distances_dbmap(knn_indices, knn_dists, n_obs, n_neighbors)


.. py:function:: approximate_n_neighbors(data, n_neighbors=15, metric='cosine', backend='hnswlib', n_jobs=10, efC=50, efS=50, M=15, p=11 / 16, dense=False, verbose=False)

   Simple function using NMSlibTransformer from topodata.ann. This implements a very fast
   and scalable approximate k-nearest-neighbors graph on spaces defined by nmslib.
   Read more about nmslib and its various available metrics at
   https://github.com/nmslib/nmslib. Read more about dbMAP at
   https://github.com/davisidarta/dbMAP.


   :param n_neighbors: this should be considered the average neighborhood size and thus vary depending
                       on your number of features, samples and data intrinsic dimensionality. Reasonable values
                       range from 5 to 100. Smaller values tend to lead to increased graph structure
                       resolution, but users should beware that a too low value may render granulated and vaguely
                       defined neighborhoods that arise as an artifact of downsampling. Defaults to 15. Larger
                       values can slightly increase computational time.
   :type n_neighbors: number of nearest-neighbors to look for. In practice,
   :param backend: Which backend to use to compute nearest-neighbors. Options for fast, approximate nearest-neighbors
                   are 'hnwslib' (default) and 'nmslib'. For exact nearest-neighbors, use 'sklearn'.
   :type backend: str (optional, default 'hnwslib')
   :param metric: Distance metric for building an approximate kNN graph. Defaults to
                  'cosine'. Users are encouraged to explore different metrics, such as 'euclidean' and 'inner_product'.
                  The 'hamming' and 'jaccard' distances are also available for string vectors.
                   Accepted metrics include NMSLib*, HNSWlib** and sklearn metrics. Some examples are:

                  -'sqeuclidean' (*, **)

                  -'euclidean' (*, **)

                  -'l1' (*)

                  -'lp' - requires setting the parameter ``p`` (*) - similar to Minkowski

                  -'cosine' (*, **)

                  -'inner_product' (**)

                  -'angular' (*)

                  -'negdotprod' (*)

                  -'levenshtein' (*)

                  -'hamming' (*)

                  -'jaccard' (*)

                  -'jansen-shan' (*)
   :type metric: str (optional, default 'cosine')
   :param p: P for the Lp metric, when ``metric='lp'``.  Can be fractional. The default 11/16 approximates
             an astroid norm with some computational efficiency (2^n bases are less painstakinly slow to compute).
             See https://en.wikipedia.org/wiki/Lp_space for some context.
   :type p: int or float (optional, default 11/16 )
   :param n_jobs:
   :type n_jobs: number of threads to be used in computation. Defaults to 10 (~5 cores).
   :param efC: accuracy of search. However this also leads to longer indexing times. A reasonable
               range is 100-2000. Defaults to 100.
   :type efC: increasing this value improves the quality of a constructed graph and leads to higher
   :param efS: retrieval time. A reasonable range is 100-2000.
   :type efS: similarly to efC, improving this value improves recall at the expense of longer
   :param M: (Hierarchical Navigable Small World Graph). However, the actual default maximum number
             of neighbors for the zero layer is 2*M. For more information on HSNW, please check
             https://arxiv.org/abs/1603.09320. HSNW is implemented in python via NMSLIB. Please check
             more about NMSLIB at https://github.com/nmslib/nmslib .
   :type M: defines the maximum number of neighbors in the zero and above-zero layers during HSNW

   :returns: *k-nearest-neighbors indices and distances. Can be customized to also return* -- return the k-nearest-neighbors graph and its gradient.

   .. rubric:: Example

   knn_indices, knn_dists = approximate_n_neighbors(data)


.. py:function:: compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos)

   Construct the membership strength data for the 1-skeleton of each local
   fuzzy simplicial set -- this is formed as a sparse matrix where each row is
   a local fuzzy simplicial set, with a membership strength for the
   1-simplex to each other data point.
   :param knn_indices: The indices on the ``n_neighbors`` closest points in the dataset.
   :type knn_indices: array of shape (n_samples, n_neighbors)
   :param knn_dists: The distances to the ``n_neighbors`` closest points in the dataset.
   :type knn_dists: array of shape (n_samples, n_neighbors)
   :param sigmas: The normalization factor derived from the metric tensor approximation.
   :type sigmas: array of shape(n_samples)
   :param rhos: The local connectivity adjustment.
   :type rhos: array of shape(n_samples)

   :returns: * **rows** (*array of shape (n_samples * n_neighbors)*) -- Row data for the resulting sparse matrix (coo format)
             * **cols** (*array of shape (n_samples * n_neighbors)*) -- Column data for the resulting sparse matrix (coo format)
             * **vals** (*array of shape (n_samples * n_neighbors)*) -- Entries for the resulting sparse matrix (coo format)


.. py:function:: smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0, bandwidth=1.0)

   Compute a continuous version of the distance to the kth nearest
   neighbor. That is, this is similar to knn-distance but allows continuous
   k values rather than requiring an integral k. In essence we are simply
   computing the distance such that the cardinality of fuzzy set we generate
   is k.
   :param distances: Distances to nearest neighbors for each samples. Each row should be a
                     sorted list of distances to a given samples nearest neighbors.
   :type distances: array of shape (n_samples, n_neighbors)
   :param k: The number of nearest neighbors to approximate for.
   :type k: float
   :param n_iter: We need to binary search for the correct distance value. This is the
                  max number of iterations to use in such a search.
   :type n_iter: int (optional, default 64)
   :param local_connectivity: The local connectivity required -- i.e. the number of nearest
                              neighbors that should be assumed to be connected at a local level.
                              The higher this value the more connected the manifold becomes
                              locally. In practice this should be not more than the local intrinsic
                              dimension of the manifold.
   :type local_connectivity: int (optional, default 1)
   :param bandwidth: The target bandwidth of the kernel, larger values will produce
                     larger return values.
   :type bandwidth: float (optional, default 1)

   :returns: * **knn_dist** (*array of shape (n_samples,)*) -- The distance to kth nearest neighbor, as suitably approximated.
             * **nn_dist** (*array of shape (n_samples,)*) -- The distance to the 1st nearest neighbor for each point.


.. py:function:: get_igraph_from_adjacency(adjacency, directed=None)

   Get igraph graph from adjacency matrix.


.. py:function:: make_epochs_per_sample(weights, n_epochs)

   Given a set of weights and number of epochs generate the number of
   epochs per sample for each weight.
   :param weights: The weights ofhow much we wish to sample each 1-simplex.
   :type weights: array of shape (n_1_simplices)
   :param n_epochs: The total number of epochs we want to train for.
   :type n_epochs: int

   :returns: *An array of number of epochs per sample, one for each 1-simplex.*


.. py:function:: simplicial_set_embedding(graph, n_components, initial_alpha, a, b, gamma, negative_sample_rate, n_epochs, init, random_state, metric, metric_kwds, densmap, densmap_kwds, output_dens, output_metric=dist.named_distances_with_gradients['euclidean'], output_metric_kwds={}, euclidean_output=True, parallel=False, verbose=False)

   Perform a fuzzy simplicial set embedding, using a specified
   initialisation method and then minimizing the fuzzy set cross entropy
   between the 1-skeletons of the high and low dimensional fuzzy simplicial
   sets.
   :param graph: The 1-skeleton of the high dimensional fuzzy simplicial set as
                 represented by a graph for which we require a sparse matrix for the
                 (weighted) adjacency matrix.
   :type graph: sparse matrix
   :param n_components: The dimensionality of the euclidean space into which to embed the data.
   :type n_components: int
   :param initial_alpha: Initial learning rate for the SGD.
   :type initial_alpha: float
   :param a: Parameter of differentiable approximation of right adjoint functor
   :type a: float
   :param b: Parameter of differentiable approximation of right adjoint functor
   :type b: float
   :param gamma: Weight to apply to negative samples.
   :type gamma: float
   :param negative_sample_rate: The number of negative samples to select per positive sample
                                in the optimization process. Increasing this value will result
                                in greater repulsive force being applied, greater optimization
                                cost, but slightly more accuracy.
   :type negative_sample_rate: int (optional, default 5)
   :param n_epochs: The number of training epochs to be used in optimizing the
                    low dimensional embedding. Larger values result in more accurate
                    embeddings. If 0 is specified a value will be selected based on
                    the size of the input dataset (200 for large datasets, 500 for small).
   :type n_epochs: int (optional, default 0)
   :param init:
                How to initialize the low dimensional embedding. Options are:
                    * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
                    * 'random': assign initial embedding positions at random.
                    * A numpy array of initial embedding positions.
   :type init: string
   :param random_state: A state capable being used as a numpy random state.
   :type random_state: numpy RandomState or equivalent
   :param metric: The metric used to measure distance in high dimensional space; used if
                  multiple connected components need to be layed out.
   :type metric: string or callable
   :param metric_kwds: Key word arguments to be passed to the metric function; used if
                       multiple connected components need to be layed out.
   :type metric_kwds: dict
   :param densmap: Whether to use the density-augmented objective function to optimize
                   the embedding according to the densMAP algorithm.
   :type densmap: bool
   :param densmap_kwds: Key word arguments to be used by the densMAP optimization.
   :type densmap_kwds: dict
   :param output_dens: Whether to output local radii in the original data and the embedding.
   :type output_dens: bool
   :param output_metric: Function returning the distance between two points in embedding space and
                         the gradient of the distance wrt the first argument.
   :type output_metric: function
   :param output_metric_kwds: Key word arguments to be passed to the output_metric function.
   :type output_metric_kwds: dict
   :param euclidean_output: Whether to use the faster code specialised for euclidean output metrics
   :type euclidean_output: bool
   :param parallel: Whether to run the computation using numba parallel.
                    Running in parallel is non-deterministic, and is not used
                    if a random seed has been set, to ensure reproducibility.
   :type parallel: bool (optional, default False)
   :param verbose: Whether to report information on the current progress of the algorithm.
   :type verbose: bool (optional, default False)

   :returns: * **embedding** (*array of shape (n_samples, n_components)*) -- The optimized of ``graph`` into an ``n_components`` dimensional
               euclidean space.
             * **aux_data** (*dict*) -- Auxiliary output returned with the embedding. When densMAP extension
               is turned on, this dictionary includes local radii in the original
               data (``rad_orig``) and in the embedding (``rad_emb``).


.. py:function:: find_ab_params(spread, min_dist)

   Fit a, b params for the differentiable curve used in lower
   dimensional fuzzy simplicial complex construction. We want the
   smooth curve (from a pre-defined family with simple gradient) that
   best matches an offset exponential decay.


.. py:function:: fuzzy_embedding(graph, n_components=2, initial_alpha=1, min_dist=0.3, spread=1.2, n_epochs=500, metric='cosine', metric_kwds={}, output_metric='euclidean', output_metric_kwds={}, gamma=1.2, negative_sample_rate=10, init='spectral', random_state=None, euclidean_output=True, parallel=True, verbose=False, a=None, b=None, densmap=False, densmap_kwds={}, output_dens=False)

   Perform a fuzzy simplicial set embedding, using a specified
   initialisation method and then minimizing the fuzzy set cross entropy
   between the 1-skeletons of the high and low dimensional fuzzy simplicial
   sets. The fuzzy simplicial set embedding was proposed and implemented by
   Leland McInnes in UMAP (see `umap-learn <https://github.com/lmcinnes/umap>`).
   Here we're using it only for the projection (layout optimization).

   :param graph: The 1-skeleton of the high dimensional fuzzy simplicial set as
                 represented by a graph for which we require a sparse matrix for the
                 (weighted) adjacency matrix.
   :type graph: sparse matrix
   :param n_components: The dimensionality of the euclidean space into which to embed the data.
   :type n_components: int
   :param initial_alpha: Initial learning rate for the SGD.
   :type initial_alpha: float
   :param a: Parameter of differentiable approximation of right adjoint functor
   :type a: float
   :param b: Parameter of differentiable approximation of right adjoint functor
   :type b: float
   :param gamma: Weight to apply to negative samples.
   :type gamma: float
   :param negative_sample_rate: The number of negative samples to select per positive sample
                                in the optimization process. Increasing this value will result
                                in greater repulsive force being applied, greater optimization
                                cost, but slightly more accuracy.
   :type negative_sample_rate: int (optional, default 5)
   :param n_epochs: The number of training epochs to be used in optimizing the
                    low dimensional embedding. Larger values result in more accurate
                    embeddings. If 0 is specified a value will be selected based on
                    the size of the input dataset (200 for large datasets, 500 for small).
   :type n_epochs: int (optional, default 0)
   :param init:
                How to initialize the low dimensional embedding. Options are:
                    * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
                    * 'random': assign initial embedding positions at random.
                    * A numpy array of initial embedding positions.
   :type init: string
   :param random_state: A state capable being used as a numpy random state.
   :type random_state: numpy RandomState or equivalent
   :param metric: The metric used to measure distance in high dimensional space; used if
                  multiple connected components need to be layed out.
   :type metric: string or callable
   :param metric_kwds: Key word arguments to be passed to the metric function; used if
                       multiple connected components need to be layed out.
   :type metric_kwds: dict
   :param densmap: Whether to use the density-augmented objective function to optimize
                   the embedding according to the densMAP algorithm.
   :type densmap: bool
   :param densmap_kwds: Key word arguments to be used by the densMAP optimization.
   :type densmap_kwds: dict
   :param output_dens: Whether to output local radii in the original data and the embedding.
   :type output_dens: bool
   :param output_metric: Function returning the distance between two points in embedding space and
                         the gradient of the distance wrt the first argument.
   :type output_metric: function
   :param output_metric_kwds: Key word arguments to be passed to the output_metric function.
   :type output_metric_kwds: dict
   :param euclidean_output: Whether to use the faster code specialised for euclidean output metrics
   :type euclidean_output: bool
   :param parallel: Whether to run the computation using numba parallel.
                    Running in parallel is non-deterministic, and is not used
                    if a random seed has been set, to ensure reproducibility.
   :type parallel: bool (optional, default False)
   :param verbose: Whether to report information on the current progress of the algorithm.
   :type verbose: bool (optional, default False)

   :returns: * **embedding** (*array of shape (n_samples, n_components)*) -- The optimized of ``graph`` into an ``n_components`` dimensional
               euclidean space.
             * **aux_data** (*dict*) -- Auxiliary output returned with the embedding. When densMAP extension
               is turned on, this dictionary includes local radii in the original
               data (``rad_orig``) and in the embedding (``rad_emb``).
                   Y_init : array of shape (n_samples, n_components)
                       The spectral initialization of ``graph`` into an ``n_components`` dimensional
                       euclidean space.


.. py:function:: _remove_anchor_anchor_edges(edges, data, anchors)


.. py:function:: IsomorphicMDE(data, attractive_penalty=penalties.Log1p, repulsive_penalty=penalties.Log, embedding_dim=2, constraint=None, n_neighbors=None, repulsive_fraction=None, max_distance=None, init='quadratic', device='cpu', verbose=False)

   Construct an MDE problem designed to preserve local structure.
   This function constructs an MDE problem for preserving the
   local structure of original data. This MDE problem is well-suited for
   visualization (using ``embedding_dim`` 2 or 3), but can also be used to
   generate features for machine learning tasks (with ``embedding_dim`` = 10,
   50, or 100, for example). It yields embeddings in which similar items
   are near each other, and dissimilar items are not near each other.
   The original data can either be a data matrix, or a graph.
   Data matrices should be torch Tensors, NumPy arrays, or scipy sparse
   matrices; graphs should be instances of ``pymde.Graph``.
   The MDE problem uses distortion functions derived from weights (i.e.,
   penalties).
   To obtain an embedding, call the ``embed`` method on the returned ``MDE``
   object. To plot it, use ``pymde.plot``.
   .. code:: python3
       embedding = pymde.preserve_neighbors(data).embed()
       pymde.plot(embedding)
   :param data: The original data, a data matrix of shape ``(n_items, n_features)`` or
                a graph. Neighbors are computed using Euclidean distance if the data is
                a matrix, or the shortest-path metric if the data is a graph.
   :type data: {torch.Tensor, numpy.ndarray, scipy.sparse matrix} or pymde.Graph
   :param embedding_dim: The embedding dimension. Use 2 or 3 for visualization.
   :type embedding_dim: int
   :param attractive_penalty: Callable that constructs a distortion function, given positive
                              weights. Typically one of the classes from ``pymde.penalties``,
                              such as ``pymde.penalties.log1p``, ``pymde.penalties.Huber``, or
                              ``pymde.penalties.Quadratic``.
   :type attractive_penalty: pymde.Function class (or factory)
   :param repulsive_penalty: Callable that constructs a distortion function, given negative
                             weights. (If ``None``, only positive weights are used.) For example,
                             ``pymde.penalties.Log`` or ``pymde.penalties.InversePower``.
   :type repulsive_penalty: pymde.Function class (or factory)
   :param constraint: Embedding constraint, like ``pymde.Standardized()`` or
                      ``pymde.Anchored(anchors, values)`` (or ``None``). Defaults to no
                      constraint when a repulsive penalty is provided, otherwise defaults to
                      ``pymde.Standardized()``.
   :type constraint: pymde.constraints.Constraint (optional)
   :param n_neighbors: The number of nearest neighbors to compute for each row (item) of
                       ``data``. A sensible value is chosen by default, depending on the
                       number of items.
   :type n_neighbors: int (optional)
   :param repulsive_fraction: How many repulsive edges to include, relative to the number
                              of attractive edges. ``1`` means as many repulsive edges as attractive
                              edges. The higher this number, the more uniformly spread out the
                              embedding will be. Defaults to ``0.5`` for standardized embeddings, and
                              ``1`` otherwise. (If ``repulsive_penalty`` is ``None``, this argument
                              is ignored.)
   :type repulsive_fraction: float (optional)
   :param max_distance: If not None, neighborhoods are restricted to have a radius
                        no greater than ``max_distance``.
   :type max_distance: float (optional)
   :param init: Initialization strategy; np.ndarray, 'quadratic' or 'random'.
   :type init: str or np.ndarray (optional, default 'quadratic')
   :param device: Device for the embedding (eg, 'cpu', 'cuda').
   :type device: str (optional)
   :param verbose: If ``True``, print verbose output.
   :type verbose: bool

   :returns: *pymde.MDE* -- A ``pymde.MDE`` object, based on the original data.


.. py:function:: IsometricMDE(data, embedding_dim=2, loss=losses.Absolute, constraint=None, max_distances=50000000.0, device='cpu', verbose=False)

   Construct an MDE problem based on original distances.
   This function constructs an MDE problem for preserving pairwise
   distances between items. This can be useful for preserving the global
   structure of the data.
   The data can be specified with either a data matrix (a NumPy array, torch
   Tensor, or sparse matrix), or a ``pymde.Graph`` instance encoding the
   distances:
       A NumPy array, torch tensor, or sparse matrix is interpreted as a
       collection of feature vectors: each row gives the feature vector for an
       item. The original distances are the Euclidean distances between the
       feature vectors.
       A ``pymde.Graph`` instance is interpreted as encoding all (n_items
       choose 2) distances: the distance between i and j is taken to be the
       length of the shortest path connecting i and j.
   When the number of items n_items is large, the total number of pairs will
   be very large. When this happens, instead of computing all pairs of
   distances, this function will sample a subset uniformly at random. The
   maximum number of distances to compute is specified by the parameter
   ``max_distances``. Depending on how many items you have (and how much
   memory your machine has), you may need to adjust this parameter.
   To obtain an embedding, call the ``embed`` method on the returned object.
   To plot it, use ``pymde.plot``.
   For example:
   .. code:: python3
       embedding = pymde.preserve_distances(data).embed()
       pymde.plot(embedding)
   :param data: The original data, a data matrix of shape ``(n_items, n_features)`` or
                a graph.
   :type data: {np.ndarray, torch.Tensor, scipy.sparse matrix} or pymde.Graph
   :param embedding_dim: The embedding dimension.
   :type embedding_dim: int
   :param loss: Callable that constructs a distortion function, given
                original distances. Typically one of the classes defined in
                ``pymde.losses``, such as ``pymde.losses.Absolute``, or
                ``pymde.losses.WeightedQuadratic``.
   :type loss: pymde.Function class (or factory)
   :param constraint: Embedding constraint, such as ``pymde.Standardized()`` or
                      ``pymde.Anchored(anchors, values)`` (or ``None``). Defaults to no
                      constraint. Note: when the constraint is ``pymde.Standardized()``,
                      the original distances will be scaled by a constant (because the
                      standardization constraint puts a limit on how large any one
                      distance can be).
   :type constraint: pymde.constraints.Constraint (optional)
   :param max_distances: Maximum number of distances to compute.
   :type max_distances: int
   :param device: Device for the embedding (eg, 'cpu', 'cuda').
   :type device: str (optional)
   :param verbose: If ``True``, print verbose output.
   :type verbose: bool

   :returns: *pymde.MDE* -- A ``pymde.MDE`` instance, based on preserving the original distances.


.. py:function:: SpectralMDE(data, edges, weights, embedding_dim=2, cg=False, max_iter=40, device='cpu')

   Performs spectral embedding (very useful for initializations).
   :param data: Input data or graph
   :type data: np.ndarray, torch.tensor, sp.csr_matrix or pymde.graph
   :param edges: Tensor of edges. Optional if `data` is a pymde.graph
   :type edges: torch.tensor, optional
   :param weights: Tensor of weights. Optional if `data` is a pymde.graph
   :type weights: torch.tensor, optional
   :param embedding_dim: Output dimension space to reduce the graph to.
   :type embedding_dim: int, optional, default 2
   :param cg: If True, uses a preconditioned CG method to find the embedding,
              which requires that the Laplacian matrix plus the identity is
              positive definite; otherwise, a Lanczos method is used. Use True when
              the Lanczos method is too slow (which might happen when the number of
              edges is very large).
   :type cg: bool
   :param max_iter: max iteration count for the CG method
   :type max_iter: int
   :param device:
   :type device: str, optional, default 'cpu'

   :returns: * *The output of an appropriately fit pymde.quadratic.spectral problem, with shape (n_items, embedding_dim).*
             * *n_items is the number of samples from the input data or graph.*


.. py:class:: LiteralMeta

   Bases: :py:obj:`type`

   .. py:method:: __getitem__(cls, values)



.. py:function:: TriMAP(X, init=None, n_dims=2, n_inliers=10, n_outliers=5, n_random=5, distance='euclidean', lr=1000.0, n_iters=400, triplets=None, weights=None, use_dist_matrix=False, knn_tuple=None, verbose=True, weight_adj=500.0, opt_method='dbd', return_seq=False)

   Dimensionality Reduction Using Triplet Constraints
   Find a low-dimensional representation of the data by satisfying the sampled
   triplet constraints from the high-dimensional features.


   Inputs
   ------
   n_dims : Number of dimensions of the embedding (default = 2)

   n_inliers : Number of inlier points for triplet constraints (default = 10)

   n_outliers : Number of outlier points for triplet constraints (default = 5)

   n_random : Number of random triplet constraints per point (default = 5)

   distance : Distance measure ('euclidean' (default), 'manhattan', 'angular',
   'hamming')

   lr : Learning rate (default = 1000.0)

   n_iters : Number of iterations (default = 400)

   use_dist_matrix : X is the pairwise distances between points (default = False)

   knn_tuple : Use the pre-computed nearest-neighbors information in form of a
   tuple (knn_nbrs, knn_distances), needs also X to compute the embedding (default = None)


   opt_method : Optimization method ('sd': steepest descent,  'momentum': GD
   with momentum, 'dbd': GD with momentum delta-bar-delta (default))

   verbose : Print the progress report (default = True)

   weight_adj : Adjusting the weights using a non-linear transformation
   (default = 500.0)

   return_seq : Return the sequence of maps recorded every 10 iterations
   (default = False)


.. py:class:: LiteralMeta

   Bases: :py:obj:`type`

   .. py:method:: __getitem__(cls, values)



.. py:function:: PaCMAP(data=None, init=None, n_dims=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, pair_neighbors=None, pair_MN=None, pair_FP=None, distance='euclidean', lr=1.0, num_iters=450, verbose=False, intermediate=False)

       Dimensionality Reduction Using Pairwise-controlled Manifold Approximation and Projectio

       Inputs
       ------
       data : np.array with the data to be reduced

       init : the initialization of the lower dimensional embedding. One of "pca" or "random", or a user-provided numpy ndarray with the shape (N, 2). Default to "random".

       n_dims :  the number of dimension of the output. Default to 2.

       n_neighbors : the number of neighbors considered in the k-Nearest Neighbor graph. Default to 10 for dataset whose
           sample size is smaller than 10000. For large dataset whose sample size (n) is larger than 10000, the default value
           is: 10 + 15 * (log10(n) - 4).

       MN_ratio :  the ratio of the number of mid-near pairs to the number of neighbors, n_MN = \lfloor n_neighbors * MN_ratio
   floor .
        Default to 0.5.

       FP_ratio : the ratio of the number of further pairs to the number of neighbors, n_FP = \lfloor n_neighbors * FP_ratio
   floor Default to 2.

       distance : Distance measure ('euclidean' (default), 'manhattan', 'angular',
       'hamming')


       lr : Optimization method ('sd': steepest descent,  'momentum': GD
       with momentum, 'dbd': GD with momentum delta-bar-delta (default))

       num_iters : number of iterations. Default to 450. 450 iterations is enough for most dataset to converge.

       pair_neighbors, pair_MN and pair_FP: pre-specified neighbor pairs, mid-near points, and further pairs. Allows user to use their own graphs. Default to None.

       verbose : controls verbosity (default False)

       intermediate : whether pacmap should also output the intermediate stages of the optimization process of the lower dimension embedding. If True, then the output will be a numpy array of the size (n, n_dims, 13), where each slice is a "screenshot" of the output embedding at a particular number of steps, from [0, 10, 30, 60, 100, 120, 140, 170, 200, 250, 300, 350, 450].

       random_state :
           RandomState object (default None)




.. py:function:: NCVis(data, n_components=2, n_jobs=-1, n_neighbors=15, distance='cosine', M=15, efC=30, random_seed=42, n_epochs=50, n_init_epochs=20, spread=1.0, min_dist=0.4, alpha=1.0, a=None, b=None, alpha_Q=1.0, n_noise=None)

   Runs Noise Contrastive Visualization ((NCVis)[https://dl.acm.org/doi/abs/10.1145/3366423.3380061])
   for dimensionality reduction and graph layout .

   :param n_components: Desired dimensionality of the embedding.
   :type n_components: int
   :param n_jobs: The maximum number of threads to use. In case n_threads < 1, it defaults to the number of available CPUs.
   :type n_jobs: int
   :param n_neighbors: Number of nearest neighbours in the high dimensional space to consider.
   :type n_neighbors: int
   :param M: The number of bi-directional links created for every new element during construction of HNSW.
             See https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
   :type M: int
   :param efC: The size of the dynamic list for the nearest neighbors (used during the search) in HNSW.
               See https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
   :type efC: int
   :param random_seed: Random seed to initialize the generators. Notice, however, that the result may still depend on the number of threads.
   :type random_seed: int
   :param n_epochs: The total number of epochs to run. During one epoch the positions of each nearest neighbors pair are updated.
   :type n_epochs: int
   :param n_init_epochs: The number of epochs used for initialization. During one epoch the positions of each nearest neighbors pair are updated.
   :type n_init_epochs: int
   :param spread: The effective scale of embedded points. In combination with ``min_dist``
                  this determines how clustered/clumped the embedded points are.
                  See https://github.com/lmcinnes/umap/blob/834184f9c0455f26db13ab148c0abd2d3767d968/umap/umap_.py#L1143
   :type spread: float
   :param min_dist: The effective minimum distance between embedded points. Smaller values
                    will result in a more clustered/clumped embedding where nearby points
                    on the manifold are drawn closer together, while larger values will
                    result on a more even dispersal of points. The value should be set
                    relative to the ``spread`` value, which determines the scale at which
                    embedded points will be spread out.
                    See https://github.com/lmcinnes/umap/blob/834184f9c0455f26db13ab148c0abd2d3767d968/umap/umap_.py#L1135
   :type min_dist: float
   :param a: More specific parameters controlling the embedding. If None these values
             are set automatically as determined by ``min_dist`` and ``spread``.
             See https://github.com/lmcinnes/umap/blob/834184f9c0455f26db13ab148c0abd2d3767d968/umap/umap_.py#L1179
   :type a: (optional, default None)
   :param b: More specific parameters controlling the embedding. If None these values
             are set automatically as determined by ``min_dist`` and ``spread``.
             See https://github.com/lmcinnes/umap/blob/834184f9c0455f26db13ab148c0abd2d3767d968/umap/umap_.py#L1183
   :type b: (optional, default None)
   :param alpha: Learning rate for the embedding positions.
   :type alpha: float
   :param alpha_Q: Learning rate for the normalization constant.
   :type alpha_Q: float
   :param n_noise:
                   Number of noise samples to use per data sample. If ndarray is provided, n_epochs is set to its length.
                    If n_noise is None, it is set to dynamic sampling with noise level gradually increasing
                     from 0 to fixed value.
   :type n_noise: int or ndarray of ints
   :param distance: Distance to use for nearest neighbors search.
   :type distance: str {'euclidean', 'cosine', 'correlation', 'inner_product'}


