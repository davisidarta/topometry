:py:mod:`topo.layouts.graph_utils`
==================================

.. py:module:: topo.layouts.graph_utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   topo.layouts.graph_utils.fuzzy_simplicial_set_ann
   topo.layouts.graph_utils.compute_diff_connectivities
   topo.layouts.graph_utils.get_sparse_matrix_from_indices_distances_dbmap
   topo.layouts.graph_utils.approximate_n_neighbors
   topo.layouts.graph_utils.compute_membership_strengths
   topo.layouts.graph_utils.smooth_knn_dist
   topo.layouts.graph_utils.get_igraph_from_adjacency
   topo.layouts.graph_utils.make_epochs_per_sample
   topo.layouts.graph_utils.simplicial_set_embedding
   topo.layouts.graph_utils.find_ab_params



Attributes
~~~~~~~~~~

.. autoapisummary::

   topo.layouts.graph_utils.ts
   topo.layouts.graph_utils.csr_unique
   topo.layouts.graph_utils.fast_knn_indices
   topo.layouts.graph_utils.optimize_layout_euclidean
   topo.layouts.graph_utils.optimize_layout_generic
   topo.layouts.graph_utils.optimize_layout_inverse
   topo.layouts.graph_utils.SMOOTH_K_TOLERANCE
   topo.layouts.graph_utils.MIN_K_DIST_SCALE
   topo.layouts.graph_utils.NPY_INFINITY
   topo.layouts.graph_utils.INT32_MIN
   topo.layouts.graph_utils.INT32_MAX


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


