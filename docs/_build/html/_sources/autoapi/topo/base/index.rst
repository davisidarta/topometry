:py:mod:`topo.base`
===================

.. py:module:: topo.base


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   ann/index.rst
   dists/index.rst
   sparse/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   topo.base.NMSlibTransformer



Functions
~~~~~~~~~

.. autoapisummary::

   topo.base.sign
   topo.base.euclidean
   topo.base.euclidean_grad
   topo.base.standardised_euclidean
   topo.base.standardised_euclidean_grad
   topo.base.manhattan
   topo.base.manhattan_grad
   topo.base.chebyshev
   topo.base.chebyshev_grad
   topo.base.minkowski
   topo.base.minkowski_grad
   topo.base.poincare
   topo.base.hyperboloid_grad
   topo.base.weighted_minkowski
   topo.base.weighted_minkowski_grad
   topo.base.mahalanobis
   topo.base.mahalanobis_grad
   topo.base.hamming
   topo.base.canberra
   topo.base.canberra_grad
   topo.base.bray_curtis
   topo.base.bray_curtis_grad
   topo.base.jaccard
   topo.base.matching
   topo.base.dice
   topo.base.kulsinski
   topo.base.rogers_tanimoto
   topo.base.russellrao
   topo.base.sokal_michener
   topo.base.sokal_sneath
   topo.base.haversine
   topo.base.haversine_grad
   topo.base.yule
   topo.base.cosine
   topo.base.cosine_grad
   topo.base.correlation
   topo.base.hellinger
   topo.base.hellinger_grad
   topo.base.approx_log_Gamma
   topo.base.log_beta
   topo.base.log_single_beta
   topo.base.ll_dirichlet
   topo.base.symmetric_kl
   topo.base.symmetric_kl_grad
   topo.base.correlation_grad
   topo.base.sinkhorn_distance
   topo.base.spherical_gaussian_energy_grad
   topo.base.diagonal_gaussian_energy_grad
   topo.base.gaussian_energy_grad
   topo.base.spherical_gaussian_grad
   topo.base.get_discrete_params
   topo.base.categorical_distance
   topo.base.hierarchical_categorical_distance
   topo.base.ordinal_distance
   topo.base.count_distance
   topo.base.levenshtein
   topo.base.parallel_special_metric
   topo.base.chunked_parallel_special_metric
   topo.base.pairwise_special_metric
   topo.base.norm
   topo.base.arr_unique
   topo.base.arr_union
   topo.base.arr_intersect
   topo.base.sparse_sum
   topo.base.sparse_diff
   topo.base.sparse_mul
   topo.base.general_sset_intersection
   topo.base.general_sset_union
   topo.base.sparse_euclidean
   topo.base.sparse_manhattan
   topo.base.sparse_chebyshev
   topo.base.sparse_minkowski
   topo.base.sparse_hamming
   topo.base.sparse_canberra
   topo.base.sparse_bray_curtis
   topo.base.sparse_jaccard
   topo.base.sparse_matching
   topo.base.sparse_dice
   topo.base.sparse_kulsinski
   topo.base.sparse_rogers_tanimoto
   topo.base.sparse_russellrao
   topo.base.sparse_sokal_michener
   topo.base.sparse_sokal_sneath
   topo.base.sparse_cosine
   topo.base.sparse_hellinger
   topo.base.sparse_correlation
   topo.base.approx_log_Gamma
   topo.base.log_beta
   topo.base.log_single_beta
   topo.base.sparse_ll_dirichlet



Attributes
~~~~~~~~~~

.. autoapisummary::

   topo.base._mock_identity
   topo.base._mock_cost
   topo.base._mock_ones
   topo.base.named_distances
   topo.base.named_distances_with_gradients
   topo.base.DISCRETE_METRICS
   topo.base.SPECIAL_METRICS
   topo.base.sparse_named_distances
   topo.base.sparse_need_n_features
   topo.base.SPARSE_SPECIAL_METRICS


.. py:class:: NMSlibTransformer(n_neighbors=15, metric='cosine', method='hnsw', n_jobs=10, p=None, M=15, efC=50, efS=50, dense=False, verbose=False)

   Bases: :py:obj:`sklearn.base.TransformerMixin`, :py:obj:`sklearn.base.BaseEstimator`

   Wrapper for using nmslib as sklearn's KNeighborsTransformer. This implements
   an escalable approximate k-nearest-neighbors graph on spaces defined by nmslib.
   Read more about nmslib and its various available metrics at
   https://github.com/nmslib/nmslib.
   Calling 'nn <- NMSlibTransformer()' initializes the class with
    neighbour search parameters.

   :param n_neighbors: number of nearest-neighbors to look for. In practice,
                       this should be considered the average neighborhood size and thus vary depending
                       on your number of features, samples and data intrinsic dimensionality. Reasonable values
                       range from 5 to 100. Smaller values tend to lead to increased graph structure
                       resolution, but users should beware that a too low value may render granulated and vaguely
                       defined neighborhoods that arise as an artifact of downsampling. Defaults to 30. Larger
                       values can slightly increase computational time.
   :type n_neighbors: int (optional, default 30)
   :param metric: Accepted NMSLIB metrics. Defaults to 'cosine'. Accepted metrics include:
                  -'sqeuclidean'
                  -'euclidean'
                  -'l1'
                  -'lp' - requires setting the parameter `p` - equivalent to minkowski distance
                  -'cosine'
                  -'angular'
                  -'negdotprod'
                  -'levenshtein'
                  -'hamming'
                  -'jaccard'
                  -'jansen-shan'
   :type metric: str (optional, default 'cosine').
   :param method:
                  approximate-neighbor search method. Available methods include:
                          -'hnsw' : a Hierarchical Navigable Small World Graph.
                          -'sw-graph' : a Small World Graph.
                          -'vp-tree' : a Vantage-Point tree with a pruning rule adaptable to non-metric distances.
                          -'napp' : a Neighborhood APProximation index.
                          -'simple_invindx' : a vanilla, uncompressed, inverted index, which has no parameters.
                          -'brute_force' : a brute-force search, which has no parameters.
                  'hnsw' is usually the fastest method, followed by 'sw-graph' and 'vp-tree'.
   :type method: str (optional, default 'hsnw').
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
               expense of longer retrieval time. A reasonable range for this parameter is 100-2000.
   :type efS: int (optional, default 100).
   :param dense: Whether to force the algorithm to use dense data, such as np.ndarrays and pandas DataFrames.
   :type dense: bool (optional, default False).

   :returns: *Class for really fast approximate-nearest-neighbors search.*

   .. rubric:: Example

   import numpy as np
   from sklearn.datasets import load_digits
   from scipy.sparse import csr_matrix
   from topo.base.ann import NMSlibTransformer
   #
   # Load the MNIST digits data, convert to sparse for speed
   digits = load_digits()
   data = csr_matrix(digits)
   #
   # Start class with parameters
   nn = NMSlibTransformer()
   nn = nn.fit(data)
   #
   # Obtain kNN graph
   knn = nn.transform(data)
   #
   # Obtain kNN indices, distances and distance gradient
   ind, dist, grad = nn.ind_dist_grad(data)
   #
   # Test for recall efficiency during approximate nearest neighbors search
   test = nn.test_efficiency(data)

   .. py:method:: fit(self, data)


   .. py:method:: transform(self, data)


   .. py:method:: ind_dist_grad(self, data, return_grad=True, return_graph=True)


   .. py:method:: test_efficiency(self, data, data_use=0.1)

      Test if NMSlibTransformer and KNeighborsTransformer give same results



   .. py:method:: update_search(self, n_neighbors)

      Updates number of neighbors for kNN distance computation.
      :param n_neighbors:
      :type n_neighbors: New number of neighbors to look for.


   .. py:method:: fit_transform(self, X)

      Fit to data, then transform it.

      Fits transformer to `X` and `y` with optional parameters `fit_params`
      and returns a transformed version of `X`.

      :param X: Input samples.
      :type X: array-like of shape (n_samples, n_features)
      :param y: Target values (None for unsupervised transformations).
      :type y: array-like of shape (n_samples,) or (n_samples, n_outputs),                 default=None
      :param \*\*fit_params: Additional fit parameters.
      :type \*\*fit_params: dict

      :returns: **X_new** (*ndarray array of shape (n_samples, n_features_new)*) -- Transformed array.



.. py:data:: _mock_identity
   

   

.. py:data:: _mock_cost
   

   

.. py:data:: _mock_ones
   

   

.. py:function:: sign(a)


.. py:function:: euclidean(x, y)

   Standard euclidean distance.
   ..math::
       D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}


.. py:function:: euclidean_grad(x, y)

   Standard euclidean distance and its gradient.
       ..math::
           D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}

   rac{dD(x, y)}{dx} = (x_i - y_i)/D(x,y)



.. py:function:: standardised_euclidean(x, y, sigma=_mock_ones)

   Euclidean distance standardised against a vector of standard
       deviations per coordinate.
       ..math::
           D(x, y) = \sqrt{\sum_i
   rac{(x_i - y_i)**2}{v_i}}



.. py:function:: standardised_euclidean_grad(x, y, sigma=_mock_ones)

   Euclidean distance standardised against a vector of standard
       deviations per coordinate with gradient.
       ..math::
           D(x, y) = \sqrt{\sum_i
   rac{(x_i - y_i)**2}{v_i}}



.. py:function:: manhattan(x, y)

   Manhattan, taxicab, or l1 distance.
   ..math::
       D(x, y) = \sum_i |x_i - y_i|


.. py:function:: manhattan_grad(x, y)

   Manhattan, taxicab, or l1 distance with gradient.
   ..math::
       D(x, y) = \sum_i |x_i - y_i|


.. py:function:: chebyshev(x, y)

   Chebyshev or l-infinity distance.
   ..math::
       D(x, y) = \max_i |x_i - y_i|


.. py:function:: chebyshev_grad(x, y)

   Chebyshev or l-infinity distance with gradient.
   ..math::
       D(x, y) = \max_i |x_i - y_i|


.. py:function:: minkowski(x, y, p=2)

   Minkowski distance.
       ..math::
           D(x, y) = \left(\sum_i |x_i - y_i|^p
   ight)^{
   rac{1}{p}}
       This is a general distance. For p=1 it is equivalent to
       manhattan distance, for p=2 it is Euclidean distance, and
       for p=infinity it is Chebyshev distance. In general it is better
       to use the more specialised functions for those distances.



.. py:function:: minkowski_grad(x, y, p=2)

   Minkowski distance with gradient.
       ..math::
           D(x, y) = \left(\sum_i |x_i - y_i|^p
   ight)^{
   rac{1}{p}}
       This is a general distance. For p=1 it is equivalent to
       manhattan distance, for p=2 it is Euclidean distance, and
       for p=infinity it is Chebyshev distance. In general it is better
       to use the more specialised functions for those distances.



.. py:function:: poincare(u, v)

   Poincare distance.
       ..math::
           \delta (u, v) = 2
   rac{ \lVert  u - v
   Vert ^2 }{ ( 1 - \lVert  u
   Vert ^2 ) ( 1 - \lVert  v
   Vert ^2 ) }
           D(x, y) = \operatorname{arcosh} (1+\delta (u,v))



.. py:function:: hyperboloid_grad(x, y)


.. py:function:: weighted_minkowski(x, y, w=_mock_ones, p=2)

   A weighted version of Minkowski distance.
       ..math::
           D(x, y) = \left(\sum_i w_i |x_i - y_i|^p
   ight)^{
   rac{1}{p}}
       If weights w_i are inverse standard deviations of data in each dimension
       then this represented a standardised Minkowski distance (and is
       equivalent to standardised Euclidean distance for p=1).



.. py:function:: weighted_minkowski_grad(x, y, w=_mock_ones, p=2)

   A weighted version of Minkowski distance with gradient.
       ..math::
           D(x, y) = \left(\sum_i w_i |x_i - y_i|^p
   ight)^{
   rac{1}{p}}
       If weights w_i are inverse standard deviations of data in each dimension
       then this represented a standardised Minkowski distance (and is
       equivalent to standardised Euclidean distance for p=1).



.. py:function:: mahalanobis(x, y, vinv=_mock_identity)


.. py:function:: mahalanobis_grad(x, y, vinv=_mock_identity)


.. py:function:: hamming(x, y)


.. py:function:: canberra(x, y)


.. py:function:: canberra_grad(x, y)


.. py:function:: bray_curtis(x, y)


.. py:function:: bray_curtis_grad(x, y)


.. py:function:: jaccard(x, y)


.. py:function:: matching(x, y)


.. py:function:: dice(x, y)


.. py:function:: kulsinski(x, y)


.. py:function:: rogers_tanimoto(x, y)


.. py:function:: russellrao(x, y)


.. py:function:: sokal_michener(x, y)


.. py:function:: sokal_sneath(x, y)


.. py:function:: haversine(x, y)


.. py:function:: haversine_grad(x, y)


.. py:function:: yule(x, y)


.. py:function:: cosine(x, y)


.. py:function:: cosine_grad(x, y)


.. py:function:: correlation(x, y)


.. py:function:: hellinger(x, y)


.. py:function:: hellinger_grad(x, y)


.. py:function:: approx_log_Gamma(x)


.. py:function:: log_beta(x, y)


.. py:function:: log_single_beta(x)


.. py:function:: ll_dirichlet(data1, data2)

   The symmetric relative log likelihood of rolling data2 vs data1
   in n trials on a die that rolled data1 in sum(data1) trials.
   ..math::
       D(data1, data2) = DirichletMultinomail(data2 | data1)


.. py:function:: symmetric_kl(x, y, z=1e-11)

       symmetrized KL divergence between two probability distributions
       ..math::
           D(x, y) =
   rac{D_{KL}\left(x \Vert y
   ight) + D_{KL}\left(y \Vert x
   ight)}{2}



.. py:function:: symmetric_kl_grad(x, y, z=1e-11)

   symmetrized KL divergence and its gradient


.. py:function:: correlation_grad(x, y)


.. py:function:: sinkhorn_distance(x, y, M=_mock_identity, cost=_mock_cost, maxiter=64)


.. py:function:: spherical_gaussian_energy_grad(x, y)


.. py:function:: diagonal_gaussian_energy_grad(x, y)


.. py:function:: gaussian_energy_grad(x, y)


.. py:function:: spherical_gaussian_grad(x, y)


.. py:function:: get_discrete_params(data, metric)


.. py:function:: categorical_distance(x, y)


.. py:function:: hierarchical_categorical_distance(x, y, cat_hierarchy=[{}])


.. py:function:: ordinal_distance(x, y, support_size=1.0)


.. py:function:: count_distance(x, y, poisson_lambda=1.0, normalisation=1.0)


.. py:function:: levenshtein(x, y, normalisation=1.0, max_distance=20)


.. py:data:: named_distances
   

   

.. py:data:: named_distances_with_gradients
   

   

.. py:data:: DISCRETE_METRICS
   :annotation: = ['categorical', 'hierarchical_categorical', 'ordinal', 'count', 'string']

   

.. py:data:: SPECIAL_METRICS
   

   

.. py:function:: parallel_special_metric(X, Y=None, metric=hellinger)


.. py:function:: chunked_parallel_special_metric(X, Y=None, metric=hellinger, chunk_size=16)


.. py:function:: pairwise_special_metric(X, Y=None, metric='hellinger', kwds=None)


.. py:function:: norm(vec)

   Compute the (standard l2) norm of a vector.
   :param vec:
   :type vec: array of shape (dim,)

   :returns: *The l2 norm of vec.*


.. py:function:: arr_unique(arr)


.. py:function:: arr_union(ar1, ar2)


.. py:function:: arr_intersect(ar1, ar2)


.. py:function:: sparse_sum(ind1, data1, ind2, data2)


.. py:function:: sparse_diff(ind1, data1, ind2, data2)


.. py:function:: sparse_mul(ind1, data1, ind2, data2)


.. py:function:: general_sset_intersection(indptr1, indices1, data1, indptr2, indices2, data2, result_row, result_col, result_val, right_complement=False, mix_weight=0.5)


.. py:function:: general_sset_union(indptr1, indices1, data1, indptr2, indices2, data2, result_row, result_col, result_val)


.. py:function:: sparse_euclidean(ind1, data1, ind2, data2)


.. py:function:: sparse_manhattan(ind1, data1, ind2, data2)


.. py:function:: sparse_chebyshev(ind1, data1, ind2, data2)


.. py:function:: sparse_minkowski(ind1, data1, ind2, data2, p=2.0)


.. py:function:: sparse_hamming(ind1, data1, ind2, data2, n_features)


.. py:function:: sparse_canberra(ind1, data1, ind2, data2)


.. py:function:: sparse_bray_curtis(ind1, data1, ind2, data2)


.. py:function:: sparse_jaccard(ind1, data1, ind2, data2)


.. py:function:: sparse_matching(ind1, data1, ind2, data2, n_features)


.. py:function:: sparse_dice(ind1, data1, ind2, data2)


.. py:function:: sparse_kulsinski(ind1, data1, ind2, data2, n_features)


.. py:function:: sparse_rogers_tanimoto(ind1, data1, ind2, data2, n_features)


.. py:function:: sparse_russellrao(ind1, data1, ind2, data2, n_features)


.. py:function:: sparse_sokal_michener(ind1, data1, ind2, data2, n_features)


.. py:function:: sparse_sokal_sneath(ind1, data1, ind2, data2)


.. py:function:: sparse_cosine(ind1, data1, ind2, data2)


.. py:function:: sparse_hellinger(ind1, data1, ind2, data2)


.. py:function:: sparse_correlation(ind1, data1, ind2, data2, n_features)


.. py:function:: approx_log_Gamma(x)


.. py:function:: log_beta(x, y)


.. py:function:: log_single_beta(x)


.. py:function:: sparse_ll_dirichlet(ind1, data1, ind2, data2)


.. py:data:: sparse_named_distances
   

   

.. py:data:: sparse_need_n_features
   :annotation: = ['hamming', 'matching', 'kulsinski', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'correlation']

   

.. py:data:: SPARSE_SPECIAL_METRICS
   

   

