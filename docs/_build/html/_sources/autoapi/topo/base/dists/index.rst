:py:mod:`topo.base.dists`
=========================

.. py:module:: topo.base.dists


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   topo.base.dists.sign
   topo.base.dists.euclidean
   topo.base.dists.euclidean_grad
   topo.base.dists.standardised_euclidean
   topo.base.dists.standardised_euclidean_grad
   topo.base.dists.manhattan
   topo.base.dists.manhattan_grad
   topo.base.dists.chebyshev
   topo.base.dists.chebyshev_grad
   topo.base.dists.minkowski
   topo.base.dists.minkowski_grad
   topo.base.dists.poincare
   topo.base.dists.hyperboloid_grad
   topo.base.dists.weighted_minkowski
   topo.base.dists.weighted_minkowski_grad
   topo.base.dists.mahalanobis
   topo.base.dists.mahalanobis_grad
   topo.base.dists.hamming
   topo.base.dists.canberra
   topo.base.dists.canberra_grad
   topo.base.dists.bray_curtis
   topo.base.dists.bray_curtis_grad
   topo.base.dists.jaccard
   topo.base.dists.matching
   topo.base.dists.dice
   topo.base.dists.kulsinski
   topo.base.dists.rogers_tanimoto
   topo.base.dists.russellrao
   topo.base.dists.sokal_michener
   topo.base.dists.sokal_sneath
   topo.base.dists.haversine
   topo.base.dists.haversine_grad
   topo.base.dists.yule
   topo.base.dists.cosine
   topo.base.dists.cosine_grad
   topo.base.dists.correlation
   topo.base.dists.hellinger
   topo.base.dists.hellinger_grad
   topo.base.dists.approx_log_Gamma
   topo.base.dists.log_beta
   topo.base.dists.log_single_beta
   topo.base.dists.ll_dirichlet
   topo.base.dists.symmetric_kl
   topo.base.dists.symmetric_kl_grad
   topo.base.dists.correlation_grad
   topo.base.dists.sinkhorn_distance
   topo.base.dists.spherical_gaussian_energy_grad
   topo.base.dists.diagonal_gaussian_energy_grad
   topo.base.dists.gaussian_energy_grad
   topo.base.dists.spherical_gaussian_grad
   topo.base.dists.get_discrete_params
   topo.base.dists.categorical_distance
   topo.base.dists.hierarchical_categorical_distance
   topo.base.dists.ordinal_distance
   topo.base.dists.count_distance
   topo.base.dists.levenshtein
   topo.base.dists.parallel_special_metric
   topo.base.dists.chunked_parallel_special_metric
   topo.base.dists.pairwise_special_metric



Attributes
~~~~~~~~~~

.. autoapisummary::

   topo.base.dists._mock_identity
   topo.base.dists._mock_cost
   topo.base.dists._mock_ones
   topo.base.dists.named_distances
   topo.base.dists.named_distances_with_gradients
   topo.base.dists.DISCRETE_METRICS
   topo.base.dists.SPECIAL_METRICS


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


