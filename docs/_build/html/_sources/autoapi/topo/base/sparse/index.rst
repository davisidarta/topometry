:py:mod:`topo.base.sparse`
==========================

.. py:module:: topo.base.sparse


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   topo.base.sparse.arr_unique
   topo.base.sparse.arr_union
   topo.base.sparse.arr_intersect
   topo.base.sparse.sparse_sum
   topo.base.sparse.sparse_diff
   topo.base.sparse.sparse_mul
   topo.base.sparse.general_sset_intersection
   topo.base.sparse.general_sset_union
   topo.base.sparse.sparse_euclidean
   topo.base.sparse.sparse_manhattan
   topo.base.sparse.sparse_chebyshev
   topo.base.sparse.sparse_minkowski
   topo.base.sparse.sparse_hamming
   topo.base.sparse.sparse_canberra
   topo.base.sparse.sparse_bray_curtis
   topo.base.sparse.sparse_jaccard
   topo.base.sparse.sparse_matching
   topo.base.sparse.sparse_dice
   topo.base.sparse.sparse_kulsinski
   topo.base.sparse.sparse_rogers_tanimoto
   topo.base.sparse.sparse_russellrao
   topo.base.sparse.sparse_sokal_michener
   topo.base.sparse.sparse_sokal_sneath
   topo.base.sparse.sparse_cosine
   topo.base.sparse.sparse_hellinger
   topo.base.sparse.sparse_correlation
   topo.base.sparse.approx_log_Gamma
   topo.base.sparse.log_beta
   topo.base.sparse.log_single_beta
   topo.base.sparse.sparse_ll_dirichlet



Attributes
~~~~~~~~~~

.. autoapisummary::

   topo.base.sparse.sparse_named_distances
   topo.base.sparse.sparse_need_n_features
   topo.base.sparse.SPARSE_SPECIAL_METRICS


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
   

   

