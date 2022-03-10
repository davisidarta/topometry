:py:mod:`topo.eval`
===================

.. py:module:: topo.eval


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   global_scores/index.rst
   local_scores/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   topo.eval.global_score_pca
   topo.eval.global_score_laplacian
   topo.eval.knn_spearman_r
   topo.eval.geodesic_distance



.. py:function:: global_score_pca(X, Y)

   Global score
   Input
   ------
   X: Instance matrix
   Y: Embedding


.. py:function:: global_score_laplacian(X, Y, k=10, data_is_graph=False, n_jobs=12, random_state=None)

   Global score
   Input
   ------
   X: Instance matrix
   Y: Embedding


.. py:function:: knn_spearman_r(data_graph, embedding_graph, path_method='D', subsample_idx=None, unweighted=False)


.. py:function:: geodesic_distance(data, method='D', unweighted=False, directed=True)


