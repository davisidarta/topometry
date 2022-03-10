:py:mod:`topo.eval.global_scores`
=================================

.. py:module:: topo.eval.global_scores


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   topo.eval.global_scores.global_loss_
   topo.eval.global_scores.global_score_pca
   topo.eval.global_scores.global_score_laplacian



.. py:function:: global_loss_(X, Y)


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


