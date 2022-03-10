:py:mod:`topo.layouts.trimap`
=============================

.. py:module:: topo.layouts.trimap


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   topo.layouts.trimap.LiteralMeta



Functions
~~~~~~~~~

.. autoapisummary::

   topo.layouts.trimap.TriMAP



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


