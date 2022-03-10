:py:mod:`topo.tpgraph.cknn`
===========================

.. py:module:: topo.tpgraph.cknn


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   topo.tpgraph.cknn.CkNearestNeighbors



Functions
~~~~~~~~~

.. autoapisummary::

   topo.tpgraph.cknn.cknn_graph



.. py:function:: cknn_graph(X, n_neighbors, delta=1.0, metric='euclidean', t='inf', include_self=False, is_sparse=True, return_instance=False)


.. py:class:: CkNearestNeighbors(n_neighbors=10, delta=1.0, metric='euclidean', t='inf', include_self=False, is_sparse=True)

   Bases: :py:obj:`object`

   This object provides the all logic of CkNN.
   :param n_neighbors: int, optional, default=5
                       Number of neighbors to estimate the density around the point.
                       It appeared as a parameter `k` in the paper.
   :param delta: float, optional, default=1.0
                 A parameter to decide the radius for each points. The combination
                 radius increases in proportion to this parameter.
   :param metric: str, optional, default='euclidean'
                  The metric of each points. This parameter depends on the parameter
                  `metric` of scipy.spatial.distance.pdist.
   :param t: 'inf' or float or int, optional, default='inf'
             The decay parameter of heat kernel. The weights are calculated as
             follow:
                 W_{ij} = exp(-(||x_{i}-x_{j}||^2)/t)
             For more infomation, read the paper 'Laplacian Eigenmaps for
             Dimensionality Reduction and Data Representation', Belkin, et. al.
   :param include_self: bool, optional, default=True
                        All diagonal elements are 1.0 if this parameter is True.
   :param is_sparse: bool, optional, default=True
                     The method `cknneighbors_graph` returns csr_matrix object if this
                     parameter is True else returns ndarray object.
   :param return_adjacency: bool, optional, default=False
                            Whether to return the adjacency matrix instead of the estimated similarity.

   .. py:method:: __repr__(self)

      Return repr(self).


   .. py:method:: cknneighbors_graph(self, X)

      A method to calculate the CkNN graph
      :param X: ndarray
                The data matrix.

      return: csr_matrix (if self.is_sparse is True)
              or ndarray(if self.is_sparse is False)



