:py:mod:`topo.layouts.mde`
==========================

.. py:module:: topo.layouts.mde


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   topo.layouts.mde._remove_anchor_anchor_edges
   topo.layouts.mde.IsomorphicMDE
   topo.layouts.mde.IsometricMDE
   topo.layouts.mde.SpectralMDE



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


