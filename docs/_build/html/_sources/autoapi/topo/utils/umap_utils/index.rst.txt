:py:mod:`topo.utils.umap_utils`
===============================

.. py:module:: topo.utils.umap_utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   topo.utils.umap_utils.eval_gaussian
   topo.utils.umap_utils.eval_density_at_point
   topo.utils.umap_utils.create_density_plot
   topo.utils.umap_utils.torus_euclidean_grad
   topo.utils.umap_utils.fast_knn_indices
   topo.utils.umap_utils.tau_rand_int
   topo.utils.umap_utils.tau_rand
   topo.utils.umap_utils.norm
   topo.utils.umap_utils.submatrix
   topo.utils.umap_utils.ts
   topo.utils.umap_utils.csr_unique
   topo.utils.umap_utils.disconnected_vertices



.. py:function:: eval_gaussian(x, pos=np.array([0, 0]), cov=np.eye(2, dtype=np.float32))


.. py:function:: eval_density_at_point(x, embedding)


.. py:function:: create_density_plot(X, Y, embedding)


.. py:function:: torus_euclidean_grad(x, y, torus_dimensions=(2 * np.pi, 2 * np.pi))

   Standard euclidean distance.

   ..math::
       D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}


.. py:function:: fast_knn_indices(X, n_neighbors)

   A fast computation of knn indices.
   :param X: The input data to compute the k-neighbor indices of.
   :type X: array of shape (n_samples, n_features)
   :param n_neighbors: The number of nearest neighbors to compute for each sample in ``X``.
   :type n_neighbors: int

   :returns: **knn_indices** (*array of shape (n_samples, n_neighbors)*) -- The indices on the ``n_neighbors`` closest points in the dataset.


.. py:function:: tau_rand_int(state)

   A fast (pseudo)-random number generator.
   :param state: The internal state of the rng
   :type state: array of int64, shape (3,)

   :returns: *A (pseudo)-random int32 value*


.. py:function:: tau_rand(state)

   A fast (pseudo)-random number generator for floats in the range [0,1]
   :param state: The internal state of the rng
   :type state: array of int64, shape (3,)

   :returns: *A (pseudo)-random float32 in the interval [0, 1]*


.. py:function:: norm(vec)

   Compute the (standard l2) norm of a vector.
   :param vec:
   :type vec: array of shape (dim,)

   :returns: *The l2 norm of vec.*


.. py:function:: submatrix(dmat, indices_col, n_neighbors)

   Return a submatrix given an orginal matrix and the indices to keep.
   :param dmat: Original matrix.
   :type dmat: array, shape (n_samples, n_samples)
   :param indices_col: Indices to keep. Each row consists of the indices of the columns.
   :type indices_col: array, shape (n_samples, n_neighbors)
   :param n_neighbors: Number of neighbors.
   :type n_neighbors: int

   :returns: **submat** (*array, shape (n_samples, n_neighbors)*) -- The corresponding submatrix.


.. py:function:: ts()


.. py:function:: csr_unique(matrix, return_index=True, return_inverse=True, return_counts=True)

   Find the unique elements of a sparse csr matrix.
   We don't explicitly construct the unique matrix leaving that to the user
   who may not want to duplicate a massive array in memory.
   Returns the indices of the input array that give the unique values.
   Returns the indices of the unique array that reconstructs the input array.
   Returns the number of times each unique row appears in the input matrix.
   matrix: a csr matrix
   return_index = bool, optional
       If true, return the row indices of 'matrix'
   return_inverse: bool, optional
       If true, return the the indices of the unique array that can be
          used to reconstruct 'matrix'.
   return_counts = bool, optional
       If true, returns the number of times each unique item appears in 'matrix'
   The unique matrix can computed via
   unique_matrix = matrix[index]
   and the original matrix reconstructed via
   unique_matrix[inverse]


.. py:function:: disconnected_vertices(model)

   Returns a boolean vector indicating which vertices are disconnected from the umap graph.
   These vertices will often be scattered across the space and make it difficult to focus on the main
   manifold.  They can either be filtered and have UMAP re-run or simply filtered from the interactive plotting tool
   via the subset_points parameter.
   Use ~disconnected_vertices(model) to only plot the connected points.
   :param model:
   :type model: a trained UMAP model

   :returns: *A boolean vector indicating which points are disconnected*


