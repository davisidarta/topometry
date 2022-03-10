:py:mod:`topo.spectral`
=======================

.. py:module:: topo.spectral


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   _spectral/index.rst
   spectral/index.rst
   umap_layouts/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   topo.spectral.spectral_layout
   topo.spectral.component_layout
   topo.spectral.multi_component_layout
   topo.spectral.LapEigenmap
   topo.spectral.tau_rand_int
   topo.spectral.clip
   topo.spectral.rdist
   topo.spectral._optimize_layout_euclidean_single_epoch
   topo.spectral._optimize_layout_euclidean_densmap_epoch_init
   topo.spectral.optimize_layout_euclidean
   topo.spectral.optimize_layout_generic
   topo.spectral.optimize_layout_inverse
   topo.spectral._optimize_layout_aligned_euclidean_single_epoch
   topo.spectral.optimize_layout_aligned_euclidean



.. py:function:: spectral_layout(graph, dim, random_state)

   Given a graph compute the spectral embedding of the graph. This is
   simply the eigenvectors of the laplacian of the graph. Here we use the
   normalized laplacian.

   :param graph: The (weighted) adjacency matrix of the graph as a sparse matrix.
   :type graph: sparse matrix
   :param dim: The dimension of the space into which to embed.
   :type dim: int
   :param random_state: A state capable being used as a numpy random state.
   :type random_state: numpy RandomState or equivalent

   :returns: **embedding** (*array of shape (n_vertices, dim)*) -- The spectral embedding of the graph.


.. py:function:: component_layout(W, n_components, component_labels, dim, norm_laplacian=True, eigen_tol=0.001)

   Provide a layout relating the separate connected components. This is done
   by taking the centroid of each component and then performing a spectral embedding
   of the centroids.
   :param W: Affinity or adjacency matrix.
   :type W: numpy.ndarray, pandas.DataFrame or scipy.sparse.csr_matrix.
   :param n_components: The number of distinct components to be layed out.
   :type n_components: int
   :param component_labels: For each vertex in the graph the label of the component to
                            which the vertex belongs.
   :type component_labels: array of shape (n_samples)
   :param dim: The chosen embedding dimension.
   :type dim: int

   :returns: **component_embedding** (*array of shape (n_components, dim)*) -- The ``dim``-dimensional embedding of the ``n_components``-many
             connected components.


.. py:function:: multi_component_layout(graph, n_components, component_labels, dim, random_state)

   Specialised layout algorithm for dealing with graphs with many connected components.
   This will first find relative positions for the components by spectrally embedding
   their centroids, then spectrally embed each individual connected component positioning
   them according to the centroid embeddings. This provides a decent embedding of each
   component while placing the components in good relative positions to one another.
   :param graph: The adjacency matrix of the graph to be embedded.
   :type graph: sparse matrix
   :param n_components: The number of distinct components to be layed out.
   :type n_components: int
   :param component_labels: For each vertex in the graph the label of the component to
                            which the vertex belongs.
   :type component_labels: array of shape (n_samples)
   :param dim: The chosen embedding dimension.
   :type dim: int

   :returns: **embedding** (*array of shape (n_samples, dim)*) -- The initial embedding of ``graph``.


.. py:function:: LapEigenmap(W, n_eigs=10, norm_laplacian=True, eigen_tol=0.001, return_evals=False)

   Performs [Laplacian Eigenmaps](https://www2.imm.dtu.dk/projects/manifold/Papers/Laplacian.pdf) on the input data.

   ----------
   :param `W`: Affinity or adjacency matrix.
   :type `W`: numpy.ndarray, pandas.DataFrame or scipy.sparse.csr_matrix.
   :param `n_eigs`: Number of eigenvectors to decompose the graph Laplacian into.
   :type `n_eigs`: int (optional, default 10).
   :param `norm_laplacian`: Whether to renormalize the graph Laplacian.
   :type `norm_laplacian`: bool (optional, default True).
   :param `return_evals`: Whether to also return the eigenvalues in a tuple of eigenvectors, eigenvalues. Defaults to False.
   :type `return_evals`: bool (optional, default False).

   ----------
   :returns: * *\* If return_evals is True* -- A tuple of eigenvectors and eigenvalues.
             * *\* If return_evals is False* -- An array of ranked eigenvectors.


.. py:function:: tau_rand_int(state)

   A fast (pseudo)-random number generator.
   :param state: The internal state of the rng
   :type state: array of int64, shape (3,)

   :returns: *A (pseudo)-random int32 value*


.. py:function:: clip(val)

   Standard clamping of a value into a fixed range (in this case -4.0 to
   4.0)
   :param val: The value to be clamped.
   :type val: float

   :returns: *The clamped value, now fixed to be in the range -4.0 to 4.0.*


.. py:function:: rdist(x, y)

   Reduced Euclidean distance.
   :param x:
   :type x: array of shape (embedding_dim,)
   :param y:
   :type y: array of shape (embedding_dim,)

   :returns: *The squared euclidean distance between x and y*


.. py:function:: _optimize_layout_euclidean_single_epoch(head_embedding, tail_embedding, head, tail, n_vertices, epochs_per_sample, a, b, rng_state, gamma, dim, move_other, alpha, epochs_per_negative_sample, epoch_of_next_negative_sample, epoch_of_next_sample, n, densmap_flag, dens_phi_sum, dens_re_sum, dens_re_cov, dens_re_std, dens_re_mean, dens_lambda, dens_R, dens_mu, dens_mu_tot)


.. py:function:: _optimize_layout_euclidean_densmap_epoch_init(head_embedding, tail_embedding, head, tail, a, b, re_sum, phi_sum)


.. py:function:: optimize_layout_euclidean(head_embedding, tail_embedding, head, tail, n_epochs, n_vertices, epochs_per_sample, a, b, rng_state, gamma=1.0, initial_alpha=1.0, negative_sample_rate=5.0, parallel=False, verbose=False, densmap=False, densmap_kwds={})

   Improve an embedding using stochastic gradient descent to minimize the
   fuzzy set cross entropy between the 1-skeletons of the high dimensional
   and low dimensional fuzzy simplicial sets. In practice this is done by
   sampling edges based on their membership strength (with the (1-p) terms
   coming from negative sampling similar to word2vec).
   :param head_embedding: The initial embedding to be improved by SGD.
   :type head_embedding: array of shape (n_samples, n_components)
   :param tail_embedding: The reference embedding of embedded points. If not embedding new
                          previously unseen points with respect to an existing embedding this
                          is simply the head_embedding (again); otherwise it provides the
                          existing embedding to embed with respect to.
   :type tail_embedding: array of shape (source_samples, n_components)
   :param head: The indices of the heads of 1-simplices with non-zero membership.
   :type head: array of shape (n_1_simplices)
   :param tail: The indices of the tails of 1-simplices with non-zero membership.
   :type tail: array of shape (n_1_simplices)
   :param n_epochs: The number of training epochs to use in optimization.
   :type n_epochs: int
   :param n_vertices: The number of vertices (0-simplices) in the dataset.
   :type n_vertices: int
   :param epochs_per_samples: A float value of the number of epochs per 1-simplex. 1-simplices with
                              weaker membership strength will have more epochs between being sampled.
   :type epochs_per_samples: array of shape (n_1_simplices)
   :param a: Parameter of differentiable approximation of right adjoint functor
   :type a: float
   :param b: Parameter of differentiable approximation of right adjoint functor
   :type b: float
   :param rng_state: The internal state of the rng
   :type rng_state: array of int64, shape (3,)
   :param gamma: Weight to apply to negative samples.
   :type gamma: float (optional, default 1.0)
   :param initial_alpha: Initial learning rate for the SGD.
   :type initial_alpha: float (optional, default 1.0)
   :param negative_sample_rate: Number of negative samples to use per positive sample.
   :type negative_sample_rate: int (optional, default 5)
   :param parallel: Whether to run the computation using numba parallel.
                    Running in parallel is non-deterministic, and is not used
                    if a random seed has been set, to ensure reproducibility.
   :type parallel: bool (optional, default False)
   :param verbose: Whether to report information on the current progress of the algorithm.
   :type verbose: bool (optional, default False)
   :param densmap: Whether to use the density-augmented densMAP objective
   :type densmap: bool (optional, default False)
   :param densmap_kwds: Auxiliary data for densMAP
   :type densmap_kwds: dict (optional, default {})

   :returns: **embedding** (*array of shape (n_samples, n_components)*) -- The optimized embedding.


.. py:function:: optimize_layout_generic(head_embedding, tail_embedding, head, tail, n_epochs, n_vertices, epochs_per_sample, a, b, rng_state, gamma=1.0, initial_alpha=1.0, negative_sample_rate=5.0, output_metric=dist.euclidean, output_metric_kwds=(), verbose=False)

   Improve an embedding using stochastic gradient descent to minimize the
   fuzzy set cross entropy between the 1-skeletons of the high dimensional
   and low dimensional fuzzy simplicial sets. In practice this is done by
   sampling edges based on their membership strength (with the (1-p) terms
   coming from negative sampling similar to word2vec).
   :param head_embedding: The initial embedding to be improved by SGD.
   :type head_embedding: array of shape (n_samples, n_components)
   :param tail_embedding: The reference embedding of embedded points. If not embedding new
                          previously unseen points with respect to an existing embedding this
                          is simply the head_embedding (again); otherwise it provides the
                          existing embedding to embed with respect to.
   :type tail_embedding: array of shape (source_samples, n_components)
   :param head: The indices of the heads of 1-simplices with non-zero membership.
   :type head: array of shape (n_1_simplices)
   :param tail: The indices of the tails of 1-simplices with non-zero membership.
   :type tail: array of shape (n_1_simplices)
   :param weight: The membership weights of the 1-simplices.
   :type weight: array of shape (n_1_simplices)
   :param n_epochs: The number of training epochs to use in optimization.
   :type n_epochs: int
   :param n_vertices: The number of vertices (0-simplices) in the dataset.
   :type n_vertices: int
   :param epochs_per_sample: A float value of the number of epochs per 1-simplex. 1-simplices with
                             weaker membership strength will have more epochs between being sampled.
   :type epochs_per_sample: array of shape (n_1_simplices)
   :param a: Parameter of differentiable approximation of right adjoint functor
   :type a: float
   :param b: Parameter of differentiable approximation of right adjoint functor
   :type b: float
   :param rng_state: The internal state of the rng
   :type rng_state: array of int64, shape (3,)
   :param gamma: Weight to apply to negative samples.
   :type gamma: float (optional, default 1.0)
   :param initial_alpha: Initial learning rate for the SGD.
   :type initial_alpha: float (optional, default 1.0)
   :param negative_sample_rate: Number of negative samples to use per positive sample.
   :type negative_sample_rate: int (optional, default 5)
   :param verbose: Whether to report information on the current progress of the algorithm.
   :type verbose: bool (optional, default False)

   :returns: **embedding** (*array of shape (n_samples, n_components)*) -- The optimized embedding.


.. py:function:: optimize_layout_inverse(head_embedding, tail_embedding, head, tail, weight, sigmas, rhos, n_epochs, n_vertices, epochs_per_sample, a, b, rng_state, gamma=1.0, initial_alpha=1.0, negative_sample_rate=5.0, output_metric=dist.euclidean, output_metric_kwds=(), verbose=False)

   Improve an embedding using stochastic gradient descent to minimize the
   fuzzy set cross entropy between the 1-skeletons of the high dimensional
   and low dimensional fuzzy simplicial sets. In practice this is done by
   sampling edges based on their membership strength (with the (1-p) terms
   coming from negative sampling similar to word2vec).
   :param head_embedding: The initial embedding to be improved by SGD.
   :type head_embedding: array of shape (n_samples, n_components)
   :param tail_embedding: The reference embedding of embedded points. If not embedding new
                          previously unseen points with respect to an existing embedding this
                          is simply the head_embedding (again); otherwise it provides the
                          existing embedding to embed with respect to.
   :type tail_embedding: array of shape (source_samples, n_components)
   :param head: The indices of the heads of 1-simplices with non-zero membership.
   :type head: array of shape (n_1_simplices)
   :param tail: The indices of the tails of 1-simplices with non-zero membership.
   :type tail: array of shape (n_1_simplices)
   :param weight: The membership weights of the 1-simplices.
   :type weight: array of shape (n_1_simplices)
   :param n_epochs: The number of training epochs to use in optimization.
   :type n_epochs: int
   :param n_vertices: The number of vertices (0-simplices) in the dataset.
   :type n_vertices: int
   :param epochs_per_sample: A float value of the number of epochs per 1-simplex. 1-simplices with
                             weaker membership strength will have more epochs between being sampled.
   :type epochs_per_sample: array of shape (n_1_simplices)
   :param a: Parameter of differentiable approximation of right adjoint functor
   :type a: float
   :param b: Parameter of differentiable approximation of right adjoint functor
   :type b: float
   :param rng_state: The internal state of the rng
   :type rng_state: array of int64, shape (3,)
   :param gamma: Weight to apply to negative samples.
   :type gamma: float (optional, default 1.0)
   :param initial_alpha: Initial learning rate for the SGD.
   :type initial_alpha: float (optional, default 1.0)
   :param negative_sample_rate: Number of negative samples to use per positive sample.
   :type negative_sample_rate: int (optional, default 5)
   :param verbose: Whether to report information on the current progress of the algorithm.
   :type verbose: bool (optional, default False)

   :returns: **embedding** (*array of shape (n_samples, n_components)*) -- The optimized embedding.


.. py:function:: _optimize_layout_aligned_euclidean_single_epoch(head_embeddings, tail_embeddings, heads, tails, epochs_per_sample, a, b, regularisation_weights, relations, rng_state, gamma, lambda_, dim, move_other, alpha, epochs_per_negative_sample, epoch_of_next_negative_sample, epoch_of_next_sample, n)


.. py:function:: optimize_layout_aligned_euclidean(head_embeddings, tail_embeddings, heads, tails, n_epochs, epochs_per_sample, regularisation_weights, relations, rng_state, a=1.576943460405378, b=0.8950608781227859, gamma=1.0, lambda_=0.005, initial_alpha=1.0, negative_sample_rate=5.0, parallel=True, verbose=False)


