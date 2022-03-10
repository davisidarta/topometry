:py:mod:`topo.spectral.umap_layouts`
====================================

.. py:module:: topo.spectral.umap_layouts


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   topo.spectral.umap_layouts.clip
   topo.spectral.umap_layouts.rdist
   topo.spectral.umap_layouts._optimize_layout_euclidean_single_epoch
   topo.spectral.umap_layouts._optimize_layout_euclidean_densmap_epoch_init
   topo.spectral.umap_layouts.optimize_layout_euclidean
   topo.spectral.umap_layouts.optimize_layout_generic
   topo.spectral.umap_layouts.optimize_layout_inverse
   topo.spectral.umap_layouts._optimize_layout_aligned_euclidean_single_epoch
   topo.spectral.umap_layouts.optimize_layout_aligned_euclidean



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


