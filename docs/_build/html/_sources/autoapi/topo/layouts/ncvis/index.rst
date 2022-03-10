:py:mod:`topo.layouts.ncvis`
============================

.. py:module:: topo.layouts.ncvis


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   topo.layouts.ncvis.NCVis



.. py:function:: NCVis(data, n_components=2, n_jobs=-1, n_neighbors=15, distance='cosine', M=15, efC=30, random_seed=42, n_epochs=50, n_init_epochs=20, spread=1.0, min_dist=0.4, alpha=1.0, a=None, b=None, alpha_Q=1.0, n_noise=None)

   Runs Noise Contrastive Visualization ((NCVis)[https://dl.acm.org/doi/abs/10.1145/3366423.3380061])
   for dimensionality reduction and graph layout .

   :param n_components: Desired dimensionality of the embedding.
   :type n_components: int
   :param n_jobs: The maximum number of threads to use. In case n_threads < 1, it defaults to the number of available CPUs.
   :type n_jobs: int
   :param n_neighbors: Number of nearest neighbours in the high dimensional space to consider.
   :type n_neighbors: int
   :param M: The number of bi-directional links created for every new element during construction of HNSW.
             See https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
   :type M: int
   :param efC: The size of the dynamic list for the nearest neighbors (used during the search) in HNSW.
               See https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
   :type efC: int
   :param random_seed: Random seed to initialize the generators. Notice, however, that the result may still depend on the number of threads.
   :type random_seed: int
   :param n_epochs: The total number of epochs to run. During one epoch the positions of each nearest neighbors pair are updated.
   :type n_epochs: int
   :param n_init_epochs: The number of epochs used for initialization. During one epoch the positions of each nearest neighbors pair are updated.
   :type n_init_epochs: int
   :param spread: The effective scale of embedded points. In combination with ``min_dist``
                  this determines how clustered/clumped the embedded points are.
                  See https://github.com/lmcinnes/umap/blob/834184f9c0455f26db13ab148c0abd2d3767d968/umap/umap_.py#L1143
   :type spread: float
   :param min_dist: The effective minimum distance between embedded points. Smaller values
                    will result in a more clustered/clumped embedding where nearby points
                    on the manifold are drawn closer together, while larger values will
                    result on a more even dispersal of points. The value should be set
                    relative to the ``spread`` value, which determines the scale at which
                    embedded points will be spread out.
                    See https://github.com/lmcinnes/umap/blob/834184f9c0455f26db13ab148c0abd2d3767d968/umap/umap_.py#L1135
   :type min_dist: float
   :param a: More specific parameters controlling the embedding. If None these values
             are set automatically as determined by ``min_dist`` and ``spread``.
             See https://github.com/lmcinnes/umap/blob/834184f9c0455f26db13ab148c0abd2d3767d968/umap/umap_.py#L1179
   :type a: (optional, default None)
   :param b: More specific parameters controlling the embedding. If None these values
             are set automatically as determined by ``min_dist`` and ``spread``.
             See https://github.com/lmcinnes/umap/blob/834184f9c0455f26db13ab148c0abd2d3767d968/umap/umap_.py#L1183
   :type b: (optional, default None)
   :param alpha: Learning rate for the embedding positions.
   :type alpha: float
   :param alpha_Q: Learning rate for the normalization constant.
   :type alpha_Q: float
   :param n_noise:
                   Number of noise samples to use per data sample. If ndarray is provided, n_epochs is set to its length.
                    If n_noise is None, it is set to dynamic sampling with noise level gradually increasing
                     from 0 to fixed value.
   :type n_noise: int or ndarray of ints
   :param distance: Distance to use for nearest neighbors search.
   :type distance: str {'euclidean', 'cosine', 'correlation', 'inner_product'}


