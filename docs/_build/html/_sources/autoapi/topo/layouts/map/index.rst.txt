:py:mod:`topo.layouts.map`
==========================

.. py:module:: topo.layouts.map


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   topo.layouts.map.LiteralMeta



Functions
~~~~~~~~~

.. autoapisummary::

   topo.layouts.map.fuzzy_embedding



.. py:class:: LiteralMeta

   Bases: :py:obj:`type`

   .. py:method:: __getitem__(cls, values)



.. py:function:: fuzzy_embedding(graph, n_components=2, initial_alpha=1, min_dist=0.3, spread=1.2, n_epochs=500, metric='cosine', metric_kwds={}, output_metric='euclidean', output_metric_kwds={}, gamma=1.2, negative_sample_rate=10, init='spectral', random_state=None, euclidean_output=True, parallel=True, verbose=False, a=None, b=None, densmap=False, densmap_kwds={}, output_dens=False)

   Perform a fuzzy simplicial set embedding, using a specified
   initialisation method and then minimizing the fuzzy set cross entropy
   between the 1-skeletons of the high and low dimensional fuzzy simplicial
   sets. The fuzzy simplicial set embedding was proposed and implemented by
   Leland McInnes in UMAP (see `umap-learn <https://github.com/lmcinnes/umap>`).
   Here we're using it only for the projection (layout optimization).

   :param graph: The 1-skeleton of the high dimensional fuzzy simplicial set as
                 represented by a graph for which we require a sparse matrix for the
                 (weighted) adjacency matrix.
   :type graph: sparse matrix
   :param n_components: The dimensionality of the euclidean space into which to embed the data.
   :type n_components: int
   :param initial_alpha: Initial learning rate for the SGD.
   :type initial_alpha: float
   :param a: Parameter of differentiable approximation of right adjoint functor
   :type a: float
   :param b: Parameter of differentiable approximation of right adjoint functor
   :type b: float
   :param gamma: Weight to apply to negative samples.
   :type gamma: float
   :param negative_sample_rate: The number of negative samples to select per positive sample
                                in the optimization process. Increasing this value will result
                                in greater repulsive force being applied, greater optimization
                                cost, but slightly more accuracy.
   :type negative_sample_rate: int (optional, default 5)
   :param n_epochs: The number of training epochs to be used in optimizing the
                    low dimensional embedding. Larger values result in more accurate
                    embeddings. If 0 is specified a value will be selected based on
                    the size of the input dataset (200 for large datasets, 500 for small).
   :type n_epochs: int (optional, default 0)
   :param init:
                How to initialize the low dimensional embedding. Options are:
                    * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
                    * 'random': assign initial embedding positions at random.
                    * A numpy array of initial embedding positions.
   :type init: string
   :param random_state: A state capable being used as a numpy random state.
   :type random_state: numpy RandomState or equivalent
   :param metric: The metric used to measure distance in high dimensional space; used if
                  multiple connected components need to be layed out.
   :type metric: string or callable
   :param metric_kwds: Key word arguments to be passed to the metric function; used if
                       multiple connected components need to be layed out.
   :type metric_kwds: dict
   :param densmap: Whether to use the density-augmented objective function to optimize
                   the embedding according to the densMAP algorithm.
   :type densmap: bool
   :param densmap_kwds: Key word arguments to be used by the densMAP optimization.
   :type densmap_kwds: dict
   :param output_dens: Whether to output local radii in the original data and the embedding.
   :type output_dens: bool
   :param output_metric: Function returning the distance between two points in embedding space and
                         the gradient of the distance wrt the first argument.
   :type output_metric: function
   :param output_metric_kwds: Key word arguments to be passed to the output_metric function.
   :type output_metric_kwds: dict
   :param euclidean_output: Whether to use the faster code specialised for euclidean output metrics
   :type euclidean_output: bool
   :param parallel: Whether to run the computation using numba parallel.
                    Running in parallel is non-deterministic, and is not used
                    if a random seed has been set, to ensure reproducibility.
   :type parallel: bool (optional, default False)
   :param verbose: Whether to report information on the current progress of the algorithm.
   :type verbose: bool (optional, default False)

   :returns: * **embedding** (*array of shape (n_samples, n_components)*) -- The optimized of ``graph`` into an ``n_components`` dimensional
               euclidean space.
             * **aux_data** (*dict*) -- Auxiliary output returned with the embedding. When densMAP extension
               is turned on, this dictionary includes local radii in the original
               data (``rad_orig``) and in the embedding (``rad_emb``).
                   Y_init : array of shape (n_samples, n_components)
                       The spectral initialization of ``graph`` into an ``n_components`` dimensional
                       euclidean space.


