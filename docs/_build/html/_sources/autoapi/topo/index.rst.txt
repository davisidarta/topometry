:py:mod:`topo`
==============

.. py:module:: topo


Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   base/index.rst
   eval/index.rst
   layouts/index.rst
   spectral/index.rst
   tpgraph/index.rst
   utils/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   _utils/index.rst
   pipes/index.rst
   plot/index.rst
   topograph/index.rst
   version/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   topo.TopOGraph



Functions
~~~~~~~~~

.. autoapisummary::

   topo.save_pkl
   topo.read_pkl
   topo.annotate_doc_types



Attributes
~~~~~~~~~~

.. autoapisummary::

   topo.__version__


.. py:class:: TopOGraph(base_knn=10, graph_knn=10, n_eigs=100, basis='diffusion', graph='diff', base_metric='cosine', graph_metric='cosine', n_jobs=1, backend='nmslib', M=15, efC=50, efS=50, verbosity=1, cache_base=True, cache_graph=True, kernel_use='decay', alpha=1, plot_spectrum=False, eigen_expansion=False, delta=1.0, t='inf', p=11 / 16, transitions=True, random_state=None)

   Bases: :py:obj:`sklearn.base.TransformerMixin`

    Main TopOMetry class for learning topological similarities, bases, graphs, and layouts from high-dimensional data.

    From data, learns topological similarity metrics, from these build orthogonal bases and from these bases learns
    topological graphs. Users can choose
    different models to achieve these topological representations, combinining either diffusion harmonics,
    continuous k-nearest-neighbors or fuzzy simplicial sets to approximate the Laplace-Beltrami Operator.
    The topological graphs can then be visualized with multiple existing layout optimization tools.

   :param base_knn: Number of k-nearest-neighbors to use when learning topological similarities.
                    Consider this as a calculus discretization threshold (i.e. approaches zero in the limit of large data).
                    For practical purposes, the minimum amount of samples one would
                    expect to constitute a neighborhood of its own. Increasing `k` can generate more globally-comprehensive metrics
                    and maps, to a certain extend, however at the expense of fine-grained resolution. In practice, the default
                    value of 10 performs quite well for almost all cases.
   :type base_knn: int (optional, default 10).
   :param graph_knn: Similar to `base_knn`, but used to learning topological graphs from the orthogonal bases.
   :type graph_knn: int (optional, default 10).
   :param n_eigs: Number of components to compute. This number can be iterated to get different views
                  from data at distinct spectral resolutions. If `basis` is set to `diffusion`, this is the number of
                  computed diffusion components. If `basis` is set to `continuous` or `fuzzy`, this is the number of
                  computed eigenvectors of the Laplacian Eigenmaps from the learned topological similarities.
   :type n_eigs: int (optional, default 100).
   :param basis: Which topological basis model to learn from data. If `diffusion`, performs an optimized, anisotropic, adaptive
                 diffusion mapping (default). If `continuous`, computes affinities from continuous k-nearest-neighbors, and a
                 topological basis with Laplacian Eigenmaps. If `fuzzy`, computes affinities using
                 fuzzy simplicial sets, and a topological basis with Laplacian Eigenmaps.
   :type basis: 'diffusion', 'continuous' or 'fuzzy' (optional, default 'diffusion').
   :param graph: Which topological graph model to learn from the built basis. If 'diff', uses a second-order diffusion process
                 to learn similarities and transition probabilities. If 'cknn', uses the continuous k-nearest-neighbors
                 algorithm. If 'fuzzy', builds a fuzzy simplicial set graph from the active basis. All these
                 algorithms learn graph-oriented topological metrics from the learned basis.
   :type graph: 'diff', 'cknn' or 'fuzzy' (optional, default 'diff').
   :param backend: Which backend to use to compute nearest-neighbors. Options for fast, approximate nearest-neighbors
                   are 'hnwslib'  and 'nmslib' (default). For exact nearest-neighbors, use 'sklearn'.

                   * If using 'nmslib', a sparse
                   csr_matrix input is expected. If using 'hnwslib' or 'sklearn', a dense array is expected.

                   * I strongly recommend you use 'hnswlib' if handling with somewhat dense, array-shaped data. If the data
                   is relatively sparse, you should use 'nmslib', which operates on sparse matrices by default on
                   TopOMetry and will automatically convert the input array to csr_matrix for performance.
   :type backend: str 'hnwslib', 'nmslib' or 'sklearn' (optional, default 'nmslib').
   :param base_metric: Distance metric for building an approximate kNN graph during topological basis construction. Defaults to
                       'cosine'. Users are encouraged to explore different metrics, such as 'euclidean' and 'inner_product'.
                       The 'hamming' and 'jaccard' distances are also available for string vectors.
                       Accepted metrics include NMSLib(*), HNSWlib(**) and sklearn(***) metrics. Some examples are:

                       -'sqeuclidean' (**, ***)

                       -'euclidean' (**, ***)

                       -'l1' (*)

                       -'lp' - requires setting the parameter ``p`` (*) - similar to Minkowski

                       -'cosine' (**, ***)

                       -'inner_product' (**)

                       -'angular' (*)

                       -'negdotprod' (*)

                       -'levenshtein' (*)

                       -'hamming' (*)

                       -'jaccard' (*)

                       -'jansen-shan' (*)
   :type base_metric: str (optional, default 'cosine')
   :param graph_metric: Similar to `base_metric`, but used for building the topological graph.
   :type graph_metric: str (optional, default 'cosine').
   :param p: P for the Lp metric, when `metric='lp'`.  Can be fractional. The default 11/16 approximates 2/3, that is,
             an astroid norm with some computational efficiency (2^n bases are less painstakinly slow to compute).
   :type p: int or float (optional, default 11/16 ).
   :param n_jobs: Number of threads to use in calculations. Set this to as much as possible for speed.
   :type n_jobs: int (optional, default 10).
   :param M: A neighborhood search parameter. Defines the maximum number of neighbors in the zero and above-zero layers
             during HSNW (Hierarchical Navigable Small World Graph). However, the actual default maximum number
             of neighbors for the zero layer is 2*M.  A reasonable range for this parameter
             is 5-100. For more information on HSNW, please check its manuscript(https://arxiv.org/abs/1603.09320).
             HSNW is implemented in python via NMSlib (https://github.com/nmslib/nmslib) and HNWSlib
             (https://github.com/nmslib/hnswlib).
   :type M: int (optional, default 15).
   :param efC: A neighborhood search parameter. Increasing this value improves the quality of a constructed graph
               and leads to higher accuracy of search. However this also leads to longer indexing times.
               A reasonable range for this parameter is 50-2000.
   :type efC: int (optional, default 50).
   :param efS: A neighborhood search parameter. Similarly to efC, increasing this value improves recall at the
               expense of longer retrieval time. A reasonable range for this parameter is 100-2000.
   :type efS: int (optional, default 50).
   :param transitions: A diffusion harmonics parameter. Whether to use the transition probabilities rather than
                       the diffusion potential when computing the diffusion harmonics model.
   :type transitions: bool (optional, default False).
   :param alpha: A diffusion harmonics parameter. Alpha in the diffusion maps literature. Controls how much the results are biased by data distribution.
                 Defaults to 1, which unbiases results from data underlying samplg distribution.
   :type alpha: int or float (optional, default 1).
   :param kernel_use: A diffusion harmonics parameter. Which type of kernel to use in the diffusion harmonics model. There are four
                      implemented, considering the adaptive decay and the neighborhood expansion,
                      written as 'simple', 'decay', 'simple_adaptive' and 'decay_adaptive'.

                          *The first, 'simple', is a locally-adaptive kernel similar to that proposed by Nadler et al.
                          (https://doi.org/10.1016/j.acha.2005.07.004) and implemented in Setty et al.
                          (https://doi.org/10.1038/s41587-019-0068-4). It is the fastest option.

                          *The 'decay' option applies an adaptive decay rate, but no neighborhood expansion.

                          *Those, followed by '_adaptive', apply the neighborhood expansion process.

                      The neighborhood expansion can impact runtime, although this is not usually expressive for datasets under 10e6 samples.
                      If you're not obtaining good separation between expect clusters, consider changing this to 'decay_adaptive' with
                      a small number of neighbors.
   :type kernel_use: str (optional, default 'simple')
   :param delta: A CkNN parameter to decide the radius for each points. The combination
                 radius increases in proportion to this parameter.
   :type delta: float (optional, default 1.0).
   :param t: A CkNN parameter encoding the decay of the heat kernel. The weights are calculated as:
             W_{ij} = exp(-(||x_{i}-x_{j}||^2)/t)
   :type t: 'inf' or float or int, optional, default='inf'
   :param verbosity: Controls verbosity. 0 for no verbosity, 1 for minimal (prints warnings and runtimes of major steps), 2 for
                     medium (also prints layout optimization messages) and 3 for full (down to neighborhood search, useful for debugging).
   :type verbosity: int (optional, default 1).
   :param cache_base: Whether to cache intermediate matrices used in computing orthogonal bases
                      (k-nearest-neighbors, diffusion harmonics etc).
   :type cache_base: bool (optional, default True).
   :param cache_graph: Whether to cache intermediate matrices used in computing topological graphs
                       (k-nearest-neighbors, diffusion harmonics etc).
   :type cache_graph: bool (optional, default True).
   :param plot_spectrum: Whether to plot the informational decay spectrum obtained during eigendecomposition of similarity matrices.
   :type plot_spectrum: bool (optional, default False).
   :param eigen_expansion: Whether to try to find a discrete eigengap during eigendecomposition. This can *severely* impact runtime,
                           as it can take numerous eigendecompositions to do so.
   :type eigen_expansion: bool (optional, default False).

   .. py:method:: __repr__(self)

      Return repr(self).


   .. py:method:: fit(self, X)

      Learn topological distances with diffusion harmonics and continuous metrics. Computes affinity operators
      that approximate the Laplace-Beltrami operator

      :param X: High-dimensional data matrix. Currently, supports only data from similar type (i.e. all bool, all float)

      :returns: * *TopoGraph instance with several slots, populated as per user settings.*
                * If `basis='diffusion'`, populates `TopoGraph.MSDiffMap` with a multiscale diffusion mapping of data, and -- `TopoGraph.DiffBasis` with a fitted `topo.tpgraph.diff.Diffusor()` class containing diffusion metrics
                  and transition probabilities, respectively stored in TopoGraph.DiffBasis.K and TopoGraph.DiffBasis.T
                * If `basis='continuous'`, populates `TopoGraph.CLapMap` with a continous Laplacian Eigenmapping of data, and -- `TopoGraph.ContBasis` with a continuous-k-nearest-neighbors model, containing continuous metrics and
                  adjacency, respectively stored in `TopoGraph.ContBasis.K` and `TopoGraph.ContBasis.A`.
                * If `basis='fuzzy'`, populates `TopoGraph.FuzzyLapMap` with a fuzzy Laplacian Eigenmapping of data, and -- `TopoGraph.FuzzyBasis` with a fuzzy simplicial set model, containing continuous metrics.


   .. py:method:: eigenspectrum(self, basis=None, use_eigs='knee', verbose=False)

      Visualize the scree plot of information entropy.

      :param `basis`: If `None`, will use the default basis. Otherwise, uses the specified basis
                      (must be 'diffusion', 'continuous' or 'fuzzy').
      :type `basis`: str (optional, default None).
      :param `use_eigs`: Number of eigenvectors to use. If 'max', expands to the maximum number of positive eigenvalues
                         (reach of numerical precision), else to the maximum amount of computed components.
                         If 'knee', uses Kneedle to find an optimal cutoff point, and expands it by ``expansion``.
                         If 'comp_gap', tries to find a discrete eigengap from the computation process.
      :type `use_eigs`: int or str (optional, default 'knee').
      :param verbose: Controls verbosity
      :type verbose: bool (optional, default False).

      :returns: *A nice scree plot .*


   .. py:method:: scree_plot(self)


   .. py:method:: transform(self, basis=None)

      Learns new affinity, topological operators from chosen basis.

      :param self: TopOGraph instance.
      :param basis: Base to use when building the topological graph. Defaults to the active base ( `TopOGraph.basis`).
                    Setting this updates the active base.
      :type basis: str, optional.

      :returns: *scipy.sparse.csr.csr_matrix, containing the similarity matrix that encodes the topological graph.*


   .. py:method:: spectral_layout(self, graph=None, n_components=2, cache=True)

      Performs a multicomponent spectral layout of the data and the target similarity matrix.

      :param graph: affinity matrix (i.e. topological graph). If None (default), uses the default graph from the default basis.
      :type graph: scipy.sparse.csr.csr_matrix.
      :param n_components: number of dimensions to embed into.
      :type n_components: int (optional, default 2).
      :param cache: Whether to cache the embedding to the `TopOGraph` object.
      :type cache: bool (optional, default True).

      :returns: *np.ndarray containing the resulting embedding.*


   .. py:method:: fuzzy_graph(self, X=None, basis=None, knn_indices=None, knn_dists=None, cache=True)

      Given a topological basis, a neighborhood size, and a measure of distance
      compute the fuzzy simplicial set (here represented as a fuzzy graph in
      the form of a sparse matrix) associated to the data. This is done by
      locally approximating geodesic distance at each point, creating a fuzzy
      simplicial set for each such point, and then combining all the local
      fuzzy simplicial sets into a global one via a fuzzy union.

      :param X: The data to be modelled as a fuzzy simplicial set. If None, defaults to the active orthogonal basis.
      :type X: np.ndarray, scipy.sparse.csr_matrix or str, 'diffusion' or 'continuous' (optional, default None).
      :param knn_indices: If the k-nearest neighbors of each point has already been calculated
                          you can pass them in here to save computation time. This should be
                          an array with the indices of the k-nearest neighbors as a row for
                          each data point.
      :type knn_indices: array of shape (n_samples, n_neighbors) (optional).
      :param knn_dists: If the k-nearest neighbors of each point has already been calculated
                        you can pass them in here to save computation time. This should be
                        an array with the distances of the k-nearest neighbors as a row for
                        each data point.
      :type knn_dists: array of shape (n_samples, n_neighbors) (optional).
      :param cache: Whether to store the fuzzy simplicial set graph in the TopOGraph object.
      :type cache: bool, optional (default True)

      :returns: **fuzzy_simplicial_set** (*coo_matrix*) -- A fuzzy simplicial set represented as a sparse matrix. The (i, j) entry of the matrix represents
                the membership strength of the 1-simplex between the ith and jth sample points.


   .. py:method:: MDE(self, basis=None, graph=None, n_components=2, n_neighbors=None, type='isomorphic', n_epochs=500, snapshot_every=30, constraint=None, init='quadratic', repulsive_fraction=None, max_distance=None, device='cpu', eps=0.001, mem_size=1)

          This function constructs a Minimum Distortion Embedding (MDE) problem for preserving the
      structure of original data. This MDE problem is well-suited for
      visualization (using ``dim`` 2 or 3), but can also be used to
      generate features for machine learning tasks (with ``dim`` = 10,
      50, or 100, for example). It yields embeddings in which similar items
      are near each other, and dissimilar items are not near each other.
      The original data can either be a data matrix, or a graph.
      Data matrices should be torch Tensors, NumPy arrays, or scipy sparse
      matrices; graphs should be instances of ``pymde.Graph``.
      The MDE problem uses distortion functions derived from weights (i.e.,
      penalties).
      To obtain an embedding, call the ``embed`` method on the returned ``MDE``
      object. To plot it, use ``pymde.plot``.

      :param basis: Which basis to use when computing the embedding. Defaults to the active basis.
      :type basis: str ('diffusion', 'continuous' or 'fuzzy').
      :param graph: The affinity matrix to embedd with. Defaults to the active graph. If init = 'spectral',
                    a fuzzy simplicial set is used, and this argument is ignored.
      :type graph: scipy.sparse matrix.
      :param n_components: The embedding dimension. Use 2 or 3 for visualization.
      :type n_components: int (optional, default 2).
      :param constraint: Constraint to use when optimizing the embedding. Options are 'standardized',
                         'centered', `None` or a `pymde.constraints.Constraint()` function.
      :type constraint: str (optional, default 'standardized').
      :param n_neighbors: The number of nearest neighbors to compute for each row (item) of
                          ``data``. A sensible value is chosen by default, depending on the
                          number of items.
      :type n_neighbors: int (optional).
      :param repulsive_fraction: How many repulsive edges to include, relative to the number
                                 of attractive edges. ``1`` means as many repulsive edges as attractive
                                 edges. The higher this number, the more uniformly spread out the
                                 embedding will be. Defaults to ``0.5`` for standardized embeddings, and
                                 ``1`` otherwise. (If ``repulsive_penalty`` is ``None``, this argument
                                 is ignored.)
      :type repulsive_fraction: float (optional).
      :param max_distance: If not None, neighborhoods are restricted to have a radius
                           no greater than ``max_distance``.
      :type max_distance: float (optional).
      :param init: Initialization strategy; np.ndarray, 'quadratic' or 'random'.
      :type init: str or np.ndarray (optional, default 'quadratic')
      :param device: Device for the embedding (eg, 'cpu', 'cuda').
      :type device: str (optional).

      :returns: *torch.tensor* -- A ``pymde.MDE`` object, based on the original data.


   .. py:method:: MAP(self, data=None, graph=None, n_components=2, min_dist=0.3, spread=1, initial_alpha=1.2, n_epochs=800, metric=None, metric_kwds={}, output_metric='euclidean', output_metric_kwds={}, gamma=1.2, negative_sample_rate=10, init='spectral', random_state=None, euclidean_output=True, parallel=True, densmap=False, densmap_kwds={}, output_dens=False, return_aux=False)

      ""

      Manifold Approximation and Projection, as proposed by Leland McInnes with an uniform distribution assumption
      in the seminal [UMAP algorithm](https://umap-learn.readthedocs.io/en/latest/index.html). Performs a fuzzy
      simplicial set embedding, using a specified initialisation method and then minimizing the fuzzy set cross
      entropy between the 1-skeletons of the high and low dimensional fuzzy simplicial sets. The fuzzy simplicial
      set embedding was proposed and implemented by Leland McInnes in UMAP (see `umap-learn
      <https://github.com/lmcinnes/umap>`). Here we're using it only for the projection (layout optimization)
      by minimizing the cross-entropy between a phenotypic map (i.e. data, TopOMetry non-uniform latent mappings) and
      its graph topological representation.


      :param data: The source data to be embedded by UMAP. If `None` (default), the active basis will be used.
      :type data: array of shape (n_samples, n_features).
      :param graph: The 1-skeleton of the high dimensional fuzzy simplicial set as
                    represented by a graph for which we require a sparse matrix for the
                    (weighted) adjacency matrix. If `None` (default), a fuzzy simplicial set
                    is computed with default parameters.
      :type graph: scipy.sparse.csr_matrix (n_samples, n_samples).
      :param n_components: The dimensionality of the euclidean space into which to embed the data.
      :type n_components: int (optional, default 2).
      :param initial_alpha: Initial learning rate for the SGD.
      :type initial_alpha: float (optional, default 1).
      :param gamma: Weight to apply to negative samples.
      :type gamma: float (optional, default 1.2).
      :param negative_sample_rate: The number of negative samples to select per positive sample
                                   in the optimization process. Increasing this value will result
                                   in greater repulsive force being applied, greater optimization
                                   cost, but slightly more accuracy.
      :type negative_sample_rate: int (optional, default 5).
      :param n_epochs: The number of training epochs to be used in optimizing the
                       low dimensional embedding. Larger values result in more accurate
                       embeddings. If 0 is specified a value will be selected based on
                       the size of the input dataset (200 for large datasets, 500 for small).
      :type n_epochs: int (optional, default 0).
      :param init:
      :type init: string (optional, default 'spectral').
      :param How to initialize the low dimensional embedding. Options are:
                                                                           * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
                                                                           * 'random': assign initial embedding positions at random.
                                                                           * A numpy array of initial embedding positions.
      :param random_state: A state capable being used as a numpy random state.
      :type random_state: numpy RandomState or equivalent.
      :param metric: The metric used to measure distance in high dimensional space; used if
                     multiple connected components need to be layed out. Defaults to `TopOGraph.graph_metric`.
      :type metric: string or callable.
      :param metric_kwds: Key word arguments to be passed to the metric function; used if
                          multiple connected components need to be layed out.
      :type metric_kwds: dict (optional, no default).
      :param densmap: Whether to use the density-augmented objective function to optimize
                      the embedding according to the densMAP algorithm.
      :type densmap: bool (optional, default False).
      :param densmap_kwds: Key word arguments to be used by the densMAP optimization.
      :type densmap_kwds: dict (optional, no default).
      :param output_dens: Whether to output local radii in the original data and the embedding.
      :type output_dens: bool (optional, default False).
      :param output_metric: Function returning the distance between two points in embedding space and
                            the gradient of the distance wrt the first argument.
      :type output_metric: function (optional, no default).
      :param output_metric_kwds: Key word arguments to be passed to the output_metric function.
      :type output_metric_kwds: dict (optional, no default).
      :param euclidean_output: Whether to use the faster code specialised for euclidean output metrics
      :type euclidean_output: bool (optional, default True).
      :param parallel: Whether to run the computation using numba parallel.
                       Running in parallel is non-deterministic, and is not used
                       if a random seed has been set, to ensure reproducibility.
      :type parallel: bool (optional, default True).
      :param return_aux: Whether to also return the auxiliary data, i.e. initialization and local radii.
      :type return_aux: bool , (optional, default False).

      :returns: * **\* embedding** (*array of shape (n_samples, n_components)*) -- The optimized of ``graph`` into an ``n_components`` dimensional
                  euclidean space.
                * *\*  return_aux is set to True* -- aux_data : dict

                  Auxiliary dictionary output returned with the embedding.
                  ``aux_data['Y_init']``: array of shape (n_samples, n_components)
                  The spectral initialization of ``graph`` into an ``n_components`` dimensional
                  euclidean space.

                  When densMAP extension is turned on, this dictionary includes local radii in the original
                  data (``aux_data['rad_orig']``) and in the embedding (``aux_data['rad_emb']``).


   .. py:method:: PaCMAP(self, data=None, init='spectral', n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, pair_neighbors=None, pair_MN=None, pair_FP=None, distance='euclidean', lr=1.0, num_iters=450, intermediate=False)

      Performs Pairwise-Controlled Manifold Approximation and Projection.

      :param data: Data to be embedded. If None, will use the active orthogonal basis.
      :type data: np.ndarray or scipy.sparse.csr_matrix (optional, default None).
      :param init: Initialization positions. Defaults to a multicomponent spectral embedding ('spectral').
                   Other options are 'pca' or 'random'.
      :type init: np.ndarray of shape (N,2) or str (optional, default 'spectral').
      :param n_components: How many components to embedd into.
      :type n_components: int (optional, default 2).
      :param n_neighbors: How many neighbors to use during embedding. If None, will use `TopOGraph.graph_knn`.
      :type n_neighbors: int (optional, default None).
      :param MN_ratio: The ratio of the number of mid-near pairs to the number of neighbors, n_MN = n_neighbors * MN_ratio.
      :type MN_ratio: float (optional, default 0.5).
      :param FP_ratio: The ratio of the number of further pairs to the number of neighbors, n_FP = n_neighbors * FP_ratio.
      :type FP_ratio: float (optional, default 2.0).
      :param distance: Distance metric to use. Options are 'euclidean', 'angular', 'manhattan' and 'hamming'.
      :type distance: float (optional, default 'euclidean').
      :param lr: Learning rate of the AdaGrad optimizer.
      :type lr: float (optional, default 1.0).
      :param num_iters: Number of iterations. The default 450 is enough for most dataset to converge.
      :type num_iters: int (optional, default 450).
      :param intermediate: Whether PaCMAP should also output the intermediate stages of the optimization process of the lower
                           dimension embedding. If True, then the output will be a numpy array of the size (n, n_dims, 13),
                           where each slice is a "screenshot" of the output embedding at a particular number of steps,
                           from [0, 10, 30, 60, 100, 120, 140, 170, 200, 250, 300, 350, 450].
      :type intermediate: bool (optional, default False).
      :param pair_neighbors: Pre-specified neighbor pairs. Allows user to use their own graphs. Default to None.
      :type pair_neighbors: optional, default None.
      :param pair_MN: Pre-specified mid-near pairs. Allows user to use their own graphs. Default to None.
      :type pair_MN: optional, default None.
      :param pair_FP: Pre-specified further pairs. Allows user to use their own graphs. Default to None.
      :type pair_FP: optional, default None.

      :returns: *PaCMAP embedding.*


   .. py:method:: TriMAP(self, basis=None, graph=None, init=None, n_components=2, n_inliers=10, n_outliers=5, n_random=5, use_dist_matrix=False, metric='euclidean', lr=1000.0, n_iters=400, triplets=None, weights=None, knn_tuple=None, weight_adj=500.0, opt_method='dbd', return_seq=False)

      Graph layout optimization using triplets.

      :param basis:
      :type basis: str (optional, default None).
      :param graph:
      :type graph: str (optional, default None).
      :param init:
      :type init: str (optional, default None).
      :param n_components: Number of dimensions of the embedding.
      :type n_components: int (optional, default 2).
      :param n_inliers: Number of inlier points for triplet constraints.
      :type n_inliers: int (optional, default 10).
      :param n_outliers: Number of outlier points for triplet constraints.
      :type n_outliers: int (optional, default 5).
      :param n_random: Number of random triplet constraints per point.
      :type n_random: int (optional, default 5).
      :param metric: Distance measure ('euclidean', 'manhattan', 'angular', 'hamming')
      :type metric: str (optional, default 'euclidean').
      :param use_dist_matrix: Use TopOMetry's learned similarities between samples. As of now, this is unstable.
      :type use_dist_matrix: bool (optional, default False).
      :param lr: Learning rate.
      :type lr: int (optional, default 1000).
      :param n_iters: Number of iterations.
      :type n_iters: int (optional, default 400)
      :param opt_method:
                         Optimization method ('sd': steepest descent,  'momentum': GD with momentum,
                          'dbd': GD with momentum delta-bar-delta).
      :type opt_method: str (optional, default 'dbd')
      :param return_seq: Return the sequence of maps recorded every 10 iterations.
      :type return_seq: bool (optional, default False)

      :returns: *TriMAP embedding.*


   .. py:method:: tSNE(self, data=None, graph=None, n_components=2, early_exaggeration=12, n_iter=1000, n_iter_early_exag=250, n_iter_without_progress=30, min_grad_norm=1e-07, init=None, random_state=None, angle=0.5, cheat_metric=True)

      The classic t-SNE embedding, usually on top of a TopOMetry topological basis.

      :param data:
      :type data: optional, default None)
      :param graph:
      :type graph: optional, default None)
      :param n_components:
      :type n_components: int (optional, default 2).
      :param early_exaggeration:
      :type early_exaggeration: sets exaggeration
      :param n_iter:
      :type n_iter: number of iterations to optmizie
      :param n_iter_early_exag:
      :type n_iter_early_exag: number of iterations in early exaggeration
      :param init: Initialisation for the optimization problem.
      :type init: np.ndarray (optional, defaults to tg.SpecLayout)
      :param random_state:
      :type random_state: optional, default None


   .. py:method:: NCVis(self, data=None, n_components=2, n_jobs=-1, n_neighbors=15, distance='cosine', M=15, efC=30, random_seed=42, n_epochs=200, n_init_epochs=20, spread=1.0, min_dist=0.4, alpha=1.0, a=None, b=None, alpha_Q=1.0, n_noise=None)


   .. py:method:: affinity_clustering(self, graph=None, damping=0.5, max_iter=200, convergence_iter=15)


   .. py:method:: plot(self, target=None, space='2D', dims_gauss=None, labels=None, pt_size=1, marker='o', opacity=1, cmap='Spectral')

      Utility function for plotting TopOGraph layouts. This is independent from the model
      and can be used to plot arbitrary layouts. Wraps around [Leland McInnes non-euclidean space
      embeddings](https://umap-learn.readthedocs.io/en/latest/embedding_space.html).

      Parameters,
      ----------
      target : np.ndarray (optional, default `None`).
          np.ndarray containing the layout to be plotted. If `None` (default), looks for
          available MDE and the MAP embedding, in this order.

      space : str (optional, default '2D').
          Projection space. Defaults to 2D space ('2D'). Options are:
              - '2D' (default);
              - '3D' ;
              - 'hyperboloid_2d' (2D hyperboloid space, 'hyperboloid' );
              - 'hyperboloid_3d' (3D hyperboloid space - note this uses a 2D input);
              - 'poincare' (Poincare disk - note this uses a 2D input);
              - 'spherical' (haversine-derived spherical space - note this uses a 2D input);
              - 'sphere_projection' (haversine-derived spherical space, projected to 2D);
              - 'toroid' (custom toroidal space);
              - 'gauss_potential' (gaussian potential, expects at least 5 dimensions, uses
                the additional parameter `dims_gauss`);

      dims_gauss : list (optional, default [2,3,4]).
          Which dimensions to use when plotting gaussian potential.

      labels : np.ndarray of int categories (optional).

      kwargs : additional kwargs for matplotlib

      :returns: 2D or 3D visualizations, depending on `space`.


   .. py:method:: run_models(self, X, bases=['diffusion', 'fuzzy', 'continuous'], graphs=['diff', 'cknn', 'fuzzy'])


   .. py:method:: run_layouts(self, X, n_components=2, bases=['diffusion', 'fuzzy', 'continuous'], graphs=['diff', 'cknn', 'fuzzy'], layouts=['tSNE', 'MAP', 'MDE', 'PaCMAP', 'TriMAP', 'NCVis'])

      Master function to easily run all combinations of possible bases and graphs that approximate the
      [Laplace-Beltrami Operator](), and the 6 layout options within TopOMetry: tSNE, MAP, MDE, PaCMAP, TriMAP,
      and NCVis.

      :param X: Data matrix.
      :type X: np.ndarray or scipy.sparse.csr_matrix
      :param n_components: Number of components for visualization.
      :type n_components: int (optional, default 2).
      :param bases: Which bases to compute. Defaults to all. To run only one or two bases, set it to
                    ['fuzzy', 'diffusion'] or ['continuous'], for exemple.
      :type bases: str (optional, default ['diffusion', 'continuous','fuzzy']).
      :param graphs: Which graphs to compute. Defaults to all. To run only one or two graphs, set it to
                     ['fuzzy', 'diff'] or ['cknn'], for exemple.
      :type graphs: str (optional, default ['diff', 'cknn','fuzzy']).
      :param layouts: Which layouts to compute. Defaults to all 6 options within TopOMetry: tSNE, MAP, MDE, PaCMAP,
                      TriMAP and NCVis. To run only one or two layouts, set it to
                      ['tSNE', 'MAP'] or ['PaCMAP'], for example.
      :type layouts: str (optional, default all ['tSNE', 'MAP', 'MDE', 'PaCMAP', 'TriMAP', 'NCVis']).

      :returns: *Populates the TopOMetry object slots.*



.. py:function:: save_pkl(TopOGraph, wd=None, filename='topograph.pkl')


.. py:function:: read_pkl(wd=None, filename='topograph.pkl')


.. py:function:: annotate_doc_types(mod, root)


.. py:data:: __version__
   :annotation: = 0.0.4.2

   

