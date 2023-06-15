#####################################
# Author: Davi Sidarta-Oliveira
# School of Medical Sciences,University of Campinas,Brazil
# contact: davisidarta[at]fcm[dot]unicamp[dot]com
# License: MIT
######################################
# Defining a projection class in a scikit-learn fashion to handle all projection methods

import numpy as np
import warnings
from scipy.sparse import csr_matrix, issparse
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator, TransformerMixin
from topo.utils._utils import get_indices_distances_from_sparse_matrix
from topo.layouts.isomap import Isomap
from topo.layouts.map import fuzzy_embedding
from topo.utils._utils import get_landmark_indices
from topo.spectral.eigen import spectral_layout
from topo.base.ann import kNN
from topo.tpgraph.kernels import Kernel

# dumb warning, suggests lilmatrix but it doesnt work
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

try:
    import hnswlib
    _have_hnswlib = True
except ImportError:
    _have_hnswlib = False
try:
    import nmslib
    _have_nmslib = True
except ImportError:
    _have_nmslib = False
try:
    import annoy
    _have_annoy = True
except ImportError:
    _have_annoy = False
try:
    import faiss
    _have_faiss = True
except ImportError:
    _have_faiss = False


class Projector(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible class that handles all projection methods. 
    Ideally, it takes in either a orthonormal eigenbasis or a graph kernel learned from such an eigenbasis.
    It is included in TopOMetry to allow custom `TopOGraph`-like pipelines (projection is the final step).

    Parameters
    ----------
    n_components : int (optional, default 2).
        Number of dimensions to optimize the layout to. Usually 2 or 3 if you're into visualizing data.

    projection_method : str (optional, default 'Isomap').
        Which projection method to use. Only 'Isomap', 't-SNE' and 'MAP' are implemented out of the box. 't-SNE' uses and 'MAP' relies
        on code that is adapted from UMAP. Current options are:
            * ['Isomap']() - one of the first manifold learning methods
            * ['t-SNE'](https://github.com/DmitryUlyanov/Multicore-TSNE) - a classic manifold learning method
            * 'MAP'- a lighter [UMAP](https://umap-learn.readthedocs.io/en/latest/index.html) with looser assumptions
            * ['UMAP'](https://umap-learn.readthedocs.io/en/latest/index.html)
            * ['PaCMAP'](http://jmlr.org/papers/v22/20-1061.html) (Pairwise-controlled Manifold Approximation and Projection) - for balanced visualizations
            * ['TriMAP'](https://github.com/eamid/trimap) - dimensionality reduction using triplets
            * 'IsomorphicMDE' - [MDE](https://github.com/cvxgrp/pymde) with preservation of nearest neighbors
            * 'IsometricMDE' - [MDE](https://github.com/cvxgrp/pymde) with preservation of pairwise distances
            * ['NCVis'](https://github.com/stat-ml/ncvis) (Noise Contrastive Visualization) - a UMAP-like method with blazing fast performance
        These are frankly quite direct to add, so feel free to make a feature request if your favorite method is not listed here.

    metric : str (optional, default 'euclidean').
        The metric to use when computing distances.
        Possible values are: 'cosine', 'euclidean' and others. Accepts precomputed distances ('precomputed').

    n_neighbors : int (optional, default 10).
        The number of neighbors to use when computing the kernel matrix. Ignored if `pairwise` set to `True`.

    landmarks : int or np.ndarray (optional, default None).
        If passed as `int`, will obtain the number of landmarks. If passed as `np.ndarray`, will use the specified indexes in the array.
        Any value other than `None` will result in only the specified landmarks being used in the layout optimization, and will
        populate the Projector.landmarks_ slot.

    landmark_method : str (optional, default 'kmeans').
        The method to use for selecting landmarks. If `landmarks` is passed as an `int`, this will be used to select the landmarks.
        Can be either 'kmeans' or 'random'.

    num_iters : int (optional, default 1000).
        Most (if not all) methods optimize the layout up to a limit number of iterations. Use this parameter to set this number.

    keep_estimator : bool (optional, default False).
        Whether to keep the used estimator as Projector.estimator_ after fitting. Useful if you want to use it later (e.g. UMAP
        allows inverse transforms and out-of-sample mapping).

    """

    def __init__(self,
                 n_components=2,
                 projection_method='MAP',
                 metric='euclidean',
                 n_neighbors=10,
                 n_jobs=1,
                 landmarks=None,
                 landmark_method='kmeans',
                 num_iters=800,
                 init='spectral',
                 nbrs_backend='nmslib',
                 keep_estimator=False,
                 random_state=None,
                 verbose=False):
        self.n_components = n_components
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.projection_method = projection_method
        self.landmarks = landmarks
        self.landmark_method = landmark_method
        self.num_iters = num_iters
        self.n_jobs = n_jobs
        self.init = init
        self.nbrs_backend = nbrs_backend
        self.keep_estimator = keep_estimator
        self.random_state = random_state
        self.verbose = verbose
        self.Y_ = None
        self.landmarks_ = None
        self.N = None
        self.M = None
        self.init_Y_ = None

    def __repr__(self):
        if self.Y_ is not None:
            if self.metric == 'precomputed':
                msg = "Projector() estimator fitted with precomputed distance matrix"
            elif (self.N is not None) and (self.M is not None):
                msg = "Projector() estimator fitted with %i samples and %i observations" % (self.N, self.M)
        else:
            msg = "Kernel() estimator without any fitted data."

        method_msg = " using %i" % self.projection_method

        msg = msg + method_msg
        return msg

    def _parse_backend(self):
        if self.nbrs_backend == 'hnswlib':
            if not _have_hnswlib:
                if _have_nmslib:
                    self.nbrs_backend == 'nmslib'
                elif _have_annoy:
                    self.nbrs_backend == 'annoy'
                elif _have_faiss:
                    self.nbrs_backend == 'faiss'
                else:
                    self.nbrs_backend == 'sklearn'
        elif self.nbrs_backend == 'nmslib':
            if not _have_nmslib:
                if _have_hnswlib:
                    self.nbrs_backend == 'hnswlib'
                elif _have_annoy:
                    self.nbrs_backend == 'annoy'
                elif _have_faiss:
                    self.nbrs_backend == 'faiss'
                else:
                    self.backend == 'sklearn'
        elif self.nbrs_backend == 'annoy':
            if not _have_annoy:
                if _have_nmslib:
                    self.nbrs_backend == 'nmslib'
                elif _have_hnswlib:
                    self.nbrs_backend == 'hnswlib'
                elif _have_faiss:
                    self.nbrs_backend == 'faiss'
                else:
                    self.nbrs_backend == 'sklearn'
        elif self.nbrs_backend == 'faiss':
            if not _have_faiss:
                if _have_nmslib:
                    self.nbrs_backend == 'nmslib'
                elif _have_hnswlib:
                    self.nbrs_backend == 'hnswlib'
                elif _have_annoy:
                    self.nbrs_backend == 'annoy'
                else:
                    self.nbrs_backend == 'sklearn'
        else:
            print(
                "Warning: no approximate nearest neighbor library found. Using sklearn's KDTree instead.")
            self.nbrs_backend == 'sklearn'

    def fit(self, X, **kwargs):
        """
        Calls the desired projection method on the specified data.

        Parameters
        -----------
        X : array-like, shape (n_samples, n_features) or topo.Kernel() class.
            The set of points to compute the kernel matrix for. Accepts np.ndarrays and scipy.sparse matrices or a `topo.Kernel()` object.
            If precomputed, assumed to be a square symmetric semidefinite matrix.

        kwargs : dict (optional, default {}).
            Additional keyword arguments for the desired projection method.

        Returns
        -----------
        Projector() class with populated Projector.Y_ attribute.

        """
        self.random_state = check_random_state(self.random_state)

        # Check inputs

        if self.projection_method not in ['Isomap', 't-SNE', 'MAP', 'UMAP', 'PaCMAP', 'TriMAP', 'IsomorphicMDE', 'IsometricMDE', 'NCVis']:
            raise ValueError(
                '\'projection_method\' must be one of \'Isomap\', \'t-SNE\', \'MAP\', \'UMAP\', \'PaCMAP\', \'TriMAP\', \'IsomorphicMDE\', \'IsometricMDE\' or \'NCVis\'.')

        if self.landmarks is not None:
            if isinstance(self.landmarks, int):
                self.landmarks_ = get_landmark_indices(
                    X, n_landmarks=self.landmarks, method=self.landmark_method, random_state=self.random_state)
            elif isinstance(self.landmarks, np.ndarray):
                self.landmarks_ = self.landmarks
            else:
                raise ValueError(
                    '\'landmarks\' must be either an integer or a numpy array.')

        if isinstance(X, Kernel):
            if self.landmarks_ is not None:
                if self.projection_method != 'Isomap':
                    self.N = self.landmarkds_
                    K = X.P[self.landmarkds_, self.landmarkds_].copy()
            else:
                self.N = X.N
            self.M = X.M
            K = X.P.copy()
        else:
            if self.metric != 'precomputed':
                if issparse(X):
                    X = X.toarray()
                if self.landmarks_ is not None:
                    if self.projection_method != 'Isomap':
                        X = X[self.landmarkds_, :]
                K = kNN(X, metric=self.metric, n_neighbors=self.n_neighbors
                        , n_jobs=self.n_jobs, backend=self.nbrs_backend)
            else:
                if self.landmarks_ is not None:
                    if self.projection_method != 'Isomap':
                        K = X[self.landmarkds_, self.landmarkds_].copy()
                    else:
                        K = X.copy()
                else:
                    K = X.copy()

        if isinstance(self.init, np.ndarray):
            self.init_Y_ = self.init
        else:
            if self.init == 'spectral':
                try:
                    self.init_Y_ = spectral_layout(
                        K, self.n_components, self.random_state, laplacian_type='random_walk', eigen_tol=10e-4, return_evals=False)
                except:
                    print(
                        'Multicomponent spectral layout initialization failed, falling back to simple spectral layout...')
                    from topo.spectral.eigen import EigenDecomposition
                    self.init_Y_ = EigenDecomposition(
                        n_components=self.n_components).fit_transform(K)
            else:
                self.init_Y_ = self.random_state.randn(
                    K.shape[0], self.n_components)
        # Fit the desired method
        if self.projection_method == 'Isomap':
            self.Y_ = Isomap(K, n_components=self.n_components,
                             n_neighbors=self.n_neighbors, metric='precomputed',
                             landmarks=self.landmarks_,
                             landmark_method=self.landmark_method,
                             eig_tol=0, n_jobs=self.n_jobs)

        elif self.projection_method == 't-SNE':
            try:
                from MulticoreTSNE import MulticoreTSNE as TSNE
                _HAS_MCTSNE = True
            except ImportError:
                _HAS_MCTSNE = False
            if not _HAS_MCTSNE:
                from sklearn.manifold import TSNE
            self.estimator_ = TSNE(n_components=self.n_components,
                                   metric='precomputed', n_iter=self.num_iters)
            self.Y_ = self.estimator_.fit_transform(K)

        elif self.projection_method == 'MAP':
            self.Y_ = fuzzy_embedding(K, n_components=self.n_components, init=self.init_Y_,
                                      n_epochs=self.num_iters, random_state=self.random_state, **kwargs)[0]

        elif self.projection_method == 'UMAP':
            try:
                import umap
            except ImportError:
                raise ImportError(
                    'UMAP is not installed. Please install UMAP with \'pip install umap\' before using this method.')
            self.estimator_ = umap.UMAP(
                n_components=self.n_components, precomputed_knn=K, init=self.init_Y_, n_epochs=self.num_iters, **kwargs)
            self.Y_ = self.estimator_.fit_transform(X)

        elif self.projection_method == 'PaCMAP':
            try:
                import pacmap
            except ImportError:
                raise ImportError(
                    'PaCMAP is not installed. Please install PaCMAP with \'pip install pacmap\' before using this method.')
            import warnings
            warnings.filterwarnings("ignore")  # PaCMAP is way too verbose...
            if self.metric == 'cosine':
                metric = 'angular'
            else:
                metric = self.metric
            self.estimator_ = pacmap.PaCMAP(n_components=self.n_components, n_neighbors=self.n_neighbors,
                                            apply_pca=False, distance=metric, num_iters=self.num_iters, verbose=self.verbose, **kwargs)
            self.Y_ = self.estimator_.fit_transform(X=X, init=self.init_Y_)

        elif self.projection_method == 'TriMAP':
            try:
                import trimap
            except ImportError:
                raise ImportError(
                    'TriMAP is not installed. Please install TriMAP with \'pip install trimap\' before using this method.')

            if self.metric == 'cosine':
                metric = 'angular'
            else:
                metric = self.metric
            self.estimator_ = trimap.TRIMAP(
                n_components=self.n_components, distance=self.metric, n_iters=self.num_iters, verbose=self.verbose, **kwargs)
            self.Y_ = self.estimator_.fit_transform(X=X, init=self.init_Y_)

        elif self.projection_method == 'IsomorphicMDE':
            try:
                import pymde
                from pymde import constraints, preprocess, problem, quadratic
                from pymde.functions import losses
            except ImportError:
                raise ImportError(
                    'pymde is not installed. Please install pymde with \'pip install pymde\' before using this method.')
            attractive_penalty = pymde.penalties.Log1p
            repulsive_penalty = pymde.penalties.Log
            loss = pymde.losses.Absolute
            graph = preprocess.graph.Graph(K)
            self.estimator_ = IsomorphicMDE(graph,
                                            attractive_penalty=attractive_penalty,
                                            repulsive_penalty=repulsive_penalty,
                                            embedding_dim=self.n_components,
                                            n_neighbors=self.n_neighbors,
                                            init='quadratic',
                                            verbose=self.verbose, **kwargs)

            self.Y_ = self.estimator_.embed(
                max_iter=self.num_iters, memory_size=10, eps=10e-4, verbose=self.verbose)

        elif self.projection_method == 'IsometricMDE':
            try:
                import pymde
                from pymde import constraints, preprocess, problem, quadratic
                from pymde.functions import losses
            except ImportError:
                raise ImportError(
                    'pymde is not installed. Please install pymde with \'pip install pymde\' before using this method.')
            attractive_penalty = pymde.penalties.Log1p
            repulsive_penalty = pymde.penalties.Log
            loss = pymde.losses.Absolute
            graph = preprocess.graph.Graph(K)
            max_distance = 5e7
            self.estimator_ = IsometricMDE(graph,
                                           embedding_dim=self.n_components,
                                           loss=loss,
                                           constraint=None,
                                           max_distances=max_distance,
                                           verbose=self.verbose, **kwargs)
            self.Y_ = self.estimator_.embed(
                max_iter=self.num_iters, memory_size=1, verbose=self.verbose)

        elif self.projection_method == 'NCVis':
            try:
                import ncvis
            except ImportError:
                raise ImportError(
                    'ncvis is not installed. Please install ncvis with \'pip install ncvis\' before using this method.')
            self.estimator_ = ncvis.NCVis(d=self.n_components, n_neighbors=self.n_neighbors,
                                          distance=self.metric, n_epochs=self.num_iters, n_threads=self.n_jobs, **kwargs)
            self.Y_ = self.estimator_.fit_transform(X)

    def transform(self, X=None):
        """
        Calls the transform method of the desired method. 
        If the desired method does not have a transform method, calls the results from the fit method.

        Returns
        ----------
        Y : np.ndarray (n_samples, n_components).
            Projection results
        """
        if self.projection_method == 'UMAP':
            return self.estimator_.transform(X)
        else:
            return self.Y_

    def fit_transform(self, X, **kwargs):
        """
        Calls the fit_transform method of the desired method. 
        If the desired method does not have a fit_transform method, calls the results from the fit method.

        Returns
        ----------
        Y : np.ndarray (n_samples, n_components).
            Projection results
        """

        self.fit(X, **kwargs)
        return self.Y_


# Check if pymde is installed
try:
    import pymde
    _HAS_PYMDE = True
except ImportError:
    _HAS_PYMDE = False


# Define custom pymde problems
if _HAS_PYMDE:
    def _remove_anchor_anchor_edges(edges, data, anchors):
        # exclude edges in which both items are anchors, since these
        # items are already pinned in place by the anchor constraint
        # NOTICE: This is exactly as implemented by Akshay Agrawal, at least for now.
        neither_anchors_mask = ~(
            (edges[:, 0][..., None] == anchors).any(-1)
            * (edges[:, 1][..., None] == anchors).any(-1)
        )
        edges = edges[neither_anchors_mask]
        data = data[neither_anchors_mask]
        return edges, data

    def IsomorphicMDE(data,
                      attractive_penalty=None,
                      repulsive_penalty=None,
                      embedding_dim=2,
                      constraint=None,
                      n_neighbors=None,
                      repulsive_fraction=None,
                      max_distance=None,
                      init='quadratic',
                      eps=1e-04,
                      max_iter=100,
                      memory_size=1,
                      print_every=None,
                      device='cpu',
                      verbose=False,
                      **kwargs):
        # Inherits from pymde.recipes.preserve_neighbors()
        """
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
        Arguments
        ---------
        data: {torch.Tensor, numpy.ndarray, scipy.sparse matrix} or pymde.Graph
            The original data, a data matrix of shape ``(n_items, n_features)`` or
            a graph. Neighbors are computed using Euclidean distance if the data is
            a matrix, or the shortest-path metric if the data is a graph.
        embedding_dim: int
            The embedding dimension. Use 2 or 3 for visualization.
        attractive_penalty: pymde.Function class (or factory)
            Callable that constructs a distortion function, given positive
            weights. Typically one of the classes from ``pymde.penalties``,
            such as ``pymde.penalties.log1p``, ``pymde.penalties.Huber``, or
            ``pymde.penalties.Quadratic``.
        repulsive_penalty: pymde.Function class (or factory)
            Callable that constructs a distortion function, given negative
            weights. (If ``None``, only positive weights are used.) For example,
            ``pymde.penalties.Log`` or ``pymde.penalties.InversePower``.
        constraint: pymde.constraints.Constraint (optional)
            Embedding constraint, like ``pymde.Standardized()`` or
            ``pymde.Anchored(anchors, values)`` (or ``None``). Defaults to no
            constraint when a repulsive penalty is provided, otherwise defaults to
            ``pymde.Standardized()``.
        n_neighbors: int (optional)
            The number of nearest neighbors to compute for each row (item) of
            ``data``. A sensible value is chosen by default, depending on the
            number of items.
        repulsive_fraction: float (optional)
            How many repulsive edges to include, relative to the number
            of attractive edges. ``1`` means as many repulsive edges as attractive
            edges. The higher this number, the more uniformly spread out the
            embedding will be. Defaults to ``0.5`` for standardized embeddings, and
            ``1`` otherwise. (If ``repulsive_penalty`` is ``None``, this argument
            is ignored.)
        max_distance: float (optional)
            If not None, neighborhoods are restricted to have a radius
            no greater than ``max_distance``.
        init: str or np.ndarray (optional, default 'quadratic')
            Initialization strategy; np.ndarray, 'quadratic' or 'random'.
        eps :float (optional)
            Residual norm threshold; quit when the residual norm is smaller than eps.
        max_iter: int
            Maximum number of iterations.
        memory_size : int 
            The quasi-Newton memory. Larger values may lead to more stable behavior, but will increase the amount of time each iteration takes.
        print_every : int (optional)
            Print verbose output every print_every iterations.
        device: str (optional)
            Device for the embedding (eg, 'cpu', 'cuda').
        verbose: bool
            If ``True``, print verbose output.
        Returns
        -------
        pymde.MDE
            A ``pymde.MDE`` object, based on the original data.
        """
        try:
            import pymde
            _have_pymde = True
        except ImportError('pyMDE is needed for this embedding. Install it with `pip install pymde`'):
            return print('pyMDE is needed for this embedding. Install it with `pip install pymde`')

        from pymde import constraints, preprocess, problem, quadratic
        from pymde.functions import penalties
        try:
            import torch
        except ImportError('pytorch is needed for this embedding. Install it with `pip install pytorch`'):
            return print('pyMDE is needed for this embedding. Install it with `pip install pymde`')

        if attractive_penalty is None:
            attractive_penalty = penalties.Log1p

        if repulsive_penalty is None:
            repulsive_penalty = penalties.Log

        if isinstance(data, preprocess.graph.Graph):
            n = data.n_items
        elif data.shape[0] <= 1:
            raise ValueError("The data matrix must have at least two rows.")
        else:
            n = data.shape[0]

        if n_neighbors is None:
            # target included edges to be ~1% of total number of edges
            n_choose_2 = n * (n - 1) / 2
            n_neighbors = int(max(min(15, n_choose_2 * 0.01 / n), 5))

        if n_neighbors > n:
            problem.LOGGER.warning(
                (
                    "Requested n_neighbors {0} > number of items {1}."
                    " Setting n_neighbors to {2}"
                ).format(n_neighbors, n, n - 1)
            )
            n_neighbors = n - 1

        if constraint is None and repulsive_penalty is not None:
            constraint = constraints.Centered()
        elif constraint is None and repulsive_penalty is None:
            constraint = constraints.Standardized()

        # enforce a max distance, otherwise may very well run out of memory
        # when n_items is large
        if max_distance is None:
            max_distance = (3 * torch.quantile(data.distances, 0.75)).item()
        if verbose:
            problem.LOGGER.info(
                f"Computing {n_neighbors}-nearest neighbors, with "
                f"max_distance={max_distance}"
            )
        knn_graph = preprocess.generic.k_nearest_neighbors(
            data,
            k=n_neighbors,
            max_distance=max_distance,
            verbose=verbose,
        )
        edges = knn_graph.edges.to(device)
        weights = knn_graph.weights.to(device)

        if isinstance(constraint, constraints.Anchored):
            # remove anchor-anchor edges before generating intialization
            edges, weights = _remove_anchor_anchor_edges(
                edges, weights, constraint.anchors
            )

        # DS: add multicomponent spectral initialization
        if isinstance(init, np.ndarray):
            X_init = torch.tensor(init)
        elif init == "quadratic":
            if verbose:
                problem.LOGGER.info("Computing quadratic initialization.")
            X_init = quadratic.spectral(
                n, embedding_dim, edges, weights, device=device
            )
            if not isinstance(
                constraint, (constraints._Centered, constraints._Standardized)
            ):
                constraint.project_onto_constraint(X_init, inplace=True)
        elif init == "random":
            X_init = constraint.initialization(n, embedding_dim, device)
        else:
            raise ValueError(
                f"Unsupported value '{init}' for keyword argument `init`; "
                "the supported values are 'quadratic' and 'random', or a np.ndarray of shape (n_items, embedding_dim)."
            )

        if repulsive_penalty is not None:
            if repulsive_fraction is None:
                if isinstance(constraint, constraints._Standardized):
                    # standardization constraint already implicity spreads,
                    # so use a lower replusion
                    repulsive_fraction = 0.5
                else:
                    repulsive_fraction = 1

            n_choose_2 = int(n * (n - 1) / 2)
            n_repulsive = int(repulsive_fraction * (edges.shape[0]))
            # cannot sample more edges than there are available
            n_repulsive = min(n_repulsive, n_choose_2 - edges.shape[0])

            negative_edges = preprocess.sample_edges(
                n, n_repulsive, exclude=edges
            ).to(device)

            negative_weights = -torch.ones(
                negative_edges.shape[0], dtype=X_init.dtype, device=device
            )

            if isinstance(constraint, constraints.Anchored):
                negative_edges, negative_weights = _remove_anchor_anchor_edges(
                    negative_edges, negative_weights, constraint.anchors
                )

            edges = torch.cat([edges, negative_edges])
            weights = torch.cat([weights, negative_weights])

            f = penalties.PushAndPull(
                weights,
                attractive_penalty=attractive_penalty,
                repulsive_penalty=repulsive_penalty,
            )
        else:
            f = attractive_penalty(weights)

        if eps is None:
            eps = 1e-6 * torch.max(weights).item()
        mde = problem.MDE(
            n_items=n,
            embedding_dim=embedding_dim,
            edges=edges,
            distortion_function=f,
            constraint=constraint,
            device=device
        )
        mde._X_init = X_init

        # Won't need to cache the graph - we have already computed it and cached with TopOMetry

        distances = mde.distances(mde._X_init)
        if (distances == 0).any():
            # pathological scenario in which at least two points overlap can yield
            # non-differentiable average distortion. perturb the initialization to
            # mitigate.
            mde._X_init += 1e-4 * torch.randn(
                mde._X_init.shape,
                device=mde._X_init.device,
                dtype=mde._X_init.dtype,
            )
        return mde

    def IsometricMDE(data,
                     embedding_dim=2,
                     loss=None,
                     constraint=None,
                     max_distances=5e7,
                     device="cpu",
                     verbose=False):
        # Inherits from pymde.recipes.preserve_distances()
        """
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
        Arguments
        ---------
        data: {np.ndarray, torch.Tensor, scipy.sparse matrix} or pymde.Graph
            The original data, a data matrix of shape ``(n_items, n_features)`` or
            a graph.
        embedding_dim: int
            The embedding dimension.
        loss: pymde.Function class (or factory)
            Callable that constructs a distortion function, given
            original distances. Typically one of the classes defined in
            ``pymde.losses``, such as ``pymde.losses.Absolute``, or
            ``pymde.losses.WeightedQuadratic``.
        constraint: pymde.constraints.Constraint (optional)
            Embedding constraint, such as ``pymde.Standardized()`` or
            ``pymde.Anchored(anchors, values)`` (or ``None``). Defaults to no
            constraint. Note: when the constraint is ``pymde.Standardized()``,
            the original distances will be scaled by a constant (because the
            standardization constraint puts a limit on how large any one
            distance can be).
        max_distances: int
            Maximum number of distances to compute.
        device: str (optional)
            Device for the embedding (eg, 'cpu', 'cuda').
        verbose: bool
            If ``True``, print verbose output.
        Returns
        -------
        pymde.MDE
            A ``pymde.MDE`` instance, based on preserving the original distances.
        """
        try:
            import pymde
            _have_pymde = True
        except ImportError('pyMDE is needed for this embedding. Install it with `pip install pymde`'):
            return print('pyMDE is needed for this embedding. Install it with `pip install pymde`')

        from pymde import constraints, preprocess, problem, quadratic
        from pymde.functions import losses
        try:
            import torch
        except ImportError('pytorch is needed for this embedding. Install it with `pip install pytorch`'):
            return print('pyMDE is needed for this embedding. Install it with `pip install pymde`')
        from scipy.sparse import issparse

        if loss is None:
            loss = losses.Absolute

        if not isinstance(
            data, (np.ndarray, torch.Tensor, preprocess.graph.Graph)
        ) and not issparse(data):
            raise ValueError(
                "`data` must be a np.ndarray/torch.Tensor/scipy.sparse matrix"
                ", or a pymde.Graph."
            )

        if isinstance(data, preprocess.graph.Graph):
            n_items = data.n_items
        else:
            n_items = data.shape[0]
        n_all_edges = (n_items) * (n_items - 1) / 2
        retain_fraction = max_distances / n_all_edges

        if isinstance(data, preprocess.graph.Graph):
            edges = data.edges.to(device)
            deviations = data.distances.to(device)
        else:
            graph = preprocess.generic.distances(
                data, retain_fraction=retain_fraction, verbose=verbose
            )
            edges = graph.edges.to(device)
            deviations = graph.distances.to(device)

        if constraint is None:
            constraint = constraints.Centered()
        elif isinstance(constraint, constraints._Standardized):
            deviations = preprocess.scale(
                deviations, constraint.natural_length(n_items, embedding_dim)
            )
        elif isinstance(constraint, constraints.Anchored):
            edges, deviations = _remove_anchor_anchor_edges(
                edges, deviations, constraint.anchors
            )

        mde = problem.MDE(
            n_items=n_items,
            embedding_dim=embedding_dim,
            edges=edges,
            distortion_function=loss(deviations),
            constraint=constraint,
            device=device,
        )
        return mde
