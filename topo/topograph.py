# TopOMetry high-level API - the TopOGraph class
#
# Author: David S Oliveira <david.oliveira(at)dpag(dot)ox(dot)ac(dot)uk>
#
import time
import warnings
import numpy as np
import gc
import copy
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.sparse as sp
from scipy.sparse import issparse, csr_matrix
from typing import Dict, Tuple, Optional, Union
from topo.base.ann import kNN
from topo.tpgraph.kernels import Kernel
from topo.spectral.eigen import EigenDecomposition, spectral_layout
from topo.layouts.projector import Projector
from topo.tpgraph.intrinsic_dim import automated_scaffold_sizing


class TopOGraph(BaseEstimator, TransformerMixin):
    """
    Main TopOMetry class for learning topological similarities, bases, graphs, and layouts from high-dimensional data.

    The public API exposes a small set of named, stable objects:
      • knn_X               : initial kNN graph in input space
      • P_of_X              : diffusion operator on input space
      • spectral_scaffold(multiscale=True|False) : msDM/DM coordinates
      • knn_msZ / knn_Z     : kNN graphs on msDM/DM scaffolds (Z spaces)
      • P_of_msZ / P_of_Z   : refined diffusion operators on msDM/DM scaffolds
      • eigenvalues         : eigenvalues of the active eigenbasis (msDM by default)
      • MAP / msMAP         : 2D MAP layouts from DM/msDM refined graphs
      • PaCMAP / msPaCMAP   : 2D PaCMAP layouts from DM/msDM refined graphs
      • global_id_mle(), global_id_fsa(), local_ids(): intrinsic-dimension details

    Legacy benchmarking and combinatorial exploration remain available through:
      • BaseKernelDict, EigenbasisDict, GraphKernelDict, ProjectionDict
      • run_models(), eval_models_layouts()

    Parameters
    ----------
    base_knn : int (optional, default 30)
        k-nearest-neighbors for the base input space.

    graph_knn : int (optional, default 30)
        k-nearest-neighbors for the scaffold (Z) space.

    n_eigs : int (optional, default 100)
        Number of eigenpairs to compute (basis size cap).

    base_kernel : topo.tpgraph.Kernel (optional, default None)
        If provided and already fitted, X is not required.

    eigenmap_method : {'DM','msDM','LE','top','bottom'} (DEPRECATED; kept for BC)
        Deprecated. The class now *always* computes both DM and msDM. This argument is
        accepted for backwards compatibility and stored internally as `_eigenmap_method`.

    laplacian_type : {'unnormalized','normalized','random_walk','geometric'} (default 'normalized')
        Laplacian for spectral computations/layout.

    base_kernel_version : str (optional, default 'bw_adaptive')
        Kernel choice for the base graph (options include: 'bw_adaptive', 'fuzzy', 'cknn',
        'bw_adaptive_alpha_decaying', 'bw_adaptive_nbr_expansion', 'bw_adaptive_alpha_decaying_nbr_expansion', 'gaussian').

    graph_kernel_version : str (optional, default 'bw_adaptive')
        Kernel choice for the scaffold graphs (applies to both DM and msDM scaffolds).

    backend : {'hnswlib','nmslib','annoy','faiss','sklearn'} (default 'hnswlib')
        ANN backend.

    base_metric : str (optional, default 'cosine')
        Distance metric for base kNN.

    graph_metric : str (optional, default 'euclidean')
        Distance metric for scaffold kNN.

    diff_t : int (optional, default 1)
        Diffusion time for DM (ignored for msDM).

    sigma : float (optional, default 0.1)
        Bandwidth if 'gaussian' kernels are used.

    delta : float (optional, default 1.0)
        'cknn' radius parameter.

    n_jobs : int (optional, default 1)
        Threads for kNN. Use -1 for all cores.

    low_memory : bool (optional, default False)
        If True, avoids caching large kernel objects in dicts.

    eigen_tol : float (optional, default 1e-8)
        Eigen solver tolerance.

    eigensolver : {'arpack','lobpcg','amg','dense'} (default 'arpack')
        Eigen solver choice.

    projection_method : str (optional, default 'MAP')
        Default projection method for `.project()`.

    cache : bool (optional, default True)
        Cache Kernel / Eigen objects in dicts.

    verbosity : int (optional, default 0)
        0: silent; 1: major steps; 2+: include layout messages; 3: full debug for neighborhoods.

    random_state : int or np.random.RandomState (optional, default 42)

    Intrinsic dimensionality (automated scaffold sizing)
    ----------------------------------------------------
    id_method : {'mle','fsa'} (default 'mle')
        Method whose estimate selects scaffold size. (Both 'mle' and 'fsa' are computed and stored.)
    id_ks : int or iterable (default 50)
    id_metric : str (default 'euclidean')
    id_quantile : float (default 0.99; for 'fsa' only)
    id_min_components : int (default 16)
    id_max_components : int (default 512)
    id_headroom : float (default 0.5)

    Attributes
    ----------
    knn_X : scipy.sparse.csr_matrix
        Initial kNN graph on X (read-only property).

    P_of_X : scipy.sparse.csr_matrix
        Diffusion operator on X (read-only property).

    knn_msZ / knn_Z : scipy.sparse.csr_matrix
        kNN graphs on the msDM/DM scaffolds (read-only properties).

    P_of_msZ / P_of_Z : scipy.sparse.csr_matrix
        Diffusion operators on the msDM/DM scaffolds (read-only properties).

    eigenvalues : np.ndarray
        Eigenvalues of the active eigenbasis (msDM by default).

    MAP / msMAP / PaCMAP / msPaCMAP : np.ndarray (n_samples, 2)
        Ready-to-plot embeddings.

    BaseKernelDict, EigenbasisDict, GraphKernelDict, ProjectionDict : dict
        Legacy storage for advanced/benchmarking use-cases.
    """

    def __init__(self,
                 base_knn=30,
                 graph_knn=30,
                 min_eigs=100,
                 n_jobs=-1,
                 base_kernel=None,
                 base_kernel_version='bw_adaptive',
                 eigenmap_method='DM',  # deprecated
                 laplacian_type='normalized',
                 projection_method='MAP',
                 graph_kernel_version='bw_adaptive',
                 base_metric='cosine',
                 graph_metric='euclidean',
                 diff_t=1,
                 delta=1.0,
                 sigma=0.1,
                 low_memory=False,
                 eigen_tol=1e-8,
                 eigensolver='arpack',
                 backend='hnswlib',
                 cache=True,
                 verbosity=0,
                 random_state=42,
                 # ID defaults (both methods computed; `id_method` selects the size used)
                 id_method='fsa',
                 id_ks=50,
                 id_metric='euclidean',
                 id_quantile=0.99,
                 id_min_components=128,
                 id_max_components=1024,
                 id_headroom=0.5,
                 ):
        # Core config
        self.projection_method = projection_method
        self.diff_t = diff_t
        self.n_eigs = min_eigs
        self.base_knn = base_knn
        self.graph_knn = graph_knn
        self.n_jobs = n_jobs
        self.low_memory = low_memory
        self.backend = backend
        self.base_metric = base_metric
        self.graph_metric = graph_metric
        self.eigen_tol = eigen_tol
        self.eigensolver = eigensolver
        self.base_kernel = base_kernel
        self.base_kernel_version = base_kernel_version
        # deprecated, kept for BC
        self._eigenmap_method = eigenmap_method
        if eigenmap_method is not None:
            warnings.warn(
                "`eigenmap_method` is deprecated. TopOGraph now computes both DM and msDM scaffolds.",
                DeprecationWarning
            )
        self.graph_kernel_version = graph_kernel_version
        self.laplacian_type = laplacian_type
        self.eigenbasis = None
        self.verbosity = verbosity
        self.sigma = sigma
        self.bases_graph_verbose = False
        self.layout_verbose = False
        self.delta = delta
        self.random_state = random_state
        self.cache = cache

        # State containers
        self.eigenbasis_knn_graph = None            # (legacy single) — kept for BC
        self.base_nbrs_class = None
        self.base_knn_graph = None

        # Legacy/benchmarking stores
        self.BaseKernelDict = {}
        self.EigenbasisDict = {}
        self.GraphKernelDict = {}
        self.ProjectionDict = {}

        self.ClustersDict = {}
        self.n = None
        self.m = None
        self.SpecLayout = None
        self.runtimes = {}
        self.current_eigenbasis = None
        self.current_graphkernel = None
        self.global_dimensionality = None
        self.local_dimensionality = None
        self.RiemannMetricDict = {}
        self.LocalScoresDict = {}
        self.temp_file = None
        self._have_hnswlib = None
        self._have_nmslib = None
        self._have_annoy = None
        self._have_faiss = None

        # Automated scaffold sizing config
        self.id_method = id_method
        self.id_ks = id_ks
        self.id_metric = id_metric
        self.id_quantile = id_quantile
        self.id_min_components = id_min_components
        self.id_max_components = id_max_components
        self.id_headroom = id_headroom

        # ID details for both methods (always collected)
        self._id_details = {'mle': None, 'fsa': None}
        self._scaffold_components_dm = None
        self._scaffold_components_ms = None

        # Dual scaffold products
        self._knn_msZ = None
        self._knn_Z = None
        self._kernel_msZ = None
        self._kernel_Z = None

        # Snapshots storage (for MAP checkpointing) — ensure attributes exist
        self.msTopoMAP_snapshots = []
        self.TopoMAP_snapshots = []

    # ---------------------------------------------------------------------
    # Representation & internals
    # ---------------------------------------------------------------------
    def __repr__(self):
        if self.base_metric == 'precomputed':
            msg = "TopOGraph object with precomputed distances from %i samples" % (self.n) + " and:"
        elif (self.n is not None) and (self.m is not None):
            msg = "TopOGraph object with %i samples and %i observations" % (self.n, self.m) + " and:"
        else:
            msg = "TopOGraph object without any fitted data."
        msg += "\n . Base Kernels:"
        for keys in self.BaseKernelDict.keys():
            msg += " \n    %s - .BaseKernelDict['%s']" % (keys, keys)
        msg += "\n . Eigenbases:"
        for keys in self.EigenbasisDict.keys():
            msg += " \n    %s - .EigenbasisDict['%s']" % (keys, keys)
        msg += "\n . Graph Kernels:"
        for keys in self.GraphKernelDict.keys():
            msg += " \n    %s - .GraphKernelDict['%s']" % (keys, keys)
        msg += "\n . Projections:"
        for keys in self.ProjectionDict.keys():
            msg += " \n    %s - .ProjectionDict['%s']" % (keys, keys)
        msg += " \n Active base kernel  -  .base_kernel"
        msg += " \n Active eigenbasis  -  .eigenbasis"
        msg += " \n Active graph kernel  -  .graph_kernel"
        return msg

    def _parse_backend(self):
        try:
            import hnswlib  # noqa: F401
            self._have_hnswlib = True
        except ImportError:
            self._have_hnswlib = False
        try:
            import nmslib  # noqa: F401
            self._have_nmslib = True
        except ImportError:
            self._have_nmslib = False
        try:
            import annoy  # noqa: F401
            self._have_annoy = True
        except ImportError:
            self._have_annoy = False
        try:
            import faiss  # noqa: F401
            self._have_faiss = True
        except ImportError:
            self._have_faiss = False

        if self.backend == 'hnswlib':
            if not self._have_hnswlib:
                if self._have_nmslib:
                    self.backend = 'nmslib'
                elif self._have_annoy:
                    self.backend = 'annoy'
                elif self._have_faiss:
                    self.backend = 'faiss'
                else:
                    self.backend = 'sklearn'
        elif self.backend == 'nmslib':
            if not self._have_nmslib:
                if self._have_hnswlib:
                    self.backend = 'hnswlib'
                elif self._have_annoy:
                    self.backend = 'annoy'
                elif self._have_faiss:
                    self.backend = 'faiss'
                else:
                    self.backend = 'sklearn'
        elif self.backend == 'annoy':
            if not self._have_annoy:
                if self._have_nmslib:
                    self.backend = 'nmslib'
                elif self._have_hnswlib:
                    self.backend = 'hnswlib'
                elif self._have_faiss:
                    self.backend = 'faiss'
                else:
                    self.backend = 'sklearn'
        elif self.backend == 'faiss':
            if not self._have_faiss:
                if self._have_nmslib:
                    self.backend = 'nmslib'
                elif self._have_hnswlib:
                    self.backend = 'hnswlib'
                elif self._have_annoy:
                    self.backend = 'annoy'
                else:
                    self.backend = 'sklearn'
        else:
            print("Warning: no approximate nearest neighbor library found. Using sklearn's KDTree instead.")
            self.backend = 'sklearn'

    def _parse_random_state(self):
        if self.random_state is None:
            self.random_state = np.random.RandomState()
        elif isinstance(self.random_state, np.random.RandomState):
            pass
        elif isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        else:
            print('RandomState error! No random state was defined!')

    # ---------------------------------------------------------------------
    # Intrinsic dimension
    # ---------------------------------------------------------------------

    def _automated_sizing(self, X):
        """
        Run automated scaffold sizing directly on X (no eigenbasis required),
        record details for both methods, and set:
           - self._scaffold_components_ms
           - self._scaffold_components_dm
           - self.n_eigs  (so eigendecomposition uses this size cap)

        Notes
        -----
        - Uses self.id_method ('fsa' or 'mle') to pick the working size.
        - Applies caps: [id_min_components, id_max_components, n-2].
        """
        # robust caps
        n = X.shape[0]
        max_cap = min(int(self.id_max_components), max(2, n - 2))

        # FSA on X (quantile/robust proxy)
        n_fsa, fsa_details = automated_scaffold_sizing(
            X,
            method='fsa',
            ks=self.id_ks,
            backend=self.backend,
            metric=self.id_metric,
            n_jobs=self.n_jobs if self.n_jobs != -1 else None,
            quantile=self.id_quantile,
            min_components=int(self.id_min_components),
            max_components=int(max_cap),
            headroom=float(self.id_headroom),
            random_state=self.random_state,
            return_details=True,
        )
        self._id_details['fsa'] = fsa_details

        # MLE on X (median-of-locals)
        n_mle, mle_details = automated_scaffold_sizing(
            X,
            method='mle',
            ks=self.id_ks,
            backend=self.backend,
            metric=self.id_metric,
            n_jobs=self.n_jobs if self.n_jobs != -1 else None,
            min_components=int(self.id_min_components),
            max_components=int(max_cap),
            headroom=float(self.id_headroom),
            random_state=self.random_state,
            use_median=True,
            return_details=True,
        )
        self._id_details['mle'] = mle_details

        # choose by configured method
        if str(self.id_method).lower().strip() == 'fsa':
            k_sel = int(n_fsa)
        else:
            k_sel = int(n_mle)

        # finalize scaffold component counts (same for msDM/DM at this stage)
        k_sel = int(max(2, min(k_sel, max_cap)))
        self._scaffold_components_ms = k_sel
        self._scaffold_components_dm = k_sel

        # IMPORTANT: ensure eigendecomposition will compute at least this many
        # (keep user-requested n_eigs if it is larger)
        self.n_eigs = int(max(self.n_eigs, k_sel))

    # ---------------------------------------------------------------------
    # High-level fit (builds everything) – dual scaffold
    # ---------------------------------------------------------------------
    def fit(self, X=None, **kwargs):
        """
        Build base kNN, base kernel P(X). Compute both msDM and DM eigenbases (dual scaffold).
        Estimate intrinsic dimensionality (both FSA and MLE; store details). Size the scaffolds.
        Build kNN and refined graphs/operators on each scaffold (msDM, DM). Prepare spectral init.
        Compute standard projections (MAP and PaCMAP) on both scaffolds when available.

        Parameters
        ----------
        X : array-like or sparse matrix
            High-dimensional data (Z-score normalized is recommended).

        **kwargs : passed to `topo.base.ann.kNN()`

        Returns
        -------
        self
        """
        # Basic checks
        if self.base_kernel_version not in ['fuzzy', 'cknn', 'bw_adaptive', 'bw_adaptive_alpha_decaying',
                                            'bw_adaptive_nbr_expansion', 'bw_adaptive_alpha_decaying_nbr_expansion', 'gaussian']:
            raise ValueError("Invalid base_kernel_version.")
        if self.graph_kernel_version not in ['fuzzy', 'cknn', 'bw_adaptive', 'bw_adaptive_alpha_decaying',
                                             'bw_adaptive_nbr_expansion', 'bw_adaptive_alpha_decaying_nbr_expansion', 'gaussian']:
            raise ValueError("Invalid graph_kernel_version.")
        if (X is not None) and ((self.n_eigs - 1 > X.shape[0]) or (self.n_eigs - 1 > X.shape[1])):
            raise ValueError("n_eigs must be less than the number of samples and observations in X")
        if not isinstance(self.n_eigs, int):
            raise ValueError("n_eigs must be an integer")
        if not isinstance(self.base_knn, int):
            raise ValueError("base_knn must be an integer")
        if not isinstance(self.graph_knn, int):
            raise ValueError("graph_knn must be an integer")

        self._parse_backend()
        self._parse_random_state()

        # n_jobs
        if self.n_jobs == -1:
            try:
                from joblib import cpu_count
                self.n_jobs = cpu_count()
            except Exception:
                pass

        # verbosity toggles
        if self.verbosity >= 2:
            self.layout_verbose = True
            self.bases_graph_verbose = (self.verbosity == 3)
        else:
            self.layout_verbose = False

        # X or pre-fitted base kernel
        if X is None:
            if self.base_kernel is None:
                raise ValueError('X was not passed and no base_kernel provided.')
            if not isinstance(self.base_kernel, Kernel):
                raise ValueError('base_kernel must be a topo.tpgraph.Kernel instance.')
            if self.base_kernel.knn_ is None:
                raise ValueError('The specified base kernel has not been fitted!')
            self.n = self.base_kernel.knn_.shape[0]
            self.m = self.base_kernel.knn_.shape[1]
        else:
            if self.base_metric == 'precomputed':
                self.base_knn_graph = X.copy()
            self.n, self.m = X.shape[0], X.shape[1]

        # Base kNN
        if self.base_knn_graph is None:
            if self.verbosity >= 1:
                print('Computing neighborhood graph (X space)...')
            t0 = time.time()
            self.base_nbrs_class, self.base_knn_graph = kNN(
                X,
                n_neighbors=self.base_knn,
                metric=self.base_metric,
                n_jobs=self.n_jobs,
                backend=self.backend,
                return_instance=True,
                verbose=self.bases_graph_verbose,
                **kwargs
            )
            self.runtimes['kNN_X'] = time.time() - t0
            if self.verbosity >= 1:
                print(f' Base kNN computed in {self.runtimes["kNN_X"]:.3f} sec')

        # Base kernel -> P(X)
        if self.base_kernel_version in self.BaseKernelDict:
            self.base_kernel = self.BaseKernelDict[self.base_kernel_version]
        else:
            t0 = time.time()
            self.base_kernel, self.BaseKernelDict = self._compute_kernel_from_version_knn(
                self.base_knn_graph,
                self.base_knn,
                self.base_kernel_version,
                self.BaseKernelDict,
                suffix='',
                low_memory=self.low_memory,
                data_for_expansion=X,
                base=True
            )
            self.runtimes['Kernel_X'] = time.time() - t0
            if self.verbosity >= 1:
                print(f' Base kernel ({self.base_kernel_version}) fitted in {self.runtimes["Kernel_X"]:.3f} sec')

        if self.metric != 'precomputed':
            self._automated_sizing(X if X is not None else self.base_kernel.X)
            if self.verbosity >= 1:
                print(f"Automated sizing (pre-eigs) → target components: {self._scaffold_components_ms} "
                    f"(n_eigs set to {self.n_eigs})")

        # Compute eigenbases
        if self.verbosity >= 1:
            print('Computing eigenbasis (once on P); deriving DM/msDM embeddings in transform()...')

        dm_key = 'DM with ' + str(self.base_kernel_version)
        ms_key = 'msDM with ' + str(self.base_kernel_version)

        if dm_key not in self.EigenbasisDict:
            t0 = time.time()
            dm_eig = EigenDecomposition(
                n_components=self.n_eigs,
                method='DM',                 # <- fit on P; no powering; DM uses λ**t in transform()
                eigensolver=self.eigensolver,
                eigen_tol=self.eigen_tol,
                drop_first=True,
                weight=True,
                t=self.diff_t,
                random_state=self.random_state,
                verbose=self.bases_graph_verbose
            ).fit(self.base_kernel)
            self.EigenbasisDict[dm_key] = dm_eig
            self.runtimes[dm_key] = time.time() - t0
            if self.verbosity >= 1:
                print(f' DM/msDM eigenpairs computed in {self.runtimes[dm_key]:.3f} sec')
        else:
            dm_eig = self.EigenbasisDict[dm_key]

        # Clone dm_eig → ms_eig (share eigenpairs; different transform rule)
        if ms_key not in self.EigenbasisDict:
            ms_eig = copy.deepcopy(dm_eig)
            ms_eig.method = 'msDM'
            self.EigenbasisDict[ms_key] = ms_eig
        else:
            ms_eig = self.EigenbasisDict[ms_key]

        # Active eigenbasis (default msDM)
        self.current_eigenbasis = ms_key
        self.eigenbasis = self.EigenbasisDict[self.current_eigenbasis]

        # kNN in msZ
        if self.verbosity >= 1:
            print('Computing neighborhood graph (msZ space)...')
        t0 = time.time()
        ms_target = ms_eig.transform()[:, :self._scaffold_components_ms]
        self._knn_msZ = kNN(
            ms_target,
            n_neighbors=self.graph_knn,
            metric=self.graph_metric,
            n_jobs=self.n_jobs,
            backend=self.backend,
            return_instance=False,
            verbose=self.bases_graph_verbose,
            **kwargs
        )
        self.runtimes['kNN_msZ'] = time.time() - t0
        if self.verbosity >= 1:
            print(f' msZ kNN computed in {self.runtimes["kNN_msZ"]:.3f} sec')

        # kNN in Z (DM)
        if self.verbosity >= 1:
            print('Computing neighborhood graph (Z [DM] space)...')
        t0 = time.time()
        dm_target = dm_eig.transform()[:, :self._scaffold_components_dm]
        self._knn_Z = kNN(
            dm_target,
            n_neighbors=self.graph_knn,
            metric=self.graph_metric,
            n_jobs=self.n_jobs,
            backend=self.backend,
            return_instance=False,
            verbose=self.bases_graph_verbose,
            **kwargs
        )
        self.runtimes['kNN_Z'] = time.time() - t0
        if self.verbosity >= 1:
            print(f' Z (DM) kNN computed in {self.runtimes["kNN_Z"]:.3f} sec')

        # Graph kernel -> P(msZ)
        t0 = time.time()
        self._kernel_msZ, self.GraphKernelDict = self._compute_kernel_from_version_knn(
            self._knn_msZ,
            self.graph_knn,
            self.graph_kernel_version,
            self.GraphKernelDict,
            suffix=' from ' + ms_key,
            low_memory=self.low_memory,
            data_for_expansion=ms_eig.transform(),
            base=False
        )
        self.runtimes['Kernel_msZ'] = time.time() - t0
        if self.verbosity >= 1:
            print(f' Graph kernel (msZ/{self.graph_kernel_version}) fitted in {self.runtimes["Kernel_msZ"]:.3f} sec')

        # Graph kernel -> P(Z) (DM)
        t0 = time.time()
        self._kernel_Z, self.GraphKernelDict = self._compute_kernel_from_version_knn(
            self._knn_Z,
            self.graph_knn,
            self.graph_kernel_version,
            self.GraphKernelDict,
            suffix=' from ' + dm_key,
            low_memory=self.low_memory,
            data_for_expansion=dm_eig.transform(),
            base=False
        )
        self.runtimes['Kernel_Z'] = time.time() - t0
        if self.verbosity >= 1:
            print(f' Graph kernel (Z/DM/{self.graph_kernel_version}) fitted in {self.runtimes["Kernel_Z"]:.3f} sec')

        # Keep legacy single "current" graph kernel pointer (msDM by default)
        self.graph_kernel = self._kernel_msZ
        self.current_graphkernel = self.graph_kernel_version + ' from ' + ms_key

        # Spectral layout (2D) using msZ by default (for convenient init)
        _ = self.spectral_layout(graph=self._kernel_msZ.K, n_components=2)

        # Projections: MAP & PaCMAP for BOTH scaffolds when available
        for proj in ['MAP', 'PaCMAP']:
            # msDM
            try:
                self.project(projection_method=proj, multiscale=True)
            except Exception as e:
                warnings.warn(f"Projection '{proj}' on msZ failed or requires extra dependency: {e}", RuntimeWarning)
            # DM
            try:
                self.project(projection_method=proj, multiscale=False)
            except Exception as e:
                warnings.warn(f"Projection '{proj}' on Z (DM) failed or requires extra dependency: {e}", RuntimeWarning)

        return self

    # ---------------------------------------------------------------------
    # Public properties / getters (stable user API)
    # ---------------------------------------------------------------------
    @property
    def knn_X(self):
        """Initial kNN graph on X."""
        if self.base_knn_graph is None:
            raise AttributeError("knn_X is not available. Call .fit(X) first.")
        return self.base_knn_graph

    @property
    def P_of_X(self):
        """Diffusion operator on X (from the base kernel)."""
        if self.base_kernel is None:
            raise AttributeError("P_of_X is not available. Call .fit(X) first.")
        return self.base_kernel.P

    def spectral_scaffold(self, multiscale: bool = True):
        """
        Return spectral scaffold coordinates.

        Parameters
        ----------
        multiscale : bool (default True)
            If True, returns msDM coordinates; else returns DM (fixed time `diff_t`) coordinates.

        Returns
        -------
        np.ndarray, shape (n_samples, n_eigs)
        """
        if multiscale:
            key = 'msDM with ' + str(self.base_kernel_version)
        else:
            key = 'DM with ' + str(self.base_kernel_version)
        if key not in self.EigenbasisDict:
            raise AttributeError("Requested spectral scaffold not found. Ensure .fit() completed.")
        return self.EigenbasisDict[key].transform(X=None)

    @property
    def eigenvalues(self):
        """Eigenvalues of the active eigenbasis (msDM by default)."""
        if self.current_eigenbasis is None:
            raise AttributeError("Eigenvalues unavailable. Call .fit() first.")
        return self.EigenbasisDict[self.current_eigenbasis].eigenvalues

    # --- Dual scaffold accessors ---
    @property
    def knn_msZ(self):
        """kNN graph in the msDM scaffold space."""
        if self._knn_msZ is None:
            raise AttributeError("knn_msZ is not available. Call .fit(X) first.")
        return self._knn_msZ

    @property
    def knn_Z(self):
        """kNN graph in the DM scaffold space."""
        if self._knn_Z is None:
            raise AttributeError("knn_Z is not available. Call .fit(X) first.")
        return self._knn_Z

    @property
    def P_of_msZ(self):
        """Diffusion operator on the msDM scaffold (refined graph kernel)."""
        if self._kernel_msZ is None:
            raise AttributeError("P_of_msZ is not available. Call .fit(X) first.")
        return self._kernel_msZ.P

    @property
    def P_of_Z(self):
        """Diffusion operator on the DM scaffold (refined graph kernel)."""
        if self._kernel_Z is None:
            raise AttributeError("P_of_Z is not available. Call .fit(X) first.")
        return self._kernel_Z.P

    # --- Intrinsic-dimension getters ---
    def global_id_mle(self):
        """Return the global intrinsic-dimension estimate from MLE (median-of-locals), msDM scaffold."""
        det = self._id_details.get('mle', None)
        if det is None:
            raise AttributeError("MLE details not available. Call .fit(X) first.")
        return float(det.get('global_id_mle', np.nan))

    def global_id_fsa(self):
        """Return the global intrinsic-dimension proxy from FSA (quantile over local estimates), msDM scaffold."""
        det = self._id_details.get('fsa', None)
        if det is None:
            raise AttributeError("FSA details not available. Call .fit(X) first.")
        # quantile of robust_cell_id before headroom; this is the global proxy used by FSA
        return float(det.get('quantile_value', np.nan))

    def local_ids(self):
        """
        Return local intrinsic-dimension arrays as a dict:
            {'mle': <local_id_mle (msDM)>, 'fsa': <robust_cell_id (msDM)>}
        """
        out = {}
        det_mle = self._id_details.get('mle', None)
        det_fsa = self._id_details.get('fsa', None)
        if det_mle is not None:
            out['mle'] = det_mle.get('local_id_mle', None)
        if det_fsa is not None:
            out['fsa'] = det_fsa.get('robust_cell_id', None)
        if not out:
            raise AttributeError("Local ID details not available. Call .fit(X) first.")
        return out

    # --- Embedding getters (properties) ---
    @property
    def TopoMAP(self):
        """2D MAP layout computed on the DM refined graph (P_of_Z)."""
        key = 'MAP of ' + (self.graph_kernel_version + ' from DM with ' + str(self.base_kernel_version))
        if key not in self.ProjectionDict:
            raise AttributeError("MAP embedding not available. Call .fit(X) first.")
        return self.ProjectionDict[key]

    @property
    def msTopoMAP(self):
        """2D MAP layout computed on the msDM refined graph (P_of_msZ)."""
        key = 'MAP of ' + (self.graph_kernel_version + ' from msDM with ' + str(self.base_kernel_version))
        if key not in self.ProjectionDict:
            raise AttributeError("msMAP embedding not available. Call .fit(X) first.")
        return self.ProjectionDict[key]

    @property
    def TopoPaCMAP(self):
        """2D PaCMAP layout computed on the DM refined graph (P_of_Z)."""
        key = 'PaCMAP of ' + 'DM with ' + str(self.base_kernel_version)
        if key not in self.ProjectionDict:
            raise AttributeError("PaCMAP embedding not available. It may require `pacmap`. Call .fit(X) first.")
        return self.ProjectionDict[key]

    @property
    def msTopoPaCMAP(self):
        """2D PaCMAP layout computed on the msDM refined graph (P_of_msZ)."""
        key = 'PaCMAP of ' + 'msDM with ' + str(self.base_kernel_version)
        if key not in self.ProjectionDict:
            raise AttributeError("msPaCMAP embedding not available. It may require `pacmap`. Call .fit(X) first.")
        return self.ProjectionDict[key]

    # Keep Y() for backwards compatibility with string aliases
    def Y(self, key: str = 'msTopoMAP'):
        """
        Return a 2D embedding by stable alias (backwards compatible).

        Aliases
       -------
        'TopoMAP'      -> MAP on DM refined graph (P_of_Z)
        'msTopoMAP'   -> MAP on msDM refined graph (P_of_msZ)
        'TopoPaCMAP'   -> PaCMAP on DM refined graph
        'msTopoPaCMAP'-> PaCMAP on msDM refined graph

        Returns
        -------
        np.ndarray shape (n_samples, 2)
        """
        dm_tag = f"{self.graph_kernel_version} from DM with {self.base_kernel_version}"
        ms_tag = f"{self.graph_kernel_version} from msDM with {self.base_kernel_version}"
        mapping = {
            'TopoMAP':        f"MAP of {dm_tag}",
            'msTopoMAP':     f"MAP of {ms_tag}",
            'TopoPaCMAP':     f"PaCMAP of {dm_tag}",
            'msTopoPaCMAP':  f"PaCMAP of {ms_tag}",
        }
        if key in mapping and mapping[key] in self.ProjectionDict:
            return self.ProjectionDict[mapping[key]]
        if key in self.ProjectionDict:  # legacy passthrough
            return self.ProjectionDict[key]
        raise KeyError(f"Unknown or unavailable embedding alias '{key}'.")

    # ---------------------------------------------------------------------
    # Plot helpers, legacy APIs, benchmarking
    # ---------------------------------------------------------------------
    def eigenspectrum(self, eigenbasis_key=None, **kwargs):
        """Scree plot helper (calls topo.plot.decay_plot)."""
        if eigenbasis_key is not None:
            if isinstance(eigenbasis_key, str):
                if eigenbasis_key in self.EigenbasisDict.keys():
                    eigenbasis = self.EigenbasisDict[eigenbasis_key]
                else:
                    raise ValueError('Eigenbasis key not in TopOGraph.EigenbasisDict.')
        else:
            eigenbasis = self.eigenbasis
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError:
            return print('Error: Matplotlib not found!')
        from topo.plot import decay_plot
        return decay_plot(evals=eigenbasis.eigenvalues, title=eigenbasis_key, **kwargs)

    def plot_eigenspectrum(self, eigenbasis_key=None, **kwargs):
        """Alias for `eigenspectrum`."""
        return self.eigenspectrum(eigenbasis_key=eigenbasis_key, **kwargs)

    def list_eigenbases(self):
        """List keys in `EigenbasisDict` (legacy/benchmarking)."""
        return list(self.EigenbasisDict.keys())

    def transform(self, X=None, **kwargs):
        """
        DEPRECATED: all computations now happen during `.fit()`.
        Returns the msDM graph kernel matrix for backward compatibility.
        """
        warnings.warn(
            "TopOGraph.transform is deprecated; all computations occur in .fit(). Returning msDM graph kernel K.",
            DeprecationWarning
        )
        if self._kernel_msZ is None:
            raise ValueError('No graph kernel computed. Call .fit() first.')
        return self._kernel_msZ.K

    def spectral_layout(self, graph=None, n_components=2):
        """
        Multicomponent spectral layout of a (precomputed) graph kernel.
        Stores result in `SpecLayout` and returns it (used for layout init).
        """
        if graph is None:
            if self._kernel_msZ is None:
                raise ValueError('No graph kernel computed. Call .fit() first.')
            graph = self._kernel_msZ.K
        t0 = time.time()
        try:
            spt_layout = spectral_layout(
                graph, n_components, self.random_state,
                laplacian_type=self.laplacian_type, eigen_tol=self.eigen_tol, return_evals=False
            )
            expansion = 10.0 / np.abs(spt_layout).max()
            spt_layout = (spt_layout * expansion).astype(np.float32) + self.random_state.normal(
                scale=0.0001, size=[graph.shape[0], n_components]
            ).astype(np.float32)
        except Exception:
            spt_layout = EigenDecomposition(n_components=n_components).fit_transform(graph)
        self.runtimes['Spectral'] = time.time() - t0
        self.SpecLayout = spt_layout
        gc.collect
        return spt_layout

    def project(
        self,
        n_components=2,
        init=None,
        projection_method=None,
        landmarks=None,
        landmark_method='kmeans',
        n_neighbors=None,
        num_iters=500,
        multiscale=True,
        # ---- NEW: checkpointing passthrough (MAP) ----
        save_every=None,                 # int or None. If int>0, store Y every `save_every` epochs
        save_limit=None,                 # optional cap on number of snapshots kept in-memory
        save_callback=None,              # optional callable(epoch:int, Y:np.ndarray) -> None
        include_init_snapshot=True,      # store epoch=0 (post-init) snapshot
        **kwargs
    ):
        """
        Compute a 2D projection and store it in `ProjectionDict`.

        Parameters
        ----------
        multiscale : bool (internal, default True)
            If True, use msDM refined graph; else use DM refined graph.

        save_every, save_limit, save_callback, include_init_snapshot :
            Passed through to MAP checkpointing. Ignored by other methods.

        Notes
        -----
        For graph-based DR methods we pass precomputed affinities from the chosen refined graph:
        {MAP, UMAP, Isomap, (Iso/Isomorphic)MDE, PaCMAP, NCVis, TriMAP, t-SNE}.
        """
        if n_neighbors is None:
            n_neighbors = self.graph_knn
        elif not isinstance(n_neighbors, int):
            raise ValueError('n_neighbors must be an integer')

        if projection_method is None:
            projection_method = self.projection_method

        # choose which refined graph to use
        if multiscale:
            input_mat = self.P_of_msZ
        else:
            input_mat = self.P_of_Z

        # Precomputed affinity path for graph-based DR methods
        if projection_method in ['MAP',  'IsomorphicMDE', 'IsometricMDE', 'Isomap']:
            metric = 'precomputed'
            input_mat = self.P_of_msZ if multiscale else self.P_of_Z
            tag = 'msDM' if multiscale else 'DM'
            key = self.graph_kernel_version + ' from ' + tag + ' with ' + str(self.base_kernel_version)
        else:
            metric = self.graph_metric
            # use corresponding scaffold coordinates
            tag = 'msDM' if multiscale else 'DM'
            eig_key = tag + ' with ' + str(self.base_kernel_version)
            input_mat = self.EigenbasisDict[eig_key].transform(X=None)
            key = eig_key

        # init
        if init is not None:
            if isinstance(init, np.ndarray):
                if np.shape(init)[1] != n_components:
                    raise ValueError('The specified initialization has the wrong number of dimensions.')
                init_Y = init
            elif isinstance(init, str):
                if init in self.ProjectionDict.keys():
                    init_Y = self.ProjectionDict[init]
                else:
                    raise ValueError('No projection found with the name ' + init + '.')
        else:
            if self.SpecLayout is not None:
                if np.shape(self.SpecLayout)[1] != n_components:
                    self.SpecLayout = self.spectral_layout(n_components=n_components)
            else:
                self.SpecLayout = self.spectral_layout(n_components=n_components)
            init_Y = self.SpecLayout

        projection_key = projection_method + ' of ' + key
        t0 = time.time()

        # 1) Build the estimator WITHOUT save_* kwargs in the constructor
        proj = Projector(
            n_components=n_components,
            projection_method=projection_method,
            metric=metric,
            n_neighbors=self.graph_knn,
            n_jobs=self.n_jobs,
            landmarks=landmarks,
            landmark_method=landmark_method,
            num_iters=num_iters,
            init=init_Y,
            nbrs_backend=self.backend,
            keep_estimator=False,
            random_state=self.random_state,
            verbose=self.layout_verbose,
            save_every=save_every,
            save_limit=save_limit,
            save_callback=save_callback,
            include_init_snapshot=include_init_snapshot,
        )

        # 2) Pass checkpointing kwargs to fit_transform (this is what MAP/fuzzy_embedding reads)
        result = proj.fit_transform(
            input_mat,
            **kwargs
        )

        # Unpack (Y, aux) if available
        if isinstance(result, tuple) and len(result) == 2:
            Y, Y_aux = result
        else:
            Y, Y_aux = result, None

        self.runtimes[projection_key] = time.time() - t0
        if self.verbosity >= 1:
            print(f' Computed {projection_method} ({ "msZ" if multiscale else "Z/DM" }) in {self.runtimes[projection_key]:.3f} sec')

        self.ProjectionDict[projection_key] = Y

        # Record snapshots if present
        if projection_method == "MAP" and Y_aux and isinstance(Y_aux, dict):
            checkpoints = Y_aux.get("checkpoints", None)
            if checkpoints:
                if multiscale:
                    self.msTopoMAP_snapshots = checkpoints
                else:
                    self.TopoMAP_snapshots = checkpoints

        return Y



    def visualize_optimization(
        self,
        num_iters: int = 600,
        save_every: int = 10,
        dpi: int = 120,
        color=None,
        *,
        multiscale: bool = True,
        filename: str = None,
        point_size: float = 3.0,
        fps: int = 20,
        include_init_snapshot: bool = True,
    ):
        """
        Produce an animated GIF of MAP training snapshots. If snapshots do not
        exist yet, they are generated via a MAP projection call with checkpointing.

        Parameters
        ----------
        num_iters : int, default 600
            Number of optimization epochs for MAP when generating snapshots.
        save_every : int, default 10
            Snapshot cadence in epochs. If <= 0, no snapshots will be made.
        dpi : int, default 120
            Figure DPI for frame rendering.
        color : None, array-like, or single color
            Per-point coloring. Options:
            * None -> uniform semi-dark gray.
            * (n,) numeric -> mapped through a colormap (viridis).
            * (n,3|4) array or list of color strings -> used directly per point.
            * single color string/tuple -> uniform.
        multiscale : bool or None
            If True, use msDM scaffold (snapshots in `msTopoMAP_snapshots`).
            If False, use DM scaffold (snapshots in `TopoMAP_snapshots`).
            If None, prefer msDM if available; else fallback to DM.
        filename : str or None
            Path for the GIF. If None, an automatic name is chosen.
        point_size : float, default 3.0
            Marker size in the scatter plot.
        fps : int, default 20
            Frames-per-second for the GIF.
        include_init_snapshot : bool, default True
            If we need to run MAP to produce snapshots, include epoch=0.

        Returns
        -------
        str
            The path to the generated GIF.
        """
        import time
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        try:
            import imageio.v2 as imageio
        except ImportError:
            raise ImportError("imageio is required to write GIFs. Please install it via `pip install imageio`.")

        # 0) Decide scaffold/snapshots source
        if multiscale is None:
            # Prefer msDM if present
            if hasattr(self, "msTopoMAP_snapshots") and self.msTopoMAP_snapshots:
                multiscale = True
            elif hasattr(self, "TopoMAP_snapshots") and self.TopoMAP_snapshots:
                multiscale = False
            else:
                multiscale = True  # default preference

        snap_attr = "msTopoMAP_snapshots" if multiscale else "TopoMAP_snapshots"
        snapshots = getattr(self, snap_attr, None)

        # 1) Ensure snapshots exist; otherwise, generate via MAP checkpointing
        if not snapshots:
            _ = self.project(
                projection_method="MAP",
                num_iters=int(num_iters),
                save_every=int(save_every),
                include_init_snapshot=bool(include_init_snapshot),
                multiscale=bool(multiscale),   # <<< FIX: use the correct kw for scaffold choice
            )
            snapshots = getattr(self, snap_attr, None)

        if not snapshots:
            raise RuntimeError(
                "No snapshots available. Ensure `save_every>0` and that MAP "
                "projection ran successfully."
            )

        # 2) Colors
        n = snapshots[-1]["embedding"].shape[0]

        def _to_rgba_array(c, n):
            # Accepts: None, scalar color, (n,) numeric, (n,3/4), list of color strings
            if c is None:
                return np.tile(np.array([0.15, 0.15, 0.15, 0.85])[None, :], (n, 1))
            if isinstance(c, (str, tuple)):
                rgba = np.array(mcolors.to_rgba(c), float)
                return np.tile(rgba[None, :], (n, 1))
            c = np.asarray(c)
            if c.ndim == 1:
                if c.shape[0] == n and np.issubdtype(c.dtype, np.number):
                    # numeric vector -> colormap
                    cmap = plt.get_cmap("viridis")
                    # normalize robustly
                    vmin, vmax = np.nanmin(c), np.nanmax(c)
                    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                        vmin, vmax = 0.0, 1.0
                    t = (c - vmin) / (vmax - vmin + 1e-12)
                    return cmap(np.clip(t, 0, 1))
                elif c.shape[0] == n:
                    # list of color strings -> map to rgba
                    return np.array([mcolors.to_rgba(ci) for ci in c], float)
                else:
                    # Single channel? Fallback to uniform gray
                    return np.tile(np.array([0.15, 0.15, 0.15, 0.85])[None, :], (n, 1))
            elif c.ndim == 2 and c.shape[0] == n and c.shape[1] in (3, 4):
                if c.shape[1] == 3:
                    # add alpha=1
                    return np.concatenate([c, np.ones((n, 1))], axis=1)
                return c.astype(float)
            else:
                # Unknown -> uniform gray
                return np.tile(np.array([0.15, 0.15, 0.15, 0.85])[None, :], (n, 1))

        point_colors = _to_rgba_array(color, n)

        # 3) Axis limits from the final snapshot
        Y_final = snapshots[-1]["embedding"]
        x_min, x_max = np.min(Y_final[:, 0]), np.max(Y_final[:, 0])
        y_min, y_max = np.min(Y_final[:, 1]), np.max(Y_final[:, 1])
        pad_x = 0.05 * (x_max - x_min + 1e-9)
        pad_y = 0.05 * (y_max - y_min + 1e-9)
        xlim = (x_min - pad_x, x_max + pad_x)
        ylim = (y_min - pad_y, y_max + pad_y)

        # 4) Render frames
        frames = []
        fig_w, fig_h = (6, 5)
        for snap in snapshots:
            Y = snap["embedding"]
            epoch = snap["epoch"]

            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

            # scatter
            ax.scatter(Y[:, 0], Y[:, 1], s=point_size, c=point_colors, linewidths=0)

            # axis limits + labels
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("TopoMAP_1")
            ax.set_ylabel("TopoMAP_2")

            # remove tick labels (optional aesthetic)
            ax.set_xticks([]); ax.set_yticks([])

            # title
            tag = "msTopoMAP" if multiscale else "TopoMAP"
            ax.set_title(f"{tag} training — epoch {epoch}")

            # adjust layout so scatter fills most of the figure
            fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.92)

            # render to numpy frame
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
            frames.append(frame)
            plt.close(fig)


        # 5) Write GIF
        if filename is None:
            tag = "msTopoMAP" if multiscale else "TopoMAP"
            filename = f"{tag}_training_{int(time.time())}.gif"
        imageio.mimsave(filename, frames, duration=1.0 / max(1, int(fps)))

        if self.verbosity >= 1:
            print(f"Wrote {filename} with {len(frames)} frames.")

        return filename




    def run_models(self, X,
                   kernels=['fuzzy', 'cknn', 'bw_adaptive'],
                   eigenmap_methods=['DM', 'LE', 'top'],
                   projections=['Isomap', 'MAP']):
        """
        Legacy power function that runs multiple models for benchmarking.
        Preserved for backward compatibility.
        """
        for kernel in kernels:
            self.base_kernel_version = kernel
            for eig_method in eigenmap_methods:
                # kept for BC, but dual scaffold is always computed anyway
                self._eigenmap_method = eig_method
                self.fit(X)
                gc.collect()
                for kernel in kernels:
                    self.graph_kernel_version = kernel
                    _ = self.transform(X)  # BC; returns current (msZ) K
                    gc.collect()
                    for projection in projections:
                        # compute on msZ and Z/DM
                        self.project(projection_method=projection, multiscale=True)
                        self.project(projection_method=projection, multiscale=False)
                        gc.collect()

    # ---------------------------------------------------------------------
    # I/O (pickle) – in-class helper kept for BC + module-level helpers below
    # ---------------------------------------------------------------------
    def write_pkl(self, filename='topograph.pkl', remove_base_class=True):
        """
        Save the TopOGraph object to a pickle file (legacy helper).
        """
        try:
            import pickle
        except ImportError:
            return print('Pickle is needed for saving the TopOGraph. Please install it with `pip3 install pickle`')

        if self.base_nbrs_class is not None:
            if remove_base_class:
                self.base_nbrs_class = None
            else:
                raise ValueError('TopOGraph cannot be pickled with the NMSlib base class.')

        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        gc.collect()
        return print('TopOGraph saved at ' + filename)

    # ---------------------------------------------------------------------
    # Kernel builder (internal)
    # ---------------------------------------------------------------------
    def _compute_kernel_from_version_knn(self, knn, n_neighbors, kernel_version, results_dict,
                                         prefix='', suffix='', low_memory=False, base=True, data_for_expansion=None):
        import gc as _gc
        _gc.collect()
        kernel_key = kernel_version
        if prefix is not None:
            kernel_key = prefix + kernel_key
        if suffix is not None:
            kernel_key = kernel_key + suffix
        if kernel_key in results_dict.keys():
            kernel = results_dict[kernel_key]
            return kernel, results_dict
        else:
            # Note: anisotropy fixed to 1.0 and semi_aniso fixed to False (kwargs removed)
            if kernel_version == 'cknn':
                kernel = Kernel(metric="precomputed",
                                n_neighbors=n_neighbors,
                                fuzzy=False,
                                cknn=True,
                                pairwise=False,
                                sigma=None,
                                adaptive_bw=True,
                                expand_nbr_search=False,
                                alpha_decaying=False,
                                backend=self.backend,
                                n_jobs=self.n_jobs,
                                laplacian_type=self.laplacian_type,
                                semi_aniso=False,
                                anisotropy=1.0,
                                cache_input=False,
                                verbose=self.bases_graph_verbose,
                                random_state=self.random_state).fit(knn)
                _gc.collect()
                results_dict[kernel_key] = kernel

            elif kernel_version == 'fuzzy':
                kernel = Kernel(metric="precomputed",
                                n_neighbors=n_neighbors,
                                fuzzy=True,
                                cknn=False,
                                pairwise=False,
                                sigma=None,
                                adaptive_bw=True,
                                expand_nbr_search=False,
                                alpha_decaying=False,
                                backend=self.backend,
                                n_jobs=self.n_jobs,
                                laplacian_type=self.laplacian_type,
                                semi_aniso=False,
                                anisotropy=1.0,
                                cache_input=False,
                                verbose=self.bases_graph_verbose,
                                random_state=self.random_state).fit(knn)
                _gc.collect()
                results_dict[kernel_key] = kernel

            elif kernel_version == 'bw_adaptive':
                kernel = Kernel(metric="precomputed",
                                n_neighbors=n_neighbors,
                                fuzzy=False,
                                cknn=False,
                                pairwise=False,
                                sigma=None,
                                adaptive_bw=True,
                                expand_nbr_search=False,
                                alpha_decaying=False,
                                backend=self.backend,
                                n_jobs=self.n_jobs,
                                laplacian_type=self.laplacian_type,
                                semi_aniso=False,
                                anisotropy=1.0,
                                cache_input=False,
                                verbose=self.bases_graph_verbose,
                                random_state=self.random_state).fit(knn)
                _gc.collect()
                results_dict[kernel_key] = kernel

            elif kernel_version == 'bw_adaptive_alpha_decaying':
                kernel = Kernel(metric="precomputed",
                                n_neighbors=n_neighbors,
                                fuzzy=False,
                                cknn=False,
                                pairwise=False,
                                sigma=None,
                                adaptive_bw=True,
                                expand_nbr_search=False,
                                alpha_decaying=True,
                                backend=self.backend,
                                n_jobs=self.n_jobs,
                                laplacian_type=self.laplacian_type,
                                semi_aniso=False,
                                anisotropy=1.0,
                                cache_input=False,
                                verbose=self.bases_graph_verbose,
                                random_state=self.random_state).fit(knn)
                _gc.collect()
                results_dict[kernel_key] = kernel

            elif kernel_version == 'bw_adaptive_nbr_expansion':
                if data_for_expansion is None:
                    raise ValueError('data_for_expansion is None. Provide data for neighborhood expansion when using `bw_adaptive_nbr_expansion`.')
                use_metric = self.base_metric if base else self.graph_metric
                kernel = Kernel(metric=use_metric,
                                n_neighbors=n_neighbors,
                                fuzzy=False,
                                cknn=False,
                                pairwise=False,
                                sigma=None,
                                adaptive_bw=True,
                                expand_nbr_search=True,
                                alpha_decaying=False,
                                backend=self.backend,
                                n_jobs=self.n_jobs,
                                laplacian_type=self.laplacian_type,
                                semi_aniso=False,
                                anisotropy=1.0,
                                cache_input=False,
                                verbose=self.bases_graph_verbose,
                                random_state=self.random_state).fit(knn)
                _gc.collect()
                results_dict[kernel_key] = kernel

            elif kernel_version == 'bw_adaptive_alpha_decaying_nbr_expansion':
                if data_for_expansion is None:
                    raise ValueError('data_for_expansion is None. Provide data for neighborhood expansion when using `bw_adaptive_alpha_decaying_nbr_expansion`.')
                use_metric = self.base_metric if base else self.graph_metric
                kernel = Kernel(metric=use_metric,
                                n_neighbors=n_neighbors,
                                fuzzy=False,
                                cknn=False,
                                pairwise=False,
                                sigma=None,
                                adaptive_bw=True,
                                expand_nbr_search=True,
                                alpha_decaying=True,
                                backend=self.backend,
                                n_jobs=self.n_jobs,
                                laplacian_type=self.laplacian_type,
                                semi_aniso=False,
                                anisotropy=1.0,
                                cache_input=False,
                                verbose=self.bases_graph_verbose,
                                random_state=self.random_state).fit(knn)
                _gc.collect()
                results_dict[kernel_key] = kernel

            elif kernel_version == 'gaussian':
                kernel = Kernel(metric="precomputed",
                                n_neighbors=n_neighbors,
                                fuzzy=False,
                                cknn=False,
                                pairwise=False,
                                sigma=self.sigma,
                                adaptive_bw=False,
                                expand_nbr_search=False,
                                alpha_decaying=False,
                                backend=self.backend,
                                n_jobs=self.n_jobs,
                                laplacian_type=self.laplacian_type,
                                semi_aniso=False,
                                anisotropy=1.0,
                                cache_input=False,
                                verbose=self.bases_graph_verbose,
                                random_state=self.random_state).fit(knn)
                _gc.collect()
                results_dict[kernel_key] = kernel

            if low_memory:
                del results_dict[kernel_key]
                _gc.collect()

        return kernel, results_dict



    # ---------------------------------
    # Intrinsic dimension access helpers
    # ---------------------------------
    def local_ids(self):
        """
        Return the local intrinsic dimensionalities computed during automated scaffold sizing,
        when available. For 'mle', this corresponds to local MLEs; for 'fsa', per-k locals.

        Returns
        -------
        dict or np.ndarray or None
            - For MLE: np.ndarray of shape (n,), local id estimates.
            - For FSA: dict {k -> np.ndarray (n,)} of local id per k, plus
                'robust_cell_id' under self._id_details (see global getters).
            - None if details are not available.
        """
        det = getattr(self, "_id_details", None)
        if det is None:
            return None
        if det.get("method") == "mle":
            return det.get("local_id_mle", None)
        elif det.get("method") == "fsa":
            return det.get("per_k_local_id", None)
        return None

    def global_id_mle(self):
        """
        Return the global MLE intrinsic dimensionality if computed.

        Returns
        -------
        float or None
        """
        det = getattr(self, "_id_details", None)
        if det is None:
            return None
        if det.get("method") == "mle":
            return det.get("global_id_mle", None)
        # If FSA was run, expose quantile-based robust estimate (with headroom) as a fallback
        if det.get("method") == "fsa":
            return float(det.get("selected_n_components", np.nan))
        return None

    def global_id_fsa(self):
        """
        Return the quantile-based FSA intrinsic dimensionality if computed.

        Returns
        -------
        float or None
        """
        det = getattr(self, "_id_details", None)
        if det is None:
            return None
        if det.get("method") == "fsa":
            return float(det.get("selected_n_components", np.nan))
        # If MLE was run, expose its selected components (with headroom) as a fallback
        if det.get("method") == "mle":
            return float(det.get("selected_n_components", np.nan))
        return None

    # ===============================
    # Analysis helpers (no Scanpy I/O)
    # ===============================

    # ---- Spectral selectivity suite ----
    # --- spectral_selectivity ---
    def spectral_selectivity(
        self,
        Z=None,
        evals=None,
        multiscale: bool = True,
        use_scaffold_components: bool = True,
        weight_mode: str = "lambda_over_one_minus_lambda",
        standardize: bool = True,
        k_neighbors: int = 30,
        metric: str = "euclidean",
        smooth_P: Optional[str] = None,   # {'X','Z','msZ'} or None
        smooth_t: int = 0,
        out_prefix: str = "spectral",
        return_dict: bool = True,
        random_state: Optional[int] = None,
    ):
        
        """
        Compute per-sample spectral selectivity diagnostics:
            - EAS (axis selectivity via spectral entropy)
            - RayScore (sigmoid of radial z-score * EAS)
            - LAC (local axial coherence = EVR1 of local PCA)
            - axis (argmax energy axis), axis_sign, radius (||Z||2)

        Parameters
        ----------
        Z : np.ndarray, optional
            Scaffold coordinates (n, m). If None, uses `spectral_scaffold(multiscale)`.
        evals : np.ndarray, optional
            Eigenvalues for weighting. If None, uses the corresponding eigenvalues from the chosen scaffold.
        multiscale : bool (default True)
            Choose msDM (True) or DM (False) scaffold if Z is None.
        use_scaffold_components : bool (default True)
            If True, slice Z to the number chosen by automated scaffold sizing.
        weight_mode : {'lambda_over_one_minus_lambda','lambda', 'none'}
            Weighting of axes in EAS.
        standardize : bool (default True)
            Column-standardize Z before metrics.
        k_neighbors : int (default 30)
            Neighborhood size for local metrics (radiality, LAC).
        metric : str (default 'euclidean')
            Metric for neighborhood search.
        smooth_P : {'X','Z','msZ', None} (default None)
            Optional diffusion smoothing of scalar fields with P^t.
        smooth_t : int (default 0)
            Number of diffusion steps for smoothing (if smooth_P is not None).
        out_prefix : str (default 'spectral')
            Keys used when storing to `self.LocalScoresDict`.
        return_dict : bool (default True)
            If True, return a dictionary with results.
        random_state : int or None
            RNG for any randomized steps.

        Returns
        -------
        dict or None
            {'EAS','RayScore','LAC','axis','axis_sign','radius'} arrays if return_dict=True.
        """
        import numpy as _np
        from sklearn.neighbors import NearestNeighbors as _NN

        def _std_cols(A, eps=1e-12):
            A = _np.asarray(A, float)
            A = A - _np.nanmean(A, axis=0, keepdims=True)
            sd = _np.nanstd(A, axis=0, keepdims=True)
            return A / (sd + eps)

        def _weights(_evals=None, m=None, mode="lambda_over_one_minus_lambda", eps=1e-12):
            if _evals is None:
                if m is None: return _np.ones(1, float)
                return _np.ones(m, float)
            ev = _np.asarray(_evals, float)
            if mode == "lambda_over_one_minus_lambda":
                return ev / (1.0 - ev + eps)
            elif mode == "lambda":
                return ev
            else:
                return _np.ones_like(ev)

        def _EAS(Zs, w=None, eps=1e-12):
            n, m = Zs.shape
            if w is None: w = _np.ones(m, float)
            E = (Zs**2) * w[None, :]
            S = _np.sum(E, axis=1, keepdims=True) + eps
            P = E / S
            H = -_np.sum(P * _np.log(P + eps), axis=1)
            Hmax = _np.log(m)
            EAS = 1.0 - (H / (Hmax + eps))
            kstar = _np.argmax(E, axis=1)
            sign_kstar = _np.sign(Zs[_np.arange(n), kstar])
            return EAS, kstar, sign_kstar, _np.sqrt((_np.square(Zs)).sum(1))

        def _radiality(Zs, k=30, metric="euclidean", eps=1e-12):
            n = Zs.shape[0]
            nn = _NN(n_neighbors=min(k, n-1), metric=metric).fit(Zs)
            _, idx = nn.kneighbors(Zs, return_distance=True)
            nbr = idx[:, 1:] if idx.shape[1] > 1 else idx
            r = _np.linalg.norm(Zs, axis=1)
            r_med = _np.median(r[nbr], axis=1)
            q75 = _np.percentile(r[nbr], 75, axis=1)
            q25 = _np.percentile(r[nbr], 25, axis=1)
            iqr = q75 - q25
            z = (r - r_med) / (iqr + eps)
            return z, r

        def _LAC(Zs, k=30, metric="euclidean", eps=1e-12):
            from numpy.linalg import svd as _svd
            n = Zs.shape[0]
            nn = _NN(n_neighbors=min(k, n-1), metric=metric).fit(Zs)
            _, idx = nn.kneighbors(Zs, return_distance=True)
            nbr = idx[:, 1:] if idx.shape[1] > 1 else idx
            out = _np.zeros(n, float)
            for i in range(n):
                Znb = Zs[nbr[i]]
                Znb = Znb - Znb.mean(0, keepdims=True)
                _, s, _ = _svd(Znb, full_matrices=False)
                num = (s[0]**2)
                den = (s**2).sum() + eps
                out[i] = num / den
            return out

        # Prepare Z and evals
        if Z is None:
            Z = self.spectral_scaffold(multiscale=multiscale)
        if use_scaffold_components and getattr(self, "_scaffold_components", None) is not None:
            Z = Z[:, :int(self._scaffold_components)]
        if evals is None:
            key = ('msDM' if multiscale else 'DM') + ' with ' + str(self.base_kernel_version)
            ev = self.EigenbasisDict[key].eigenvalues
            # eigenvalues include lambda_0; match columns (drop first if eigenvectors drop-first)
            evals = ev[1:Z.shape[1]+1] if ev.shape[0] >= Z.shape[1] + 1 else ev[:Z.shape[1]]

        Zs = _std_cols(Z) if standardize else _np.asarray(Z, float)
        w = _weights(evals, m=Z.shape[1], mode=weight_mode)

        # Metrics
        EAS, kstar, sign_k, r = _EAS(Zs, w=w)
        z_rad, _ = _radiality(Zs, k=k_neighbors, metric=metric)
        LAC = _LAC(Zs, k=k_neighbors, metric=metric)
        RayScore = (1.0 / (1.0 + _np.exp(-z_rad))) * EAS

        # Optional diffusion smoothing of scalar fields
        if smooth_P is not None and int(smooth_t) > 0:
            P = self._select_P_operator(which=smooth_P)
            def _smooth(v):
                u = _np.asarray(v, float).copy()
                for _ in range(int(smooth_t)):
                    u = P @ u
                return _np.asarray(u).ravel()
            EAS = _smooth(EAS)
            RayScore = _smooth(RayScore)
            LAC = _smooth(LAC)

        # Store to LocalScoresDict
        self.LocalScoresDict[f'{out_prefix}_EAS'] = EAS
        self.LocalScoresDict[f'{out_prefix}_RayScore'] = RayScore
        self.LocalScoresDict[f'{out_prefix}_LAC'] = LAC
        self.LocalScoresDict[f'{out_prefix}_axis'] = kstar.astype(int)
        self.LocalScoresDict[f'{out_prefix}_axis_sign'] = (sign_k > 0).astype(int)
        self.LocalScoresDict[f'{out_prefix}_radius'] = r

        if return_dict:
            return dict(EAS=EAS, RayScore=RayScore, LAC=LAC, axis=kstar, axis_sign=(sign_k > 0).astype(int), radius=r)

    # ---- Graph diffusion filtering ----
    def _select_P_operator(self, which: str = 'msZ'):
        """
        Internal: choose a diffusion operator.

        Parameters
        ----------
        which : {'X','Z','msZ'}
            - 'X'   : P_of_X (input space)
            - 'Z'   : P_of_Z (DM-based refined operator)
            - 'msZ' : P_of_msZ (msDM-based refined operator)
        """
        which = str(which).lower()
        if which == 'x':
            return self.P_of_X
        elif which == 'z':
            # DM-based refined graph (if available), else fallback to current
            Pz = getattr(self, "P_of_Z", None)
            if Pz is None:
                raise ValueError("P_of_Z not available. Ensure DM-based refined graph has been computed.")
            return Pz
        elif which == 'msz':
            Pms = getattr(self, "P_of_msZ", None)
            if Pms is None:
                # for backward compat, msZ is the default refined graph in current design
                return self.P_of_Z
            return Pms
        else:
            raise ValueError("`which` must be one of {'X','Z','msZ'}.")

    def filter_signal(self, signal, t: int = 8, which: str = 'msZ'):
        """
        Diffusion-filter a 1D signal over the chosen graph operator.

        Parameters
        ----------
        signal : array-like, shape (n,)
            Scalar per-sample values to be smoothed.
        t : int (default 8)
            Number of diffusion steps (applications of P).
        which : {'X','Z','msZ'} (default 'msZ')
            Which operator to use (see `_select_P_operator`).

        Returns
        -------
        np.ndarray, shape (n,)
            Filtered signal.
        """
        import numpy as _np
        P = self._select_P_operator(which)
        y = _np.asarray(signal, float).copy().ravel()
        for _ in range(int(t)):
            y = P @ y
        return _np.asarray(y).ravel()

    # ---- Pseudotime estimation from spectral scaffold ----
    def pseudotime(
        self,
        root: Optional[int] = None,
        labels: Optional[np.ndarray] = None,
        label_value: Optional[object] = None,
        multiscale: bool = True,
        k: int = 64,
        weight_mode: str = "lambda_over_one_minus_lambda",
        null_n_seeds: int = 0,
        random_state: Optional[int] = 42,
        return_null: bool = True,
    ):

        """
        Compute a diffusion-based pseudotime on the chosen spectral scaffold.

        Parameters
        ----------
        root : int or None
            Index of the root cell. If None and `labels`/`label_value` is given,
            a random root from that subset is chosen; otherwise a global random root is used.
        labels : array-like or None
            Per-sample labels for root selection (optional).
        label_value : any or None
            Choose root from samples where labels == label_value.
        multiscale : bool (default True)
            Use msDM scaffold if True, else DM.
        k : int (default 64)
            Number of spectral coordinates to use (after dropping the trivial one).
        weight_mode : {'lambda_over_one_minus_lambda','lambda','none'}
            Axis weighting for MSDD coordinates.
        null_n_seeds : int (default 0)
            If >0, compute null mean/std over randomized roots.
        random_state : int or None
            RNG seed.
        return_null : bool (default True)
            If True, return (mean,std) of null as well.

        Returns
        -------
        dict
            {'pseudotime','root', 'null_mean','null_std'} (the latter two present if requested)
        """
        import numpy as _np
        rng = _np.random.default_rng(random_state)

        # Scaffold + eigenvalues
        Z = self.spectral_scaffold(multiscale=multiscale)
        ev_key = ('msDM' if multiscale else 'DM') + ' with ' + str(self.base_kernel_version)
        evals = self.EigenbasisDict[ev_key].eigenvalues
        # Build MSDD coordinates Psi = phi_1..k * w
        k_eff = int(min(k, Z.shape[1]-1))
        lam = _np.asarray(evals[1:k_eff+1], float)
        if weight_mode == "lambda_over_one_minus_lambda":
            w = (lam / (1.0 - lam))[None, :]
        elif weight_mode == "lambda":
            w = lam[None, :]
        else:
            w = _np.ones((1, k_eff), float)
        Psi = Z[:, 0:k_eff] * w

        # Root selection
        n = Psi.shape[0]
        if root is None:
            if labels is not None and label_value is not None:
                lab = _np.asarray(labels)
                pool = _np.where(lab == label_value)[0]
                if pool.size > 0:
                    root = int(rng.choice(pool))
                else:
                    root = int(rng.integers(0, n))
            else:
                root = int(rng.integers(0, n))

        d2 = _np.sum((Psi - Psi[root, :])**2, axis=1)
        pt = (d2 - d2.min()) / (d2.max() - d2.min() + 1e-12)

        out = {'pseudotime': pt, 'root': int(root)}
        if int(null_n_seeds) > 0 and return_null:
            pts = _np.empty((int(null_n_seeds), n), dtype=float)
            for s in range(int(null_n_seeds)):
                r0 = int(rng.integers(0, n))
                d2 = _np.sum((Psi - Psi[r0, :])**2, axis=1)
                v = (d2 - d2.min()) / (d2.max() - d2.min() + 1e-12)
                pts[s] = v
            out['null_mean'] = pts.mean(axis=0)
            out['null_std'] = pts.std(axis=0)
        return out

    # ---- Matrix imputation via diffusion ----
    def impute(
        self,
        X,
        t: int = 8,
        which: str = 'msZ',
        output: str = 'auto',   # {'auto','sparse','dense'}
        dtype=np.float64,
    ):
        ...

        """
        Diffusion-based imputation using P^t.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Data matrix to be imputed.
        t : int (default 8)
            Number of diffusion steps.
        which : {'X','Z','msZ'} (default 'msZ')
            Which diffusion operator to use.
        output : {'auto','sparse','dense'} (default 'auto')
            Output format. 'auto' preserves input sparsity.
        dtype : numpy dtype (default float64)
            Computation dtype.

        Returns
        -------
        scipy.sparse.csr_matrix or np.ndarray
            Imputed matrix.
        """
        import numpy as _np
        import scipy.sparse as _sp

        P = self._select_P_operator(which)
        # make CSR view
        if _sp.issparse(X):
            Xc = X.tocsr(copy=True).astype(dtype)
            for _ in range(int(t)):
                Xc = P @ Xc
            if output in ('auto', 'sparse'):
                return Xc
            return Xc.toarray()
        else:
            Xd = _np.asarray(X, dtype=dtype)
            for _ in range(int(t)):
                Xd = P @ Xd
            if output in ('auto', 'dense'):
                return Xd
            # convert to sparse CSR
            return _sp.csr_matrix(Xd)

    # ---- Riemannian diagnostics (metric + scalars) ----
    def riemann_diagnostics(
        self,
        Y: Optional[np.ndarray] = None,
        L: Optional[np.ndarray] = None,
        center: str = "median",
        diffusion_t: int = 0,
        diffusion_op: Optional[str] = None,  # {'X','Z','msZ'} or None
        normalize: str = "symmetric",
        clip_percentile: float = 2.0,
        return_limits: bool = True,
        compute_metric: bool = True,
        compute_scalars: bool = True,
        random_state: Optional[int] = 7,
    ):
        ...

        """
        Compute Riemann metric in 2D and derived scalars (anisotropy, log det G, deformation).

        Parameters
        ----------
        Y : np.ndarray, shape (n,2), optional
            2D embedding. If None, uses msMAP if available, else MAP.
        L : array-like (Laplacian) or None
            Graph Laplacian. If None, defaults to base-kernel Laplacian.
        center : {'median','mean'}
            Centering for deformation calculation.
        diffusion_t : int (default 0)
            Optional diffusion smoothing steps for deformation maps.
        diffusion_op : {'X','Z','msZ', None}
            Operator for smoothing scalar fields (if diffusion_t > 0).
        normalize : str (default 'symmetric')
            Normalization option passed to deformation routine.
        clip_percentile : float (default 2.0)
            Percentile for clipping in deformation calculation.
        return_limits : bool (default True)
            Return color-scale limits (vmin, vmax) alongside values.
        compute_metric : bool (default True)
            Compute local metric tensor G.
        compute_scalars : bool (default True)
            Compute anisotropy and logdet.

        Returns
        -------
        dict
            {
                'G': ndarray of shape (n,2,2) if computed,
                'anisotropy': ndarray (n,) if computed,
                'logdetG': ndarray (n,) if computed,
                'deformation': ndarray (n,),
                'limits': (vmin,vmax) if return_limits
            }
        """
        import numpy as _np
        from topo.eval.rmetric import RiemannMetric, calculate_deformation

        # Defaults
        if Y is None:
            try:
                Y = self.TopoMAP
            except Exception:
                try:
                    Y = self.msTopoMAP
                except Exception:
                    try:
                        Y = self.TopoPaCMAP
                    except Exception:
                        try:
                            Y = self.msTopoPaCMAP
                        except Exception:
                            warnings.warn("No projection found; computing new projection.")
                            Y = self.project(projection_method='MAP', multiscale=False, random_state=random_state)
                            Y = self.TopoMAP
        if L is None:
            # Base Laplacian for geometry (as in the demo)
            L = self.base_kernel.L

        # Metric
        out = {}
        if compute_metric:
            G = RiemannMetric(Y, L).get_rmetric()
            out['G'] = G
        else:
            G = None

        # Scalars (anisotropy and logdet from G)
        if compute_scalars:
            if G is None:
                G = RiemannMetric(Y, L).get_rmetric()
                out['G'] = G
            lam = _np.linalg.eigvalsh(G)
            lam = _np.clip(lam, 1e-12, None)
            out['anisotropy'] = _np.log(lam[:, -1] / lam[:, 0])
            out['logdetG'] = _np.sum(_np.log(lam), axis=1)

        # Deformation (centered log det(G) with optional diffusion smoothing)
        P = None
        if diffusion_t and diffusion_op is not None:
            P = self._select_P_operator(diffusion_op)
        deform_vals, limits = calculate_deformation(
            Y, L,
            center=center,
            diffusion_t=int(max(0, diffusion_t)),
            diffusion_op=P,
            normalize=normalize,
            clip_percentile=float(clip_percentile),
            return_limits=True,
        )
        out['deformation'] = deform_vals
        if return_limits:
            out['limits'] = limits

        # Cache for later reuse
        self.RiemannMetricDict['last'] = out
        return out


    # ---------------------------------------------------------------------
    # Evaluation helper (unchanged, BC)
    # ---------------------------------------------------------------------
    def eval_models_layouts(self, X,
                            landmarks=None,
                            kernels=['cknn', 'bw_adaptive'],
                            eigenmap_methods=['msDM', 'DM', 'LE'],
                            projections=['MAP'],
                            additional_eigenbases=None,
                            additional_projections=None,
                            landmark_method='random',
                            n_neighbors=5, n_jobs=-1,
                            cor_method='spearman', **kwargs):
        """
        Evaluate orthogonal bases, topological graphs and layouts against geodesic correlations
        and a PCA baseline. Kept for backward compatibility.
        """
        from scipy.stats import spearmanr, kendalltau
        from scipy.spatial.distance import squareform
        from topo.utils._utils import get_landmark_indices
        from topo.eval.global_scores import global_score_pca
        from topo.eval.local_scores import geodesic_distance

        if self.verbosity > 0:
            print('Running specified models...')
        self.run_models(X, kernels, eigenmap_methods, projections)
        gc.collect()

        # landmarks
        if landmarks is not None:
            if isinstance(landmarks, int):
                landmark_indices = get_landmark_indices(
                    self.base_knn_graph, n_landmarks=landmarks, method=landmark_method, random_state=self.random_state)
                if landmark_indices.shape[0] == self.base_knn_graph.shape[0]:
                    landmark_indices = None
            elif isinstance(landmarks, np.ndarray):
                landmark_indices = landmarks
            else:
                raise ValueError('\'landmarks\' must be either an integer or a numpy array.')
        else:
            landmark_indices = None

        # base geodesics
        if self.verbosity > 0:
            print('Computing base geodesics...')
        base_graph = self.base_knn_graph if landmark_indices is None else self.base_knn_graph[landmark_indices, :][:, landmark_indices]
        base_geodesics = squareform(geodesic_distance(base_graph, directed=False, n_jobs=n_jobs))
        gc.collect()

        # eigenbases
        EigenbasisLocalResults = {}
        EigenbasisGlobalResults = {}
        for key in self.EigenbasisDict.keys():
            if self.verbosity > 0:
                print(f"Computing geodesics for eigenbasis '{key}...'")
            emb = self.EigenbasisDict[key].results()
            emb_graph = kNN(emb, n_neighbors=n_neighbors, metric=self.base_metric, n_jobs=n_jobs,
                            backend=self.backend, return_instance=False, verbose=False, **kwargs)
            if landmark_indices is not None:
                emb_graph = emb_graph[landmark_indices, :][:, landmark_indices]
            embedding_geodesics = squareform(geodesic_distance(emb_graph, directed=False, n_jobs=n_jobs))
            if cor_method == 'spearman':
                EigenbasisLocalResults[key], _ = spearmanr(base_geodesics, embedding_geodesics)
            else:
                EigenbasisLocalResults[key], _ = kendalltau(base_geodesics, embedding_geodesics)
            EigenbasisGlobalResults[key] = global_score_pca(X, emb)
            if self.verbosity > 0:
                print('Finished for eigenbasis', key)
            gc.collect()

        # projections
        ProjectionLocalResults = {}
        ProjectionGlobalResults = {}
        for key in self.ProjectionDict.keys():
            if self.verbosity > 0:
                print(f"Computing geodesics for projection '{key}...'")
            emb_graph = kNN(self.ProjectionDict[key], n_neighbors=n_neighbors, metric=self.graph_metric,
                            n_jobs=n_jobs, backend=self.backend, return_instance=False, verbose=False, **kwargs)
            if landmark_indices is not None:
                emb_graph = emb_graph[landmark_indices, :][:, landmark_indices]
            embedding_geodesics = squareform(geodesic_distance(emb_graph, directed=False, n_jobs=n_jobs))
            if cor_method == 'spearman':
                ProjectionLocalResults[key], _ = spearmanr(base_geodesics, embedding_geodesics)
            else:
                ProjectionLocalResults[key], _ = kendalltau(base_geodesics, embedding_geodesics)
            ProjectionGlobalResults[key] = global_score_pca(X, self.ProjectionDict[key])
            gc.collect()

        # PCA baseline
        from sklearn.decomposition import PCA
        if self.verbosity >= 1:
            print('Computing PCA for comparison...')
        if issparse(X) is True:
            if isinstance(X, csr_matrix):
                data = X.todense()
            else:
                data = X
        else:
            if not isinstance(X, np.ndarray):
                import pandas as pd
                if isinstance(X, pd.DataFrame):
                    data = np.asarray(X.values.T)
                else:
                    return print('Unknown data format.')
            else:
                data = X
        pca_emb = PCA(n_components=self.n_eigs).fit_transform(data)
        emb_graph = kNN(pca_emb, n_neighbors=n_neighbors, metric=self.graph_metric, n_jobs=n_jobs,
                        backend=self.backend, return_instance=False, verbose=False, **kwargs)
        if landmark_indices is not None:
            emb_graph = emb_graph[landmark_indices, :][:, landmark_indices]
        embedding_geodesics = squareform(geodesic_distance(emb_graph, directed=False, n_jobs=n_jobs))
        if self.verbosity > 0:
            print('Computing Spearman R for PCA...')
        EigenbasisLocalResults['PCA'], _ = spearmanr(base_geodesics, embedding_geodesics)
        ProjectionGlobalResults['PCA'] = global_score_pca(X, pca_emb)

        res_dict = {'EigenbasisLocal': EigenbasisLocalResults,
                    'EigenbasisGlobal': EigenbasisGlobalResults,
                    'ProjectionLocal': ProjectionLocalResults,
                    'ProjectionGlobal': ProjectionGlobalResults}
        return res_dict


# -----------------------------------------------------------------------------
# Module-level pickle helpers (read/write TopOGraph objects)
# -----------------------------------------------------------------------------

def save_topograph(tg: TopOGraph, filename: str = 'topograph.pkl', remove_base_class: bool = True):
    """
    Save a TopOGraph object to a pickle file.

    Parameters
    ----------
    tg : TopOGraph
        The object to save.
    filename : str
        Destination path.
    remove_base_class : bool
        If True, clears `tg.base_nbrs_class` before pickling to avoid non-serializable ANN handles.
    """
    try:
        import pickle
    except ImportError:
        print('Pickle is needed for saving the TopOGraph. Please install it with `pip3 install pickle`')
        return

    if not isinstance(tg, TopOGraph):
        raise TypeError("`tg` must be a TopOGraph instance.")

    if tg.base_nbrs_class is not None and remove_base_class:
        tg.base_nbrs_class = None

    with open(filename, 'wb') as f:
        pickle.dump(tg, f, pickle.HIGHEST_PROTOCOL)
    print(f'TopOGraph saved at {filename}')


def load_topograph(filename: str) -> TopOGraph:
    """
    Load a TopOGraph object from a pickle file.

    Parameters
    ----------
    filename : str
        Path to a previously pickled TopOGraph.

    Returns
    -------
    TopOGraph
        The loaded object.
    """
    try:
        import pickle
    except ImportError:
        raise ImportError('Pickle is needed for loading the TopOGraph. Please install it with `pip3 install pickle`')

    with open(filename, 'rb') as f:
        obj = pickle.load(f)

    if not isinstance(obj, TopOGraph):
        warnings.warn("Loaded object is not a TopOGraph; returning as-is.", RuntimeWarning)
        return obj
    return obj



