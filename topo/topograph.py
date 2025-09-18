# TopoMetry high-level API - the TopOGraph class
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
      • global_id, local_ids: intrinsic-dimension details

    Legacy benchmarking and combinatorial exploration remain available through:
      • BaseKernelDict, EigenbasisDict, GraphKernelDict, ProjectionDict
      • run_models(), eval_models_layouts()

    Parameters
    ----------
    base_knn : int (optional, default 30)
        k-nearest-neighbors for the base input space.

    graph_knn : int (optional, default 30)
        k-nearest-neighbors for the refined graph built on the spectral scaffold (Z) space.

    min_eigs : int (optional, default 100)
        Minimum number of eigenpairs to compute (spectral scaffold size cap).

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
    id_min_components : int (default 128)
    id_max_components : int (default 1024)
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

    # --- Updated constructor (__init__) with UoM flag & state ----------------
    def __init__(self,
                base_knn=30,
                graph_knn=30,
                min_eigs=128,
                n_jobs=-1,
                projection_methods=['MAP','PaCMAP'],
                base_kernel=None,
                base_kernel_version='bw_adaptive',
                eigenmap_method=None,  # deprecated
                laplacian_type='normalized', # deprecated
                graph_kernel_version='bw_adaptive',
                base_metric='cosine',
                graph_metric='euclidean',
                diff_t=0,
                delta=1.0,
                sigma=0.1,
                low_memory=False,
                eigen_tol=1e-8,
                eigensolver='arpack',
                backend='hnswlib',
                cache=True,
                verbosity=0,
                random_state=0,
                # ID defaults (both methods computed; `id_method` selects the size used)
                id_method='fsa',
                id_ks=50,
                id_metric='euclidean',
                id_quantile=0.99,
                id_min_components=128,
                id_max_components=1024,
                id_headroom=0.5,
                uom=False,
                ):
        # Core config
        self.projection_methods = projection_methods
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

        # Snapshots storage (for MAP checkpointing)
        self.msTopoMAP_snapshots = []
        self.TopoMAP_snapshots = []

        # ------------------ Union-of-Manifolds state ------------------
        self.uom_enabled = bool(uom)
        self.uom_comp_labels_ = None                 # shape (n,)
        self.uom_components_ = None                  # list of np.ndarray indices (per component)

        # Per-component stores (lists aligned with uom_components_)
        self.uom_knn_X_list = None   # list[csr_matrix] per component (kNN on X_i)
        self.knn_X_uom = None        # aggregated block-diagonal kNN(X)
        self.P_of_X_uom = None       # aggregated block-diagonal P_of_X
        self.uom_BaseKernel_list = None              # list[Kernel] (P_of_Xi)
        self.uom_DMEig_list = None                   # list[EigenDecomposition]
        self.uom_msDMEig_list = None                 # list[EigenDecomposition]
        self.uom_eigenvalues_dm_list = None   # list[np.ndarray], per-component DM eigenvalues
        self.uom_eigenvalues_ms_list = None   # list[np.ndarray], per-component msDM eigenvalues
        self._uom_active_mode = "msDM"        # track which mode is considered 'active' for summaries
        self.uom_Z_list = None                       # list[np.ndarray]
        self.uom_msZ_list = None                     # list[np.ndarray]
        self.uom_knn_Z_list = None                   # list[csr_matrix]
        self.uom_knn_msZ_list = None                 # list[csr_matrix]
        self.uom_Kernel_Z_list = None                # list[Kernel]
        self.uom_Kernel_msZ_list = None              # list[Kernel]

        # Aggregated UoM views
        self.Z_uom = None
        self.msZ_uom = None
        self.knn_Z_uom = None
        self.knn_msZ_uom = None
        self.P_of_Z_uom = None
        self.P_of_msZ_uom = None
        self._uom_axis_slices = None                 # list[(start,end)] per component
        # --- UoM cached components (user- or auto-computed) ---
        self.uom_comp_labels_: Optional[np.ndarray] = None   # shape (n,) or None
        self.uom_components_: Optional[list[np.ndarray]] = None

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

        def _noANN_lib(self):
            print("Warning: no approximate nearest neighbor library found. Using sklearn's KDTree instead.")
            self.backend = 'sklearn'

        if self.backend == 'hnswlib':
            if not self._have_hnswlib:
                if self._have_nmslib:
                    self.backend = 'nmslib'
                else:
                    self._noANN_lib()
        elif self.backend == 'nmslib':
            if self._have_hnswlib:
                self.backend = 'hnswlib'
            else:
                self._noANN_lib()
        else:
            self._noANN_lib()

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
        n_eigs_automated, id_details = automated_scaffold_sizing(
            X,
            method=self.id_method,
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
        self._id_details[self.id_method] = id_details

        # finalize scaffold component counts (same for msDM/DM at this stage)
        k_sel = int(max(2, min(n_eigs_automated, max_cap)))
        self._scaffold_components_ms = k_sel
        self._scaffold_components_dm = k_sel

        self.n_eigs = int(max(self.n_eigs, k_sel))
        local   = id_details.get('local_id_mle', None)

        # Cache for downstream report / accessors
        self.global_dimensionality = k_sel
        self.local_dimensionality  = local

        # Keep details available (back-compat)
        self._id_details[self.id_method] = id_details


    def _csr(self, A): 
        return A if sp.isspmatrix_csr(A) else A.tocsr()

    def _to_float32_csr(self, A: sp.csr_matrix) -> sp.csr_matrix:
        A = self._csr(A)
        if A.dtype != np.float32:
            A = A.astype(np.float32, copy=False)
        return A

    def _symmetrize_geometric(self, P: sp.csr_matrix) -> sp.csr_matrix:
        """S_ij = sqrt(P_ij * P_ji) on overlapping support (CSR float32)."""
        P = self._to_float32_csr(P)
        S = P.multiply(P.T)
        if S.nnz == 0:
            return S
        S.data = np.sqrt(S.data.astype(np.float64)).astype(np.float32, copy=False)
        S.eliminate_zeros()
        return S

    def _normalized_laplacian(self, A: sp.csr_matrix) -> sp.csr_matrix:
        A = self._to_float32_csr(A)
        n = A.shape[0]
        d = np.asarray(A.sum(axis=1)).ravel().astype(np.float64).clip(min=1e-12)
        Dmh = sp.diags((1.0 / np.sqrt(d)).astype(np.float32))
        return sp.eye(n, dtype=np.float32, format="csr") - (Dmh @ A @ Dmh)

    def _eigengap_k(self, vals: np.ndarray, k_max: int, k_min: int = 2) -> int:
        vals = np.asarray(vals, dtype=float)
        if vals.size <= 2: return max(k_min, min(k_max, 2))
        gaps = np.diff(vals)
        if gaps.size <= 1: return max(k_min, min(k_max, 2))
        j = int(np.argmax(gaps[1:])) + 1
        return int(max(k_min, min(k_max, j + 1)))

    def _mbkm(self, X: np.ndarray, n_clusters: int, random_state: int = 0) -> np.ndarray:
        from sklearn.cluster import MiniBatchKMeans
        n_use = int(max(2, n_clusters))
        batch = int(min(2048, max(256, 8 * n_use * n_use)))
        km = MiniBatchKMeans(
            n_clusters=n_use, batch_size=batch, n_init=10, max_no_improvement=30,
            reassignment_ratio=0.01, random_state=random_state, verbose=0,
        )
        return km.fit_predict(X.astype(np.float32, copy=False))

    def _consolidate_macros_via_conductance(self, W: sp.csr_matrix, labels: np.ndarray, max_iters: int = 100):
        W = self._to_float32_csr(W)
        labels = labels.copy()
        it = 0
        while it < max_iters:
            it += 1
            uniq = np.unique(labels)
            if uniq.size <= 2:
                break
            deg = np.asarray(W.sum(axis=1)).ravel().astype(np.float64)
            phi, idx_list = [], []
            # Merge macros whose volume is tiny vs median (e.g., < 0.6 * median)
            vols = [float(W[idx, :].sum()) for idx in idx_list]
            med_vol = np.median(vols) if len(vols) else 0.0
            tiny = [i for i, v in enumerate(vols) if v < 0.6 * med_vol and len(idx_list[i]) > 0]

            for t in tiny:
                idx_t = idx_list[t]
                # find strongest neighbor to merge into
                best_neighbor, best_w = None, -1.0
                for j, idx_other in enumerate(idx_list):
                    if j == t or len(idx_other) == 0:
                        continue
                    w = float(W[np.ix_(idx_t, idx_other)].sum())
                    if w > best_w:
                        best_w = w
                        best_neighbor = j
                if best_neighbor is not None:
                    labels[idx_t] = uniq[best_neighbor]
            # reindex after tiny merges before computing phi
            _, labels = np.unique(labels, return_inverse=True)
            for g in uniq:
                idx = np.where(labels == g)[0]
                idx_list.append(idx)
                comp = W[np.ix_(idx, idx)]
                internal = (comp.sum() - comp.diagonal().sum()).astype(np.float64)
                ext = (deg[idx].sum() - 2.0 * internal).astype(np.float64)
                phi.append(float(ext / (ext + 2.0 * internal + 1e-12)))
            phi = np.array(phi, dtype=float)
            if phi.size < 3: break
            q1, q3 = np.percentile(phi, [25, 75]); thr = q3 + 1.0 * (q3 - q1)
            mask = phi > thr
            if not mask.any(): break
            worst_pos = int(np.argmax(phi * mask))
            idx_worst = idx_list[worst_pos]
            best_neighbor, best_w = None, -1.0
            for pos, idx_other in enumerate(idx_list):
                if pos == worst_pos: continue
                w = float(W[np.ix_(idx_worst, idx_other)].sum())
                if w > best_w: best_w, best_neighbor = w, uniq[pos]
            if best_neighbor is None: break
            labels[idx_worst] = best_neighbor
        _, labels_new = np.unique(labels, return_inverse=True)
        return labels_new

    # -------------------- Louvain (pure SciPy) --------------------
    def _louvain_micro(self, S: sp.csr_matrix, random_state: int = 0, max_passes: int = 100, gamma: float = 0.85):
        """
        Greedy Louvain (modularity) on weighted undirected graph S (CSR).
        Returns micro labels (ints). No external deps.
        """
        rng = np.random.RandomState(int(random_state))
        S = self._to_float32_csr(S)
        n = S.shape[0]
        if n <= 2 or S.nnz == 0:
            return np.zeros(n, dtype=int)

        # Modularity constants
        w = float(S.sum())                # total graph weight (sum of all edges)
        if w <= 0: 
            return np.zeros(n, dtype=int)
        m2 = w                            # 2m in standard notation if S has each edge once per direction
        ki = np.asarray(S.sum(axis=1)).ravel().astype(np.float64)  # degree/strength per node

        # init: each node in its own community
        labels = np.arange(n, dtype=int)
        # community strength (sum of degrees) and internal weights
        com_deg = ki.copy()               # sum of degrees in community
        com_in = np.zeros(n, dtype=np.float64)   # 2 * internal edge weight per community

        # Precompute neighbors lists (& weights) for speed
        S = S.tocsr()
        indptr, indices, data = S.indptr, S.indices, S.data

        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            order = np.arange(n); rng.shuffle(order)

            for v in order:
                v_lab = labels[v]
                k_v = ki[v]
                # remove v from its community
                # contribution removal: we’ll adjust when computing deltas; implicit here

                # neighborhood community weights: sum of weights from v to each neighbor's community
                start, end = indptr[v], indptr[v+1]
                nbrs = indices[start:end]
                wts  = data[start:end]

                # accumulate weights per community
                com_w = {}
                for u, wvu in zip(nbrs, wts):
                    cu = labels[u]
                    com_w[cu] = com_w.get(cu, 0.0) + float(wvu)

                # compute best move (ΔQ)
                best_c, best_dq = v_lab, 0.0
                # remove v from its current community: compute com_deg without v
                com_deg_v_removed = com_deg[v_lab] - k_v

                for c, k_v_in in com_w.items():
                    if c == v_lab and com_deg_v_removed <= 0:
                        # staying put, but treat as candidate with dq=0
                        continue
                    # ΔQ (modularity) for moving v -> c (Newman-Girvan, weighted)
                    # ΔQ = [k_v_in - k_v * (sum_deg_c) / m2] / m2 * 2  (simplified constants folded)
                    dq = k_v_in - gamma * (k_v * (com_deg[c]) / m2)

                    if c != v_lab and dq > best_dq:
                        best_dq, best_c = dq, c
                if best_c != v_lab and best_dq > 1e-12:
                    # apply move
                    labels[v] = best_c
                    com_deg[v_lab] -= k_v
                    com_deg[best_c] += k_v
                    improved = True

            # optional: relabel communities to be compact
            _, labels = np.unique(labels, return_inverse=True)

        return labels

    # -------------------- main API (Louvain micro; rest unchanged) --------------------
    def uom_find_components(self, P: sp.csr_matrix, random_state: int = 0, consolidate: bool = True, max_passes: int = 100, gamma: float = 0.85):
        """
        Discover disconnected "macro" components under the Union-of-Manifolds (UoM) hypothesis
        using the refined TopoMetry operator (self.graph_kernel.P).

        Workflow
        --------
        1. Symmetrize P via geometric mean → conservative similarity S.
        2. If S already disconnected, return its connected components.
        3. Micro partition: greedy Louvain clustering on S, with resolution γ < 1.
        4. Build a supergraph W (micro × micro), edge weights = sum of S across partitions.
        5. Macro partition: eigengap spectral clustering on W with MiniBatchKMeans.
        6. Optional consolidation: merge fragile macros using a conductance outlier rule.
        7. Propagate macro labels back to all cells.

        Parameters
        ----------
        random_state : int or None, default=None
            Random seed for stochastic parts (MiniBatchKMeans initialization,
            node visiting order in Louvain).  
            If None, falls back to `self.random_state`.

        consolidate : bool, default=True
            Whether to run a final **consolidation pass** on the macro components.  
            Uses conductance-based outlier detection to merge flimsy macros into stronger
            neighbors. Helps avoid over-splitting into many weak components.  
            Set False to return the raw spectral partition.

        gamma : float, default=0.85
            Resolution parameter for Louvain modularity at the **micro** partition step.  
            - γ < 1 → favors fewer, larger micro-communities (coarser).  
            - γ > 1 → favors more, smaller micro-communities (finer).  
            Lowering below 1 is generally helpful for UoM, as it avoids too many spurious
            micro splits that later cascade into excess macro components.

        max_passes : int, default=100
            Maximum number of Louvain refinement passes over all nodes.  
            Each pass shuffles nodes and attempts greedy modularity improvements.  
            Larger values increase stability at the cost of runtime. Usually
            20–50 passes are sufficient; 100 is a safe ceiling.

        Returns
        -------
        n_comp : int
            Number of discovered UoM components (macro-level).

        labels : ndarray of shape (n_samples,)
            Integer component label for each cell (0 .. n_comp-1).
            These labels reflect the final macro components after optional consolidation.
        """
        from scipy.sparse.csgraph import connected_components
        from scipy.sparse.linalg import eigsh

        S = self._symmetrize_geometric(P)
        n = S.shape[0]
        if S.nnz == 0:
            return np.zeros(n, dtype=int)

        n_cc, cc_labels = connected_components(S, directed=False, return_labels=True)
        if n_cc > 2:
            return n_cc, cc_labels

        # (3) micro via Louvain with resolution γ < 1
        micro = self._louvain_micro(S, random_state=random_state, max_passes=max_passes, gamma=gamma)
        _, micro_labels = np.unique(micro, return_inverse=True)
        k = micro_labels.max() + 1

        # (4) supergraph W (unchanged)
        rows, cols = S.nonzero()
        vals = np.asarray(S[rows, cols]).ravel()
        mr, mc = micro_labels[rows], micro_labels[cols]
        u = mr < mc
        if not np.any(u):
            return np.zeros(n, dtype=int)

        r, c, w = mr[u], mc[u], vals[u]
        idx = r * k + c
        acc = np.bincount(idx, weights=w, minlength=k * k).astype(np.float32, copy=False).reshape(k, k)
        W = sp.csr_matrix(acc + acc.T, dtype=np.float32)

        if W.nnz == 0 or k <= 2:
            return np.zeros(n, dtype=int) if k <= 1 else micro_labels

        # (5) macro spectral + MBKM with slightly smaller k_max
        Lw = self._normalized_laplacian(W)
        k_max = int(min(8, max(3, np.floor(np.sqrt(k) + 1))))   # <- tweak
        nev = int(min(k_max + 1, max(2, k - 1)))
        vals_w, vecs_w = eigsh(Lw, k=nev, which="SM")
        order = np.argsort(vals_w)
        vals_w, vecs_w = vals_w[order], vecs_w[:, order]
        k_macro = self._eigengap_k(vals_w[:nev], k_max=k_max, k_min=2)
        Uw = vecs_w[:, :k_macro]
        Uw /= (np.linalg.norm(Uw, axis=1, keepdims=True) + 1e-12)
        macro = self._mbkm(Uw, n_clusters=k_macro, random_state=random_state)

        # (6) stronger consolidation
        if consolidate and np.unique(macro).size > 2:
            macro = self._consolidate_macros_via_conductance(W, macro)

        labels = macro[micro_labels]
        n_comp = int(np.unique(labels).size)
        self.uom_comp_labels_ = labels
        self.uom_components_  = [np.where(labels == c)[0] for c in np.unique(labels)]
        return n_comp, labels



        
    # ---------------------------------------------------------------------
    # High-level fit (builds everything) – dual scaffold
    # ---------------------------------------------------------------------
    def fit(self, X=None, **kwargs):
        """
        Build base kNN, base kernel P(X). Compute both msDM and DM eigenbases (dual scaffold).
        Optionally (uom=True), detect disconnected components and build per-component
        scaffolds and refined graphs; aggregate them into block-diagonal operators and
        concatenated coordinates with no cross-component edges.
        """
        # Basic checks (as before)
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

        # Global automated sizing (diagnostics / BC)
        if self.base_metric != 'precomputed':
            self._automated_sizing(X if X is not None else self.base_kernel.X)
            if self.verbosity >= 1:
                print(f"Automated sizing (pre-eigs) → target components: {self._scaffold_components_ms} "
                    f"(n_eigs set to {self.n_eigs})")

        self.uom_eigenvalues_dm_list, self.uom_eigenvalues_ms_list = [], []

        # --- UoM branch: detect components and build per-component pipelines ---
        if self.uom_enabled:
            if self.verbosity >= 1:
                print('UoM: detecting disconnected components in P(X) and building per-component scaffolds/graphs...')

            class _ProxyKernel:
                __slots__ = ("P", "K")
                def __init__(self, P): self.P = P; self.K = P

            if (self.uom_comp_labels_ is not None) and (self.uom_comp_labels_.shape[0] == self.n):
                labels = self.uom_comp_labels_
                n_comp = int(np.unique(labels).size)
                if self.verbosity >= 1:
                    print(f"UoM: using precomputed component labels (n={n_comp}).")
            else:
                # Build components on the refined operator (already available now)
                n_comp, labels = self.uom_find_components(P=self.base_kernel.P)
                if self.verbosity >= 1:
                    print(f"UoM: computed component labels on refined graph (n={n_comp}).")

            self.uom_comp_labels_ = labels
            self.uom_components_  = [np.where(labels == c)[0] for c in np.unique(labels)]

            # Prepare per-component containers
            self.uom_knn_X_list, self.uom_BaseKernel_list, self.uom_DMEig_list, self.uom_msDMEig_list = [], [], [], []
            self.uom_Z_list, self.uom_msZ_list = [], []
            self.uom_knn_Z_list, self.uom_knn_msZ_list = [], []
            self.uom_Kernel_Z_list, self.uom_Kernel_msZ_list = [], []

            # Helper for per-component sizing (without touching global state)
            def _local_size(Xi, n_max):
                n_i = Xi.shape[0]
                cap = min(int(self.id_max_components), max(2, n_i - 2))
                k_auto, _det = automated_scaffold_sizing(
                    Xi,
                    method=self.id_method,
                    ks=self.id_ks,
                    backend=self.backend,
                    metric=self.id_metric,
                    n_jobs=self.n_jobs if self.n_jobs != -1 else None,
                    quantile=self.id_quantile,
                    min_components=int(min(self.id_min_components, cap)),
                    max_components=int(min(cap, n_max)),
                    headroom=float(self.id_headroom),
                    random_state=self.random_state,
                    return_details=True,
                )
                return int(max(2, min(k_auto, cap)))

            # Build per-component products
            for I in self.uom_components_:
                n_i = int(I.size)
                # Tiny components: fallback to trivial coordinates & identity operator
                if n_i < 3:
                    # Tiny components: fallback to trivial coordinates & identity operator
                    Zi = np.zeros((n_i, 1), dtype=np.float32)
                    msZi = Zi.copy()
                    self.uom_Z_list.append(Zi)
                    self.uom_msZ_list.append(msZi)

                    # Graphs: 1-NN (self-loop) / identity diffusion
                    P_block = sp.eye(n_i, format='csr', dtype=np.float32)
                    self.uom_knn_Z_list.append(P_block.copy())
                    self.uom_knn_msZ_list.append(P_block.copy())

                    # Use proxy kernels instead of Kernel()
                    KZ   = _ProxyKernel(P_block)
                    KmsZ = _ProxyKernel(P_block)
                    Ki   = _ProxyKernel(P_block)

                    self.uom_Kernel_Z_list.append(KZ)
                    self.uom_Kernel_msZ_list.append(KmsZ)
                    self.uom_BaseKernel_list.append(Ki)
                    self.uom_DMEig_list.append(None)
                    self.uom_msDMEig_list.append(None)
                    continue

                # Build a fresh per-component kNN 
                k_neighbors_i = min(self.base_knn, max(1, n_i - 1))
                if self.base_metric == 'precomputed':
                    Xi = (X[np.ix_(I, I)] if X is not None else
                        (self.base_kernel.X[np.ix_(I, I)] if getattr(self.base_kernel, 'X', None) is not None else None))
                else:
                    Xi = (X[I] if (X is not None) else
                        (self.base_kernel.X[I] if getattr(self.base_kernel, 'X', None) is not None else None))

                knn_i = kNN(
                    Xi,
                    n_neighbors=k_neighbors_i,
                    metric=self.base_metric,
                    n_jobs=self.n_jobs,
                    backend=self.backend,
                    return_instance=False,
                    verbose=False,
                    **kwargs
                )
                self.uom_knn_X_list.append(knn_i)

                Ki, _ = self._compute_kernel_from_version_knn(
                    knn_i,
                    min(self.base_knn, max(1, n_i - 1)),
                    self.base_kernel_version,
                    {} if self.low_memory else self.BaseKernelDict,
                    suffix=f'_uom_X[{n_i}]',
                    low_memory=self.low_memory,
                    data_for_expansion=Xi,
                    base=True
                )
                self.uom_BaseKernel_list.append(Ki)

                # Per-component sizing (initial suggestion)
                k_i = _local_size(Xi if Xi is not None else knn_i, n_max=n_i - 2)

                # --- NEW: clamp against the actual N seen by the eigensolver ---
                # (ARPACK requires k < N; use the Kernel's matrix shape, not just n_i)
                Ki_mat = getattr(Ki, "K", None)
                if Ki_mat is None:
                    Ki_mat = getattr(Ki, "P", None)
                N_i = int(Ki_mat.shape[0])

                # Tiny safety: if something slipped through
                if N_i <= 2:
                    # fall back to the tiny-component path (same as your earlier branch)
                    Zi = np.zeros((N_i, 1), dtype=np.float32)
                    msZi = Zi.copy()
                    self.uom_Z_list.append(Zi)
                    self.uom_msZ_list.append(msZi)
                    P_block = sp.eye(N_i, format='csr', dtype=np.float32)
                    self.uom_knn_Z_list.append(P_block.copy())
                    self.uom_knn_msZ_list.append(P_block.copy())
                    self.uom_Kernel_Z_list.append(_ProxyKernel(P_block))
                    self.uom_Kernel_msZ_list.append(_ProxyKernel(P_block))
                    self.uom_BaseKernel_list.append(_ProxyKernel(P_block))
                    self.uom_DMEig_list.append(None)
                    self.uom_msDMEig_list.append(None)
                    continue

                # Final requested k: strictly less than N_i
                k_req = int(min(max(k_i, 2), N_i - 1, self.n_eigs))
                k_req = max(1, k_req)  # absolute lower bound

                # Eigens on P_of_Xi with guaranteed k_req < N_i
                eig_dm_i = EigenDecomposition(
                    n_components=k_req,
                    method='DM',
                    eigensolver=self.eigensolver,
                    eigen_tol=self.eigen_tol,
                    drop_first=True,
                    weight=True,
                    t=self.diff_t,
                    random_state=self.random_state,
                    verbose=False
                ).fit(Ki)

                eig_ms_i = copy.deepcopy(eig_dm_i); eig_ms_i.method = 'msDM'
                # Store per-component eigenvalues (DM & msDM; same spectrum, different transforms at use-time)
                self.uom_eigenvalues_dm_list.append(np.array(eig_dm_i.eigenvalues, copy=True))
                self.uom_eigenvalues_ms_list.append(np.array(eig_ms_i.eigenvalues, copy=True))
                
                k_avail = eig_dm_i.eigenvalues.shape[0]
                k_use = min(k_i, k_avail)
                Zi   = eig_dm_i.transform()[:, :k_use]
                eig_ms_i = copy.deepcopy(eig_dm_i); eig_ms_i.method = 'msDM'
                msZi = eig_ms_i.transform()[:, :k_use]

                self.uom_DMEig_list.append(eig_dm_i); self.uom_msDMEig_list.append(eig_ms_i)
                self.uom_Z_list.append(Zi); self.uom_msZ_list.append(msZi)

                # kNN on Zi / msZi
                k_graph_i = min(self.graph_knn, max(1, n_i - 1))
                knn_Z_i = kNN(Zi, n_neighbors=k_graph_i, metric=self.graph_metric,
                            n_jobs=self.n_jobs, backend=self.backend, return_instance=False,
                            verbose=False, **kwargs)
                knn_msZ_i = kNN(msZi, n_neighbors=k_graph_i, metric=self.graph_metric,
                                n_jobs=self.n_jobs, backend=self.backend, return_instance=False,
                                verbose=False, **kwargs)
                self.uom_knn_Z_list.append(knn_Z_i); self.uom_knn_msZ_list.append(knn_msZ_i)

                # Refined kernels on Zi / msZi
                KZ_i, _ = self._compute_kernel_from_version_knn(
                    knn_Z_i, k_graph_i, self.graph_kernel_version,
                    {} if self.low_memory else self.GraphKernelDict,
                    suffix=f'_uom_Z[{n_i}]',
                    low_memory=self.low_memory,
                    data_for_expansion=Zi,
                    base=False
                )
                KmsZ_i, _ = self._compute_kernel_from_version_knn(
                    knn_msZ_i, k_graph_i, self.graph_kernel_version,
                    {} if self.low_memory else self.GraphKernelDict,
                    suffix=f'_uom_msZ[{n_i}]',
                    low_memory=self.low_memory,
                    data_for_expansion=msZi,
                    base=False
                )
                self.uom_Kernel_Z_list.append(KZ_i); self.uom_Kernel_msZ_list.append(KmsZ_i)

            # Aggregate (no cross-edges): place blocks at original indices
            n = self.n
            # Coordinates: concatenate columns with per-component slices
            total_cols_Z = int(sum(z.shape[1] for z in self.uom_Z_list))
            total_cols_msZ = int(sum(z.shape[1] for z in self.uom_msZ_list))
            self.Z_uom = np.zeros((n, total_cols_Z), dtype=np.float32)
            self.msZ_uom = np.zeros((n, total_cols_msZ), dtype=np.float32)
            self._uom_axis_slices = []
            c0 = 0
            for I, Zi in zip(self.uom_components_, self.uom_Z_list):
                c1 = c0 + Zi.shape[1]
                self.Z_uom[I, c0:c1] = Zi
                self._uom_axis_slices.append((c0, c1))
                c0 = c1
            c0 = 0
            for I, msZi in zip(self.uom_components_, self.uom_msZ_list):
                c1 = c0 + msZi.shape[1]
                self.msZ_uom[I, c0:c1] = msZi
                c0 = c1

            # Graphs/operators: sparse assembly with block placement at (I, I)
            def _place_blocks(block_list):
                M = sp.lil_matrix((n, n), dtype=np.float32)
                for I, B in zip(self.uom_components_, block_list):
                    M[np.ix_(I, I)] = B
                return M.tocsr()
            self.knn_X_uom = _place_blocks(self.uom_knn_X_list)
            self.P_of_X_uom = _place_blocks([K.P for K in self.uom_BaseKernel_list])
            self.knn_Z_uom = _place_blocks(self.uom_knn_Z_list)
            self.knn_msZ_uom = _place_blocks(self.uom_knn_msZ_list)
            self.P_of_Z_uom = _place_blocks([K.P for K in self.uom_Kernel_Z_list])
            self.P_of_msZ_uom = _place_blocks([K.P for K in self.uom_Kernel_msZ_list])

            # Set an 'active eigenbasis' label for consistency in getters that check it
            self.current_eigenbasis = f"UoM_{self._uom_active_mode}"
            self.eigenbasis = None  # not a single global object; per-component lists are stored instead

            # Set active refined graphs to UoM aggregates
            self._knn_Z = self.knn_Z_uom
            self._knn_msZ = self.knn_msZ_uom

            # Lightweight proxy kernels for properties
            self._kernel_Z = _ProxyKernel(self.P_of_Z_uom)
            self._kernel_msZ = _ProxyKernel(self.P_of_msZ_uom)

            # Spectral layout using UoM msZ operator by default
            _ = self.spectral_layout(graph=self._kernel_msZ.K, n_components=2)

            # Projections for both scaffolds (UoM-backed)
            for proj in self.projection_methods:
                try:
                    self.project(projection_method=proj, multiscale=True)
                except Exception as e:
                    warnings.warn(f"Projection '{proj}' on msZ (UoM) failed or requires extra dependency: {e}", RuntimeWarning)
                try:
                    self.project(projection_method=proj, multiscale=False)
                except Exception as e:
                    warnings.warn(f"Projection '{proj}' on Z/DM (UoM) failed or requires extra dependency: {e}", RuntimeWarning)

            # Optionally free per-component heavy objects if low_memory
            if self.low_memory:
                self.uom_BaseKernel_list = None
                self.uom_DMEig_list = None
                self.uom_msDMEig_list = None
                self.uom_Kernel_Z_list = None
                self.uom_Kernel_msZ_list = None

            return self  # UoM path ends here
        else:
            # --- Compute global eigenbases (kept for BC/diagnostics) -------------
            if self.verbosity >= 1:
                print('Computing eigenbasis (once on P); deriving DM/msDM embeddings in transform()...')

            dm_key = 'DM with ' + str(self.base_kernel_version)
            ms_key = 'msDM with ' + str(self.base_kernel_version)

            if dm_key not in self.EigenbasisDict:
                t0 = time.time()
                dm_eig = EigenDecomposition(
                    n_components=self.n_eigs,
                    method='DM',
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

            # Graph kernels -> P(msZ), P(Z)
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

            # Spectral layout and projections (global)
            _ = self.spectral_layout(graph=self._kernel_msZ.K, n_components=2)
            for proj in self.projection_methods:
                try:
                    self.project(projection_method=proj, multiscale=True)
                except Exception as e:
                    warnings.warn(f"Projection '{proj}' on msZ failed or requires extra dependency: {e}", RuntimeWarning)
                try:
                    self.project(projection_method=proj, multiscale=False)
                except Exception as e:
                    warnings.warn(f"Projection '{proj}' on Z (DM) failed or requires extra dependency: {e}", RuntimeWarning)

        return self


    # ---------------------------------------------------------------------
    # Public properties / getters (stable user API)
    # ---------------------------------------------------------------------

    def spectral_scaffold(self, multiscale: bool = True):
        """
        Return spectral scaffold coordinates.
        If UoM is enabled, return the aggregated UoM coordinates (concatenated columns,
        rows in original sample order). Otherwise return the global scaffold.

        Parameters
        ----------
        multiscale : bool (default True)
            If True, returns msDM coordinates; else returns DM (fixed time `diff_t`) coordinates.

        Returns
        -------
        np.ndarray, shape (n_samples, n_eigs)
        """
        if self.uom_enabled:
            if multiscale:
                if self.msZ_uom is None:
                    raise AttributeError("UoM msZ scaffold not available. Call .fit(X) with uom=True.")
                return self.msZ_uom
            else:
                if self.Z_uom is None:
                    raise AttributeError("UoM Z scaffold not available. Call .fit(X) with uom=True.")
                return self.Z_uom
        # Non-UoM
        if multiscale:
            key = 'msDM with ' + str(self.base_kernel_version)
        else:
            key = 'DM with ' + str(self.base_kernel_version)
        if key not in self.EigenbasisDict:
            raise AttributeError("Requested spectral scaffold not found. Ensure .fit() completed.")
        return self.EigenbasisDict[key].transform(X=None)
    
    @property
    def eigenvalues(self):
        """
        Eigenvalues of the active eigenbasis.

        Returns
        -------
        np.ndarray or dict
            - Standard mode (uom=False): 1-D np.ndarray (msDM by default).
            - UoM mode (uom=True): dict with per-component eigenvalue arrays:
                {
                'mode': 'msDM' or 'DM',
                'per_component': [np.ndarray, ...],       # eigenvalues for each component
                'component_sizes': [int, ...]             # n_i for each component
                }
        """
        if getattr(self, "uom_enabled", False) and self.uom_eigenvalues_ms_list is not None:
            mode = getattr(self, "_uom_active_mode", "msDM")
            per_comp = self.uom_eigenvalues_ms_list if mode == "msDM" else self.uom_eigenvalues_dm_list
            sizes = [int(ix.size) for ix in (self.uom_components_ or [])]
            return {
                "mode": mode,
                "per_component": per_comp,
                "component_sizes": sizes,
            }

        # Non-UoM (legacy/global)
        if self.current_eigenbasis is None:
            raise AttributeError("Eigenvalues unavailable. Call .fit() first.")
        return self.EigenbasisDict[self.current_eigenbasis].eigenvalues

    # --- Dual scaffold accessors ---
    @property
    def knn_msZ(self):
        """
        kNN graph in the multiscale Diffusion Map (msDM) scaffold space. (UoM-aggregated if enabled).

        Returns
        -------
        scipy.sparse.csr_matrix
            k-nearest neighbor adjacency matrix for the msDM scaffold.

        Raises
        ------
        AttributeError
            If the msDM kNN graph is not available (e.g., `.fit(X)` has not been called).
        """
        if self.uom_enabled:
            if self.knn_msZ_uom is None:
                raise AttributeError("UoM knn_msZ not available. Call .fit(X) with uom=True.")
            return self.knn_msZ_uom
        if self._knn_msZ is None:
            raise AttributeError("knn_msZ is not available. Call .fit(X) first.")
        return self._knn_msZ

    @property
    def knn_Z(self):
        """
        kNN graph in the standard Diffusion Map (DM) scaffold space (UoM-aggregated if enabled)..

        Returns
        -------
        scipy.sparse.csr_matrix
            k-nearest neighbor adjacency matrix for the DM scaffold.

        Raises
        ------
        AttributeError
            If the DM kNN graph is not available (e.g., `.fit(X)` has not been called).
        """
        if self.uom_enabled:
            if self.knn_Z_uom is None:
                raise AttributeError("UoM knn_Z not available. Call .fit(X) with uom=True.")
            return self.knn_Z_uom
        if self._knn_Z is None:
            raise AttributeError("knn_Z is not available. Call .fit(X) first.")
        return self._knn_Z

    @property
    def P_of_msZ(self):
        """
        Diffusion operator on the msDM scaffold (UoM block-diagonal if enabled).

        Returns
        -------
        scipy.sparse.csr_matrix
            Refined diffusion operator (row-stochastic) constructed on the msDM scaffold.

        Raises
        ------
        AttributeError
            If the msDM diffusion operator is not available (e.g., `.fit(X)` has not been called).
        """
        if self.uom_enabled:
            if self.P_of_msZ_uom is None:
                raise AttributeError("UoM P_of_msZ not available. Call .fit(X) with uom=True.")
            return self.P_of_msZ_uom
        if self._kernel_msZ is None:
            raise AttributeError("P_of_msZ is not available. Call .fit(X) first.")
        return self._kernel_msZ.P

    @property
    def P_of_Z(self):
        """
        Diffusion operator on the DM scaffold (UoM block-diagonal if enabled).

        Returns
        -------
        scipy.sparse.csr_matrix
            Refined diffusion operator (row-stochastic) constructed on the DM scaffold.

        Raises
        ------
        AttributeError
            If the DM diffusion operator is not available (e.g., `.fit(X)` has not been called).
        """
        if self.uom_enabled:
            if self.P_of_Z_uom is None:
                raise AttributeError("UoM P_of_Z not available. Call .fit(X) with uom=True.")
            return self.P_of_Z_uom
        if self._kernel_Z is None:
            raise AttributeError("P_of_Z is not available. Call .fit(X) first.")
        return self._kernel_Z.P


    @property
    def knn_X(self):
        """
        Initial kNN graph in the input (X) space (UoM block-diagonal if enabled).

        Returns
        -------
        scipy.sparse.csr_matrix
            k-nearest neighbor adjacency matrix built directly from the input data.

        Raises
        ------
        AttributeError
            If the base kNN graph is not available (e.g., `.fit(X)` has not been called).
        """
        if self.uom_enabled and (self.knn_X_uom is not None):
            return self.knn_X_uom
        if self.base_knn_graph is None:
            raise AttributeError("knn_X is not available. Call .fit(X) first.")
        return self.base_knn_graph

    @property
    def P_of_X(self):
        """
        Diffusion operator on the input (X) space (UoM block-diagonal if enabled).

        Returns
        -------
        scipy.sparse.csr_matrix
            Diffusion operator (row-stochastic) constructed on the base input kNN graph.

        Raises
        ------
        AttributeError
            If the diffusion operator on X is not available (e.g., `.fit(X)` has not been called).
        """
        if self.uom_enabled and (self.P_of_X_uom is not None):
            return self.P_of_X_uom
        if self.base_kernel is None:
            raise AttributeError("P_of_X is not available. Call .fit(X) first.")
        return self.base_kernel.P

    @property
    def global_id(self):
        """
        Global intrinsic dimensionality estimated by the Maximum Likelihood Estimator (MLE).

        Returns
        -------
        float
            The global MLE dimension estimate, as computed during `.fit(X)`.

        Raises
        ------
        AttributeError
            If MLE details are not available (e.g., `.fit(X)` has not been called).
        """
        return self.global_dimensionality


    @property
    def local_ids(self):
        """
        Local intrinsic dimensionality estimates per sample.

        Returns
        -------
        dict
            Dictionary with per-sample ID vectors:
              • 'mle' → per-sample estimates from Maximum Likelihood Estimator (MLE).
              • 'fsa' → per-sample estimates from Fisher Separability Analysis (FSA).

        Raises
        ------
        AttributeError
            If local ID details are not available (e.g., `.fit(X)` has not been called).
        """
        out = {}
        det = getattr(self, "_id_details", {}).get(self.id_method, None)
        if det is not None:
            out[self.id_method] = det.get('local_id', (self.local_dimensionality or {}).get(self.id_method, None))
        if not out:
            raise AttributeError("Local ID details not available. Call .fit(X) first.")
        return out


    # --- Embedding getters (properties) ---
    @property
    def TopoMAP(self):
        """2D MAP layout computed on the DM refined graph (P_of_Z)."""
        key_std = 'MAP of ' + (self.graph_kernel_version + ' from DM with ' + str(self.base_kernel_version))
        key_uom = 'MAP of UoM DM with ' + str(self.base_kernel_version)
        if key_std in self.ProjectionDict:
            return self.ProjectionDict[key_std]
        if key_uom in self.ProjectionDict:  # back-compat (older runs)
            return self.ProjectionDict[key_uom]
        raise AttributeError("MAP embedding not available. Call .fit(X) first.")

    @property
    def msTopoMAP(self):
        """2D MAP layout computed on the msDM refined graph (P_of_msZ)."""
        key_std = 'MAP of ' + (self.graph_kernel_version + ' from msDM with ' + str(self.base_kernel_version))
        key_uom = 'MAP of UoM msDM with ' + str(self.base_kernel_version)
        if key_std in self.ProjectionDict:
            return self.ProjectionDict[key_std]
        if key_uom in self.ProjectionDict:
            return self.ProjectionDict[key_uom]
        raise AttributeError("msMAP embedding not available. Call .fit(X) first.")

    @property
    def TopoPaCMAP(self):
        """2D PaCMAP layout computed on the DM refined graph (P_of_Z)."""
        key_std = 'PaCMAP of ' + 'DM with ' + str(self.base_kernel_version)
        key_uom = 'PaCMAP of UoM DM with ' + str(self.base_kernel_version)
        if key_std in self.ProjectionDict:
            return self.ProjectionDict[key_std]
        if key_uom in self.ProjectionDict:
            return self.ProjectionDict[key_uom]
        raise AttributeError("PaCMAP embedding not available. Call .fit(X) first.")

    @property
    def msTopoPaCMAP(self):
        """2D PaCMAP layout computed on the msDM refined graph (P_of_msZ)."""
        key_std = 'PaCMAP of ' + 'msDM with ' + str(self.base_kernel_version)
        key_uom = 'PaCMAP of UoM msDM with ' + str(self.base_kernel_version)
        if key_std in self.ProjectionDict:
            return self.ProjectionDict[key_std]
        if key_uom in self.ProjectionDict:
            return self.ProjectionDict[key_uom]
        raise AttributeError("msPaCMAP embedding not available. Call .fit(X) first.")

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
        """
        Scree plot helper (calls topo.plot.decay_plot).

        Behavior
        --------
        - UoM enabled: plots one scree per disconnected component using the active mode
        (self._uom_active_mode, default 'msDM'). Titles include component index and size.
        - Non-UoM: behaves as before, plotting the selected/global eigenbasis.
        """
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError:
            return print('Error: Matplotlib not found!')
        from topo.plot import decay_plot

        # UoM path: per-component plots
        if getattr(self, "uom_enabled", False) and self.uom_eigenvalues_ms_list is not None:
            mode = getattr(self, "_uom_active_mode", "msDM")
            ev_lists = self.uom_eigenvalues_ms_list if mode == "msDM" else self.uom_eigenvalues_dm_list
            sizes = [int(ix.size) for ix in (self.uom_components_ or [])]

            figs = []
            for j, ev in enumerate(ev_lists):
                title = f"Component {j} (n={sizes[j]}) · {mode}"
                figs.append(decay_plot(evals=ev, title=title, **kwargs))
            return figs  # list of figure handles (or None if decay_plot handles display)

        # Non-UoM: keep original behavior
        if eigenbasis_key is not None:
            if isinstance(eigenbasis_key, str):
                if eigenbasis_key in self.EigenbasisDict.keys():
                    eigenbasis = self.EigenbasisDict[eigenbasis_key]
                else:
                    raise ValueError('Eigenbasis key not in TopOGraph.EigenbasisDict.')
        else:
            eigenbasis = self.eigenbasis
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
        In UoM mode, defaults to the UoM msZ operator if no graph is provided.
        """
        if graph is None:
            if self.uom_enabled:
                if self._kernel_msZ is None:
                    raise ValueError('No UoM msZ kernel computed. Call .fit() first.')
                graph = self._kernel_msZ.K
            else:
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
        gc.collect()
        return spt_layout

    def project(
        self,
        n_components=2,
        init=None,
        projection_method=None,
        landmarks=None,
        landmark_method='kmeans',
        n_neighbors=None,
        num_iters=300,
        multiscale=True,
        save_every=None,                 # int or None. If int>0, store Y every `save_every` epochs
        save_limit=None,                 # optional cap on number of snapshots kept in-memory
        save_callback=None,              # optional callable(epoch:int, Y:np.ndarray) -> None
        include_init_snapshot=True,      # store epoch=0 (post-init) snapshot
        **kwargs
    ):
        """
        Compute a 2D projection and store it in `ProjectionDict`.
        In UoM mode, graph-based methods use UoM block-diagonal affinities; coordinate-based
        methods use UoM concatenated scaffolds. This guarantees zero cross-component edges.

        Parameters
        ----------
        n_components : int (default 2)
            Number of output dimensions.
        init : np.ndarray or str (optional)
            Initial coordinates for layout optimization.
            If a string, must be a key in `ProjectionDict`.
            If None, spectral layout is used.
        projection_method : str (optional, default 'Isomap').
            Which projection method to use. Only 'Isomap', 't-SNE' and 'MAP' are implemented out of the box. 't-SNE' uses and 'MAP' relies
            on code that is adapted from UMAP. Current options are:
                * 'Isomap' - one of the first manifold learning methods
                * ['t-SNE'](https://github.com/DmitryUlyanov/Multicore-TSNE) - a classic manifold learning method
                * 'MAP'- a lighter [UMAP](https://umap-learn.readthedocs.io/en/latest/index.html) with looser assumptions
                * ['UMAP'](https://umap-learn.readthedocs.io/en/latest/index.html)
                * ['PaCMAP'](http://jmlr.org/papers/v22/20-1061.html) (Pairwise-controlled Manifold Approximation and Projection) - for balanced visualizations
                * ['TriMAP'](https://github.com/eamid/trimap) - dimensionality reduction using triplets
                * 'IsomorphicMDE' - [MDE](https://github.com/cvxgrp/pymde) with preservation of nearest neighbors
                * 'IsometricMDE' - [MDE](https://github.com/cvxgrp/pymde) with preservation of pairwise distances
                * 'NCVis' - [Noise Contrastive Visualization](https://github.com/stat-ml/ncvis) - a UMAP-like method with blazing fast performance
            These are frankly quite direct to add, so feel free to make a feature request if your favorite method is not listed here.
        landmarks : int or np.ndarray (optional)
            Number of landmarks or indices of landmark samples.
            If None, no landmarks are used.
        landmark_method : str (default 'kmeans')
            Landmark selection method (if `landmarks` is an int).
            One of {'random', 'kmeans').
        n_neighbors : int (optional)
            Number of neighbors for graph-based methods.
            If None, uses `self.graph_knn`.
        num_iters : int (default 300)
            Number of optimization epochs for layout optimization.
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
            projection_method = self.projection_methods[0]

        # choose which refined graph / scaffold to use
        if projection_method in ['MAP', 'IsomorphicMDE', 'IsometricMDE', 'Isomap']:
            metric = 'precomputed'
            input_mat = self.P_of_msZ if multiscale else self.P_of_Z
            tag = 'msDM' if multiscale else 'DM'
            # Standardize keys even in UoM mode (no "UoM" prefix)
            key = f"{self.graph_kernel_version} from {tag} with {self.base_kernel_version}"
        else:
            metric = self.graph_metric
            tag = 'msDM' if multiscale else 'DM'
            if self.uom_enabled:
                input_mat = self.msZ_uom if multiscale else self.Z_uom
            else:
                eig_key = f"{tag} with {self.base_kernel_version}"
                input_mat = self.EigenbasisDict[eig_key].transform(X=None)
            # Standardized key (no "UoM" prefix)
            key = f"{tag} with {self.base_kernel_version}"

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

        result = proj.fit_transform(input_mat, **kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            Y, Y_aux = result
        else:
            Y, Y_aux = result, None

        self.runtimes[projection_key] = time.time() - t0
        if self.verbosity >= 1:
            print(f' Computed {projection_method} ({ "msZ" if multiscale else "Z/DM" }{" [UoM]" if self.uom_enabled else ""}) in {self.runtimes[projection_key]:.3f} sec')

        self.ProjectionDict[projection_key] = Y

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
        P = self._select_P_operator(which)
        # make CSR view
        if sp.issparse(X):
            Xc = X.tocsr(copy=True).astype(dtype)
            for _ in range(int(t)):
                Xc = P @ Xc
            if output in ('auto', 'sparse'):
                return Xc
            return Xc.toarray()
        else:
            Xd = np.asarray(X, dtype=dtype)
            for _ in range(int(t)):
                Xd = P @ Xd
            if output in ('auto', 'dense'):
                return Xd
            # convert to sparse CSR
            return sp.csr_matrix(Xd)

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



