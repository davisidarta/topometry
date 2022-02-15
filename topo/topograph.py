# TopOMetry high-level models API
# Author: Davi Sidarta-Oliveira <davisidarta(at)gmail(dot)com>
# School of Medical Sciences, University of Campinas, Brazil
#
import time
import numpy as np
from sklearn.base import TransformerMixin
from scipy.sparse import issparse
import topo.spectral._spectral as spt
from topo.base.ann import kNN
from topo.layouts.graph_utils import fuzzy_simplicial_set_ann
from topo.tpgraph.cknn import cknn_graph
from topo.tpgraph.diffusion import Diffusor
from topo.tpgraph.multiscale import decay_plot

try:
    from typing import Literal
except ImportError:
    try:
        from typing_extensions import Literal
    except ImportError:

        class LiteralMeta(type):
            def __getitem__(cls, values):
                if not isinstance(values, tuple):
                    values = (values,)
                return type('Literal_', (Literal,), dict(__args__=values))


        class Literal(metaclass=LiteralMeta):
            pass

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


class TopOGraph(TransformerMixin):
    """

     Main TopOMetry class for learning topological similarities, bases, graphs, and layouts from high-dimensional data.

     From data, learns topological similarity metrics, from these build orthogonal bases and from these bases learns
     topological graphs. Users can choose
     different models to achieve these topological representations, combinining either diffusion harmonics,
     continuous k-nearest-neighbors or fuzzy simplicial sets to approximate the Laplace-Beltrami Operator.
     The topological graphs can then be visualized with multiple existing layout optimization tools.

    Parameters
    ----------
    base_knn : int (optional, default 10).
         Number of k-nearest-neighbors to use when learning topological similarities.
         Consider this as a calculus discretization threshold (i.e. approaches zero in the limit of large data).
         For practical purposes, the minimum amount of samples one would
         expect to constitute a neighborhood of its own. Increasing `k` can generate more globally-comprehensive metrics
         and maps, to a certain extend, however at the expense of fine-grained resolution. In practice, the default
         value of 10 performs quite well for almost all cases.

    graph_knn : int (optional, default 10).
         Similar to `base_knn`, but used to learning topological graphs from the orthogonal bases.

    n_eigs : int (optional, default 100).
         Number of components to compute. This number can be iterated to get different views
         from data at distinct spectral resolutions. If `basis` is set to `diffusion`, this is the number of 
         computed diffusion components. If `basis` is set to `continuous` or `fuzzy`, this is the number of
         computed eigenvectors of the Laplacian Eigenmaps from the learned topological similarities.

    basis : 'diffusion', 'continuous' or 'fuzzy' (optional, default 'diffusion').
         Which topological basis model to learn from data. If `diffusion`, performs an optimized, anisotropic, adaptive
         diffusion mapping (default). If `continuous`, computes affinities from continuous k-nearest-neighbors, and a 
         topological basis with Laplacian Eigenmaps. If `fuzzy`, computes affinities using
         fuzzy simplicial sets, and a topological basis with Laplacian Eigenmaps.

    graph : 'diff', 'cknn' or 'fuzzy' (optional, default 'diff').
         Which topological graph model to learn from the built basis. If 'diff', uses a second-order diffusion process
         to learn similarities and transition probabilities. If 'cknn', uses the continuous k-nearest-neighbors
         algorithm. If 'fuzzy', builds a fuzzy simplicial set graph from the active basis. All these
         algorithms learn graph-oriented topological metrics from the learned basis.

    backend : str 'hnwslib', 'nmslib' or 'sklearn' (optional, default 'nmslib').
         Which backend to use to compute nearest-neighbors. Options for fast, approximate nearest-neighbors
         are 'hnwslib'  and 'nmslib' (default). For exact nearest-neighbors, use 'sklearn'.

         * If using 'nmslib', a sparse
         csr_matrix input is expected. If using 'hnwslib' or 'sklearn', a dense array is expected.

         * I strongly recommend you use 'hnswlib' if handling with somewhat dense, array-shaped data. If the data
         is relatively sparse, you should use 'nmslib', which operates on sparse matrices by default on
         TopOMetry and will automatically convert the input array to csr_matrix for performance.



    base_metric : str (optional, default 'cosine')
        Distance metric for building an approximate kNN graph during topological basis construction. Defaults to
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



    graph_metric : str (optional, default 'cosine').
         Similar to `base_metric`, but used for building the topological graph.

    p : int or float (optional, default 11/16 ).
         P for the Lp metric, when `metric='lp'`.  Can be fractional. The default 11/16 approximates 2/3, that is,
         an astroid norm with some computational efficiency (2^n bases are less painstakinly slow to compute).

    n_jobs : int (optional, default 10).
         Number of threads to use in calculations. Set this to as much as possible for speed.

    M : int (optional, default 15).
        A neighborhood search parameter. Defines the maximum number of neighbors in the zero and above-zero layers
        during HSNW (Hierarchical Navigable Small World Graph). However, the actual default maximum number
        of neighbors for the zero layer is 2*M.  A reasonable range for this parameter
        is 5-100. For more information on HSNW, please check its manuscript(https://arxiv.org/abs/1603.09320).
        HSNW is implemented in python via NMSlib (https://github.com/nmslib/nmslib) and HNWSlib
        (https://github.com/nmslib/hnswlib).

    efC : int (optional, default 50).
        A neighborhood search parameter. Increasing this value improves the quality of a constructed graph
        and leads to higher accuracy of search. However this also leads to longer indexing times.
        A reasonable range for this parameter is 50-2000.

    efS : int (optional, default 50).
        A neighborhood search parameter. Similarly to efC, increasing this value improves recall at the
        expense of longer retrieval time. A reasonable range for this parameter is 100-2000.

    transitions : bool (optional, default False).
        A diffusion harmonics parameter. Whether to use the transition probabilities rather than
        the diffusion potential when computing the diffusion harmonics model.

    alpha : int or float (optional, default 1).
         A diffusion harmonics parameter. Alpha in the diffusion maps literature. Controls how much the results are biased by data distribution.
         Defaults to 1, which unbiases results from data underlying samplg distribution.

    kernel_use : str (optional, default 'simple')
         A diffusion harmonics parameter. Which type of kernel to use in the diffusion harmonics model. There are four
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

    delta : float (optional, default 1.0).
        A CkNN parameter to decide the radius for each points. The combination
        radius increases in proportion to this parameter.

    t : 'inf' or float or int, optional, default='inf'
        A CkNN parameter encoding the decay of the heat kernel. The weights are calculated as:
        W_{ij} = exp(-(||x_{i}-x_{j}||^2)/t)

    verbosity : int (optional, default 1).
        Controls verbosity. 0 for no verbosity, 1 for minimal (prints warnings and runtimes of major steps), 2 for
        medium (also prints layout optimization messages) and 3 for full (down to neighborhood search, useful for debugging).

    cache_base : bool (optional, default True).
        Whether to cache intermediate matrices used in computing orthogonal bases
        (k-nearest-neighbors, diffusion harmonics etc).

    cache_graph : bool (optional, default True).
        Whether to cache intermediate matrices used in computing topological graphs
        (k-nearest-neighbors, diffusion harmonics etc).

    plot_spectrum : bool (optional, default False).
        Whether to plot the informational decay spectrum obtained during eigendecomposition of similarity matrices.

    eigen_expansion : bool (optional, default False).
        Whether to try to find a discrete eigengap during eigendecomposition. This can *severely* impact runtime,
        as it can take numerous eigendecompositions to do so.

    """

    def __init__(self,
                 base_knn=10,
                 graph_knn=10,
                 n_eigs=100,
                 basis='diffusion',
                 graph='diff',
                 base_metric='cosine',
                 graph_metric='cosine',
                 n_jobs=1,
                 backend='nmslib',
                 M=15,
                 efC=50,
                 efS=50,
                 verbosity=1,
                 cache_base=True,
                 cache_graph=True,
                 kernel_use='decay',
                 alpha=1,
                 plot_spectrum=False,
                 eigen_expansion=False,
                 delta=1.0,
                 t='inf',
                 p=11 / 16,
                 transitions=True,
                 random_state=None
                 ):
        self.graph = graph
        self.basis = basis
        self.n_eigs = n_eigs
        self.base_knn = base_knn
        self.graph_knn = graph_knn
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.backend = backend
        self.base_metric = base_metric
        self.graph_metric = graph_metric
        self.p = p
        self.M = M
        self.efC = efC
        self.efS = efS
        self.kernel_use = kernel_use
        self.norm = False
        self.transitions = transitions
        self.eigen_expansion = eigen_expansion
        self.verbosity = verbosity
        self.bases_graph_verbose = False
        self.layout_verbose = False
        self.plot_spectrum = plot_spectrum
        self.delta = delta
        self.t = t
        self.cache_base = cache_base
        self.cache_graph = cache_graph
        self.random_state = random_state
        self.base_nbrs_class = None
        self.base_knn_graph = None
        self.graph_nbrs_class = None
        self.graph_knn_graph = None
        self.DiffBasis = None
        self.computed_LapGraph = False
        self.MSDiffMap = None
        self.MSDiffMap_evals = None
        self.ContBasis = None
        self.CLapMap = None
        self.CLapMap_evals = None
        self.FuzzyBasis = None
        self.FuzzyLapMap = None
        self.FuzzyLapMap_evals = None
        self.n = None
        self.m = None
        self.MDE_problem = None
        self.SpecLayout = None
        self.db_fuzzy_graph = None
        self.db_cknn_graph = None
        self.db_diff_graph = None
        self.fb_fuzzy_graph = None
        self.fb_cknn_graph = None
        self.fb_diff_graph = None
        self.cb_fuzzy_graph = None
        self.cb_cknn_graph = None
        self.cb_diff_graph = None
        self.db_fuzzy_clusters = None
        self.db_cknn_clusters = None
        self.db_diff_clusters = None
        self.fb_fuzzy_clusters = None
        self.fb_cknn_clusters = None
        self.fb_diff_clusters = None
        self.cb_fuzzy_clusters = None
        self.cb_cknn_clusters = None
        self.cb_diff_clusters = None
        self.db_diff_MAP = None
        self.db_fuzzy_MAP = None
        self.db_cknn_MAP = None
        self.db_diff_MDE = None
        self.db_fuzzy_MDE = None
        self.db_cknn_MDE = None
        self.db_PaCMAP = None
        self.db_TriMAP = None
        self.db_tSNE = None
        self.fb_diff_MAP = None
        self.fb_fuzzy_MAP = None
        self.fb_cknn_MAP = None
        self.fb_diff_MDE = None
        self.fb_fuzzy_MDE = None
        self.fb_cknn_MDE = None
        self.fb_PaCMAP = None
        self.fb_TriMAP = None
        self.fb_tSNE = None
        self.cb_diff_MAP = None
        self.cb_fuzzy_MAP = None
        self.cb_cknn_MAP = None
        self.cb_diff_MDE = None
        self.cb_fuzzy_MDE = None
        self.cb_cknn_MDE = None
        self.cb_PaCMAP = None
        self.cb_TriMAP = None
        self.cb_tSNE = None
        self.db_TriMAP = None
        self.cb_TriMAP = None
        self.fb_TriMAP = None
        self.db_NCVis = None
        self.cb_NCVis = None
        self.fb_NCVis = None
        self.runtimes = {}

    def __repr__(self):
        if (self.n is not None) and (self.m is not None):
            msg = "TopOGraph object with %i samples and %i observations" % (self.n, self.m) + " and:"
        else:
            msg = "TopOGraph object without any fitted data."
        msg = msg + "\n . Orthogonal bases:"
        if self.MSDiffMap is not None:
            msg = msg + " \n .. Multiscale Diffusion Maps fitted - .MSDiffMap"
            msg = msg + " \n    With similarity metrics stored at - .DiffBasis.K and .DiffBasis.T"
            if (self.db_PaCMAP is not None) or (self.db_TriMAP is not None) or \
                    (self.db_tSNE is not None) or (self.db_NCVis is not None):
                msg = msg + " \n    With layouts:"
            if self.db_PaCMAP is not None:
                msg = msg + " \n         PaCMAP - .db_PaCMAP"
            if self.db_TriMAP is not None:
                msg = msg + " \n         TriMAP - .db_TriMAP"
            if self.db_tSNE is not None:
                msg = msg + " \n         tSNE - .db_tSNE"
            if self.db_NCVis is not None:
                msg = msg + " \n         NCVis - .db_NCVis"
            msg = msg + "\n     And the downstream topological graphs:"
            if self.db_diff_graph is not None:
                msg = msg + " \n       Diffusion graph - .db_diff_graph"
                msg = msg + "\n          Graph layouts:"
                if self.db_diff_MAP is not None:
                    msg = msg + " \n         MAP - .db_diff_MAP"
                if self.db_diff_MDE is not None:
                    msg = msg + " \n         MDE - .db_diff_MDE"
                if (self.db_diff_MAP is None) and (self.db_diff_MDE is None):
                    msg = msg + " none fitted."
            if self.db_fuzzy_graph is not None:
                msg = msg + " \n       Fuzzy graph - .db_fuzzy_graph"
                msg = msg + " \n         Graph layouts:"
                if self.db_fuzzy_MAP is not None:
                    msg = msg + " \n         MAP - .db_fuzzy_MAP"
                if self.db_fuzzy_MDE is not None:
                    msg = msg + " \n         MDE - .db_fuzzy_MDE"
                if (self.db_fuzzy_MAP is None) and (self.db_fuzzy_MDE is None):
                    msg = msg + " none fitted."
            if self.db_cknn_graph is not None:
                msg = msg + " \n       CkNN graph - .db_cknn_graph"
                msg = msg + " \n         Graph layouts:"
                if self.db_cknn_MAP is not None:
                    msg = msg + " \n         MAP - .db_cknn_MAP"
                if self.db_cknn_MDE is not None:
                    msg = msg + " \n         MDE - .db_cknn_MDE"
                if (self.db_cknn_MAP is None) and (self.db_cknn_MDE is None):
                    msg = msg + " none fitted."
            if (self.db_diff_graph is None) and (self.db_fuzzy_graph is None) and (self.db_cknn_graph is None):
                msg = msg + " none fitted."
            msg = msg + "\n"

        if self.CLapMap is not None:
            msg = msg + " \n .. Continuous (CkNN) Laplacian Eigenmaps fitted - .CLapMap"
            msg = msg + " \n    With similarity metrics stored at - .ContBasis"
            if (self.cb_PaCMAP is not None) or (self.cb_TriMAP is not None) or \
                    (self.cb_tSNE is not None) or (self.cb_NCVis is not None):
                msg = msg + " \n    With layouts:"
            if self.cb_PaCMAP is not None:
                msg = msg + " \n         PaCMAP - .cb_PaCMAP"
            if self.cb_TriMAP is not None:
                msg = msg + " \n         TriMAP - .cb_TriMAP"
            if self.cb_tSNE is not None:
                msg = msg + " \n         tSNE - .cb_tSNE"
            if self.cb_NCVis is not None:
                msg = msg + " \n         NCVis - .cb_NCVis"
            msg = msg + "\n     And the downstream topological graphs:"
            if self.cb_diff_graph is not None:
                msg = msg + " \n      Diffusion graph - .cb_diff_graph"
                msg = msg + " \n        Graph layouts:"
                if self.cb_diff_MAP is not None:
                    msg = msg + " \n         MAP - .cb_diff_MAP"
                if self.cb_diff_MDE is not None:
                    msg = msg + " \n         MDE - .cb_diff_MDE"
                if (self.cb_diff_MAP is None) and (self.cb_diff_MDE is None):
                    msg = msg + " none fitted."
            if self.cb_fuzzy_graph is not None:
                msg = msg + " \n      Fuzzy graph - .cb_fuzzy_graph"
                msg = msg + " \n        Graph layouts:"
                if self.cb_fuzzy_MAP is not None:
                    msg = msg + " \n         MAP - .cb_fuzzy_MAP"
                if self.cb_fuzzy_MDE is not None:
                    msg = msg + " \n         MDE - .cb_fuzzy_MDE"
                if (self.cb_fuzzy_MAP is None) and (self.cb_fuzzy_MDE is None):
                    msg = msg + " none fitted."
            if self.cb_cknn_graph is not None:
                msg = msg + " \n      CkNN graph - .cb_cknn_graph"
                msg = msg + " \n        Graph layouts:"
                if self.cb_cknn_MAP is not None:
                    msg = msg + " \n         MAP - .cb_cknn_MAP"
                if self.cb_cknn_MDE is not None:
                    msg = msg + " \n         MDE - .cb_cknn_MDE"
                if (self.cb_cknn_MAP is None) and (self.cb_cknn_MDE is None):
                    msg = msg + " none fitted."
            if (self.cb_fuzzy_graph is None) and (self.cb_cknn_graph is None):
                msg = msg + " none fitted."
            msg = msg + "\n"

        if self.FuzzyLapMap is not None:
            msg = msg + "\n .. Fuzzy (simplicial sets) Laplacian Eigenmaps fitted - .FuzzyLapMap"
            msg = msg + "\n    With similarity metrics stored at - .FuzzyBasis"
            if (self.fb_PaCMAP is not None) or (self.fb_TriMAP is not None) or \
                    (self.fb_tSNE is not None) or (self.fb_NCVis is not None):
                msg = msg + " \n    With layouts:"
            if self.fb_PaCMAP is not None:
                msg = msg + " \n         PaCMAP - .fb_PaCMAP"
            if self.fb_TriMAP is not None:
                msg = msg + " \n         TriMAP - .fb_TriMAP"
            if self.fb_tSNE is not None:
                msg = msg + " \n         tSNE - .fb_tSNE"
            if self.fb_NCVis is not None:
                msg = msg + " \n         NCVis - .fb_NCVis"
            msg = msg + "\n     And the downstream topological graphs:"
            if self.fb_diff_graph is not None:
                msg = msg + " \n      Diffusion graph - .fb_diff_graph"
                msg = msg + " \n        Graph layouts:"
                if self.fb_diff_MAP is not None:
                    msg = msg + " \n         MAP - .fb_diff_MAP"
                if self.fb_diff_MDE is not None:
                    msg = msg + " \n         MDE - .fb_diff_MDE"
                if (self.fb_diff_MAP is None) and (self.fb_diff_MDE is None):
                    msg = msg + " none fitted."
            if self.fb_fuzzy_graph is not None:
                msg = msg + " \n      Fuzzy graph - .fb_fuzzy_graph"
                msg = msg + " \n        Graph layouts:"
                if self.fb_fuzzy_MAP is not None:
                    msg = msg + " \n         MAP - .fb_fuzzy_MAP"
                if self.fb_fuzzy_MDE is not None:
                    msg = msg + " \n         MDE - .fb_fuzzy_MDE"
                if (self.fb_fuzzy_MAP is None) and (self.fb_fuzzy_MDE is None):
                    msg = msg + " none fitted."
            if self.fb_cknn_graph is not None:
                msg = msg + " \n      CkNN graph - .fb_cknn_graph"
                msg = msg + " \n        Graph layouts:"
                if self.fb_cknn_MAP is not None:
                    msg = msg + " \n         MAP - .fb_cknn_MAP"
                if self.fb_cknn_MDE is not None:
                    msg = msg + " \n         MDE - .fb_cknn_MDE"
                if (self.fb_cknn_MAP is None) and (self.fb_cknn_MDE is None):
                    msg = msg + " none fitted."
            if (self.fb_fuzzy_graph is None) and (self.fb_cknn_graph is None):
                msg = msg + " none fitted."
            msg = msg + "\n"

        if (self.MSDiffMap is None) and (self.CLapMap is None) and (self.FuzzyLapMap is None):
            msg = msg + " none fitted."
        msg = msg + " \n "
        msg = msg + " \n Active basis: " + str(self.basis) + ' basis.'
        msg = msg + " \n Active graph: " + str(self.graph) + ' graph.'
        return msg

    def fit(self, X):
        """
        Learn topological distances with diffusion harmonics and continuous metrics. Computes affinity operators
        that approximate the Laplace-Beltrami operator

        Parameters
        ----------
        X :
            High-dimensional data matrix. Currently, supports only data from similar type (i.e. all bool, all float)

        Returns
        -------

        TopoGraph instance with several slots, populated as per user settings.
        If `basis='diffusion'`, populates `TopoGraph.MSDiffMap` with a multiscale diffusion mapping of data, and
                `TopoGraph.DiffBasis` with a fitted `topo.tpgraph.diff.Diffusor()` class containing diffusion metrics
                and transition probabilities, respectively stored in TopoGraph.DiffBasis.K and TopoGraph.DiffBasis.T

        If `basis='continuous'`, populates `TopoGraph.CLapMap` with a continous Laplacian Eigenmapping of data, and
                `TopoGraph.ContBasis` with a continuous-k-nearest-neighbors model, containing continuous metrics and
                adjacency, respectively stored in `TopoGraph.ContBasis.K` and `TopoGraph.ContBasis.A`.

        If `basis='fuzzy'`, populates `TopoGraph.FuzzyLapMap` with a fuzzy Laplacian Eigenmapping of data, and
                `TopoGraph.FuzzyBasis` with a fuzzy simplicial set model, containing continuous metrics.

        """
        if self.verbosity >= 2:
            self.layout_verbose = True
            if self.verbosity == 3:
                self.bases_graph_verbose = True
            else:
                self.bases_graph_verbose = False
        else:
            self.layout_verbose = False

        if self.backend == 'hnswlib':
            if not _have_hnswlib:
                if _have_nmslib:
                    self.backend == 'nmslib'
                else:
                    self.backend == 'sklearn'
        if self.backend == 'nmslib':
            if not _have_nmslib:
                if _have_hnswlib:
                    self.backend == 'hnswlib'
                else:
                    self.backend == 'sklearn'

        self.n = X.shape[0]
        self.m = X.shape[1]
        if self.random_state is None:
            self.random_state = np.random.RandomState()

        # First build a kNN graph:
        if self.kernel_use == 'simple' or self.kernel_use == 'decay':
            if self.basis == 'diffusion':
                if self.base_knn_graph is None:
                    if self.verbosity >= 1:
                        print('Computing neighborhood graph...')
                    start = time.time()
                    self.base_nbrs_class, self.base_knn_graph = kNN(X, n_neighbors=self.base_knn,
                                                                    metric=self.base_metric,
                                                                    n_jobs=self.n_jobs,
                                                                    backend=self.backend,
                                                                    M=self.M,
                                                                    efC=self.efC,
                                                                    efS=self.efS,
                                                                    return_instance=True,
                                                                    verbose=self.bases_graph_verbose)
                    end = time.time()
                    self.runtimes['kNN'] = end - start
                    if self.verbosity >= 1:
                        print(
                            ' Base kNN graph computed in %f (sec)' % (
                                    end - start))

        if self.verbosity >= 1:
            print('Building topological basis...' + 'using ' + str(self.basis) + ' model.')
        if self.basis == 'diffusion':
            self.DiffBasis = Diffusor(n_eigs=self.n_eigs,
                                              n_neighbors=self.base_knn,
                                              alpha=self.alpha,
                                              n_jobs=self.n_jobs,
                                              backend=self.backend,
                                              metric=self.base_metric,
                                              p=self.p,
                                              M=self.M,
                                              efC=self.efC,
                                              efS=self.efS,
                                              kernel_use=self.kernel_use,
                                              norm=self.norm,
                                              transitions=self.transitions,
                                              eigen_expansion=self.eigen_expansion,
                                              verbose=self.bases_graph_verbose,
                                              plot_spectrum=False,
                                              cache=self.cache_base)
            if self.kernel_use == 'simple' or self.kernel_use == 'decay':
                self.DiffBasis.metric = 'precomputed'
            start = time.time()
            self.MSDiffMap = self.DiffBasis.fit_transform(self.base_knn_graph)
            self.MSDiffMap_evals = self.DiffBasis.res['EigenValues']
            end = time.time()
            self.runtimes['DB'] = end - start
            if self.verbosity >= 1:
                print(' Topological basis fitted with multiscale self-adaptive diffusion maps in %f (sec)' %
                      (end - start))

        elif self.basis == 'continuous':
            start = time.time()
            # # Enforce symmetry
            # knn = self.base_knn_graph.toarray()
            # knn[(np.arange(knn.shape[0]), np.arange(knn.shape[0]))] = 0
            # knn = (knn + knn.T) / 2
            if issparse(X):
                data_use = X.toarray()
            else:
                data_use = X
            self.ContBasis = cknn_graph(data_use,
                                        n_neighbors=self.base_knn,
                                        delta=self.delta,
                                        metric=self.base_metric,
                                        t=self.t,
                                        include_self=True,
                                        is_sparse=True,
                                        return_instance=False
                                        )
            del data_use
            self.CLapMap, self.CLapMap_evals = spt.LapEigenmap(
                self.ContBasis,
                self.n_eigs,
                norm_laplacian=True,
                return_evals=True
            )
            end = time.time()
            self.runtimes['CB'] = end - start
            self.CLapMap_evals = -self.CLapMap_evals
            if self.verbosity >= 1:
                print(
                    ' Topological basis fitted with Laplacian Eigenmaps from Continuous-k-Nearest-Neighbors in %f (sec)' % (
                            end - start))

        elif self.basis == 'fuzzy':
            # some tweaks to use the precomputed distances
            if self.base_knn_graph is not None:
                knn_inds, knn_distances, grad, knn_graph = self.base_nbrs_class.ind_dist_grad(X)
                start = time.time()
                fuzzy_results = fuzzy_simplicial_set_ann(X,
                                                         n_neighbors=self.base_knn,
                                                         knn_indices=knn_inds,  # I can still improve the performance of this
                                                         knn_dists=knn_distances,    # FIXME
                                                         backend=self.backend,
                                                         metric=self.base_metric,
                                                         n_jobs=self.n_jobs,
                                                         efC=self.efC,
                                                         efS=self.efS,
                                                         M=self.M,
                                                         set_op_mix_ratio=1.0,
                                                         local_connectivity=1.0,
                                                         apply_set_operations=True,
                                                         return_dists=False,
                                                         verbose=self.bases_graph_verbose)
            else:
                start = time.time()
                fuzzy_results = fuzzy_simplicial_set_ann(X,
                                                         n_neighbors=self.base_knn,
                                                         knn_indices=None,  # I can still improve the performance of this
                                                         knn_dists=None,    # FIXME
                                                         backend=self.backend,
                                                         metric=self.base_metric,
                                                         n_jobs=self.n_jobs,
                                                         efC=self.efC,
                                                         efS=self.efS,
                                                         M=self.M,
                                                         set_op_mix_ratio=1.0,
                                                         local_connectivity=1.0,
                                                         apply_set_operations=True,
                                                         return_dists=False,
                                                         verbose=self.bases_graph_verbose)
            self.FuzzyBasis = fuzzy_results[0]

            self.FuzzyLapMap, self.FuzzyLapMap_evals = spt.LapEigenmap(
                self.FuzzyBasis,
                self.n_eigs,
                norm_laplacian=True,
                return_evals=True
            )
            end = time.time()
            self.runtimes['FB'] = end - start
            self.FuzzyLapMap_evals = -self.FuzzyLapMap_evals
            if self.verbosity >= 1:
                print(' Topological basis fitted with Laplacian Eigenmaps from fuzzy simplicial sets in %f (sec)' % (
                        end - start))

        else:
            return print(
                ' Error: \'basis\' must be either \'diffusion\', \'continuous\' or \'fuzzy\'! Returning empty TopOGraph.')

        if self.plot_spectrum:
            self.scree_plot()

        return self

    def scree_plot(self, basis=None, use_eigs='knee', verbose=False):
        """
        Visualize the scree plot of information entropy.

        Parameters
        ----------
        `basis` : str (optional, default None).
            If `None`, will use the default basis. Otherwise, uses the specified basis
            (must be 'diffusion', 'continuous' or 'fuzzy').

        `use_eigs` : int or str (optional, default 'knee').
            Number of eigenvectors to use. If 'max', expands to the maximum number of positive eigenvalues
            (reach of numerical precision), else to the maximum amount of computed components.
            If 'knee', uses Kneedle to find an optimal cutoff point, and expands it by ``expansion``.
            If 'comp_gap', tries to find a discrete eigengap from the computation process.

        verbose : bool (optional, default False).
            Controls verbosity

        Returns
        -------
        A nice plot.

        """
        if basis is not None:
            if isinstance(basis, str):
                self.basis = basis
        if self.basis == 'diffusion':
            return self.DiffBasis.spectrum_plot()
        else:
            if self.basis == 'continuous':
                use_evals = self.CLapMap_evals
                curve = 'concave'
            elif self.basis == 'fuzzy':
                use_evals = self.FuzzyLapMap_evals
                curve = 'concave'
            else:
                return print(
                    'Error: No computed basis available!')
            return decay_plot(evals=use_evals, curve=curve, verbose=verbose)


    def transform(self, basis=None):
        """
        Learns new affinity, topological operators from chosen basis.

        Parameters
        ----------
        self :
            TopOGraph instance.

        basis : str, optional.
            Base to use when building the topological graph. Defaults to the active base ( `TopOGraph.basis`).
            Setting this updates the active base.


        Returns
        -------
        scipy.sparse.csr.csr_matrix, containing the similarity matrix that encodes the topological graph.

        """
        if basis is not None:
            if isinstance(basis, str):
                self.basis = basis
            elif isinstance(basis, np.ndarray):
                if self.verbosity >= 1:
                    print('    Using provided data...')
        if self.verbosity >= 1:
            print('    Building topological graph...')
        if isinstance(basis, np.ndarray):
            use_basis = basis
        else:
            if self.basis == 'continuous':
                use_basis = self.CLapMap
            elif self.basis == 'diffusion':
                use_basis = self.MSDiffMap
            elif self.basis == 'fuzzy':
                use_basis = self.FuzzyLapMap
            else:
                return print(
                    'Error: No computed basis available! Compute a topological basis before fitting a topological graph.')

        if self.graph == 'diff':
            start = time.time()
            DiffGraph = Diffusor(n_neighbors=self.graph_knn,
                                 alpha=self.alpha,
                                 n_jobs=self.n_jobs,
                                 backend=self.backend,
                                 metric=self.graph_metric,
                                 p=self.p,
                                 M=self.M,
                                 efC=self.efC,
                                 efS=self.efS,
                                 kernel_use=self.kernel_use,
                                 norm=self.norm,
                                 transitions=self.transitions,
                                 eigen_expansion=self.eigen_expansion,
                                 verbose=self.bases_graph_verbose,
                                 plot_spectrum=self.plot_spectrum,
                                 cache=False
                                 ).fit(use_basis)
            end = time.time()
            if self.basis == 'diffusion':
                self.runtimes['db_diff_graph'] = end - start
            if self.basis == 'fuzzy':
                self.runtimes['fb_diff_graph'] = end - start
            if self.basis == 'continuous':
                self.runtimes['fb_diff_graph'] = end - start
            if self.cache_graph:
                if self.basis == 'diffusion':
                    if self.transitions:
                        self.db_diff_graph = DiffGraph.T
                    else:
                        self.db_diff_graph = DiffGraph.K
                if self.basis == 'continuous':
                    if self.transitions:
                        self.cb_diff_graph = DiffGraph.T
                    else:
                        self.cb_diff_graph = DiffGraph.K
                if self.basis == 'fuzzy':
                    if self.transitions:
                        self.fb_diff_graph = DiffGraph.T
                    else:
                        self.fb_diff_graph = DiffGraph.K


        elif self.graph == 'cknn':
            start = time.time()
            base_nbrs_class, graph_knn_graph = kNN(use_basis, n_neighbors=self.graph_knn,
                                                            metric=self.graph_metric,
                                                            n_jobs=self.n_jobs,
                                                            backend=self.backend,
                                                            M=self.M,
                                                            efC=self.efC,
                                                            efS=self.efS,
                                                            return_instance=True,
                                                            verbose=self.bases_graph_verbose)
            # Enforce symmetry
            knn = graph_knn_graph.toarray()
            knn[(np.arange(knn.shape[0]), np.arange(knn.shape[0]))] = 0
            knn = (knn + knn.T) / 2
            CknnGraph = cknn_graph(knn,
                                   n_neighbors=self.graph_knn,
                                   delta=self.delta,
                                   metric='precomputed',
                                   t=self.t,
                                   include_self=True,
                                   is_sparse=True)
            end = time.time()
            if self.basis == 'diffusion':
                self.runtimes['db_cknn_graph'] = end - start
            if self.basis == 'fuzzy':
                self.runtimes['cb_cknn_graph'] = end - start
            if self.basis == 'continuous':
                self.runtimes['fb_cknn_graph'] = end - start
            if self.cache_graph:
                if self.basis == 'diffusion':
                    self.db_cknn_graph = CknnGraph
                if self.basis == 'continuous':
                    self.cb_cknn_graph = CknnGraph
                if self.basis == 'fuzzy':
                    self.fb_cknn_graph = CknnGraph

        elif self.graph == 'fuzzy':
            start = time.time()
            FuzzyGraph = fuzzy_simplicial_set_ann(use_basis,
                                                  n_neighbors=self.graph_knn,
                                                  backend=self.backend,
                                                  metric=self.graph_metric,
                                                  n_jobs=self.n_jobs,
                                                  efC=self.efC,
                                                  efS=self.efS,
                                                  M=self.M,
                                                  set_op_mix_ratio=1.0,
                                                  local_connectivity=1.0,
                                                  apply_set_operations=True,
                                                  return_dists=False,
                                                  verbose=self.bases_graph_verbose)
            end = time.time()
            if self.basis == 'diffusion':
                self.runtimes['db_fuzzy_graph'] = end - start
            if self.basis == 'fuzzy':
                self.runtimes['cb_fuzzy_graph'] = end - start
            if self.basis == 'continuous':
                self.runtimes['fb_fuzzy_graph'] = end - start
            FuzzyGraph = FuzzyGraph[0]
            if self.cache_graph:
                if self.basis == 'diffusion':
                    self.db_fuzzy_graph = FuzzyGraph
                if self.basis == 'continuous':
                    self.cb_fuzzy_graph = FuzzyGraph
                if self.basis == 'fuzzy':
                    self.fb_fuzzy_graph = FuzzyGraph
        else:
            return print('Error: \'graph\' must be \'diff\', \'cknn\' or \'fuzzy\'!')

        if self.verbosity >= 1:
            print('     Topological `' + str(self.graph) + '` graph extracted in = %f (sec)' % (end - start))
        if self.graph == 'diff':
            return DiffGraph.T
        elif self.graph == 'cknn':
            return CknnGraph
        elif self.graph == 'fuzzy':
            return FuzzyGraph
        else:
            return self

    def spectral_layout(self, graph=None, n_components=2, cache=True):
        """

        Performs a multicomponent spectral layout of the data and the target similarity matrix.

        Parameters
        ----------
        graph : scipy.sparse.csr.csr_matrix.
            affinity matrix (i.e. topological graph). If None (default), uses the default graph from the default basis.
        n_components : int (optional, default 2).
            number of dimensions to embed into.
        cache : bool (optional, default True).
            Whether to cache the embedding to the `TopOGraph` object.
        Returns
        -------
        np.ndarray containing the resulting embedding.

        """
        if graph is None:
            if self.basis == 'diffusion':
                if self.MSDiffMap is None:
                    return print('Error: Basis set to \'diffusion\', but the diffusion basis is not computed!')
                else:
                    data = self.MSDiffMap
                if graph is None:
                    if self.graph == 'diff':
                        if self.db_diff_graph is None:
                            self.transform()
                        graph = self.db_diff_graph
                    if self.graph == 'fuzzy':
                        if self.db_fuzzy_graph is None:
                            self.transform()
                        graph = self.db_fuzzy_graph
                    if self.graph == 'cknn':
                        if self.db_cknn_graph is None:
                            self.transform()
                        graph = self.db_cknn_graph

            elif self.basis == 'continuous':
                if self.CLapMap is None:
                    return print(
                        'Error: Basis set to \'continuous\', but the continuous basis is not computed!')
                else:
                    data = self.CLapMap
                if graph is None:
                    if self.graph == 'diff':
                        if self.cb_diff_graph is None:
                            self.transform()
                        graph = self.cb_diff_graph
                    if self.graph == 'fuzzy':
                        if self.cb_fuzzy_graph is None:
                            self.transform()
                        graph = self.cb_fuzzy_graph
                    if self.graph == 'cknn':
                        if self.cb_cknn_graph is None:
                            self.transform()
                        graph = self.cb_cknn_graph

            elif self.basis == 'fuzzy':
                if self.FuzzyLapMap is None:
                    return print('Error: Basis set to \'fuzzy\', but the fuzzy basis is not computed!')
                else:
                    data = self.FuzzyLapMap
                if graph is None:
                    if self.graph == 'diff':
                        if self.fb_diff_graph is None:
                            self.transform()
                        graph = self.fb_diff_graph
                    if self.graph == 'fuzzy':
                        if self.fb_fuzzy_graph is None:
                            self.transform()
                        graph = self.fb_fuzzy_graph
                    if self.graph == 'cknn':
                        if self.fb_cknn_graph is None:
                            self.transform()
                        graph = self.fb_cknn_graph

        start = time.time()
        spt_layout = spt.spectral_layout(graph=graph, dim=n_components, random_state=self.random_state)
        expansion = 10.0 / np.abs(spt_layout).max()
        spt_layout = (spt_layout * expansion).astype(
            np.float32
        ) + self.random_state.normal(
            scale=0.0001, size=[graph.shape[0], n_components]
        ).astype(np.float32)
        end = time.time()
        self.runtimes['Spectral'] = end - start
        if cache:
            self.SpecLayout = spt_layout

        return spt_layout

    def fuzzy_graph(self,
                    X=None,
                    basis=None,
                    knn_indices=None,
                    knn_dists=None,
                    cache=True):
        """

        Given a topological basis, a neighborhood size, and a measure of distance
        compute the fuzzy simplicial set (here represented as a fuzzy graph in
        the form of a sparse matrix) associated to the data. This is done by
        locally approximating geodesic distance at each point, creating a fuzzy
        simplicial set for each such point, and then combining all the local
        fuzzy simplicial sets into a global one via a fuzzy union.

        Parameters
        ----------
        X : np.ndarray, scipy.sparse.csr_matrix or str, 'diffusion' or 'continuous' (optional, default None).
            The data to be modelled as a fuzzy simplicial set. If None, defaults to the active orthogonal basis.

        knn_indices : array of shape (n_samples, n_neighbors) (optional).
            If the k-nearest neighbors of each point has already been calculated
            you can pass them in here to save computation time. This should be
            an array with the indices of the k-nearest neighbors as a row for
            each data point.

        knn_dists : array of shape (n_samples, n_neighbors) (optional).
            If the k-nearest neighbors of each point has already been calculated
            you can pass them in here to save computation time. This should be
            an array with the distances of the k-nearest neighbors as a row for
            each data point.

        cache : bool, optional (default True)
            Whether to store the fuzzy simplicial set graph in the TopOGraph object.

        Returns
        -------
        fuzzy_simplicial_set : coo_matrix
            A fuzzy simplicial set represented as a sparse matrix. The (i, j) entry of the matrix represents
            the membership strength of the 1-simplex between the ith and jth sample points.

        """

        if X is None:
            if basis == 'diffusion':
                if self.MSDiffMap is None:
                    return print('Error: Basis set to \'diffusion\', but the diffusion basis is not computed!')
                else:
                    X = self.MSDiffMap
            elif basis == 'continuous':
                if self.CLapMap is None:
                    return print('Error: Basis set to \'continuous\', but the continuous basis is not computed!')
                else:
                    X = self.CLapMap
            elif basis == 'fuzzy':
                if self.FuzzyLapMap is None:
                    return print('Error: Basis set to \'fuzzy\', but the fuzzy basis is not computed!')
                else:
                    X = self.FuzzyLapMap

        fuzzy_set = fuzzy_simplicial_set_ann(X,
                                             n_neighbors=self.graph_knn,
                                             knn_indices=knn_indices,
                                             knn_dists=knn_dists,
                                             backend=self.backend,
                                             metric=self.graph_metric,
                                             n_jobs=self.n_jobs,
                                             efC=self.efC,
                                             efS=self.efS,
                                             M=self.M,
                                             set_op_mix_ratio=1.0,
                                             local_connectivity=1.0,
                                             apply_set_operations=True,
                                             return_dists=False,
                                             verbose=self.bases_graph_verbose)
        if cache:
            if self.basis == 'diffusion':
                self.db_fuzzy_graph = fuzzy_set[0]
            if self.basis == 'continuous':
                self.cb_fuzzy_graph = fuzzy_set[0]
            if self.basis == 'fuzzy':
                self.fb_fuzzy_graph = fuzzy_set[0]
        return fuzzy_set[0]

    def MDE(self,
            basis=None,
            graph=None,
            n_components=2,
            n_neighbors=None,
            type='isomorphic',
            n_epochs=500,
            snapshot_every=30,
            constraint=None,
            init='quadratic',
            repulsive_fraction=None,
            max_distance=None,
            device='cpu',
            eps=10e-4,
            mem_size=1):

        """

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

        Parameters
        ----------
        basis :  str ('diffusion', 'continuous' or 'fuzzy').
            Which basis to use when computing the embedding. Defaults to the active basis.

        graph : scipy.sparse matrix.
            The affinity matrix to embedd with. Defaults to the active graph. If init = 'spectral',
            a fuzzy simplicial set is used, and this argument is ignored.

        n_components : int (optional, default 2).
            The embedding dimension. Use 2 or 3 for visualization.

        constraint : str (optional, default 'standardized').
            Constraint to use when optimizing the embedding. Options are 'standardized',
            'centered', `None` or a `pymde.constraints.Constraint()` function.

        n_neighbors : int (optional).
            The number of nearest neighbors to compute for each row (item) of
            ``data``. A sensible value is chosen by default, depending on the
            number of items.

        repulsive_fraction : float (optional).
            How many repulsive edges to include, relative to the number
            of attractive edges. ``1`` means as many repulsive edges as attractive
            edges. The higher this number, the more uniformly spread out the
            embedding will be. Defaults to ``0.5`` for standardized embeddings, and
            ``1`` otherwise. (If ``repulsive_penalty`` is ``None``, this argument
            is ignored.)

        max_distance : float (optional).
            If not None, neighborhoods are restricted to have a radius
            no greater than ``max_distance``.

        init : str or np.ndarray (optional, default 'quadratic')
            Initialization strategy; np.ndarray, 'quadratic' or 'random'.

        device : str (optional).
            Device for the embedding (eg, 'cpu', 'cuda').

        Returns
        -------
        torch.tensor
            A ``pymde.MDE`` object, based on the original data.

        """
        try:
            from pymde.functions import penalties, losses
            from pymde.preprocess import Graph
            from topo.layouts import mde
        except ImportError:
            return print("pymde is required for this. Install it with `pip install pymde`")
        attractive_penalty = penalties.Log1p
        repulsive_penalty = penalties.Log
        loss = losses.Absolute

        if n_neighbors is None:
            n_neighbors = self.graph_knn

        if basis is not None:
            if isinstance(basis, str):
                self.basis = basis
            else:
                return print('Error: `basis` must be \'diffusion\', \'continuous\' or \'fuzzy\'!')
        if self.basis == 'diffusion':
            if self.MSDiffMap is None:
                return print('Error: Basis set to \'diffusion\', but the diffusion basis is not computed!')
            else:
                data = self.MSDiffMap
            if graph is None:
                if self.graph == 'diff':
                    if self.db_diff_graph is None:
                        self.transform()
                    graph = self.db_diff_graph
                if self.graph == 'fuzzy':
                    if self.db_fuzzy_graph is None:
                        self.transform()
                    graph = self.db_fuzzy_graph
                if self.graph == 'cknn':
                    if self.db_cknn_graph is None:
                        self.transform()
                    graph = self.db_cknn_graph

        elif self.basis == 'continuous':
            if self.CLapMap is None:
                return print(
                    'Error: Basis set to \'continuous\', but the continuous basis is not computed!')
            else:
                data = self.CLapMap
            if graph is None:
                if self.graph == 'diff':
                    if self.cb_diff_graph is None:
                        self.transform()
                    graph = self.cb_diff_graph
                if self.graph == 'fuzzy':
                    if self.cb_fuzzy_graph is None:
                        self.transform()
                    graph = self.cb_fuzzy_graph
                if self.graph == 'cknn':
                    if self.cb_cknn_graph is None:
                        self.transform()
                    graph = self.cb_cknn_graph

        elif self.basis == 'fuzzy':
            if self.FuzzyLapMap is None:
                return print('Error: Basis set to \'fuzzy\', but the fuzzy basis is not computed!')
            else:
                data = self.FuzzyLapMap
            if graph is None:
                if self.graph == 'diff':
                    if self.fb_diff_graph is None:
                        self.transform()
                    graph = self.fb_diff_graph
                if self.graph == 'fuzzy':
                    if self.fb_fuzzy_graph is None:
                        self.transform()
                    graph = self.fb_fuzzy_graph
                if self.graph == 'cknn':
                    if self.fb_cknn_graph is None:
                        self.transform()
                    graph = self.fb_cknn_graph

        start = time.time()
        graph = Graph(graph)
        if type == 'isomorphic':
            emb = mde.IsomorphicMDE(graph,
                                    attractive_penalty=attractive_penalty,
                                    repulsive_penalty=repulsive_penalty,
                                    embedding_dim=n_components,
                                    constraint=constraint,
                                    n_neighbors=n_neighbors,
                                    repulsive_fraction=repulsive_fraction,
                                    max_distance=max_distance,
                                    init=init,
                                    device=device,
                                    verbose=self.layout_verbose)

        elif type == 'isometric':
            if max_distance is None:
                max_distance = 5e7
            emb = mde.IsometricMDE(graph,
                                   embedding_dim=n_components,
                                   loss=loss,
                                   constraint=constraint,
                                   max_distances=max_distance,
                                   device=device,
                                   verbose=self.layout_verbose)
        else:
            return print(
                'Error: The tg.MDE problem must be \'isomorphic\' or \'isometric\'. Alternatively, build your custom '
                'MDE problem with `pyMDE` (i.g. pymde.MDE())')
        self.MDE_problem = emb

        emb_Y = emb.embed(max_iter=n_epochs,
                          memory_size=mem_size,
                          snapshot_every=snapshot_every,
                          eps=eps,
                          verbose=self.layout_verbose)
        end = time.time()
        if self.verbosity >= 1:
            print('         Obtained MDE embedding in = %f (sec)' % (end - start))

        MDE_Y = np.array(emb_Y)

        if self.basis == 'diffusion':
            if self.graph == 'diff':
                self.db_diff_MDE = MDE_Y
                self.runtimes['db_diff_MDE'] = end - start
            if self.graph == 'cknn':
                self.db_cknn_MDE = MDE_Y
                self.runtimes['db_cknn_MDE'] = end - start
            if self.graph == 'fuzzy':
                self.db_fuzzy_MDE = MDE_Y
                self.runtimes['db_fuzzy_MDE'] = end - start
        if self.basis == 'continuous':
            if self.graph == 'diff':
                self.cb_diff_MDE = MDE_Y
                self.runtimes['cb_diff_MDE'] = end - start
            if self.graph == 'cknn':
                self.cb_cknn_MDE = MDE_Y
                self.runtimes['cb_cknn_MDE'] = end - start
            if self.graph == 'fuzzy':
                self.cb_fuzzy_MDE = MDE_Y
                self.runtimes['cb_fuzzy_MDE'] = end - start
        if self.basis == 'fuzzy':
            if self.graph == 'diff':
                self.fb_diff_MDE = MDE_Y
                self.runtimes['fb_diff_MDE'] = end - start
            if self.graph == 'cknn':
                self.fb_cknn_MDE = MDE_Y
                self.runtimes['fb_cknn_MDE'] = end - start
            if self.graph == 'fuzzy':
                self.fb_fuzzy_MDE = MDE_Y
                self.runtimes['fb_fuzzy_MDE'] = end - start

        return MDE_Y

    def MAP(self,
            data=None,
            graph=None,
            n_components=2,
            min_dist=0.3,
            spread=1,
            initial_alpha=1.2,
            n_epochs=800,
            metric=None,
            metric_kwds={},
            output_metric='euclidean',
            output_metric_kwds={},
            gamma=1.2,
            negative_sample_rate=10,
            init='spectral',
            random_state=None,
            euclidean_output=True,
            parallel=True,
            densmap=False,
            densmap_kwds={},
            output_dens=False,
            return_aux=False
            ):
        """""
        
        Manifold Approximation and Projection, as proposed by Leland McInnes with an uniform distribution assumption
        in the seminal [UMAP algorithm](https://umap-learn.readthedocs.io/en/latest/index.html). Performs a fuzzy
        simplicial set embedding, using a specified initialisation method and then minimizing the fuzzy set cross 
        entropy between the 1-skeletons of the high and low dimensional fuzzy simplicial sets. The fuzzy simplicial 
        set embedding was proposed and implemented by Leland McInnes in UMAP (see `umap-learn
        <https://github.com/lmcinnes/umap>`). Here we're using it only for the projection (layout optimization)
        by minimizing the cross-entropy between a phenotypic map (i.e. data, TopOMetry non-uniform latent mappings) and 
        its graph topological representation.


        Parameters
        ----------
        data : array of shape (n_samples, n_features).
            The source data to be embedded by UMAP. If `None` (default), the active basis will be used.
            
        graph : scipy.sparse.csr_matrix (n_samples, n_samples).
            The 1-skeleton of the high dimensional fuzzy simplicial set as
            represented by a graph for which we require a sparse matrix for the
            (weighted) adjacency matrix. If `None` (default), a fuzzy simplicial set 
            is computed with default parameters.
            
        n_components : int (optional, default 2).
            The dimensionality of the euclidean space into which to embed the data.
            
        initial_alpha: float (optional, default 1).
            Initial learning rate for the SGD.
            
        gamma : float (optional, default 1.2).
            Weight to apply to negative samples.
            
        negative_sample_rate : int (optional, default 5).
            The number of negative samples to select per positive sample
            in the optimization process. Increasing this value will result
            in greater repulsive force being applied, greater optimization
            cost, but slightly more accuracy.
            
        n_epochs : int (optional, default 0).
            The number of training epochs to be used in optimizing the
            low dimensional embedding. Larger values result in more accurate
            embeddings. If 0 is specified a value will be selected based on
            the size of the input dataset (200 for large datasets, 500 for small).
            
        init : string (optional, default 'spectral').
        How to initialize the low dimensional embedding. Options are:
                * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
                * 'random': assign initial embedding positions at random.
                * A numpy array of initial embedding positions.
                
        random_state : numpy RandomState or equivalent.
            A state capable being used as a numpy random state.
            
        metric : string or callable.
            The metric used to measure distance in high dimensional space; used if
            multiple connected components need to be layed out. Defaults to `TopOGraph.graph_metric`.
            
        metric_kwds : dict (optional, no default).
            Key word arguments to be passed to the metric function; used if
            multiple connected components need to be layed out.
            
        densmap : bool (optional, default False).
            Whether to use the density-augmented objective function to optimize
            the embedding according to the densMAP algorithm.
            
        densmap_kwds : dict (optional, no default).
            Key word arguments to be used by the densMAP optimization.
            
        output_dens : bool (optional, default False).
            Whether to output local radii in the original data and the embedding.
            
        output_metric : function (optional, no default).
            Function returning the distance between two points in embedding space and
            the gradient of the distance wrt the first argument.
            
        output_metric_kwds : dict (optional, no default).
            Key word arguments to be passed to the output_metric function.
            
        euclidean_output : bool (optional, default True).
            Whether to use the faster code specialised for euclidean output metrics
            
        parallel : bool (optional, default True).
            Whether to run the computation using numba parallel.
            Running in parallel is non-deterministic, and is not used
            if a random seed has been set, to ensure reproducibility.
            
        return_aux : bool , (optional, default False).
            Whether to also return the auxiliary data, i.e. initialization and local radii.
            
        Returns
        -------
        * embedding : array of shape (n_samples, n_components)
            The optimized of ``graph`` into an ``n_components`` dimensional
            euclidean space.

        *  return_aux is set to True :
            aux_data : dict
            
            Auxiliary dictionary output returned with the embedding.
            ``aux_data['Y_init']``: array of shape (n_samples, n_components)
            The spectral initialization of ``graph`` into an ``n_components`` dimensional
            euclidean space.
            
            When densMAP extension is turned on, this dictionary includes local radii in the original
            data (``aux_data['rad_orig']``) and in the embedding (``aux_data['rad_emb']``).


        """""
        from topo.layouts import map
        if output_metric == 'torus':
            from topo.utils.umap_utils import torus_euclidean_grad
            output_metric = torus_euclidean_grad
        if metric is None:
            metric = self.graph_metric
        if data is None:
            if self.basis == 'diffusion':
                if self.MSDiffMap is None:
                    return print('Error: Basis set to \'diffusion\', but the diffusion basis is not computed!')
                else:
                    data = self.MSDiffMap
                if graph is None:
                    if self.graph == 'diff':
                        if self.db_diff_graph is None:
                            self.transform()
                        graph = self.db_diff_graph
                    if self.graph == 'fuzzy':
                        if self.db_fuzzy_graph is None:
                            self.transform()
                        graph = self.db_fuzzy_graph
                    if self.graph == 'cknn':
                        if self.db_cknn_graph is None:
                            self.transform()
                        graph = self.db_cknn_graph

            elif self.basis == 'continuous':
                if self.CLapMap is None:
                    return print(
                        'Error: Basis set to \'continuous\', but the continuous basis is not computed!')
                else:
                    data = self.CLapMap
                if graph is None:
                    if self.graph == 'diff':
                        if self.cb_diff_graph is None:
                            self.transform()
                        graph = self.cb_diff_graph
                    if self.graph == 'fuzzy':
                        if self.cb_fuzzy_graph is None:
                            self.transform()
                        graph = self.cb_fuzzy_graph
                    if self.graph == 'cknn':
                        if self.cb_cknn_graph is None:
                            self.transform()
                        graph = self.cb_cknn_graph

            elif self.basis == 'fuzzy':
                if self.FuzzyLapMap is None:
                    return print('Error: Basis set to \'fuzzy\', but the fuzzy basis is not computed!')
                else:
                    data = self.FuzzyLapMap
                if graph is None:
                    if self.graph == 'diff':
                        if self.fb_diff_graph is None:
                            self.transform()
                        graph = self.fb_diff_graph
                    if self.graph == 'fuzzy':
                        if self.fb_fuzzy_graph is None:
                            self.transform()
                        graph = self.fb_fuzzy_graph
                    if self.graph == 'cknn':
                        if self.fb_cknn_graph is None:
                            self.transform()
                        graph = self.fb_cknn_graph

        if isinstance(init, str) and init == 'spectral':
            if self.SpecLayout is None:
                if self.verbosity >= 1:
                    print('         Computing spectral layout...')
                if self.basis == 'diffusion':
                    if self.db_fuzzy_graph is None:
                        self.db_fuzzy_graph = self.fuzzy_graph(data)
                    init = self.spectral_layout(graph=self.db_fuzzy_graph, n_components=n_components)
                if self.basis == 'continuous':
                    if self.cb_fuzzy_graph is None:
                        self.cb_fuzzy_graph = self.fuzzy_graph(data)
                    init = self.spectral_layout(graph=self.cb_fuzzy_graph, n_components=n_components)
                if self.basis == 'fuzzy':
                    if self.fb_fuzzy_graph is None:
                        self.fb_fuzzy_graph = self.fuzzy_graph(data)
                    init = self.spectral_layout(graph=self.fb_fuzzy_graph, n_components=n_components)
            else:
                init = self.SpecLayout


        start = time.time()
        results = map.fuzzy_embedding(graph,
                                      n_components=n_components,
                                      initial_alpha=initial_alpha,
                                      min_dist=min_dist,
                                      spread=spread,
                                      n_epochs=n_epochs,
                                      metric=metric,
                                      metric_kwds=metric_kwds,
                                      output_metric=output_metric,
                                      output_metric_kwds=output_metric_kwds,
                                      gamma=gamma,
                                      negative_sample_rate=negative_sample_rate,
                                      init=init,
                                      random_state=random_state,
                                      euclidean_output=euclidean_output,
                                      parallel=parallel,
                                      verbose=self.layout_verbose,
                                      a=None,
                                      b=None,
                                      densmap=densmap,
                                      densmap_kwds=densmap_kwds,
                                      output_dens=output_dens)

        end = time.time()
        if self.verbosity >= 1:
            print('         Optimized MAP embedding in = %f (sec)' % (end - start))

        if self.basis == 'diffusion':
            if self.graph == 'diff':
                self.db_diff_MAP = results[0]
                self.runtimes['db_diff_MAP'] = end - start
            if self.graph == 'cknn':
                self.db_cknn_MAP = results[0]
                self.runtimes['db_cknn_MAP'] = end - start
            if self.graph == 'fuzzy':
                self.db_fuzzy_MAP = results[0]
                self.runtimes['db_fuzzy_MAP'] = end - start
        if self.basis == 'continuous':
            if self.graph == 'diff':
                self.cb_diff_MAP = results[0]
                self.runtimes['cb_diff_MAP'] = end - start
            if self.graph == 'cknn':
                self.cb_cknn_MAP = results[0]
                self.runtimes['cb_cknn_MAP'] = end - start
            if self.graph == 'fuzzy':
                self.cb_fuzzy_MAP = results[0]
                self.runtimes['cb_fuzzy_MAP'] = end - start
        if self.basis == 'fuzzy':
            if self.graph == 'diff':
                self.fb_diff_MAP = results[0]
                self.runtimes['fb_diff_MAP'] = end - start
            if self.graph == 'cknn':
                self.fb_cknn_MAP = results[0]
                self.runtimes['fb_cknn_MAP'] = end - start
            if self.graph == 'fuzzy':
                self.fb_fuzzy_MAP = results[0]
                self.runtimes['fb_fuzzy_MAP'] = end - start

        if return_aux:
            return results
        else:
            return results[0]

    def PaCMAP(self, data=None,
               init='spectral',
               n_components=2,
               n_neighbors=None,
               MN_ratio=0.5,
               FP_ratio=2.0,
               pair_neighbors=None,
               pair_MN=None,
               pair_FP=None,
               distance="euclidean",
               lr=1.0,
               num_iters=450,
               intermediate=False):
        """
        Performs Pairwise-Controlled Manifold Approximation and Projection.

        Parameters
        ----------

        data : np.ndarray or scipy.sparse.csr_matrix (optional, default None).
            Data to be embedded. If None, will use the active orthogonal basis.

        init : np.ndarray of shape (N,2) or str (optional, default 'spectral').
            Initialization positions. Defaults to a multicomponent spectral embedding ('spectral').
            Other options are 'pca' or 'random'.

        n_components : int (optional, default 2).
            How many components to embedd into.

        n_neighbors : int (optional, default None).
            How many neighbors to use during embedding. If None, will use `TopOGraph.graph_knn`.

        MN_ratio : float (optional, default 0.5).
            The ratio of the number of mid-near pairs to the number of neighbors, n_MN = n_neighbors * MN_ratio.

        FP_ratio : float (optional, default 2.0).
            The ratio of the number of further pairs to the number of neighbors, n_FP = n_neighbors * FP_ratio.

        distance : float (optional, default 'euclidean').
            Distance metric to use. Options are 'euclidean', 'angular', 'manhattan' and 'hamming'.

        lr : float (optional, default 1.0).
            Learning rate of the AdaGrad optimizer.

        num_iters : int (optional, default 450).
            Number of iterations. The default 450 is enough for most dataset to converge.

        intermediate : bool (optional, default False).
            Whether PaCMAP should also output the intermediate stages of the optimization process of the lower
            dimension embedding. If True, then the output will be a numpy array of the size (n, n_dims, 13),
            where each slice is a "screenshot" of the output embedding at a particular number of steps,
            from [0, 10, 30, 60, 100, 120, 140, 170, 200, 250, 300, 350, 450].

        pair_neighbors : optional, default None.
            Pre-specified neighbor pairs. Allows user to use their own graphs. Default to None.

        pair_MN : optional, default None.
            Pre-specified mid-near pairs. Allows user to use their own graphs. Default to None.

        pair_FP : optional, default None.
            Pre-specified further pairs. Allows user to use their own graphs. Default to None.

        Returns
        -------

        PaCMAP embedding.

        """
        from topo.layouts import pairwise
        if data is None:
            if self.basis == 'diffusion':
                if self.MSDiffMap is None:
                    return print('Error: `basis` set to \'diffusion\', but the diffusion basis is not computed!')
                else:
                    data = self.MSDiffMap
            elif self.basis == 'continuous':
                if self.CLapMap is None:
                    return print('Error: `basis` set to \'continuous\', but the continuous basis is not computed!')
                else:
                    data = self.CLapMap
            elif self.basis == 'fuzzy':
                if self.FuzzyLapMap is None:
                    return print('Error: `basis` set to \'fuzzy\', but the fuzzy basis is not computed!')
                else:
                    data = self.FuzzyLapMap
            else:
                return print('No computed basis or data is provided!')

        if isinstance(init, str) and init == 'spectral':
            if self.SpecLayout is None:
                if self.verbosity >= 1:
                    print('         Computing spectral layout...')
                if self.basis == 'diffusion':
                    if self.db_fuzzy_graph is None:
                        self.db_fuzzy_graph = self.fuzzy_graph(data)
                    init = self.spectral_layout(graph=self.db_fuzzy_graph, n_components=n_components)
                if self.basis == 'continuous':
                    if self.cb_fuzzy_graph is None:
                        self.cb_fuzzy_graph = self.fuzzy_graph(data)
                    init = self.spectral_layout(graph=self.cb_fuzzy_graph, n_components=n_components)
                if self.basis == 'fuzzy':
                    if self.fb_fuzzy_graph is None:
                        self.fb_fuzzy_graph = self.fuzzy_graph(data)
                    init = self.spectral_layout(graph=self.fb_fuzzy_graph, n_components=n_components)
            else:
                init = self.SpecLayout

        start = time.time()
        results = pairwise.PaCMAP(data=data,
                                  init=init,
                                  n_dims=n_components,
                                  n_neighbors=n_neighbors,
                                  MN_ratio=MN_ratio,
                                  FP_ratio=FP_ratio,
                                  pair_neighbors=pair_neighbors,
                                  pair_MN=pair_MN,
                                  pair_FP=pair_FP,
                                  distance=distance,
                                  lr=lr,
                                  num_iters=num_iters,
                                  verbose=self.layout_verbose,
                                  intermediate=intermediate)

        end = time.time()
        if self.verbosity >= 1:
            print('         Obtained PaCMAP embedding in = %f (sec)' % (end - start))

        if self.basis == 'diffusion':
            self.db_PaCMAP = results
            self.runtimes['db_PaCMAP'] = end - start
        if self.basis == 'continuous':
            self.cb_PaCMAP = results
            self.runtimes['cb_PaCMAP'] = end - start
        if self.basis == 'fuzzy':
            self.fb_PaCMAP = results
            self.runtimes['fb_PaCMAP'] = end - start

        return results

    def TriMAP(self, basis=None,
               graph=None,
               init=None,
               n_components=2,
               n_inliers=10,
               n_outliers=5,
               n_random=5,
               use_dist_matrix=False,
               metric='euclidean',
               lr=1000.0,
               n_iters=400,
               triplets=None,
               weights=None,
               knn_tuple=None,
               weight_adj=500.0,
               opt_method="dbd",
               return_seq=False):
        """
        Graph layout optimization using triplets.

        Parameters
        ----------
        basis : str (optional, default None).
        graph : str (optional, default None).
        init : str (optional, default None).
        n_components : int (optional, default 2).
            Number of dimensions of the embedding.
        n_inliers : int (optional, default 10).
            Number of inlier points for triplet constraints.
        n_outliers : int (optional, default 5).
            Number of outlier points for triplet constraints.
        n_random : int (optional, default 5).
            Number of random triplet constraints per point.
        metric : str (optional, default 'euclidean').
             Distance measure ('euclidean', 'manhattan', 'angular', 'hamming')
        use_dist_matrix : bool (optional, default False).
            Use TopOMetry's learned similarities between samples. As of now, this is unstable.
        lr : int (optional, default 1000).
            Learning rate.
        n_iters : int (optional, default 400)
            Number of iterations.
        opt_method : str (optional, default 'dbd')
             Optimization method ('sd': steepest descent,  'momentum': GD with momentum,
              'dbd': GD with momentum delta-bar-delta).
        return_seq : bool (optional, default False)
             Return the sequence of maps recorded every 10 iterations.

        Returns
        -------
        TriMAP embedding.

        """
        if basis is not None:
            if isinstance(basis, str):
                self.basis = basis
            else:
                return print('Error: `basis` must be \'diffusion\', \'continuous\' or \'fuzzy\'!')
        if use_dist_matrix:
            if graph is None:
                if self.basis == 'diffusion':
                    if self.graph == 'diff':
                        if self.db_diff_graph is None:
                            self.transform()
                        data = self.db_diff_graph
                    if self.graph == 'fuzzy':
                        if self.db_fuzzy_graph is None:
                            self.transform()
                        data = self.db_fuzzy_graph
                    if self.graph == 'cknn':
                        if self.db_cknn_graph is None:
                            self.transform()
                        data = self.db_cknn_graph
                elif self.basis == 'continuous':
                    if self.graph == 'diff':
                        if self.cb_diff_graph is None:
                            self.transform()
                        data = self.cb_diff_graph
                    if self.graph == 'fuzzy':
                        if self.cb_fuzzy_graph is None:
                            self.transform()
                        data = self.cb_fuzzy_graph
                    if self.graph == 'cknn':
                        if self.cb_cknn_graph is None:
                            self.transform()
                        data = self.cb_cknn_graph
                elif self.basis == 'fuzzy':
                    if self.graph == 'diff':
                        if self.fb_diff_graph is None:
                            self.transform()
                        data = self.fb_diff_graph
                    if self.graph == 'fuzzy':
                        if self.fb_fuzzy_graph is None:
                            self.transform()
                        data = self.fb_fuzzy_graph
                    if self.graph == 'cknn':
                        if self.fb_cknn_graph is None:
                            self.transform()
                        data = self.fb_cknn_graph
            else:
                data = graph
            data = data.toarray()

        else:
            if self.basis == 'diffusion':
                if self.MSDiffMap is None:
                    return print('Error: Basis set to \'diffusion\', but the diffusion basis is not computed!')
                else:
                    data = self.MSDiffMap
            elif self.basis == 'continuous':
                if self.CLapMap is None:
                    return print(
                        'Error: Basis set to \'continuous\', but the continuous basis is not computed!')
                else:
                    data = self.CLapMap
            elif self.basis == 'fuzzy':
                if self.FuzzyLapMap is None:
                    return print('Error: Basis set to \'fuzzy\', but the fuzzy basis is not computed!')
                else:
                    data = self.FuzzyLapMap

        if isinstance(init, str) and init == 'spectral':
            if self.SpecLayout is None:
                if self.verbosity >= 1:
                    print('         Computing spectral layout...')
                if self.basis == 'diffusion':
                    if self.db_fuzzy_graph is None:
                        self.db_fuzzy_graph = self.fuzzy_graph(data)
                    init = self.spectral_layout(graph=self.db_fuzzy_graph, n_components=n_components)
                if self.basis == 'continuous':
                    if self.cb_fuzzy_graph is None:
                        self.cb_fuzzy_graph = self.fuzzy_graph(data)
                    init = self.spectral_layout(graph=self.cb_fuzzy_graph, n_components=n_components)
                if self.basis == 'fuzzy':
                    if self.fb_fuzzy_graph is None:
                        self.fb_fuzzy_graph = self.fuzzy_graph(data)
                    init = self.spectral_layout(graph=self.fb_fuzzy_graph, n_components=n_components)
            else:
                init = self.SpecLayout

        from topo.layouts import trimap
        start = time.time()
        results = trimap.TriMAP(X=data,
                                init=init,
                                n_dims=n_components,
                                n_inliers=n_inliers,
                                n_outliers=n_outliers,
                                n_random=n_random,
                                distance=metric,
                                use_dist_matrix=use_dist_matrix,
                                lr=lr,
                                n_iters=n_iters,
                                triplets=triplets,
                                weights=weights,
                                knn_tuple=knn_tuple,
                                verbose=self.layout_verbose,
                                weight_adj=weight_adj,
                                opt_method=opt_method,
                                return_seq=return_seq)

        end = time.time()
        if self.verbosity >= 1:
            print('         Obtained TriMAP embedding in = %f (sec)' % (end - start))
        if self.basis == 'diffusion':
            if use_dist_matrix:
                if self.graph == 'diff':
                    self.db_diff_TriMAP = results
                    self.runtimes['db_diff_TriMAP'] = end - start
                if self.graph == 'cknn':
                    self.db_cknn_TriMAP = results
                    self.runtimes['db_cknn_TriMAP'] = end - start
                if self.graph == 'fuzzy':
                    self.db_fuzzy_TriMAP = results
                    self.runtimes['db_fuzzy_TriMAP'] = end - start
            else:
                self.db_TriMAP = results
                self.runtimes['db_TriMAP'] = end - start

        if self.basis == 'continuous':
            if use_dist_matrix:
                if self.graph == 'diff':
                    self.cb_diff_TriMAP = results
                    self.runtimes['cb_diff_TriMAP'] = end - start
                if self.graph == 'cknn':
                    self.cb_cknn_TriMAP = results
                    self.runtimes['cb_cknn_TriMAP'] = end - start
                if self.graph == 'fuzzy':
                    self.cb_fuzzy_TriMAP = results
                    self.runtimes['cb_fuzzy_TriMAP'] = end - start
            else:
                self.cb_TriMAP = results
                self.runtimes['cb_TriMAP'] = end - start

        if self.basis == 'fuzzy':
            if use_dist_matrix:
                if self.graph == 'diff':
                    self.fb_diff_TriMAP = results
                    self.runtimes['fb_diff_TriMAP'] = end - start
                if self.graph == 'cknn':
                    self.fb_cknn_TriMAP = results
                    self.runtimes['fb_cknn_TriMAP'] = end - start
                if self.graph == 'fuzzy':
                    self.fb_fuzzy_TriMAP = results
                    self.runtimes['fb_fuzzy_TriMAP'] = end - start
            else:
                self.fb_TriMAP = results
                self.runtimes['fb_TriMAP'] = end - start

        return results


    def tSNE(self, data=None,
             graph=None,
             n_components=2,
             early_exaggeration=12,
             n_iter=1000,
             n_iter_early_exag=250,
             n_iter_without_progress=30,
             min_grad_norm=1e-07,
             init='random',
             random_state=None,
             angle=0.5,
             cheat_metric=True):
        """

        The classic t-SNE embedding, usually on top of a TopOMetry topological basis.

        Parameters
        ----------
        data : optional, default None)
        graph : optional, default None)
        n_components: int (optional, default 2).
        early_exaggeration : sets exaggeration
        n_iter : number of iterations to optmizie
        n_iter_early_exag : number of iterations in early exaggeration
        init : np.ndarray (optional, defaults to tg.SpecLayout)
            Initialisation for the optimization problem.
        random_state : optional, default None

        Returns
        -------

        """
        try:
            from MulticoreTSNE import MulticoreTSNE as TSNE
            _have_mc_tsne = True
        except ImportError:
            _have_mc_tsne = False
            return print('No MulticoreTSNE installation found. Exiting.')
        if self.SpecLayout is not None:
            init = self.SpecLayout
        if data is None:
            if self.basis == 'diffusion':
                if self.MSDiffMap is None:
                    return print('Error: `basis` set to \'diffusion\', but the diffusion basis is not computed!')
                else:
                    data = self.MSDiffMap
            elif self.basis == 'continuous':
                if self.CLapMap is None:
                    return print('Error: `basis` set to \'continuous\', but the continuous basis is not computed!')
                else:
                    data = self.CLapMap
            elif self.basis == 'fuzzy':
                if self.FuzzyLapMap is None:
                    return print('Error: `basis` set to \'fuzzy\', but the fuzzy basis is not computed!')
                else:
                    data = self.FuzzyLapMap
            else:
                return print('Error: no computed basis or data is provided!')

        if isinstance(init, str) and init == 'spectral':
            if self.SpecLayout is None:
                if self.verbosity >= 1:
                    print('         Computing spectral layout...')
                if self.basis == 'diffusion':
                    if self.db_fuzzy_graph is None:
                        self.db_fuzzy_graph = self.fuzzy_graph(data)
                    init = self.spectral_layout(graph=self.db_fuzzy_graph, n_components=n_components)
                if self.basis == 'continuous':
                    if self.cb_fuzzy_graph is None:
                        self.cb_fuzzy_graph = self.fuzzy_graph(data)
                    init = self.spectral_layout(graph=self.cb_fuzzy_graph, n_components=n_components)
                if self.basis == 'fuzzy':
                    if self.fb_fuzzy_graph is None:
                        self.fb_fuzzy_graph = self.fuzzy_graph(data)
                    init = self.spectral_layout(graph=self.fb_fuzzy_graph, n_components=n_components)
            else:
                init = self.SpecLayout

        if data is None:
            data = graph.toarray()
            metric = 'precomputed'
        else:
            metric = self.graph_metric
        if self.layout_verbose:
            verbose = 1
        else:
            verbose = 0
        if int(self.n) <= 3000:
            perplexity = 30
        else:
            perplexity = self.n // 1000
        start = time.time()
        tsne = TSNE(n_components=n_components,
                    perplexity=perplexity,
                    early_exaggeration=early_exaggeration,
                    learning_rate=self.n // 12,
                    n_iter=n_iter,
                    n_iter_early_exag=n_iter_early_exag,
                    n_iter_without_progress=n_iter_without_progress,
                    min_grad_norm=min_grad_norm,
                    metric=metric,
                    init=init,
                    verbose=verbose,
                    random_state=random_state,
                    method='barnes_hut',
                    angle=angle,
                    n_jobs=self.n_jobs,
                    cheat_metric=cheat_metric)
        Y = tsne.fit_transform(data)
        end = time.time()
        if self.verbosity >= 1:
            print('         Obtained tSNE embedding in = %f (sec)' % (end - start))
        tSNE_Y = Y

        if self.basis == 'diffusion':
            self.db_tSNE = tSNE_Y
            self.runtimes['db_tSNE'] = end - start
        if self.basis == 'continuous':
            self.cb_tSNE = tSNE_Y
            self.runtimes['cb_tSNE'] = end - start
        if self.basis == 'fuzzy':
            self.fb_tSNE = tSNE_Y
            self.runtimes['fb_tSNE'] = end - start

        return tSNE_Y


    def NCVis(self,
              data=None,
              n_components=2,
              n_jobs=-1,
              n_neighbors=15,
              distance="cosine",
              M=15,
              efC=30,
              random_seed=42,
              n_epochs=200,
              n_init_epochs=20,
              spread=1.0,
              min_dist=0.4,
              alpha=1.0,
              a=None,
              b=None,
              alpha_Q=1.,
              n_noise=None):
        from topo.layouts import NCVis as ncvis
        if data is None:
            if self.basis == 'diffusion':
                if self.MSDiffMap is None:
                    return print('Error: `basis` set to \'diffusion\', but the diffusion basis is not computed!')
                else:
                    data = self.MSDiffMap
            elif self.basis == 'continuous':
                if self.CLapMap is None:
                    return print('Error: `basis` set to \'continuous\', but the continuous basis is not computed!')
                else:
                    data = self.CLapMap
            elif self.basis == 'fuzzy':
                if self.FuzzyLapMap is None:
                    return print('Error: `basis` set to \'fuzzy\', but the fuzzy basis is not computed!')
                else:
                    data = self.FuzzyLapMap
            else:
                return print('Error: no computed basis or data is provided!')

        start = time.time()
        NCVis_Y = ncvis(data, n_components=n_components, n_jobs=n_jobs, n_neighbors=n_neighbors, distance=distance,
                          M=M, efC=efC, random_seed=random_seed, n_epochs=n_epochs, n_init_epochs=n_init_epochs,
                          spread=spread, min_dist=min_dist, alpha=alpha, alpha_Q=alpha_Q, a=a, b=b, n_noise=n_noise)

        end = time.time()
        if self.verbosity >= 1:
            print('         Obtained NCVis embedding in = %f (sec)' % (end - start))

        if self.basis == 'diffusion':
            self.db_NCVis = NCVis_Y
            self.runtimes['db_NCVis'] = end - start
        if self.basis == 'continuous':
            self.cb_NCVis = NCVis_Y
            self.runtimes['cb_NCVis'] = end - start
        if self.basis == 'fuzzy':
            self.fb_NCVis = NCVis_Y
            self.runtimes['fb_NCVis'] = end - start
        return NCVis_Y


    def affinity_clustering(self, graph=None, damping=0.5, max_iter=200, convergence_iter=15):
        from sklearn.cluster import AffinityPropagation
        if self.layout_verbose:
            verbose = True
        else:
            verbose = False
        if graph is None:
            if self.basis == 'diffusion':
                if self.graph == 'diff':
                    if self.db_diff_graph is None:
                        self.transform()
                    graph = self.db_diff_graph
                if self.graph == 'fuzzy':
                    if self.db_fuzzy_graph is None:
                        self.transform()
                    graph = self.db_fuzzy_graph
                if self.graph == 'cknn':
                    if self.db_cknn_graph is None:
                        self.transform()
                    graph = self.db_cknn_graph
            elif self.basis == 'continuous':
                if self.graph == 'diff':
                    if self.cb_diff_graph is None:
                        self.transform()
                    graph = self.cb_diff_graph
                if self.graph == 'fuzzy':
                    if self.cb_fuzzy_graph is None:
                        self.transform()
                    graph = self.cb_fuzzy_graph
                if self.graph == 'cknn':
                    if self.cb_cknn_graph is None:
                        self.transform()
                    graph = self.cb_cknn_graph
            elif self.basis == 'fuzzy':
                if self.graph == 'diff':
                    if self.fb_diff_graph is None:
                        self.transform()
                    graph = self.fb_diff_graph
                if self.graph == 'fuzzy':
                    if self.fb_fuzzy_graph is None:
                        self.transform()
                    graph = self.fb_fuzzy_graph
                if self.graph == 'cknn':
                    if self.fb_cknn_graph is None:
                        self.transform()
                    graph = self.fb_cknn_graph

        start = time.time()
        labels = AffinityPropagation(damping=damping, max_iter=max_iter,
                                     convergence_iter=convergence_iter,
                                     copy=False, preference=None, verbose=verbose,
                                     affinity='precomputed', random_state=self.random_state).fit_predict(
            graph.toarray())

        end = time.time()
        if self.verbosity >= 1:
            print('         Affinity clustering performed in = %f (sec)' % (end - start))


        if self.basis == 'diffusion':
            if self.graph == 'diff':
                self.db_diff_clusters = labels
            if self.graph == 'cknn':
                self.db_cknn_clusters = labels
            if self.graph == 'fuzzy':
                self.db_fuzzy_clusters = labels
        if self.basis == 'continuous':
            if self.graph == 'diff':
                self.cb_diff_clusters = labels
            if self.graph == 'cknn':
                self.cb_cknn_clusters = labels
            if self.graph == 'fuzzy':
                self.cb_fuzzy_clusters = labels
        if self.basis == 'fuzzy':
            if self.graph == 'diff':
                self.fb_diff_clusters = labels
            if self.graph == 'cknn':
                self.fb_cknn_clusters = labels
            if self.graph == 'fuzzy':
                self.fb_fuzzy_clusters = labels

        return labels

    def plot(self,
             target=None,
             space='2D',
             dims_gauss=None,
             labels=None,
             pt_size=1,
             marker='o',
             opacity=1,
             cmap='Spectral'
             ):
        """

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

        Returns
        -------

        2D or 3D visualizations, depending on `space`.

        """
        import topo.plot as pt
        if space == '2d' or space == '3d':
            if space == '2d':
                return pt.scatter(target,
                                  labels=labels,
                                  pt_size=pt_size,
                                  marker=marker,
                                  opacity=opacity,
                                  cmap=cmap
                                  )
            else:
                return pt.scatter_3d(target,
                                     labels=labels,
                                     pt_size=pt_size,
                                     marker=marker,
                                     opacity=opacity,
                                     cmap=cmap
                                     )
        elif space == 'hyperboloid':
            return pt.hyperboloid_3d(target,
                                     labels=labels,
                                     pt_size=pt_size,
                                     marker=marker,
                                     opacity=opacity
                                     )

        elif space == 'poincare':
            return pt.poincare_disk(target,
                                    labels=labels,
                                    pt_size=pt_size,
                                    marker=marker,
                                    opacity=opacity,
                                    cmap=cmap
                                    )
        elif space == 'sphere':
            return pt.sphere_3d(target,
                                labels=labels,
                                pt_size=pt_size,
                                marker=marker,
                                opacity=opacity,
                                cmap=cmap
                                )

        elif space == 'sphere_projection':
            return pt.sphere_projection(target,
                                        labels=labels,
                                        pt_size=pt_size,
                                        marker=marker,
                                        opacity=opacity,
                                        cmap=cmap
                                        )


        elif space == 'toroid':
            return pt.toroid_3d(target,
                                labels=labels,
                                pt_size=pt_size,
                                marker=marker,
                                opacity=opacity,
                                cmap=cmap
                                )

        elif space == 'gauss_potential':
            if dims_gauss is None:
                if target.shape[1] >= 5:
                    dims_gauss = [2, 3, 4]
                else:
                    return print('Error: could not find at least 5 dimensions.')

            return pt.gaussian_potential(target,
                                         dims=dims_gauss,
                                         labels=labels,
                                         pt_size=pt_size,
                                         marker=marker,
                                         opacity=opacity
                                         )


    def run_models(self, X,
                   bases=['diffusion', 'fuzzy', 'continuous'],
                   graphs=['diff', 'cknn', 'fuzzy']):
        if ('diffusion' in bases) and (self.MSDiffMap is None):
            run_db = True
        else:
            run_db = False
        if ('continuous' in bases) and (self.MSDiffMap is None):
            run_cb = True
        else:
            run_cb = False
        if ('fuzzy' in bases) and (self.MSDiffMap is None):
            run_fb = True
        else:
            run_fb = False
        if 'diff' in graphs:
            run_diff = True
        else:
            run_diff = False
        if 'cknn' in graphs:
            run_cknn = True
        else:
            run_cknn = False
        if 'fuzzy' in graphs:
            run_fuzzy = True
        else:
            run_fuzzy = False
        if run_db:
            self.basis = 'diffusion'
            self.fit(X)
            if run_diff:
                self.graph = 'diff'
            if run_cknn:
                self.graph = 'cknn'
            if run_fuzzy:
                self.graph = 'fuzzy'
            self.transform(X)
        if run_cb:
            self.basis = 'continuous'
            self.fit(X)
            if run_diff:
                self.graph = 'diff'
            if run_cknn:
                self.graph = 'cknn'
            if run_fuzzy:
                self.graph = 'fuzzy'
            self.transform(X)
        if run_fb:
            self.basis = 'fuzzy'
            self.fit(X)
            if run_diff:
                self.graph = 'diff'
            if run_cknn:
                self.graph = 'cknn'
            if run_fuzzy:
                self.graph = 'fuzzy'
            self.transform(X)

        return self

    def run_layouts(self, X, n_components=2,
                    bases=['diffusion', 'fuzzy', 'continuous'],
                    graphs=['diff', 'cknn', 'fuzzy'],
                    layouts=['tSNE', 'MAP', 'MDE', 'PaCMAP', 'TriMAP', 'NCVis']):
        """

        Master function to easily run all combinations of possible bases and graphs that approximate the
        [Laplace-Beltrami Operator](), and the 6 layout options within TopOMetry: tSNE, MAP, MDE, PaCMAP, TriMAP,
        and NCVis.

        Parameters
        ----------
        X : np.ndarray or scipy.sparse.csr_matrix
         Data matrix.
        n_components : int (optional, default 2).
         Number of components for visualization.
        bases : str (optional, default ['diffusion', 'continuous','fuzzy'])
         Which bases to compute. Defaults to all. To run only one or two bases, set it to
         ['fuzzy', 'diffusion'] or ['continuous'], for exemple.
        graphs : str (optional, default ['diff', 'cknn','fuzzy'])
         Which graphs to compute. Defaults to all. To run only one or two graphs, set it to
         ['fuzzy', 'diff'] or ['cknn'], for exemple.
        layouts : str (optional, default all ['tSNE', 'MAP', 'MDE', 'PaCMAP', 'TriMAP', 'NCVis'])
         Which layouts to compute. Defaults to all 6 options within TopOMetry: tSNE, MAP, MDE, PaCMAP,
         TriMAP and NCVis. To run only one or two layouts, set it to
         ['tSNE', 'MAP'] or ['PaCMAP'], for exemple.

        Returns
        -------

        Populates the TopOMetry object slots

        """
        if str('diffusion') in bases:
            run_db = True
        else:
            run_db = False
        if str('continuous') in bases:
            run_cb = True
        else:
            run_cb = False
        if str('fuzzy') in bases:
            run_fb = True
        else:
            run_fb = False
        if str('diff') in graphs:
            run_diff = True
        else:
            run_diff = False
        if str('cknn') in graphs:
            run_cknn = True
        else:
            run_cknn = False
        if str('fuzzy') in graphs:
            run_fuzzy = True
        else:
            run_fuzzy = False
        if str('tSNE') in layouts:
            run_tSNE = True
        else:
            run_tSNE = False
        if str('MAP') in layouts:
            run_MAP = True
        else:
            run_MAP = False
        if str('MDE') in layouts:
            run_MDE = True
        else:
            run_MDE = False
        if str('PaCMAP') in layouts:
            run_PaCMAP = True
        else:
            run_PaCMAP = False
        if str('TriMAP') in layouts:
            run_TriMAP = True
        else:
            run_TriMAP = False
        if str('NCVis') in layouts:
            run_NCVis = True
        else:
            run_NCVis = False
        # Run all models and layouts
        if run_db:
            self.basis = 'diffusion'
            if self.MSDiffMap is None:
                self.fit(X=X)
            if run_PaCMAP:
                if self.db_PaCMAP is None:
                    self.db_PaCMAP = self.PaCMAP(n_components=n_components)
            if run_TriMAP:
                if self.db_TriMAP is None:
                    self.db_TriMAP = self.TriMAP(n_components=n_components)
            if run_tSNE:
                if self.db_tSNE is None:
                    self.db_tSNE = self.tSNE(n_components=n_components)
            if run_NCVis:
                if self.db_NCVis is None:
                    self.db_NCVis = self.NCVis(n_components=n_components)
            if run_diff:
                self.graph = 'diff'
                if self.db_diff_graph is None:
                    self.db_diff_graph = self.transform()
                    self.SpecLayout = self.spectral_layout(graph=self.db_diff_graph, n_components=n_components)
                if run_MAP:
                    if self.db_diff_MAP is None:
                        self.db_diff_MAP = self.MAP()
                if run_MDE:
                    if self.db_diff_MDE is None:
                        self.db_diff_MDE = self.MDE(n_components=n_components)
            if run_cknn:
                self.graph = 'cknn'
                if self.db_cknn_graph is None:
                    self.db_cknn_graph = self.transform()
                    self.SpecLayout = self.spectral_layout(graph=self.db_cknn_graph, n_components=n_components)
                if run_MAP:
                    if self.db_cknn_MAP is None:
                        self.db_cknn_MAP = self.MAP()
                if run_MDE:
                    if self.db_cknn_MDE is None:
                        self.db_cknn_MDE = self.MDE(n_components=n_components)
            if run_fuzzy:
                self.graph = 'fuzzy'
                if self.db_fuzzy_graph is None:
                    self.db_fuzzy_graph = self.transform()
                    self.SpecLayout = self.spectral_layout(graph=self.db_fuzzy_graph, n_components=n_components)
                if run_MAP:
                    if self.db_fuzzy_MAP is None:
                        self.db_fuzzy_MAP = self.MAP()
                if run_MDE:
                    if self.db_fuzzy_MDE is None:
                        self.db_fuzzy_MDE = self.MDE(n_components=n_components)
        if run_cb:
            self.basis = 'continuous'
            if self.CLapMap is None:
                self.fit(X)
            if run_PaCMAP:
                if self.cb_PaCMAP is None:
                    self.cb_PaCMAP = self.PaCMAP(n_components=n_components)
            if run_TriMAP:
                if self.cb_TriMAP is None:
                    self.cb_TriMAP = self.TriMAP(n_components=n_components)
            if run_tSNE:
                if self.cb_tSNE is None:
                    self.cb_tSNE = self.tSNE(n_components=n_components)
            if run_NCVis:
                if self.cb_NCVis is None:
                    self.cb_NCVis = self.NCVis(n_components=n_components)
            if run_diff:
                self.graph = 'diff'
                if self.cb_diff_graph is None:
                    self.cb_diff_graph = self.transform()
                    self.SpecLayout = self.spectral_layout(graph=self.cb_diff_graph, n_components=n_components)
                if run_MAP:
                    if self.cb_diff_MAP is None:
                        self.cb_diff_MAP = self.MAP()
                if run_MDE:
                    if self.cb_diff_MDE is None:
                        self.cb_diff_MDE = self.MDE(n_components=n_components)
            if run_cknn:
                self.graph = 'cknn'
                if self.cb_cknn_graph is None:
                    self.cb_cknn_graph = self.transform()
                    self.SpecLayout = self.spectral_layout(graph=self.cb_cknn_graph, n_components=n_components)
                if run_MAP:
                    if self.cb_cknn_MAP is None:
                        self.cb_cknn_MAP = self.MAP()
                if run_MDE:
                    if self.cb_cknn_MDE is None:
                        self.cb_cknn_MDE = self.MDE(n_components=n_components)
            if run_fuzzy:
                self.graph = 'fuzzy'
                if self.cb_fuzzy_graph is None:
                    self.cb_fuzzy_graph = self.transform()
                    self.SpecLayout = self.spectral_layout(graph=self.cb_fuzzy_graph, n_components=n_components)
                if run_MAP:
                    if self.cb_fuzzy_MAP is None:
                        self.cb_fuzzy_MAP = self.MAP()
                if run_MDE:
                    if self.cb_fuzzy_MDE is None:
                        self.cb_fuzzy_MDE = self.MDE(n_components=n_components)
        if run_fb:
            self.basis = 'fuzzy'
            if self.FuzzyLapMap is None:
                self.fit(X)
            if run_PaCMAP:
                if self.fb_PaCMAP is None:
                    self.fb_PaCMAP = self.PaCMAP(n_components=n_components)
            if run_TriMAP:
                if self.fb_TriMAP is None:
                    self.fb_TriMAP = self.TriMAP(n_components=n_components)
            if run_tSNE:
                if self.fb_tSNE is None:
                    self.fb_tSNE = self.tSNE(n_components=n_components)
            if run_NCVis:
                if self.fb_NCVis is None:
                    self.fb_NCVis = self.NCVis(n_components=n_components)
            if run_diff:
                self.graph = 'diff'
                if self.fb_diff_graph is None:
                    self.fb_diff_graph = self.transform()
                    self.SpecLayout = self.spectral_layout(graph=self.fb_diff_graph, n_components=n_components)
                if run_MAP:
                    if self.fb_diff_MAP is None:
                        self.fb_diff_MAP = self.MAP()
                if run_MDE:
                    if self.fb_diff_MDE is None:
                        self.fb_diff_MDE = self.MDE(n_components=n_components)
            if run_cknn:
                self.graph = 'cknn'
                if self.fb_diff_graph is None:
                    self.fb_cknn_graph = self.transform()
                    self.SpecLayout = self.spectral_layout(graph=self.fb_cknn_graph, n_components=n_components)
                if run_MAP:
                    if self.fb_cknn_MAP is None:
                        self.fb_cknn_MAP = self.MAP()
                if run_MDE:
                    if self.fb_cknn_MDE is None:
                        self.fb_cknn_MDE = self.MDE(n_components=n_components)
            if run_fuzzy:
                self.graph = 'fuzzy'
                if self.fb_diff_graph is None:
                    self.fb_fuzzy_graph = self.transform()
                    self.SpecLayout = self.spectral_layout(graph=self.fb_fuzzy_graph, n_components=n_components)
                if run_MAP:
                    if self.fb_fuzzy_MAP is None:
                        self.fb_fuzzy_MAP = self.MAP()
                if run_MDE:
                    if self.fb_fuzzy_MDE is None:
                        self.fb_fuzzy_MDE = self.MDE(n_components=n_components)

        return self



    # def plot_all_layouts(self, labels=None, pt_size=5, marker='o', opacity=1, cmap='Spectral'):
    #     """
    #
    #     Convenience function to plotting all computed layouts.
    #
    #     Parameters
    #     ----------
    #     labels : list of colors or int encoding groups, clusters or values to color by.
    #
    #     pt_size : int (default 5). Controls point size.
    #
    #     marker : str (default 'o'). Controls shape.
    #
    #     opacity : int or float (default 1). Controls opacity.
    #
    #     cmap : str (default 'Spectral'). Any of matplotlib colormaps. 'tab20' can be a good choice if you have
    #     lots of clusters.
    #
    #     Returns
    #     -------
    #     A matplotlib.pyplot with all layouts.
    #
    #     """
    #
    #     n_bases = 0
    #     n_graphs = 0
    #     n_layouts = 0
    #     embeddings = list()
    #     if self.MSDiffMap is not None:
    #         n_bases = n_bases + 1
    #     if self.CLapMap is not None:
    #         n_bases = n_bases + 1
    #     if self.FuzzyLapMap is not None:
    #         n_bases = n_bases + 1
    #     if self.DiffGraph is not None:
    #         n_graphs = n_graphs + 1
    #     if self.FuzzyGraph is not None:
    #         n_graphs = n_graphs + 1
    #     if self.CknnGraph is not None:
    #         n_graphs = n_graphs + 1
    #     if self.tSNE_Y is not None:
    #         n_layouts = n_layouts + 1
    #     if self.MAP_Y is not None:
    #         n_layouts = n_layouts + 1
    #     if self.TriMAP_Y is not None:
    #         n_layouts = n_layouts + 1
    #     if self.PaCMAP_Y is not None:
    #         n_layouts = n_layouts + 1
    #     if self.MDE_Y is not None:
    #         n_layouts = n_layouts + 1
    #     if self.db_diff_MAP is not None:
    #         embeddings.append(self.db_diff_MAP)
    #     if self.db_diff_tSNE is not None:
    #         embeddings.append(self.db_diff_tSNE)
    #     if self.db_diff_PaCMAP is not None:
    #         embeddings.append(self.db_diff_PaCMAP)
    #     if self.db_diff_TriMAP is not None:
    #         embeddings.append(self.db_diff_TriMAP)
    #     if self.db_diff_MDE is not None:
    #         embeddings.append(self.db_diff_MDE)
    #     if self.db_fuzzy_MAP is not None:
    #         embeddings.append(self.db_fuzzy_MAP)
    #     if self.db_fuzzy_tSNE is not None:
    #         embeddings.append(self.db_fuzzy_tSNE)
    #     if self.db_fuzzy_PaCMAP is not None:
    #         embeddings.append(self.db_fuzzy_PaCMAP)
    #     if self.db_fuzzy_TriMAP is not None:
    #         embeddings.append(self.db_fuzzy_TriMAP)
    #     if self.db_fuzzy_MDE is not None:
    #         embeddings.append(self.db_fuzzy_MDE)
    #     if self.db_cknn_MAP is not None:
    #         embeddings.append(self.db_cknn_MAP)
    #     if self.db_cknn_tSNE is not None:
    #         embeddings.append(self.db_cknn_tSNE)
    #     if self.db_cknn_PaCMAP is not None:
    #         embeddings.append(self.db_cknn_PaCMAP)
    #     if self.db_cknn_TriMAP is not None:
    #         embeddings.append(self.db_cknn_TriMAP)
    #     if self.db_cknn_MDE is not None:
    #         embeddings.append(self.db_cknn_MDE)
    #     if self.fb_diff_MAP is not None:
    #         embeddings.append(self.fb_diff_MAP)
    #     if self.fb_diff_tSNE is not None:
    #         embeddings.append(self.fb_diff_tSNE)
    #     if self.fb_diff_PaCMAP is not None:
    #         embeddings.append(self.fb_diff_PaCMAP)
    #     if self.fb_diff_TriMAP is not None:
    #         embeddings.append(self.fb_diff_TriMAP)
    #     if self.fb_diff_MDE is not None:
    #         embeddings.append(self.fb_diff_MDE)
    #     if self.fb_fuzzy_MAP is not None:
    #         embeddings.append(self.fb_fuzzy_MAP)
    #     if self.fb_fuzzy_tSNE is not None:
    #         embeddings.append(self.fb_fuzzy_tSNE)
    #     if self.fb_fuzzy_PaCMAP is not None:
    #         embeddings.append(self.fb_fuzzy_PaCMAP)
    #     if self.fb_fuzzy_TriMAP is not None:
    #         embeddings.append(self.fb_fuzzy_TriMAP)
    #     if self.fb_fuzzy_MDE is not None:
    #         embeddings.append(self.fb_fuzzy_MDE)
    #     if self.fb_cknn_MAP is not None:
    #         embeddings.append(self.fb_cknn_MAP)
    #     if self.fb_cknn_tSNE is not None:
    #         embeddings.append(self.fb_cknn_tSNE)
    #     if self.fb_cknn_PaCMAP is not None:
    #         embeddings.append(self.fb_cknn_PaCMAP)
    #     if self.fb_cknn_TriMAP is not None:
    #         embeddings.append(self.fb_cknn_TriMAP)
    #     if self.fb_cknn_MDE is not None:
    #         embeddings.append(self.fb_cknn_MDE)
    #     if self.cb_diff_MAP is not None:
    #         embeddings.append(self.cb_diff_MAP)
    #     if self.cb_diff_tSNE is not None:
    #         embeddings.append(self.cb_diff_tSNE)
    #     if self.cb_diff_PaCMAP is not None:
    #         embeddings.append(self.cb_diff_PaCMAP)
    #     if self.cb_diff_TriMAP is not None:
    #         embeddings.append(self.cb_diff_TriMAP)
    #     if self.cb_diff_MDE is not None:
    #         embeddings.append(self.cb_diff_MDE)
    #     if self.cb_fuzzy_MAP is not None:
    #         embeddings.append(self.cb_fuzzy_MAP)
    #     if self.cb_fuzzy_tSNE is not None:
    #         embeddings.append(self.cb_fuzzy_tSNE)
    #     if self.cb_fuzzy_PaCMAP is not None:
    #         embeddings.append(self.cb_fuzzy_PaCMAP)
    #     if self.cb_fuzzy_TriMAP is not None:
    #         embeddings.append(self.cb_fuzzy_TriMAP)
    #     if self.cb_fuzzy_MDE is not None:
    #         embeddings.append(self.cb_fuzzy_MDE)
    #     if self.cb_cknn_MAP is not None:
    #         embeddings.append(self.cb_cknn_MAP)
    #     if self.cb_cknn_tSNE is not None:
    #         embeddings.append(self.cb_cknn_tSNE)
    #     if self.cb_cknn_PaCMAP is not None:
    #         embeddings.append(self.cb_cknn_PaCMAP)
    #     if self.cb_cknn_TriMAP is not None:
    #         embeddings.append(self.cb_cknn_TriMAP)
    #     if self.cb_cknn_MDE is not None:
    #         embeddings.append(self.cb_cknn_MDE)
    #
    #     emb_number = len(embeddings)
    #
    #     fig, axes_tuple = plt.subplots(n_graphs, n_layouts)
    #
    #     if n_graphs > 1:
    #         def row_range(emb_number, n_graphs, nrow):
    #             rr = range((emb_number // n_graphs) * nrow, emb_number)
    #             return rr
    #
    #         for i in range(n_graphs):
    #             axs_tuple = axes_tuple[i]
    #             for e in range(n_layouts):
    #                 j = row_range(emb_number, n_graphs, i)
    #                 j = j[e]
    #                 axs_tuple[e].scatter(
    #                     embeddings[j][:, 0],
    #                     embeddings[j][:, 1],
    #                     cmap=cmap,
    #                     c=labels,
    #                     s=pt_size,
    #                     marker=marker,
    #                     alpha=opacity)
    #
    #
    #     else:
    #         for i in range(emb_number):
    #             axes_tuple[i].scatter(
    #                 embeddings[i][:, 0],
    #                 embeddings[i][:, 1],
    #                 cmap=cmap,
    #                 c=labels,
    #                 s=pt_size,
    #                 marker=marker,
    #                 alpha=opacity)
    #
    #     return plt.show()
    # def plot_runtime_comparison(self):

