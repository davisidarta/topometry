# TopOMetry high-level API - the TopOGraph class
#
# Author: Davi Sidarta-Oliveira <davisidarta(at)gmail(dot)com>
# School of Medical Sciences, University of Campinas, Brazil
#
import time
import numpy as np
import gc
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import issparse, csr_matrix
from topo.base.ann import kNN
from topo.tpgraph.kernels import Kernel
from topo.spectral.eigen import EigenDecomposition, spectral_layout
from topo.layouts.projector import Projector



class TopOGraph(BaseEstimator, TransformerMixin):
    """
    Main TopOMetry class for learning topological similarities, bases, graphs, and layouts from high-dimensional data.

    From data, learns topological similarity metrics, from these build orthonormal eigenbases and from these eigenbases learns
    new topological graphs. Users can choose different adaptive kernels to achieve these topological representations,
    which can approximate the Laplace-Beltrami Operator via their Laplacian or Diffusion operators.

    The eigenbasis or their topological graphs can then be visualized with multiple existing layout optimization tools.

    Parameters
    ----------
    base_knn : int (optional, default 30).
        Number of k-nearest-neighbors to use when learning topological similarities.

    graph_knn : int (optional, default 30).
        Similar to `base_knn`, but used to learning topological graphs from the orthogonal bases.

    n_eigs : int (optional, default 100).
        Number of eigenpairs to compute. 

    base_kernel : topo.tpgraph.Kernel (optional, default None).
        An optional Kernel object already fitted with the data. If available, the original input data X is not required.

    eigenmap_method : str (optional, default 'DM').
        Which eigenmap method to use. Defaults to 'DM', which is the diffusion maps method and will use the diffusion
        operator learned from the used kernel. Options include:
        * 'DM' - uses the diffusion operator learned from the used kernel (TopOGraph.kernel.P). By default, uses
        the multiscale Diffusion Maps version, which accounts for all possible diffusion timescales. Finds top eigenpairs (with highest eigenvalues).
        * 'LE' - uses the graph Laplacian learned from the used kernel (TopOGraph.kernel.L). Finds bottom eigenpairs (with lowest eigenvalues).
        * 'top' - uses the kernel matrix (TopOGraph.kernel.K) as the affinity matrix. Finds top eigenpairs (with highest eigenvalues).
        * 'bottom' - uses the kernel matrix (TopOGraph.kernel.K) as the affinity matrix. Finds bottom eigenpairs (with lowest eigenvalues).

    alpha : int or float (optional, default 1).
         Anisotropy. Alpha in the diffusion maps literature. Controls how much the results are biased by data distribution.
         Defaults to 1, which unbiases results from data underlying sampling distribution.

    laplacian_type : str (optional, default 'random_walk').
        Which Laplacian to use by default with the kernel. Defaults to 'random_walk', but 'normalized' is the most common.
        Options include 'unnormalized' (also referred to as the combinatorial Laplacian), 'normalized', 'random_walk' and 'geometric'.

    base_kernel_version : str (optional, default 'bw_adaptive')
        Which method to use for learning affinities to use in the eigenmap strategy. Defaults to 'bw_adaptive', which  employs an adaptive bandwidth. 
        There are several other options available to learn affinities, including:
        * 'fuzzy' - uses fuzzy simplicial sets as per [UMAP](). It is grounded on solid theory.
        * 'cknn' - uses continuous k-nearest-neighbors as per [Berry et al.](). As 'fuzzy', it is grounded on solid theory, and is guaranteed
        to approaximate the Laplace-Beltrami Operator on the underlying manifold.
        * 'bw_adaptive' - uses an adaptive bandwidth without adaptive exponentiation. It is a locally-adaptive kernel similar
        to that proposed by [Nadler et al.](https://doi.org/10.1016/j.acha.2005.07.004) and implemented in [Setty et al.](https://doi.org/10.1038/s41587-019-0068-4).
        * 'bw_adaptive_alpha_decaying' - uses an adaptive bandwidth and an adaptive decaying exponentiation (alpha). This is the default.
        * 'bw_adaptive_nbr_expansion' - first uses a simple adaptive bandwidth and then uses it to learn an adaptive number of neighbors
        to expand neighborhood search to learn a second adaptive bandwidth kernel. The neighborhood expansion can impact runtime, although
        this is not usually expressive for datasets under 10e6 samples. The neighborhood expansion was proposed in the TopOMetry paper.
        * 'bw_adaptive_alpha_decaying_nbr_expansion' - first uses a simple adaptive bandwidth and then uses it to learn an adaptive number of neighbors
        to expand neighborhood search to learn a second adaptive bandwidth kernel. The neighborhood expansion was proposed in the TopOMetry paper.
        * 'gaussian' - uses a fixed Gaussian kernel with a fixed bandwidth. This is the most naive approach.

    graph_kernel_version : str (optional, default 'bw_adaptive_alpha_decaying')
        Which method to use for learning affinities from the eigenbasis. Same as `base_kernel_version`, but for the graph construction
        after learning an eigenbasis.

    backend : str (optional, default 'nmslib').
        Which backend to use for neighborhood search. Options are 'nmslib', 'hnswlib', 'pynndescent','annoy', 'faiss' and 'sklearn'.
        By default it will check what you have available, in this order.

    base_metric : str (optional, default 'cosine')
        Distance metric for building an approximate kNN graph during topological basis construction. Defaults to
        'cosine'. When using scaled data (zero mean and unit variance) the cosine similarity metric is highly recommended.
        The 'hamming' and 'jaccard' distances are also available for string vectors. 

        NOTE: not all k-nearest-neighbors backends have the same metrics available! The algorithm will expect you to input a metric compatible with your backend.
        Example of accepted metrics in NMSLib(*), HNSWlib(**) and sklearn(***):

        * 'sqeuclidean' (**, ***)

        * 'euclidean' (**, ***)

        * 'l1' (*)

        * 'lp' - requires setting the parameter ``p`` (*) - similar to Minkowski

        * 'cosine' (**, ***)

        * 'inner_product' (**)

        * 'angular' (*)

        * 'negdotprod' (*)

        * 'levenshtein' (*)

        * 'hamming' (*)

        * 'jaccard' (*)

        * 'jansen-shan' (*)

    graph_metric : str (optional, default 'euclidean').
         Similar to `base_metric`, but used for building a new topological graph on top of the learned orthonormal eigenbasis.

    sigma : float (optional, default None).
        Scaling factor if using fixed bandwidth kernel (only used if `base_kernel_version` or `graph_kernel_version` is set to `gaussian`).

    delta : float (optional, default 1.0).
        A parameter of the 'cknn' kernel version. It is ignored if `graph_kernel_version` or `base_kernel_version` are not set to 'cknn'.
        It is used to decide the local radius in the continuous kNN algorithm. The combination radius increases in proportion to this parameter.

    low_memory : bool (optional, default False).
        Whether to use a low-memory implementation of the algorithm. Everything will quite much run the same way, but the TopOGraph object will
        not store the full kernel matrices obtained at each point of the algorithm. A few take-away points:

        * If you are running a single model, you should always keep this set this to `False` unless you have very little memory available.
        
        * This parameter is particularly useful when analysing very large datasets, and for cross-validation of algorithmic combinations 
        (e.g. different `base_kernel_version` and `graph_kernel_version`) when using the `TopOGraph.run_models_layouts()` function.
        This is because the kernel matrices are stored in memory, and can be quite large. After defining the best algorithmic combination, recomputing only it costs a
        fraction of the time and memory. 
        

    n_jobs : int (optional, default 1).
        Number of threads to use in neighborhood calculations. Set this to as much as possible for speed. Setting to `-1` uses all available threads.

    eigensolver : string (optional, default 'arpack').
        Method for computing the eigendecomposition. Can be either 'arpack', 'lobpcg', 'amg' or 'dense'.
        * 'dense' :
            use standard dense matrix operations for the eigenvalue decomposition.
            For this method, M must be an array or matrix type.
            This method should be avoided for large problems.
        * 'arpack' :
            use arnoldi iteration in shift-invert mode. For this method,
            M may be a dense matrix, sparse matrix, or general linear operator.
        * 'lobpcg' :
            Locally Optimal Block Preconditioned Conjugate Gradient Method.
            A preconditioned eigensolver for large symmetric positive definite
            (SPD) generalized eigenproblems.
        * 'amg' :
            Algebraic Multigrid solver (requires ``pyamg`` to be installed)
            It can be faster on very large, sparse problems, but requires
            setting a random seed for better reproducibility.

    eigen_tol : float (optional, default 0.0).
        Error tolerance for the eigenvalue solver. If 0, machine precision is used.

    diff_t : int (optional, default None).
        Time parameter for the diffusion operator, if `eigenmap_method` is 'DM'. Also works with 'eigenmap_method' being 'LE'. 
        The diffusion operator or the graph Laplacian will be powered by t steps (a diffusion process). Ignored for other methods.
        If None, the number of k-nearest neighbors is used.

    semi_aniso : bool (optional, default False).
        Whether to use semi-anisotropic diffusion, if `eigenmap_method` is 'DM'. This reweights the original kernel (not the renormalized kernel) by the renormalized degree.

    projection_method : str (optional, default 'MAP').
        Which projection method to use. Only 'Isomap', 't-SNE' and 'MAP' are implemented out of the box without need to import packages.
        't-SNE' uses scikit-learn if the user does not have multicore-tsne and 'MAP' relies on code that is adapted from UMAP for efficiency.
        Current options are:
            * ['Isomap']() - one of the first manifold learning methods
            * ['t-SNE'](https://github.com/DmitryUlyanov/Multicore-TSNE) - a classic manifold learning method
            * 'MAP'- a lighter [UMAP](https://umap-learn.readthedocs.io/en/latest/index.html) with looser assumptions
            * ['UMAP'](https://umap-learn.readthedocs.io/en/latest/index.html)
            * ['PaCMAP'](http://jmlr.org/papers/v22/20-1061.html) (Pairwise-controlled Manifold Approximation and Projection) - for balanced visualizations (requires installing `pacmap`)
            * ['TriMAP'](https://github.com/eamid/trimap) - dimensionality reduction using triplets (requires installing `trimap`)
            * 'IsomorphicMDE' - [MDE](https://github.com/cvxgrp/pymde) with preservation of nearest neighbors (requires installing `pymde`)
            * 'IsometricMDE' - [MDE](https://github.com/cvxgrp/pymde) with preservation of pairwise distances (requires installing `pymde`)
            * ['NCVis'](https://github.com/stat-ml/ncvis) (Noise Contrastive Visualization) - a UMAP-like method with blazing fast performance (requires installing `ncvis`)
        These are frankly quite direct to add, so feel free to make a feature request if your favorite method is not listed here.

    verbosity : int (optional, default 1).
        Controls verbosity. 0 for no verbosity, 1 for minimal (prints warnings and runtimes of major steps), 2 for
        medium (also prints layout optimization messages) and 3 for full (down to neighborhood search, useful for debugging).

    cache :  bool (optional, default True).
        Whether to cache kernel and eigendecomposition classes in the results dictionaries.
        Set to `False` to avoid unnecessary duplications if you're using a single model with large data.

    random_state : int or numpy.random.RandomState() (optional, default None).
        A pseudo random number generator. Used in eigendecomposition when `eigensolver` is 'amg',

    Attributes
    ----------
        BaseKernelDict : dict
            A dictionary of base kernel classes, with keys being the kernel version and values being the kernel class.

        EigenbasisDict : dict
            A dictionary of eigendecomposition classes, with keys referring to the kernel version and eigenmap method
            and values being the eigendecomposition class.

        GraphKernelDict : dict  
            A dictionary of graph kernel classes, with keys referring to the base kernel, eigendecomposition method,
            and graph kernel version used, and and values being the kernel class.

        self.SpecLayout : np.ndarray (n_samples, n_components)
            The spectral layout of the data.

        self.ProjectionDict : dict
            A dictionary of graph projection coordinates, with keys referring to the projection method and values being the projection class.

        ClustersDict : dict
            A dictionary of cluster labels, with keys referring to the eigenbasis used and values being the cluster labels.

        current_basekernel : `topo.tpgraph.Kernel` object
            The current base kernel class. 

        current_eigenbasis : `topo.eigen.EigenDecomposition` object
            The current eigendecomposition class.

        current_graphkernel : `topo.tpgraph.Kernel` object
            The current graph kernel class (Kernel class fitted on the eigendecomposition results).

        global_dimensionality : float or None
            The global dimensionality of the data, if it has been estimated.

        local_dimensionality : np.ndarray (n_samples,)
            The local pairwise dimensionality of the data, if it has been estimated.


    """

    def __init__(self,
                 base_knn=30,
                 graph_knn=30,
                 n_eigs=100,
                 base_kernel=None,
                 base_kernel_version='bw_adaptive',
                 eigenmap_method='DM',
                 laplacian_type='normalized',
                 projection_method='MAP',
                 graph_kernel_version='bw_adaptive',
                 base_metric='cosine',
                 graph_metric='euclidean',
                 alpha=1.0,
                 diff_t=None,
                 semi_aniso=False,
                 delta=1.0,
                 sigma=0.1,
                 n_jobs=1,
                 low_memory=False,
                 eigen_tol=1e-4,
                 eigensolver='arpack',
                 backend='hnswlib',
                 cache=True,
                 verbosity=1,
                 random_state=42,
                 ):
        self.projection_method = projection_method
        self.diff_t = diff_t
        self.n_eigs = n_eigs
        self.base_knn = base_knn
        self.graph_knn = graph_knn
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.low_memory = low_memory
        self.backend = backend
        self.base_metric = base_metric
        self.graph_metric = graph_metric
        self.eigen_tol = eigen_tol
        self.eigensolver = eigensolver
        self.base_kernel = base_kernel
        self.base_kernel_version = base_kernel_version
        self.eigenmap_method = eigenmap_method
        self.graph_kernel_version = graph_kernel_version
        self.laplacian_type = laplacian_type
        self.semi_aniso = semi_aniso
        self.eigenbasis = None
        self.graph_kernel = None
        self.verbosity = verbosity
        self.sigma = sigma
        self.bases_graph_verbose = False
        self.layout_verbose = False
        self.delta = delta
        self.random_state = random_state
        self.eigenbasis_knn_graph = None
        self.base_nbrs_class = None
        self.base_knn_graph = None
        self.cache = cache
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
        
    def __repr__(self):
        if self.base_metric == 'precomputed':
            msg = "TopOGraph object with precomputed distances from %i samples" % (
                self.n) + " and:"
        elif (self.n is not None) and (self.m is not None):
            msg = "TopOGraph object with %i samples and %i observations" % (
                self.n, self.m) + " and:"
        else:
            msg = "TopOGraph object without any fitted data."
        msg = msg + "\n . Base Kernels:"
        for keys in self.BaseKernelDict.keys():
            msg = msg + " \n    %s - .BaseKernelDict['%s']" % (keys, keys)
        msg = msg + "\n . Eigenbases:"
        for keys in self.EigenbasisDict.keys():
            msg = msg + " \n    %s - .EigenbasisDict['%s']" % (keys, keys)
        msg = msg + "\n . Graph Kernels:"
        for keys in self.GraphKernelDict.keys():
            msg = msg + " \n    %s - .GraphKernelDict['%s']" % (keys, keys)
        msg = msg + "\n . Projections:"
        for keys in self.ProjectionDict.keys():
            msg = msg + " \n    %s - .ProjectionDict['%s']" % (keys, keys)

        msg = msg + " \n Active base kernel  -  .base_kernel"
        msg = msg + " \n Active eigenbasis  -  .eigenbasis"
        msg = msg + " \n Active graph kernel  -  .graph_kernel"

        return msg

    def _parse_backend(self):
        try:
            import hnswlib
            self._have_hnswlib = True
        except ImportError:
            self._have_hnswlib = False
        try:
            import nmslib
            self._have_nmslib = True
        except ImportError:
            self._have_nmslib = False
        try:
            import annoy
            self._have_annoy = True
        except ImportError:
            self._have_annoy = False
        try:
            import faiss
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
            print(
                "Warning: no approximate nearest neighbor library found. Using sklearn's KDTree instead.")
            self.backend == 'sklearn'

    def _parse_random_state(self):
        if self.random_state is None:
            self.random_state = np.random.RandomState()
        elif isinstance(self.random_state, np.random.RandomState):
            pass
        elif isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        else:
            print('RandomState error! No random state was defined!')

    def fit(self, X=None, **kwargs):
        """
        Learn distances, computes similarities with various kernels and applies eigendecompositions to them or their Laplacian or diffusion operators.
        The learned operators approximate the Laplace-Beltrami Operator and learn the topology of the underlying manifold. The eigenbasis learned from
        such eigendecomposition represents the data in a lower-dimensional space (with up to some hundreds dimensions),
        where the euclidean distances between points are approximated by the geodesic distances on the manifold. Because the eigenbasis is equivalent
        to a Fourier basis, one needs at least `n+1` to eigenvectors to represent a dataset that can be optimically divided into `n` clusters.


        Parameters
        ----------
        X : High-dimensional data matrix.
             Currently, supports only data from similar type (i.e. all bool, all float).
             Not required if the `base_kernel` parameter is specified and corresponds to a fitted `topo.tpgraph.Kernel()` object.

        **kwargs : Additional parameters to be passed to the `topo.base.ann.kNN()` function.

        Returns
        -------

        TopoGraph object with a populated `TopOGraph.EigenbasisDict` dictionary. The keys of the dictionary are the names of the eigendecompositions
        that were performed. The values are the corresponding `topo.base.eigen.Eigenbasis()` objects. The latest set eigenbasis is
        also accessible through the `TopOGraph.eigenbasis` attribute.


        """
        if self.base_kernel_version not in ['fuzzy', 'cknn', 'bw_adaptive', 'bw_adaptive_alpha_decaying', 'bw_adaptive_nbr_expansion', 'bw_adaptive_alpha_decaying_nbr_expansion', 'gaussian']:
            raise ValueError(
                "base_kernel_version must be one of ['fuzzy', 'cknn', 'bw_adaptive', 'bw_adaptive_alpha_decaying', 'bw_adaptive_nbr_expansion', 'bw_adaptive_alpha_decaying_nbr_expansion', 'gaussian']")
        if self.graph_kernel_version not in ['fuzzy', 'cknn', 'bw_adaptive', 'bw_adaptive_alpha_decaying', 'bw_adaptive_nbr_expansion', 'bw_adaptive_alpha_decaying_nbr_expansion', 'gaussian']:
            raise ValueError(
                "graph_kernel_version must be one of ['fuzzy', 'cknn', 'bw_adaptive', 'bw_adaptive_alpha_decaying', 'bw_adaptive_nbr_expansion', 'bw_adaptive_alpha_decaying_nbr_expansion', 'gaussian']")
        if self.eigenmap_method not in ['msDM', 'DM', 'LE', 'top', 'bottom']:
            raise ValueError(
                "eigenmap_method must be one of ['msDM', 'DM', 'LE', 'top', 'bottom']")
        if (self.n_eigs - 1 > X.shape[0]) or (self.n_eigs - 1 > X.shape[1]):
            raise ValueError(
                "n_eigs must be less than the number of samples and observations in X")
        if not isinstance(self.n_eigs, int):
            raise ValueError("n_eigs must be an integer")
        if not isinstance(self.base_knn, int):
            raise ValueError("base_knn must be an integer")
        if not isinstance(self.graph_knn, int):
            raise ValueError("graph_knn must be an integer")
        
        self._parse_backend()
        self._parse_random_state()
        if self.diff_t is None:
            self.diff_t = self.base_knn
        if X is None:
            if self.base_kernel is None:
                raise ValueError('X was not passed!')
            elif not isinstance(self.base_kernel, Kernel):
                raise ValueError(
                    'The specified base kernel is not a topo.tpgraph.Kernel object!')
            else:
                if self.base_kernel.knn_ is None:
                    raise ValueError(
                        'The specified base kernel has not been fitted!')
                else:
                    self.n = self.base_kernel.knn_.shape[0]
                    self.m = self.base_kernel.knn_.shape[1]
        else:
            if self.base_metric == 'precomputed':
                self.base_knn_graph = X.copy()
            self.n = X.shape[0]
            self.m = X.shape[1]

        # parse other inputs
        if self.n_jobs == -1:
            from joblib import cpu_count
            self.n_jobs = cpu_count()

        if self.verbosity >= 2:
            self.layout_verbose = True
            if self.verbosity == 3:
                self.bases_graph_verbose = True
            else:
                self.bases_graph_verbose = False
        else:
            self.layout_verbose = False

        if self.base_kernel is not None:
            if not isinstance(self.base_kernel, Kernel):
                print(
                    "WARNING: kernel must be an instance of the Kernel class. Creating a default kernel...")
                self.base_kernel = None
            if self.base_kernel.knn_ is not None:
                self.base_knn_graph = self.base_kernel.knn_

        if self.base_knn_graph is None:
            if self.verbosity >= 1:
                print('Computing neighborhood graph...')
            start = time.time()
            self.base_nbrs_class, self.base_knn_graph = kNN(X, n_neighbors=self.base_knn,
                                                            metric=self.base_metric,
                                                            n_jobs=self.n_jobs,
                                                            backend=self.backend,
                                                            return_instance=True,
                                                            verbose=self.bases_graph_verbose, **kwargs)
            end = time.time()
            gc.collect()
            self.runtimes['kNN'] = end - start
            if self.verbosity >= 1:
                print(' Base kNN graph computed in %f (sec)' % (end - start))

        if self.base_kernel_version in self.BaseKernelDict.keys():
            self.base_kernel = self.BaseKernelDict[self.base_kernel_version]
        else:
            start = time.time()
            self.base_kernel, self.BaseKernelDict = self._compute_kernel_from_version_knn(self.base_knn_graph,
                                                                                          self.base_knn,
                                                                                          self.base_kernel_version,
                                                                                          self.BaseKernelDict,
                                                                                          suffix='',
                                                                                          low_memory=self.low_memory,
                                                                                          data_for_expansion=X,
                                                                                          base=True)
            end = time.time()
            gc.collect()
            if self.verbosity >= 1:
                print(' Fitted the ' + (self.base_kernel_version) +
                      ' kernel in %f (sec)' % (end - start))

        if self.verbosity >= 1:
            print('Computing eigenbasis...')

        if self.eigenmap_method == 'msDM':
            basis_key = 'msDM with ' + str(self.base_kernel_version)
            if basis_key in self.EigenbasisDict.keys():
                self.eigenbasis = self.EigenbasisDict[basis_key]
                self.current_eigenbasis = basis_key
            else:
                start = time.time()
                self.eigenbasis = EigenDecomposition(n_components=self.n_eigs,
                                                     method=self.eigenmap_method,
                                                     eigensolver=self.eigensolver,
                                                     eigen_tol=self.eigen_tol,
                                                     drop_first=True,
                                                     weight=True,
                                                     t=self.diff_t,
                                                     random_state=self.random_state,
                                                     verbose=self.bases_graph_verbose).fit(self.base_kernel)
                self.EigenbasisDict[basis_key] = self.eigenbasis
                self.current_eigenbasis = basis_key
                end = time.time()
                gc.collect()
                self.runtimes[basis_key] = end - start
                if self.verbosity >= 1:
                    print(' Fitted eigenbasis with multiscale Diffusion Maps from the ' +
                          str(self.base_kernel_version) + ' kernel in %f (sec)' % (end - start))

        if self.eigenmap_method == 'DM':
            basis_key = 'DM with ' + str(self.base_kernel_version)
            if basis_key in self.EigenbasisDict.keys():
                self.eigenbasis = self.EigenbasisDict[basis_key]
                self.current_eigenbasis = basis_key
            else:
                start = time.time()
                self.eigenbasis = EigenDecomposition(n_components=self.n_eigs,
                                                     method=self.eigenmap_method,
                                                     eigensolver=self.eigensolver,
                                                     eigen_tol=self.eigen_tol,
                                                     drop_first=True,
                                                     weight=True,
                                                     t=self.diff_t,
                                                     random_state=self.random_state,
                                                     verbose=self.bases_graph_verbose).fit(self.base_kernel)
                self.EigenbasisDict[basis_key] = self.eigenbasis
                self.current_eigenbasis = basis_key
                end = time.time()
                gc.collect()
                self.runtimes[basis_key] = end - start
                if self.verbosity >= 1:
                    print(' Fitted eigenbasis with Diffusion Maps from the ' +
                          str(self.base_kernel_version) + ' kernel in %f (sec)' % (end - start))

        elif self.eigenmap_method == 'LE':
            basis_key = 'LE with ' + str(self.base_kernel_version)
            if basis_key in self.EigenbasisDict.keys():
                self.eigenbasis = self.EigenbasisDict[basis_key]
                self.current_eigenbasis = basis_key
            else:
                start = time.time()
                self.eigenbasis = EigenDecomposition(n_components=self.n_eigs,
                                                     method=self.eigenmap_method,
                                                     eigensolver=self.eigensolver,
                                                     eigen_tol=self.eigen_tol,
                                                     drop_first=True,
                                                     weight=True,
                                                     t=self.diff_t,
                                                     random_state=self.random_state,
                                                     verbose=self.bases_graph_verbose).fit(self.base_kernel)
                self.EigenbasisDict[basis_key] = self.eigenbasis
                self.current_eigenbasis = basis_key
                end = time.time()
                gc.collect()
                self.runtimes[basis_key] = end - start
                if self.verbosity >= 1:
                    print(' Fitted eigenbasis with Laplacian Eigenmaps from the ' +
                          str(self.base_kernel_version) + ' in %f (sec)' % (end - start))

        elif self.eigenmap_method == 'top':
            basis_key = 'Top eigenpairs with ' + str(self.base_kernel_version)
            if basis_key in self.EigenbasisDict.keys():
                self.eigenbasis = self.EigenbasisDict[basis_key]
                self.current_eigenbasis = basis_key
            else:
                start = time.time()
                self.eigenbasis = EigenDecomposition(n_components=self.n_eigs,
                                                     method=self.eigenmap_method,
                                                     eigensolver=self.eigensolver,
                                                     eigen_tol=self.eigen_tol,
                                                     drop_first=True,
                                                     weight=True,
                                                     t=self.diff_t,
                                                     random_state=self.random_state,
                                                     verbose=self.bases_graph_verbose).fit(self.base_kernel)
                self.EigenbasisDict[basis_key] = self.eigenbasis
                self.current_eigenbasis = basis_key
                end = time.time()
                gc.collect()
                self.runtimes[basis_key] = end - start
                if self.verbosity >= 1:
                    print(' Fitted eigenbasis with top eigenpairs from the ' +
                          str(self.base_kernel_version) + ' in %f (sec)' % (end - start))

        elif self.eigenmap_method == 'bottom':
            if basis_key in self.EigenbasisDict.keys():
                self.eigenbasis = self.EigenbasisDict[basis_key]
                self.current_eigenbasis = basis_key
            else:
                basis_key = 'Bottom eigenpairs with ' + \
                    str(self.base_kernel_version)
            start = time.time()
            self.eigenbasis = EigenDecomposition(n_components=self.n_eigs,
                                                 method=self.eigenmap_method,
                                                 eigensolver=self.eigensolver,
                                                 eigen_tol=self.eigen_tol,
                                                 drop_first=True,
                                                 weight=True,
                                                 t=self.diff_t,
                                                 random_state=self.random_state,
                                                 verbose=self.bases_graph_verbose).fit(self.base_kernel)
            basis_key = 'Bottom eigenpairs with ' + \
                str(self.base_kernel_version)
            self.EigenbasisDict[basis_key] = self.eigenbasis
            self.current_eigenbasis = basis_key
            end = time.time()
            gc.collect()
            self.runtimes[basis_key] = end - start
            if self.verbosity >= 1:
                print(' Fitted eigenbasis with bottom eigenpairs from from the ' +
                      str(self.base_kernel_version) + ' in %f (sec)' % (end - start))

        return self

    def fit_transform(self, X=None):
        self.fit(X)
        gc.collect()
        return self.transform(X=None)

    def eigenspectrum(self, eigenbasis_key=None, **kwargs):
        """
        Visualize the eigenspectrum decay. Corresponds to a scree plot of information entropy.
        Useful to indirectly estimate the intrinsic dimensionality of a dataset.

        Parameters
        ----------
        `eigenbasis_key` : str (optional, default None).
            If `None`, will use the default eigenbasis at `TopOGraph.eigenbasis`. Otherwise, uses the specified eigenbasis.

        Returns
        -------
        A nice eigenspectrum decay plot ('scree plot').

        """
        if eigenbasis_key is not None:
            if isinstance(eigenbasis_key, str):
                if eigenbasis_key in self.EigenbasisDict.keys():
                    eigenbasis = self.EigenbasisDict[eigenbasis_key]
                else:
                    raise ValueError(
                        'Eigenbasis key not in TopOGraph.EigenbasisDict.')
        else:
            eigenbasis = self.eigenbasis
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return print('Error: Matplotlib not found!')
        from topo.plot import decay_plot
        return decay_plot(evals=eigenbasis.eigenvalues, title=eigenbasis_key, **kwargs)
    
    def plot_eigenspectrum(self, eigenbasis_key=None, **kwargs):
        """
        An anlias for `TopOGraph.eigenspectrum`. Visualize the eigenspectrum decay. Corresponds to a scree plot of information entropy.
        Useful to indirectly estimate the intrinsic dimensionality of a dataset.

        Parameters
        ----------
        `eigenbasis_key` : str (optional, default None).
            If `None`, will use the default eigenbasis at `TopOGraph.eigenbasis`. Otherwise, uses the specified eigenbasis.

        Returns
        -------
        A nice eigenspectrum decay plot ('scree plot').

        """
        return self.eigenspectrum(eigenbasis_key=eigenbasis_key, **kwargs)

    def list_eigenbases(self):
        """
        List the eigenbases in the `TopOGraph.EigenbasisDict`.

        Returns
        -------
        A list of eigenbasis keys in the `TopOGraph.EigenbasisDict`.

        """
        return list(self.EigenbasisDict.keys())

    def transform(self, X=None, **kwargs):
        """
        Learns new affinity, topological operators from chosen eigenbasis. This does not strictly follow scikit-learn's API,
        no new data can be transformed with this method. Instead, this method is used to learn new topological operators
        on top of the learned eigenbasis, analagous to spectral clustering methods.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features), optional
            New data to transform. This is not used in this method, but is included for compatibility with scikit-learn's API.


        Returns
        -------
        scipy.sparse.csr.csr_matrix, containing the similarity matrix that encodes the topological graph.

        """
        if self.eigenbasis is not None:
            eigenbasis = self.eigenbasis
        else:
            raise ValueError('No eigenbasis computed. Call .fit() first.')
        if self.verbosity >= 1:
            print('    Building topological graph from eigenbasis...')
        if self.verbosity >= 1:
            print('        Computing neighborhood graph...')
        target = eigenbasis.transform(X=None)[:, 0:eigenbasis.eigengap]
        start = time.time()
        self.eigenbasis_knn_graph = kNN(target, n_neighbors=self.graph_knn,
                                        metric=self.graph_metric,
                                        n_jobs=self.n_jobs,
                                        backend=self.backend,
                                        return_instance=False,
                                        verbose=self.bases_graph_verbose, **kwargs)
        end = time.time()
        gc.collect()
        self.runtimes['Graph kNN'] = end - start
        if self.verbosity >= 1:
            print(' Computed in %f (sec)' % (end - start))
        start = time.time()
        self.graph_kernel, self.GraphKernelDict = self._compute_kernel_from_version_knn(self.eigenbasis_knn_graph,
                                                                                        self.graph_knn,
                                                                                        self.graph_kernel_version,
                                                                                        self.GraphKernelDict,
                                                                                        suffix=' from ' + self.current_eigenbasis,
                                                                                        low_memory=self.low_memory,
                                                                                        data_for_expansion=eigenbasis.transform(X=None),
                                                                                        base=False)
        
        end = time.time()
        gc.collect()
        self.current_graphkernel = self.graph_kernel_version + \
            ' from ' + self.current_eigenbasis
        if self.verbosity >= 1:
            print(' Fitted the ' + str(self.graph_kernel_version) +
                  ' graph kernel in %f (sec)' % (end - start))

    def spectral_layout(self, graph=None, n_components=2):
        """

        Performs a multicomponent spectral layout of the data and the target similarity matrix.

        Parameters
        ----------
        graph : str indicating a `TopOGraph.EigenbasisDict` key or or array-like, optional.
            Graph kernel to use for the spectral layout. Defaults to the active graph kernel (`TopOGraph.graph_kernel`).

        n_components : int (optional, default 2).
            Number of dimensions to embed into.

        Returns
        -------
        np.ndarray containing the resulting embedding.

        """
        if graph is None:
            if self.graph_kernel is None:
                raise ValueError(
                    'No graph kernel computed. Call .fit() first.')
            graph = self.graph_kernel.K
        start = time.time()
        try:
            spt_layout = spectral_layout(graph, n_components, self.random_state,
                                        laplacian_type=self.laplacian_type, eigen_tol=self.eigen_tol, return_evals=False)
            expansion = 10.0 / np.abs(spt_layout).max()
            spt_layout = (spt_layout * expansion).astype(
                np.float32
            ) + self.random_state.normal(
                scale=0.0001, size=[graph.shape[0], n_components]
            ).astype(np.float32)
        except:
            spt_layout = EigenDecomposition(
                            n_components=n_components).fit_transform(graph)
        end = time.time()
        self.runtimes['Spectral'] = end - start
        self.SpecLayout = spt_layout
        gc.collect()
        return spt_layout

    def project(self, n_components=2, init=None, projection_method=None, landmarks=None, landmark_method='kmeans', n_neighbors=None, num_iters=500, **kwargs):
        """
        Projects the data into a lower dimensional space using the specified projection method. Calls topo.layout.Projector().

        Parameters
        ----------
        n_components : int (optional, default 2).
            Number of dimensions to optimize the layout to. Usually 2 or 3 if you're into visualizing data.

        init : str or np.ndarray (optional, default None).
            If passed as `str`, will use the specified layout as initialization. If passed as `np.ndarray`, will use the array as initialization.

        projection_method : str (optional, default 'MAP').
            Which projection method to use. Only 'Isomap', 't-SNE' and 'MAP' are implemented out of the box. 't-SNE' uses scikit-learn implementation
            and 'MAP' relies on code that is adapted from UMAP. Current options are:
            * 'Isomap'
            * 't-SNE'
            * 'MAP'
            * 'UMAP'
            * 'PaCMAP'
            * 'TriMAP'
            * 'IsomorphicMDE' - MDE with preservation of nearest neighbors
            * 'IsometricMDE' - MDE with preservation of pairwise distances
            * 'NCVis'
            These are frankly quite direct to add, so feel free to make a feature request if your favorite method is not listed here.

        landmarks : int or np.ndarray (optional, default None).
            If passed as `int`, will obtain the number of landmarks. If passed as `np.ndarray`, will use the specified indexes in the array.
            Any value other than `None` will result in only the specified landmarks being used in the layout optimization, and will
            populate the Projector.landmarks_ slot.

        landmark_method : str (optional, default 'kmeans').
            The method to use for selecting landmarks. If `landmarks` is passed as an `int`, this will be used to select the landmarks.
            Can be either 'kmeans' or 'random'.

        num_iters : int (optional, default 1000).
            Most (if not all) methods optimize the layout up to a limit number of iterations. Use this parameter to set this number.

        **kwargs : dict (optional).
            Additional keyword arguments to pass to the projection method during fit.

        Returns
        -------
        np.ndarray containing the resulting embedding. Also stores it in the `TopOGraph.ProjectionDict` slot.

        """
        if n_neighbors is None:
            n_neighbors = self.graph_knn
        elif not isinstance(n_neighbors,int):
            raise ValueError('n_neighbors must be an integer')
        
        if projection_method is None:
            projection_method = self.projection_method

        if projection_method in ['MAP', 'UMAP', 'MDE', 'Isomap']:
            metric = 'precomputed'
            input = self.graph_kernel.P
            key = self.current_graphkernel
        else:
            metric = self.graph_metric
            input = self.eigenbasis.transform(X=None)
            key = self.current_eigenbasis
        if init is not None:
            if isinstance(init, np.ndarray):
                if np.shape(init)[1] != n_components:
                    raise ValueError(
                        'The specified initialization has the wrong number of dimensions.')
                else:
                    init_Y = init
            elif isinstance(init, str):
                if init in self.ProjectionDict.keys():
                    init_Y = self.ProjectionDict[init]
                else:
                    raise ValueError(
                        'No projection found with the name ' + init + '.')
        else:
            if self.SpecLayout is not None:
                if np.shape(self.SpecLayout)[1] != n_components:
                    self.SpecLayout = self.spectral_layout(
                            n_components=n_components)
            else:
                self.SpecLayout = self.spectral_layout(
                n_components=n_components)  
            init_Y = self.SpecLayout

        projection_key = projection_method + ' of ' + key
        start = time.time()
        Y = Projector(n_components=n_components,
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
                      verbose=self.layout_verbose).fit_transform(input, **kwargs)
        end = time.time()
        gc.collect()
        self.runtimes[projection_key] = end - start
        if self.verbosity >= 1:
            print(' Computed ' + projection_method +
                  ' in %f (sec)' % (end - start))
        self.ProjectionDict[projection_key] = Y
        return Y

    
    def run_models(self, X,
                   kernels=['fuzzy', 'cknn', 'bw_adaptive'],
                   eigenmap_methods=['DM', 'LE', 'top'],
                   projections=['Isomap', 'MAP']):
        """
        Power function that runs all models in TopOMetry.
        It iterates through the specified kernel versions, eigenmap methods and projection methods.
        As expected, it can take a while to run, depending on how many kernels and methods you specify.
        At least one kernel, one eigenmap method and one projection *must* be specified.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data. Use a sparse matrix for efficiency.

        kernels : list of str (optional, default ['fuzzy', 'cknn', 'bw_adaptive']).
            List of kernel versions to run. These will be used to learn an eigenbasis and to learn a new graph kernel from it.
            Options are:
            * 'fuzzy'
            * 'cknn'
            * 'bw_adaptive'
            * 'bw_adaptive_alpha_decaying'
            * 'bw_adaptive_nbr_expansion'
            * 'bw_adaptive_alpha_decaying_nbr_expansion'
            * 'gaussian'
            Will not run all by default to avoid long waiting times in reckless calls.

        eigenmap_methods : list of str (optional, default ['DM', 'LE', 'top']).
            List of eigenmap methods to run. Options are:
            * 'DM'
            * 'LE'
            * 'top'
            * 'bottom'

        projections : list of str (optional, default ['Isomap', 'MAP']).
            List of projection methods to run. Options are the same of the `topo.layouts.Projector()` object:
            * ['(L)Isomap']() - one of the first manifold learning methods
            * ['t-SNE'](https://github.com/DmitryUlyanov/Multicore-TSNE) - a classic manifold learning method
            * 'MAP'- a lighter [UMAP](https://umap-learn.readthedocs.io/en/latest/index.html) with looser assumptions
            * ['UMAP'](https://umap-learn.readthedocs.io/en/latest/index.html)
            * ['PaCMAP'](http://jmlr.org/papers/v22/20-1061.html) (Pairwise-controlled Manifold Approximation and Projection) - for balanced visualizations
            * ['TriMAP'](https://github.com/eamid/trimap) - dimensionality reduction using triplets
            * 'IsomorphicMDE' - [MDE](https://github.com/cvxgrp/pymde) with preservation of nearest neighbors
            * 'IsometricMDE' - [MDE](https://github.com/cvxgrp/pymde) with preservation of pairwise distances
            * ['NCVis'](https://github.com/stat-ml/ncvis) (Noise Contrastive Visualization) - a UMAP-like method with blazing fast performance

        Returns
        -------
        A `TopOGraph` object with results stored at different slots.
            * Base kernel results are stored at `TopOGraph.BaseKernelDict`.
            * Eigenbases results are stored at `TopOGraph.EigenbasisDict`.
            * Graph kernel results learned from eigenbases are stored at `TopOGraph.GraphKernelDict`.
            * Projection results are stored at `TopOGraph.ProjectionDict`.
        """
        for kernel in kernels:
            self.base_kernel_version = kernel
            for eig_method in eigenmap_methods:
                self.eigenmap_method = eig_method
                self.fit(X)
                gc.collect()
                for kernel in kernels:
                    self.graph_kernel_version = kernel
                    self.transform(X)
                    gc.collect()
                    for projection in projections:
                        self.project(projection_method=projection)
                        gc.collect()

    def write_pkl(self, filename='topograph.pkl', remove_base_class=True):
        try:
            import pickle
        except ImportError:
            return (print('Pickle is needed for saving the TopOGraph. Please install it with `pip3 install pickle`'))

        if self.base_nbrs_class is not None:
            if remove_base_class:
                self.base_nbrs_class = None
            else:
                raise ValueError(
                    'TopOGraph cannot be pickled with the NMSlib base class.')

        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        gc.collect()
        return print('TopOGraph saved at ' + filename)

    # def memory_saver(self, save_temp=True):
    #     """
    #     Removes all the intermediate results from the `TopOGraph` object.
    #     Optionally, also saves these results in a temporary file.

    #     """
    #     if save_temp:
    #         import tempfile
    #         import pickle
    #         temp = tempfile.NamedTemporaryFile(delete=False)
    #         self.write_pkl(filename=temp.name, remove_base_class=True)
    #         self.temp_file = temp.name

    #     self.base_nbrs_class = None
    #     self.BaseKernelDict = {}
    #     self.EigenbasisDict = {}
    #     self.GraphKernelDict = {}
    #     self.ProjectionDict = {}




    def _compute_kernel_from_version_knn(self, knn, n_neighbors, kernel_version, results_dict, prefix='', suffix='', low_memory=False, base=True, data_for_expansion=None):
        import gc
        gc.collect()
        kernel_key = kernel_version
        if prefix is not None:
            kernel_key = prefix + kernel_key
        if suffix is not None:
            kernel_key = kernel_key + suffix
        if kernel_key in results_dict.keys():
            kernel = results_dict[kernel_key]
            return kernel, results_dict
        else:
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
                                semi_aniso=self.semi_aniso,
                                anisotropy=self.alpha,
                                cache_input=False,
                                verbose=self.bases_graph_verbose,
                                random_state=self.random_state).fit(knn)
                gc.collect()
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
                                semi_aniso=self.semi_aniso,
                                anisotropy=self.alpha,
                                cache_input=False,
                                verbose=self.bases_graph_verbose,
                                random_state=self.random_state).fit(knn)
                gc.collect()
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
                                semi_aniso=self.semi_aniso,
                                anisotropy=self.alpha,
                                cache_input=False,
                                verbose=self.bases_graph_verbose,
                                random_state=self.random_state).fit(knn)
                gc.collect()
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
                                semi_aniso=self.semi_aniso,
                                anisotropy=self.alpha,
                                cache_input=False,
                                verbose=self.bases_graph_verbose,
                                random_state=self.random_state).fit(knn)
                gc.collect()
                results_dict[kernel_key] = kernel

            elif kernel_version == 'bw_adaptive_nbr_expansion':
                if data_for_expansion is None:
                    raise ValueError('data_for_expansion is None. Please provide data for neighborhood expansion when using the `bw_adaptive_nbr_expansion`.')
                if base:
                    use_metric = self.base_metric
                else:
                    use_metric = self.graph_metric
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
                                semi_aniso=self.semi_aniso,
                                anisotropy=self.alpha,
                                cache_input=False,
                                verbose=self.bases_graph_verbose,
                                random_state=self.random_state).fit(knn)
                gc.collect()
                results_dict[kernel_key] = kernel

            elif kernel_version == 'bw_adaptive_alpha_decaying_nbr_expansion':
                if data_for_expansion is None:
                    raise ValueError('data_for_expansion is None. Please provide data for neighborhood expansion when using the `bw_adaptive_nbr_expansion`.')
                if base:
                    use_metric = self.base_metric
                else:
                    use_metric = self.graph_metric
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
                                semi_aniso=self.semi_aniso,
                                anisotropy=self.alpha,
                                cache_input=False,
                                verbose=self.bases_graph_verbose,
                                random_state=self.random_state).fit(knn)
                gc.collect()
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
                                semi_aniso=self.semi_aniso,
                                anisotropy=self.alpha,
                                cache_input=False,
                                verbose=self.bases_graph_verbose,
                                random_state=self.random_state).fit(knn)
                gc.collect()
                results_dict[kernel_key] = kernel
            if low_memory:
                import gc
                del results_dict[kernel_key]
                gc.collect()

        return kernel, results_dict

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
        Evaluates all orthogonal bases, topological graphs and layouts in the TopOGraph object.
        Compares results with PCA and PCA-derived layouts (i.e. t-SNE, UMAP etc).

        Parameters
        --------------

        X : data matrix. Expects either numpy.ndarray or scipy.sparse.csr_matrix.

        landmarks : optional (int, default None).
            If specified, subsamples the TopOGraph object and/or data matrix X to a number of landmark samples
            before computing results and scores. Useful if dealing with large datasets (>30,000 samples).

        kernels : list of str (optional, default ['fuzzy', 'cknn', 'bw_adaptive_alpha_decaying']).
            List of kernel versions to run and evaluate. These will be used to learn an eigenbasis and to learn a new graph kernel from it.
            Options are:
            * 'fuzzy'
            * 'cknn'
            * 'bw_adaptive'
            * 'bw_adaptive_alpha_decaying'
            * 'bw_adaptive_nbr_expansion'
            * 'bw_adaptive_alpha_decaying_nbr_expansion'
            * 'gaussian'
            Will not run all by default to avoid long waiting times in reckless calls.

        eigenmap_methods : list of str (optional, default ['msDM', 'DM', 'LE']).
            List of eigenmap methods to run and evaluate. Options are:
            * 'msDM' - multiscale diffusion maps
            * 'DM' - diffusion maps
            * 'LE' - Laplacian eigenmaps
            * 'top' - top eingenfunctions (with largest eigenvalues)
            * 'bottom' - bottom eigenfunctions (with smallest eigenvalues)

        projections : list of str (optional, default ['Isomap', 'MAP']).
            List of projection methods to run and evaluate. Options are the same of the `topo.layouts.Projector()` object:
            * '(L)Isomap'
            * 't-SNE'
            * 'MAP'
            * 'UMAP'
            * 'PaCMAP'
            * 'TriMAP'
            * 'IsomorphicMDE' - MDE with preservation of nearest neighbors
            * 'IsometricMDE' - MDE with preservation of pairwise distances
            * 'NCVis'

        additional_eigenbases : dict (optional, default None).
            Dictionary containing named additional eigenbases (e.g. factor analysis, VAEs, ICA, etc) to be evaluated.

        additional_projections : dict (optional, default None).
            Dictionary containing named additional projections (e.g. t-SNE, UMAP, etc) to be evaluated.

        n_neighbors : int (optional, default 5).
            Number of nearest neighbors to use for the kNN graph.

        n_jobs : int (optional, default -1).
            Number of jobs to use for parallelization. If -1, uses all available cores.

        cor_method : str (optional, default 'spearman').
            Correlation method to use for local scores. Options are 'spearman' and 'kendall'.

        landmark_method : str (optional, default 'random').
            Method to use for landmark selection. Options are 'random' and 'kmeans'.

        kwargs : dict (optional, default {}).
            Additional keyword arguments to pass to the `topo.base.ann.kNN()` function.


        Returns
        -------

        Populates the TopOGraph object and returns a dictionary of dictionaries with the results


        """
        from scipy.stats import spearmanr, kendalltau
        from scipy.spatial.distance import squareform
        from topo.utils._utils import get_landmark_indices
        from topo.eval.global_scores import global_score_pca
        from topo.eval.local_scores import geodesic_distance
        # Run modselfels
        if self.verbosity > 0:
            print('Running specified models...')
        self.run_models(X, kernels, eigenmap_methods, projections)
        gc.collect()
        # Define landmarks if applicable
        if landmarks is not None:
            if isinstance(landmarks, int):
                landmark_indices = get_landmark_indices(
                    self.base_knn_graph, n_landmarks=landmarks, method=landmark_method, random_state=self.random_state)
                if landmark_indices.shape[0] == self.base_knn_graph.shape[0]:
                    landmark_indices = None
            elif isinstance(landmarks, np.ndarray):
                landmark_indices = landmarks
            else:
                raise ValueError(
                    '\'landmarks\' must be either an integer or a numpy array.')

        # Compute geodesics
        gc.collect()
        EigenbasisLocalResults = {}
        EigenbasisGlobalResults = {}
        if self.verbosity > 0:
            print('Computing base geodesics...')
        if landmarks is not None:
            base_graph = self.base_knn_graph[landmark_indices, :][:, landmark_indices]
        else:
            base_graph = self.base_knn_graph
        base_geodesics = squareform(geodesic_distance(
            base_graph, directed=False, n_jobs=n_jobs))
        gc.collect()
        for key in self.EigenbasisDict.keys():
            if self.verbosity > 0:
                print('Computing geodesics for eigenbasis \'{}...\''.format(key))
            emb_graph = kNN(self.EigenbasisDict[key].results(), n_neighbors=n_neighbors,
                            metric=self.base_metric,
                            n_jobs=n_jobs,
                            backend=self.backend,
                            return_instance=False,
                            verbose=False, **kwargs)
            if landmarks is not None:
                emb_graph = emb_graph[landmark_indices, :][:, landmark_indices]
                gc.collect()
            embedding_geodesics = squareform(geodesic_distance(
                emb_graph, directed=False, n_jobs=n_jobs))
            gc.collect()
            if cor_method == 'spearman':
                if self.verbosity > 0:
                    print('Computing Spearman R for eigenbasis \'{}...\''.format(key))
                EigenbasisLocalResults[key], _ = spearmanr(
                    base_geodesics, embedding_geodesics)
            else:
                if self.verbosity > 0:
                    print('Computing Kendall Tau for eigenbasis \'{}...\''.format(key))
                EigenbasisLocalResults[key], _ = kendalltau(
                    base_geodesics, embedding_geodesics)
            gc.collect()
            EigenbasisGlobalResults[key] = global_score_pca(
                X, self.EigenbasisDict[key].results())
            if self.verbosity > 0:
                print('Finished for eigenbasis {}'.format(key))
            gc.collect()
        ProjectionLocalResults = {}
        ProjectionGlobalResults = {}
        for key in self.ProjectionDict.keys():
            if self.verbosity > 0:
                print('Computing geodesics for projection \' {}...\''.format(key))
            emb_graph = kNN(self.ProjectionDict[key],
                            n_neighbors=n_neighbors,
                            metric=self.graph_metric,
                            n_jobs=n_jobs,
                            backend=self.backend,
                            return_instance=False,
                            verbose=False, **kwargs)
            if landmarks is not None:
                emb_graph = emb_graph[landmark_indices, :][:, landmark_indices]
                gc.collect()
            embedding_geodesics = squareform(geodesic_distance(
                emb_graph, directed=False, n_jobs=n_jobs))
            gc.collect()
            if cor_method == 'spearman':
                if self.verbosity > 0:
                    print('Computing Spearman R for projection \'{}...\''.format(key))
                ProjectionLocalResults[key], _ = spearmanr(
                    base_geodesics, embedding_geodesics)
            else:
                if self.verbosity > 0:
                    print('Computing Kendall Tau for projection \'{}...\''.format(key))
                ProjectionLocalResults[key], _ = kendalltau(
                    base_geodesics, embedding_geodesics)
            gc.collect()
            ProjectionGlobalResults[key] = global_score_pca(
                X, self.ProjectionDict[key])
            gc.collect()
        from sklearn.decomposition import PCA
        if self.verbosity >= 1:
            print('Computing PCA for comparison...')
        import numpy as np
        if issparse(X) == True:
            if isinstance(X, csr_matrix):
                data = X.todense()
        if issparse(X) == False:
            if not isinstance(X, np.ndarray):
                import pandas as pd
                if isinstance(X, pd.DataFrame):
                    data = np.asarray(X.values.T)
                else:
                    return print('Uknown data format.')
            else:
                data = X
        pca_emb = PCA(n_components=self.n_eigs).fit_transform(data)
        gc.collect()
        emb_graph = kNN(pca_emb,
                        n_neighbors=n_neighbors,
                        metric=self.graph_metric,
                        n_jobs=n_jobs,
                        backend=self.backend,
                        return_instance=False,
                        verbose=False, **kwargs)
        gc.collect()
        if landmarks is not None:
            emb_graph = emb_graph[landmark_indices, :][:, landmark_indices]
            gc.collect()
        embedding_geodesics = squareform(geodesic_distance(
            emb_graph, directed=False, n_jobs=n_jobs))
        gc.collect()
        if self.verbosity > 0:
            print('Computing Spearman R for PCA...')
        EigenbasisLocalResults['PCA'], _ = spearmanr(
            base_geodesics, embedding_geodesics)
        gc.collect()
        ProjectionGlobalResults['PCA'] = global_score_pca(X, pca_emb)
        gc.collect()
        if additional_eigenbases is not None:
            for key in additional_eigenbases.keys():
                if self.verbosity > 0:
                    print('Computing geodesics for additional eigenbasis \'{}...\''.format(key))
                emb_graph = kNN(additional_eigenbases[key],
                                n_neighbors=n_neighbors,
                                metric=self.base_metric,
                                n_jobs=n_jobs,
                                backend=self.backend,
                                return_instance=False,
                                verbose=False, **kwargs)
                gc.collect()
                if landmarks is not None:
                    emb_graph = emb_graph[landmark_indices, :][:, landmark_indices]
                    gc.collect()
                embedding_geodesics = squareform(geodesic_distance(
                    emb_graph, directed=False, n_jobs=n_jobs))
                gc.collect()
                if cor_method == 'spearman':
                    if self.verbosity > 0:
                        print('Computing Spearman R for additional eigenbasis \'{}...\''.format(key))
                    EigenbasisLocalResults[key], _ = spearmanr(
                        base_geodesics, embedding_geodesics)
                else:
                    if self.verbosity > 0:
                        print('Computing Kendall Tau for additional eigenbasis \'{}...\''.format(key))
                    EigenbasisLocalResults[key], _ = kendalltau(
                        base_geodesics, embedding_geodesics)
                gc.collect()
                EigenbasisGlobalResults[key] = global_score_pca(
                    X, additional_eigenbases[key])
                if self.verbosity > 0:
                    print('Finished for eigenbasis {}'.format(key))
                gc.collect()
        if additional_projections is not None:
            for key in additional_projections.keys():
                if self.verbosity > 0:
                    print('Computing geodesics for additional projection \' {}...\''.format(key))
                emb_graph = kNN(additional_projections[key],
                                n_neighbors=n_neighbors,
                                metric=self.graph_metric,
                                n_jobs=n_jobs,
                                backend=self.backend,
                                return_instance=False,
                                verbose=False, **kwargs)
                if landmarks is not None:
                    emb_graph = emb_graph[landmark_indices, :][:, landmark_indices]
                embedding_geodesics = squareform(geodesic_distance(
                    emb_graph, directed=False, n_jobs=n_jobs))
                gc.collect()
                if cor_method == 'spearman':
                    if self.verbosity > 0:
                        print('Computing Spearman R for additional projection \'{}...\''.format(key))
                    ProjectionLocalResults[key], _ = spearmanr(
                        base_geodesics, embedding_geodesics)
                else:
                    if self.verbosity > 0:
                        print('Computing Kendall Tau for additional projection \'{}...\''.format(key))
                    ProjectionLocalResults[key], _ = kendalltau(
                        base_geodesics, embedding_geodesics)
                gc.collect()
                ProjectionGlobalResults[key] = global_score_pca(
                    X, additional_projections[key])
                gc.collect()
        res_dict = {'EigenbasisLocal': EigenbasisLocalResults,
                    'EigenbasisGlobal': EigenbasisGlobalResults,
                    'ProjectionLocal': ProjectionLocalResults,
                    'ProjectionGlobal': ProjectionGlobalResults}

        return res_dict
