# TopOMetry high-level models API
# Author: Davi Sidarta-Oliveira <davisidarta(at)gmail(dot)com>
# School of Medical Sciences, University of Campinas, Brazil
#
import sys
import time
import numpy as np
import torch

from numpy import random
from pymde.functions import penalties, losses
from sklearn.base import TransformerMixin, BaseEstimator
from pymde.preprocess import Graph

import topo.plot as pt
from topo.layouts import map, mde
from topo.layouts.graph_utils import fuzzy_simplicial_set_ann
from topo.spectral import spectral as spt
from topo.tpgraph.cknn import cknn_graph
from topo.tpgraph.diffusion import Diffusor

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Matplotlib is required for the plotting functions.")
    sys.exit()

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

class TopOGraph(TransformerMixin, BaseEstimator):
    """

     Main TopOMetry class for building, clustering and visualizing n-order topological graphs.

     From data, builds a topologically-oriented basis and from this basis learns a topological graph. Users can choose
     different models to achieve these topological representations, combinining either diffusion harmonics,
     continuous k-nearest-neighbors or fuzzy simplicial sets to approximate the Laplace-Beltrami Operator.
     The topological graphs can then be visualized with multiple layout tools.

     TopOGraph has three built-in options
     of previously proposed, state-of-the-art algorithms for graph layout: spectral layouts (multicomponent [Laplacian
     Eigenmaps](http://www2.imm.dtu.dk/projects/manifold/Papers/Laplacian.pdf)),
     Manifold Approximation and Projection (a generalization of the seminal and robust
     [UMAP](https://umap-learn.readthedocs.io/en/latest/index.html)) and
     [Minimum Distortion Embedding](https://pymde.org/). When properly tuned, these algorithms tend to reach
     very similar layouts.

    Parameters
     ----------
     base_knn : int (optional, default 10).
         Number of k-nearest-neighbors to compute the ``Diffusor`` base operator on.
         The adaptive kernel will normalize distances by each cell distance of its median neighbor. Nonetheless,
         this hyperparameter remains as an user input regarding the minimal sample neighborhood resolution that drives
         the computation of the diffusion metrics. For practical purposes, the minimum amount of samples one would
         expect to constitute a neighborhood of its own. Increasing `k` can generate more globally-comprehensive metrics
         and maps, to a certain extend, however at the expense of fine-grained resolution. More generally,
          consider this a calculus discretization threshold.

     graph_knn : int (optional, default 10).
         Number of k-nearest-neighbors to compute the graph operator on.
         The adaptive kernel will normalize distances by each cell distance of its median neighbor. Nonetheless, this
         hyperparameter remains as an user input regarding the minimal sample neighborhood resolution that drives the 
         computation of the diffusion metrics. For practical purposes, the minimum amount of samples one would expect
         to constitute a neighborhood of its own. Increasing `k` can generate more globally-comprehensive metrics 
         and maps, to a certain extend,
         however at the expense of fine-grained resolution. More generally, consider this a calculus
         discretization threshold.

     n_eigs : int (optional, default 50).
         Number of components to compute. This number can be iterated to get different views
         from data at distinct spectral resolutions. If `basis` is set to `diffusion`, this is the number of 
         computed diffusion components. If `basis` is set to `continuous`, this is the number of computed eigenvectors
         of the Laplacian Eigenmaps from the continuous affinity matrix.

     basis : 'diffusion', 'continuous' or 'fuzzy' (optional, default 'diffusion').
         Which topological basis to build from data. If `diffusion`, performs an optimized, anisotropic, adaptive
         diffusion mapping (default). If `continuous`, computes affinities from continuous k-nearest-neighbors, and a 
         topological basis from the Laplacian Eigenmaps of such metric.

     graph : 'diff', 'cknn' or 'fuzzy' (optional, default 'diff').
         Which topological graph to learn from the built basis. If 'diff', uses a second-order diffusion process to learn
         similarities and transition probabilities. If 'cknn', uses the continuous k-nearest-neighbors algorithms. Both
         algorithms learn graph-oriented topological metrics from the learned basis. If 'fuzzy', builds a fuzzy simplicial
         set from the active basis.

    backend : str 'hnwslib', 'nmslib' or 'sklearn' (optional, default 'hnwslib')
        Which backend to use to compute nearest-neighbors. Options for fast, approximate nearest-neighbors
        are 'hnwslib' (default) and 'nmslib'. For exact nearest-neighbors, use 'sklearn'.
        I strongly recommend you use 'hnswlib' if handling with somewhat dense, array-shaped data. If the data
        is relatively sparse, you should consider using 'nmslib', which will automatically convert np.arrays to
        a csr_matrix for performance.

    base_metric : str (optional, default 'cosine')
        Distance metric for building an approximate kNN graph during topological basis construction. Defaults to
        'cosine'. Users are encouraged to explore different metrics, such as 'euclidean' and 'inner_product'.
        The 'hamming' and 'jaccard' distances are also available for string vectors.
         Accepted metrics include NMSLib*, HNSWlib** and sklearn metrics. Some examples are:

        -'sqeuclidean' (*, **)

        -'euclidean' (*, **)

        -'l1' (*)

        -'lp' - requires setting the parameter ``p`` (*) - similar to Minkowski

        -'cosine' (*, **)

        -'inner_product' (**)

        -'angular' (*)

        -'negdotprod' (*)

        -'levenshtein' (*)

        -'hamming' (*)

        -'jaccard' (*)

        -'jansen-shan' (*)
         
     graph_metric : str (optional, default 'cosine').
         Exactly the same as base_matric, but used for building the topological graph.
     
     p : int or float (optional, default 11/16 )
         P for the Lp metric, when `metric='lp'`.  Can be fractional. The default 11/16 approximates
         an astroid norm with some computational efficiency (2^n bases are less painstakinly slow to compute).

     transitions : bool (optional, default False)
         Whether to estimate the diffusion transitions graph. If `True`, maps a basis encoding neighborhood
          transitions probability during eigendecomposition. If 'False' (default), maps the diffusion kernel.

     alpha : int or float (optional, default 1)
         Alpha in the diffusion maps literature. Controls how much the results are biased by data distribution.
             Defaults to 1, which is suitable for normalized data.

     kernel_use : str (optional, default 'decay_adaptive')
         Which type of kernel to use in the diffusion approach. There are four implemented, considering the adaptive 
         decay and the neighborhood expansion, written as 'simple', 'decay', 'simple_adaptive' and 'decay_adaptive'.
         The first, 'simple', is a locally-adaptive kernel similar to that proposed by Nadler et al.
         (https://doi.org/10.1016/j.acha.2005.07.004) and implemented in Setty et al. 
         (https://doi.org/10.1038/s41587-019-0068-4).
         The 'decay' option applies an adaptive decay rate, but no neighborhood expansion.
         Those, followed by '_adaptive', apply the neighborhood expansion process. The default and recommended is 'decay_adaptive'.
         The neighborhood expansion can impact runtime, although this is not usually expressive for datasets under 10e6 samples.

     transitions : bool (optional, default False).
         Whether to decompose the transition graph when fitting the diffusion basis.
     n_jobs : int.
         Number of threads to use in calculations. Defaults to all but one.
     verbose : bool (optional, default False).
         Controls verbosity.
     cache : bool (optional, default True).
         Whether to cache nearest-neighbors (before fit) and to store diffusion matrices after mapping (before transform).

     """

    def __init__(self,
                 base_knn=10,
                 graph_knn=10,
                 n_eigs=50,
                 basis='diffusion',
                 graph='diff',
                 base_metric='cosine',
                 graph_metric='cosine',
                 n_jobs=1,
                 backend='hnwslib',
                 M=15,
                 efC=50,
                 efS=50,
                 verbose=False,
                 cache_base=True,
                 cache_graph=True,
                 kernel_use='decay_adaptive',
                 alpha=1,
                 plot_spectrum=False,
                 eigen_expansion=False,
                 delta=1.0,
                 t='inf',
                 p=11 / 16,
                 norm=False,
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
        self.norm = norm
        self.transitions = transitions
        self.eigen_expansion = eigen_expansion
        self.verbose = verbose
        self.plot_spectrum = plot_spectrum
        self.delta = delta
        self.t = t
        self.cache_base = cache_base
        self.cache_graph = cache_graph
        self.random_state = random_state
        self.DiffBasis = None
        self.clusters = None
        self.computed_LapGraph = False
        self.MSDiffMap = None
        self.ContBasis = None
        self.CLapMap = None
        self.DLapMap = None
        self.CknnGraph = None
        self.DiffGraph = None
        self.n = None
        self.m = None
        self.fitted_MAP = None
        self.MDE_problem = None
        self.SpecLayout = None
        self.FuzzyBasis = None
        self.FuzzyLapMap = None
        self.FuzzyGraph = None

    def __repr__(self):
        if (self.n is not None) and (self.m is not None):
            msg = "TopoGraph object with %i samples and %i observations" % (self.n, self.m) + " and:"
        else:
            msg = "TopoGraph object without any fitted data."
        if self.DiffBasis is not None:
            msg = msg + " \n    Diffusion basis fitted - .DiffBasis"
        if self.ContBasis is not None:
            msg = msg + " \n    Continuous basis fitted - .ContBasis"
        if self.FuzzyBasis is not None:
            msg = msg + " \n    Fuzzy basis fitted - .FuzzyBasis"
        if self.MSDiffMap is not None:
            msg = msg + " \n    Multiscale Diffusion Maps fitted - .MSDiffMap"
        if self.CLapMap is not None:
            msg = msg + " \n    Continuous Laplacian Eigenmaps fitted - .CLapMap"
        if self.FuzzyLapMap is not None:
            msg = msg + " \n    Fuzzy Laplacian Eigenmaps fitted - .FuzzyLapMap"
        if self.DiffGraph is not None:
            msg = msg + " \n    Diffusion graph fitted - .DiffGraph"
        if self.CknnGraph is not None:
            msg = msg + " \n    Continuous graph fitted - .CknnGraph"
        if self.FuzzyGraph is not None:
            msg = msg + " \n    Fuzzy graph fitted - .FuzzyGraph"
        if self.fitted_MAP is not None:
            msg = msg + " \n    Manifold Approximation and Projection fitted - .fitted_MAP"
        if self.MDE_problem is not None:
            msg = msg + " \n    Minimum Distortion Embedding set up - .MDE_problem"
        if self.SpecLayout is not None:
            msg = msg + " \n    Spectral layout fitted - .SpecLayout"
        if self.clusters is not None:
            msg = msg + " \n    Clustering fitted"
        msg = msg + " \n Active basis: " + str(self.basis) + ' basis.'
        msg = msg + " \n Active graph: " + str(self.graph) + ' graph.'
        return msg

    def fit(self, data):
        """
        Learn topological distances with diffusion harmonics and continuous metrics. Computes affinity operators
        that approximate the Laplace-Beltrami operator

        Parameters
        ----------
        data :
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

        self.n = data.shape[0]
        self.m = data.shape[1]
        if self.random_state is None:
            self.random_state = random.RandomState()
        print('Building topological basis...')
        if self.basis == 'diffusion':
            start = time.time()
            self.DiffBasis = Diffusor(n_components=self.n_eigs,
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
                                      verbose=self.verbose,
                                      plot_spectrum=self.plot_spectrum,
                                      cache=self.cache_base)
            self.MSDiffMap = self.DiffBasis.fit_transform(data)
            end = time.time()
            print('Topological basis fitted with diffusion mappings in %f (sec)' % (end - start))

        elif self.basis == 'continuous':
            start = time.time()
            self.ContBasis = cknn_graph(data,
                                        n_neighbors=self.base_knn,
                                        delta=self.delta,
                                        metric=self.base_metric,
                                        t=self.t,
                                        include_self=True,
                                        is_sparse=True,
                                        return_instance=True
                                        )
            self.CLapMap = spt.LapEigenmap(
                self.ContBasis.K,
                self.n_eigs,
                self.random_state,
            )
            expansion = 10.0 / np.abs(self.CLapMap).max()
            self.CLapMap = (self.CLapMap * expansion).astype(
                np.float32
            ) + self.random_state.normal(
                scale=0.0001, size=[self.ContBasis.K.shape[0], self.n_eigs]
            ).astype(
                np.float32
            )
            end = time.time()
            print('Topological basis fitted with continuous mappings in %f (sec)' % (end - start))

        elif self.basis == 'fuzzy':
            start = time.time()
            fuzzy_results = fuzzy_simplicial_set_ann(data,
                                                       n_neighbors=self.base_knn,
                                                       knn_indices=None,
                                                       knn_dists=None,
                                                       backend=self.backend,
                                                       metric=self.base_metric,
                                                       n_jobs=self.n_jobs,
                                                       efC=self.efC,
                                                       efS=self.efS,
                                                       M=self.M,
                                                       set_op_mix_ratio=1.0,
                                                       local_connectivity=1.0,
                                                       apply_set_operations=True,
                                                       return_dists=True,
                                                       verbose=self.verbose)
            self.FuzzyBasis = fuzzy_results[0]
            self.FuzzyLapMap = self.spectral_layout(X=data,
                                                    target=self.FuzzyBasis,
                                                    dim=self.n_eigs,
                                                    metric=self.base_metric)
            expansion = 10.0 / np.abs(self.FuzzyLapMap).max()
            self.FuzzyLapMap = (self.FuzzyLapMap * expansion).astype(
                np.float32
            ) + self.random_state.normal(
                scale=0.0001, size=[self.FuzzyBasis.shape[0], self.n_eigs]
            ).astype(
                np.float32
            )
            end = time.time()
            print('Topological basis fitted with fuzzy mappings in %f (sec)' % (end - start))

        else:
            return print('\'basis\' must be either \'diffusion\', \'continuous\' or \'fuzzy\'! Returning empty TopOGraph.')

        return self

    def transform(self, basis=None):
        """
        Learns new affinity, topological operators from chosen basis.

        Parameters
        ----------
        self :
            TopOGraph instance.

        base : str, optional.
            Base to use when building the topological graph. Defaults to the active base ( `TopOGraph.basis`).
            Setting this updates the active base.


        Returns
        -------
        scipy.sparse.csr.csr_matrix, containing the similarity matrix that encodes the topological graph.

        """
        if basis is not None:
            self.basis = basis
        print('Building topological graph...')
        start = time.time()
        if self.basis == 'continuous':
            use_basis = self.CLapMap
        elif self.basis == 'diffusion':
            use_basis = self.MSDiffMap
        elif self.basis == 'fuzzy':
            use_basis = self.FuzzyLapMap
        else:
            return print('No computed basis available! Compute a topological basis before fitting a topological graph.')
        if self.graph == 'diff':
            DiffGraph = Diffusor(n_neighbors=self.graph_knn,
                                 alpha=self.alpha,
                                 n_jobs=self.n_jobs,
                                 backend=self.backend,
                                 metric=self.graph_metric,
                                 p=self.p,
                                 M=self.M,
                                 efC=self.efC,
                                 efS=self.efS,
                                 kernel_use='simple_adaptive',
                                 norm=self.norm,
                                 transitions=self.transitions,
                                 eigen_expansion=self.eigen_expansion,
                                 verbose=self.verbose,
                                 plot_spectrum=self.plot_spectrum,
                                 cache=False
                                 ).fit(use_basis)
            if self.cache_graph:
                self.DiffGraph = DiffGraph.T
        elif self.graph == 'cknn':
            CknnGraph = cknn_graph(use_basis,
                                   n_neighbors=self.graph_knn,
                                   delta=self.delta,
                                   metric=self.graph_metric,
                                   t=self.t,
                                   include_self=True,
                                   is_sparse=True)
            if self.cache_graph:
                self.CknnGraph = CknnGraph

        elif self.graph == 'fuzzy':
            FuzzyGraph = fuzzy_simplicial_set_ann(use_basis,
                                                  n_neighbors=self.graph_knn,
                                                  knn_indices=None,
                                                  knn_dists=None,
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
                                                  verbose=self.verbose)
            FuzzyGraph = FuzzyGraph[0]
            if self.cache_graph:
                self.FuzzyGraph = FuzzyGraph
        else:
            return print('\'graph\' must be \'diff\', \'cknn\' or \'fuzzy\'!')

        end = time.time()
        print('Topological graph extracted in = %f (sec)' % (end - start))
        if self.graph == 'diff':
            return DiffGraph.T
        elif self.graph == 'cknn':
            return CknnGraph
        elif self.graph == 'fuzzy':
            return FuzzyGraph
        else:
            return self

    def spectral_layout(self, X=None, basis=None, target=None, dim=2, metric='cosine', cache=True):
        """

        Performs a multicomponent spectral layout of the data and the target similarity matrix.

        Parameters
        ----------
        basis :
            which basis to use.
        target : scipy.sparse.csr.csr_matrix.
            target similarity matrix. If None (default), computes a fuzzy simplicial set with default parameters.
        dim : int (optional, default 2).
            number of dimensions to embed into.
        cache : bool (optional, default True).
            Whether to cache the embedding to the `TopOGraph` object.
        Returns
        -------
        np.ndarray containing the resulting embedding.

        """
        if X is None:
            if basis is None:
                basis = self.basis
            if basis == 'diffusion':
                if self.MSDiffMap is None:
                    return print('Basis set to \'diffusion\', but the diffusion basis is not computed!')
                else:
                    X = self.MSDiffMap
            elif basis == 'continuous':
                if self.CLapMap is None:
                    return print('Basis set to \'continuous\', but the continuous basis is not computed!')
                else:
                    X = self.CLapMap
            elif basis == 'fuzzy':
                if self.FuzzyLapMap is None:
                    return print('Basis set to \'fuzzy\', but the fuzzy basis is not computed!')
                else:
                    X = self.FuzzyLapMap
            if target is None:
                if self.FuzzyGraph is None:
                    target = self.fuzzy_graph(X)
                else:
                    target = self.FuzzyGraph
            else:
                return print('No computed basis or data is provided!')

        spt_layout = spt.spectral_layout(X, target, dim, self.random_state, metric=metric, metric_kwds={})
        expansion = 10.0 / np.abs(spt_layout).max()
        spt_layout = (spt_layout * expansion).astype(
            np.float32
        ) + self.random_state.normal(
            scale=0.0001, size=[target.shape[0], dim]
        ).astype(
            np.float32
        )
        if cache:
            self.SpecLayout = spt_layout
        return spt_layout


    def fuzzy_graph(self,
                    X=None,
                    basis=None,
                    graph_knn=None,
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
            X : str, 'diffusion' or 'continuous'.
                The data to be modelled as a fuzzy simplicial set.
            graph_knn : int.
                The number of neighbors to use to approximate geodesic distance.
                Larger numbers induce more global estimates of the manifold that can
                miss finer detail, while smaller values will focus on fine manifold
                structure to the detriment of the larger picture. Defaults to `TopOGraph.graph_knn`
            nmslib_metric : str (optional, default `TopOGraph.graph_knn`).
                accepted NMSLIB metrics. Accepted metrics include:
                        -'sqeuclidean'
                        -'euclidean'
                        -'l1'
                        -'l1_sparse'
                        -'cosine'
                        -'angular'
                        -'negdotprod'
                        -'levenshtein'
                        -'hamming'
                        -'jaccard'
                        -'jansen-shan'
            nmslib_n_jobs : int (optional, default None).
                Number of threads to use for approximate-nearest neighbor search.
            nmslib_efC : int (optional, default 100).
                increasing this value improves the quality of a constructed graph and leads to higher
                accuracy of search. However this also leads to longer indexing times. A reasonable
                range is 100-2000.
            nmslib_efS : int (optional, default 100).
                similarly to efC, improving this value improves recall at the expense of longer
                retrieval time. A reasonable range is 100-2000.
            nmslib_M : int (optional, default 30).
                defines the maximum number of neighbors in the zero and above-zero layers during HSNW
                (Hierarchical Navigable Small World Graph). However, the actual default maximum number
                of neighbors for the zero layer is 2*M. For more information on HSNW, please check
                https://arxiv.org/abs/1603.09320. HSNW is implemented in python via NMSLIB. Please check
                more about NMSLIB at https://github.com/nmslib/nmslib .    n_epochs: int (optional, default None)
                The number of training epochs to be used in optimizing the
                low dimensional embedding. Larger values result in more accurate
                embeddings. If None is specified a value will be selected based on
                the size of the input dataset (200 for large datasets, 500 for small).
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
            set_op_mix_ratio : float (optional, default 1.0).
                Interpolate between (fuzzy) union and intersection as the set operation
                used to combine local fuzzy simplicial sets to obtain a global fuzzy
                simplicial sets. Both fuzzy set operations use the product t-norm.
                The value of this parameter should be between 0.0 and 1.0; a value of
                1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
                intersection.
            local_connectivity : int (optional, default 1).
                The local connectivity required -- i.e. the number of nearest
                neighbors that should be assumed to be connected at a local level.
                The higher this value the more connected the manifold becomes
                locally. In practice this should be not more than the local intrinsic
                dimension of the manifold.
            verbose : bool (optional, default False).
                Whether to report information on the current progress of the algorithm.
            cache: bool

            Returns
            -------
            fuzzy_simplicial_set : coo_matrix
            A fuzzy simplicial set represented as a sparse matrix. The (i,
            j) entry of the matrix represents the membership strength of the
            1-simplex between the ith and jth sample points.
        """

        if X is None:
            if basis == 'diffusion':
                if self.MSDiffMap is None:
                    return print('Basis set to \'diffusion\', but the diffusion basis is not computed!')
                else:
                    X = self.MSDiffMap
            elif basis == 'continuous':
                if self.CLapMap is None:
                    return print('Basis set to \'continuous\', but the continuous basis is not computed!')
                else:
                    X = self.CLapMap
            elif basis == 'fuzzy':
                if self.FuzzyLapMap is None:
                    return print('Basis set to \'fuzzy\', but the fuzzy basis is not computed!')
                else:
                    X = self.FuzzyLapMap
            else:
                return print('No computed basis or data is provided!')

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
                                             verbose=self.verbose)
        if cache:
            self.FuzzyGraph = fuzzy_set[0]
        return fuzzy_set[0]

    def MDE(self,
            basis=None,
            target=None,
            dim=2,
            n_neighbors=None,
            type='isomorphic',
            Y_init=None,
            n_epochs=500,
            snapshot_every=30,
            constraint=None,
            init='quadratic',
            attractive_penalty=penalties.Log1p,
            repulsive_penalty=penalties.Log,
            loss=losses.Absolute,
            repulsive_fraction=None,
            max_distance=None,
            device='cpu',
            eps=10e-5,
            mem_size=10,
            verbose=False):
        """
        This function constructs an MDE problem for preserving the
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
        target : scipy.sparse matrix
            The affinity matrix to embedd with. Defaults to the active graph. If init = 'spectral',
            a fuzzy simplicial set is used, and this argument is ignored.
        dim : int.
            The embedding dimension. Use 2 or 3 for visualization.
        attractive_penalty : pymde.Function class (or factory).
            Callable that constructs a distortion function, given positive
            weights. Typically one of the classes from ``pymde.penalties``,
            such as ``pymde.penalties.log1p``, ``pymde.penalties.Huber``, or
            ``pymde.penalties.Quadratic``.
        repulsive_penalty : pymde.Function class (or factory).
            Callable that constructs a distortion function, given negative
            weights. (If ``None``, only positive weights are used.) For example,
            ``pymde.penalties.Log`` or ``pymde.penalties.InversePower``.
        constraint : str (optional), default 'standardized'.
            Constraint to use when optimizing the embedding. Options are 'standardized',
            'centered', `None` or a `pymde.constraints.Constraint()` function.
        n_neighbors : int (optional)
            The number of nearest neighbors to compute for each row (item) of
            ``data``. A sensible value is chosen by default, depending on the
            number of items.
        repulsive_fraction : float (optional)
            How many repulsive edges to include, relative to the number
            of attractive edges. ``1`` means as many repulsive edges as attractive
            edges. The higher this number, the more uniformly spread out the
            embedding will be. Defaults to ``0.5`` for standardized embeddings, and
            ``1`` otherwise. (If ``repulsive_penalty`` is ``None``, this argument
            is ignored.)
        max_distance : float (optional)
            If not None, neighborhoods are restricted to have a radius
            no greater than ``max_distance``.
        init : str or np.ndarray (optional, default 'quadratic')
            Initialization strategy; np.ndarray, 'quadratic' or 'random'.
        device : str (optional)
            Device for the embedding (eg, 'cpu', 'cuda').
        verbose : bool
            If ``True``, print verbose output.

        Returns
        -------
        torch.tensor
            A ``pymde.MDE`` object, based on the original data.
        """
        if n_neighbors is None:
            n_neighbors = self.graph_knn
        X = None
        if basis is not None:
            if isinstance(basis, str):
                self.basis = basis
            elif isinstance(basis, np.ndarray):
                import torch
                X = basis
            else:
                print('\'basis\' must be either a str (\'diffusion\', \'continuous\', \'fuzzy\') or a np.ndarray!'
                      '\n Setting to default basis...')
            if isinstance(basis, str):
                if self.basis == 'diffusion':
                    if self.MSDiffMap is None:
                        return print('Basis set to \'diffusion\', but the diffusion basis is not computed!')
                    else:
                        X = self.MSDiffMap
                elif self.basis == 'continuous':
                    if self.CLapMap is None:
                        return print('Basis set to \'continuous\', but the continuous basis is not computed!')
                    else:
                        X = self.CLapMap
                elif self.basis == 'fuzzy':
                    if self.FuzzyLapMap is None:
                        return print('Basis set to \'fuzzy\', but the fuzzy basis is not computed!')
                    else:
                        X = self.FuzzyLapMap
                else:
                    return print('No computed basis or data was provided!')

        if isinstance(init, str) and init == 'spectral':
            if self.SpecLayout is None:
                if verbose:
                    print('Spectral layout not stored at TopOGraph.SpecLayout. Trying to compute...')
                if self.FuzzyGraph is None:
                    target = self.fuzzy_graph(X, basis='fuzzy')
                    init = self.spectral_layout(X, target, dim, metric=self.graph_metric)
                else:
                    target = self.FuzzyGraph
                    init = self.spectral_layout(X, target, dim, metric=self.graph_metric)
            else:
                init = self.SpecLayout

        if target is None:
            if self.graph == 'diff':
                if self.DiffGraph is None:
                    return print('Graph set to \'diff\', but the diffusion graph is not computed!')
                else:
                    target = self.DiffGraph
            elif self.graph == 'cknn':
                if self.CknnGraph is None:
                    return print('Graph set to \'cknn\', but the continuous graph is not computed!')
                else:
                    target = self.CknnGraph
            elif self.graph == 'fuzzy':
                if self.FuzzyGraph is None:
                    return print('Graph set to \'fuzzy\', but the fuzzy graph is not computed!')
                else:
                    target = self.FuzzyGraph
            elif X is not None:
                target = self.fuzzy_graph(X)
            else:
                return print('Could not find a computed graph or basis!')

        graph = Graph(target)
        if type == 'isomorphic':
            emb = mde.IsomorphicMDE(graph,
                                    attractive_penalty=attractive_penalty,
                                    repulsive_penalty=repulsive_penalty,
                                    embedding_dim=dim,
                                    constraint=constraint,
                                    n_neighbors=n_neighbors,
                                    repulsive_fraction=repulsive_fraction,
                                    max_distance=max_distance,
                                    init=init,
                                    device=device,
                                    verbose=verbose)

        elif type == 'isometric':
            if max_distance is None:
                max_distance = 5e7
            emb = mde.IsometricMDE(graph,
                                   embedding_dim=dim,
                                   loss=loss,
                                   constraint=constraint,
                                   max_distances=max_distance,
                                   device=device,
                                   verbose=verbose)
        else:
            return print('The tg.MDE problem must be \'isomorphic\' or \'isometric\'. Alternatively, build your own '
                         'MDE problem with `pyMDE` (i.g. pymde.MDE())')
        self.MDE_problem = emb
        if Y_init is not None:
            Y_init = torch.tensor(Y_init)
        emb_Y = emb.embed(X=Y_init,
                          max_iter=n_epochs,
                          memory_size=mem_size,
                          snapshot_every=snapshot_every,
                          eps=eps,
                          verbose=verbose)
        self.MDE_Y = emb_Y
        return emb_Y

    def MAP(self,
            data=None,
            graph=None,
            n_components=2,
            min_dist=0.3,
            spread=1.5,
            initial_alpha=1.5,
            n_epochs=400,
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
            njobs=-1,
            verbose=False,
            densmap=False,
            densmap_kwds={},
            output_dens=False,
            ):
        """""

        Manifold Approximation and Projection, as proposed by Leland McInnes with an uniform distribution assumption in
        the seminal [UMAP algorithm](https://umap-learn.readthedocs.io/en/latest/index.html). Perform a fuzzy simplicial set embedding, using a
        specified initialisation method and then minimizing the fuzzy set cross entropy between the 1-skeletons of the high
        and low dimensional fuzzy simplicial sets. The fuzzy simplicial set embedding was proposed and implemented by
        Leland McInnes in UMAP (see `umap-learn <https://github.com/lmcinnes/umap>`). Here we're using it only for the
        projection (layout optimization) by minimizing the cross-entropy between a phenotypic map (i.e. data, TopOMetry latent mappings)
        and its graph topological representation.


        Parameters
        ----------
        data : array of shape (n_samples, n_features)
            The source data to be embedded by UMAP. If `None` (default), the active basis will be used.
        graph : sparse matrix
            The 1-skeleton of the high dimensional fuzzy simplicial set as
            represented by a graph for which we require a sparse matrix for the
            (weighted) adjacency matrix. If `None` (default), a fuzzy simplicial set 
            is computed with default parameters.
        n_components : int
            The dimensionality of the euclidean space into which to embed the data.
        initial_alpha: float
            Initial learning rate for the SGD.
        a : float
            Parameter of differentiable approximation of right adjoint functor
        b : float
            Parameter of differentiable approximation of right adjoint functor
        gamma : float
            Weight to apply to negative samples.
        negative_sample_rate : int (optional, default 5)
            The number of negative samples to select per positive sample
            in the optimization process. Increasing this value will result
            in greater repulsive force being applied, greater optimization
            cost, but slightly more accuracy.
        n_epochs : int (optional, default 0)
            The number of training epochs to be used in optimizing the
            low dimensional embedding. Larger values result in more accurate
            embeddings. If 0 is specified a value will be selected based on
            the size of the input dataset (200 for large datasets, 500 for small).
        init : string
            How to initialize the low dimensional embedding. Options are:
                * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
                * 'random': assign initial embedding positions at random.
                * A numpy array of initial embedding positions.
        random_state : numpy RandomState or equivalent
            A state capable being used as a numpy random state.
        metric : string or callable
            The metric used to measure distance in high dimensional space; used if
            multiple connected components need to be layed out. Defaults to `TopOGraph.graph_metric`.
        metric_kwds : dict
            Key word arguments to be passed to the metric function; used if
            multiple connected components need to be layed out.
        densmap : bool
            Whether to use the density-augmented objective function to optimize
            the embedding according to the densMAP algorithm.
        densmap_kwds : dict
            Key word arguments to be used by the densMAP optimization.
        output_dens : bool
            Whether to output local radii in the original data and the embedding.
        output_metric : function
            Function returning the distance between two points in embedding space and
            the gradient of the distance wrt the first argument.
        output_metric_kwds : dict
            Key word arguments to be passed to the output_metric function.
        euclidean_output : bool
            Whether to use the faster code specialised for euclidean output metrics
        parallel : bool (optional, default False)
            Whether to run the computation using numba parallel.
            Running in parallel is non-deterministic, and is not used
            if a random seed has been set, to ensure reproducibility.
        return_aux : bool , (optional, default False)
            Whether to also return the auxiliary data, i.e. initialization and local radii.
        verbose : bool (optional, default False)
            Whether to report information on the current progress of the algorithm.
        Returns
        -------
        embedding : array of shape (n_samples, n_components)
            The optimized of ``graph`` into an ``n_components`` dimensional
            euclidean space.
        
        If return_aux is set to True :
            aux_data : dict
                Auxiliary dictionary output returned with the embedding.
                ``aux_data['Y_init']``: array of shape (n_samples, n_components)
                    The spectral initialization of ``graph`` into an ``n_components`` dimensional
                    euclidean space.
                When densMAP extension is turned on, this dictionary includes local radii in the original
                data (``aux_data['rad_orig']``) and in the embedding (``aux_data['rad_emb']``).


        """""
        if output_metric == 'torus':
            from topo.utils.umap_utils import torus_euclidean_grad
            output_metric = torus_euclidean_grad
        if metric is None:
            metric = self.graph_metric
        if self.SpecLayout is not None:
            init = self.SpecLayout
        if data is None:
            if self.basis == 'diffusion':
                if self.MSDiffMap is None:
                    return print('Basis set to \'diffusion\', but the diffusion basis is not computed!')
                else:
                    data = self.MSDiffMap
            elif self.basis == 'continuous':
                if self.CLapMap is None:
                    return print('Basis set to \'continuous\', but the continuous basis is not computed!')
                else:
                    data = self.CLapMap
            elif self.basis == 'fuzzy':
                if self.FuzzyLapMap is None:
                    return print('Basis set to \'fuzzy\', but the fuzzy basis is not computed!')
                else:
                    data = self.FuzzyLapMap
            else:
                return print('No computed basis or data is provided!')
        if graph is None:
            if self.graph == 'diff':
                if self.DiffGraph is None:
                    return print('Graph set to \'diff\', but the diffusion graph is not computed!')
                else:
                    graph = self.DiffGraph
            elif self.graph == 'cknn':
                if isinstance(init, str):
                    if init == 'spectral':
                        print('Graph set to \'cknn\', but the spectral initialisation requires \'diff\' or \'fuzzy\'!'
                                  '\n Computing layout with \'fuzzy\' graph...')
                        if self.FuzzyGraph is None:
                            graph = self.fuzzy_graph(X=data)
                        else:
                            graph = self.FuzzyGraph
                    else:
                        if self.CknnGraph is None:
                            return print('Graph set to \'cknn\', but the continuous graph is not computed!')
                        else:
                            graph = self.CknnGraph
                else:
                    if self.CknnGraph is None:
                        return print('Graph set to \'cknn\', but the continuous graph is not computed!')
                    else:
                        graph = self.CknnGraph
            elif self.graph == 'fuzzy':
                if self.FuzzyGraph is None:
                    return print('Graph set to \'fuzzy\', but the fuzzy simplicial set graph is not computed!')
                else:
                    graph = self.FuzzyGraph
            else:
                print('Could not find a computed graph! Computing a fuzzy simplicial'
                      ' set graph with default parameters on the active basis.')
                graph = self.fuzzy_graph(X=data)

        start = time.time()
        results = map.fuzzy_embedding(data, graph,
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
                                      njobs=njobs,
                                      verbose=verbose,
                                      a=None,
                                      b=None,
                                      densmap=densmap,
                                      densmap_kwds=densmap_kwds,
                                      output_dens=output_dens)

        end = time.time()
        print('Fuzzy layout optimization embedding in = %f (sec)' % (end - start))
        self.MAP_Y = results
        return results

    def plot(self,
             target=None,
             space='2D',
             dims_gauss=None,
             labels=None,
             title=None,
             pt_size=1,
             fontsize=18,
             marker='o',
             opacity=1,
             cmap='Spectral'
             ):
        """

        Utility function for plotting TopOGraph layouts. This is independent from the model
        and can be used to plot arbitrary layouts. Wraps around [Leland McInnes non-euclidean space
        embeddings](https://umap-learn.readthedocs.io/en/latest/embedding_space.html).

        Parameters
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


        if target is None:
            if self.MDE_Y is not None:
                target = self.MDE_Y
                used_target = 'MDE'
            elif self.MAP_Y is not None:
                target = self.MAP_Y
                used_target = 'MAP'
            else:
                raise Exception('Could not find a computed embedding at TopOGraph.MDE_Y '
                                'or TopOGraph.MAP_Y.')
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
                   n_eigs=None,
                   base_knn=None,
                   graph_knn=None,
                   verbose=None,
                   basis=['diffusion', 'fuzzy', 'continuous'],
                   graphs=['diff', 'cknn', 'fuzzy']):
        if str('diffusion') in basis:
            run_db = True
        if str('continuous') in basis:
            run_cb = True
        if str('fuzzy') in basis:
            run_fb = True
        if str('diff') in graphs:
            run_diff = True
        if str('cknn') in basis:
            run_cknn = True
        if str('fuzzy') in basis:
            run_fuzzy = True
        if n_eigs is not None:
            self.n_eigs = n_eigs
        if base_knn is not None:
            self.base_knn = base_knn
        if n_eigs is not None:
            self.graph_knn = graph_knn
        if verbose is not None:
            self.verbose = verbose
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
