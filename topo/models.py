# TopOMetry models API

import numpy as np
import time
from sklearn.cluster import AffinityPropagation
from sklearn.manifold import SpectralEmbedding
from topo.base import ann as nn
from topo.tpgraph.diffusion import Diffusor
from topo.tpgraph.cknn import CkNearestNeighbors, cknn_graph
from topo.spectral import spectral as spt
from topo.layouts import uni, mde
from topo import plot as pl
from sklearn.base import TransformerMixin, BaseEstimator

from numpy import random

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

class TopoGraph(TransformerMixin, BaseEstimator):
    def __init__(self,
                 base_knn=10,
                 graph_knn=10,
                 n_eigs=100,
                 basis='diffusion',
                 graph='dgraph',
                 kernel_use='simple_adaptive',
                 base_metric='cosine',
                 graph_metric='cosine',
                 transitions=False,
                 alpha=1,
                 plot_spectrum=False,
                 eigengap=True,
                 delta=1.0,
                 t='inf',
                 p=11 / 16,
                 n_jobs=1,
                 ann=True,
                 M=30,
                 efC=100,
                 efS=100,
                 verbose=False,
                 cache_base=True,
                 cache_graph=True,
                 norm=True,
                 random_state=None
                 ):
        self.graph = graph
        self.basis = basis
        self.n_eigs = n_eigs
        self.base_knn = base_knn
        self.graph_knn = graph_knn
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.ann = ann
        self.base_metric = base_metric
        self.graph_metric = graph_metric
        self.p = p
        self.M = M
        self.efC = efC
        self.efS = efS
        self.kernel_use = kernel_use
        self.norm = norm
        self.transitions = transitions
        self.eigengap = eigengap
        self.verbose = verbose
        self.plot_spectrum = plot_spectrum
        self.delta = delta
        self.t = t
        self.cache_base = cache_base
        self.cache_graph = cache_graph
        self.DiffBasis = None
        self.LapGraph = None
        self.clusters = None
        self.computed_LapGraph = False
        self.MSDiffMap = None
        self.ContBasis = None
        self.CLapMap = None
        self.DLapMap = None
        self.DiffCknnGraph = None
        self.CknnGraph = None
        self.CDiffGraph = None
        self.DiffGraph = None
        self.random_state = random_state

    def __repr__(self):
        msg = "TopoGraph object with %i samples and %i observations:" % (self.N , self.M)
        if self.DiffBasis is not None:
            msg = msg + " \n Diffusion basis fitted"
        if self.ContBasis is not None:
            msg = msg + " \n Continuous basis fitted"
        if self.MSDiffMap is not None:
            msg = msg + " \n Multiscale diffusion maps fitted"
        if self.LapGraph is not None:
            msg = msg + " \n Laplacian graph fitted"
        if self.DLapMap is not None:
            msg = msg + " \n Diffuse Laplacian graph fitted"
        if self.CLapMap is not None:
            msg = msg + " \n Continuous Laplacian graph fitted"
        if self.DiffGraph is not None:
            msg = msg + " \n Diffusion graph fitted"
        if self.CknnGraph is not None:
            msg = msg + " \n Continuous graph fitted"
        if self.DiffCknnGraph is not None:
            msg = msg + " \n Diffuse continuous graph fitted"
        if self.DiffGraph is not None:
            msg = msg + " \n Diffusion graph fitted"
        if self.CDiffGraph is not None:
            msg = msg + " \n Continuous diffusion graph fitted"
        if self.clusters is not None:
            msg = msg + " \n Clustering fitted"

        return msg

    """""""""
    Convenient TopOMetry class for building, clustering and visualizing n-order topological graphs.

    From data, builds a topologically-oriented basis with  optimized diffusion maps or a continuous k-nearest-neighbors
    Laplacian Eigenmap, and from this basis learns a topological graph (using a new diffusion process or a continuous 
    kNN kernel). This model approximates the Laplace-Beltrami Operator multiple ways by different ways, depending on
    the user setup. The topological graph can then be visualized in two or three dimensions with multiple
    classes for graph layout optimization in `topo.layout`.

   Parameters
    ----------
    base_knn : int (optional, default 10)
        Number of k-nearest-neighbors to compute the ``Diffusor`` base operator on.
        The adaptive kernel will normalize distances by each cell distance of its median neighbor. Nonetheless,
        this hyperparameter remains as an user input regarding the minimal sample neighborhood resolution that drives
        the computation of the diffusion metrics. For practical purposes, the minimum amount of samples one would
        expect to constitute a neighborhood of its own. Increasing `k` can generate more globally-comprehensive metrics
        and maps, to a certain extend, however at the expense of fine-grained resolution. More generally,
         consider this a calculus discretization threshold.

    graph_knn: int (optional, default 10)
        Number of k-nearest-neighbors to compute the graph operator on.
        The adaptive kernel will normalize distances by each cell
        distance of its median neighbor. Nonetheless, this hyperparameter remains as an user input regarding
        the minimal sample neighborhood resolution that drives the computation of the diffusion metrics. For
        practical purposes, the minimum amount of samples one would expect to constitute a neighborhood of its
        own. Increasing `k` can generate more globally-comprehensive metrics and maps, to a certain extend,
        however at the expense of fine-grained resolution. More generally, consider this a calculus
        discretization threshold.

    n_eigs : int (optional, default 50)
        Number of components to compute. This number can be iterated to get different views
        from data at distinct spectral resolutions. If `basis` is set to `diffusion`, this is the number of 
        computed diffusion components. If `basis` is set to `continuous`, this is the number of computed eigenvectors
        of the Laplacian Eigenmaps from the continuous affinity matrix.

    basis: `diffusion` or `continuous` (optional, default `diffusion`)
        Which topological basis to build from data. If `diffusion`, performs an optimized, anisotropic, adaptive
        diffusion mapping. If `continuous`, computes affinities from continuous k-nearest-neighbors, and a topological
        basis from the Laplacian Eigenmaps of such metric.

    ann : bool (optional, default True)
        Whether to use approximate nearest neighbors for graph construction. If `False`, uses `sklearn` default implementation.

    graph_metric, base_metric : str (optional, default 'cosine')
        Distance metrics for building a approximate kNN graphs. Defaults to 'cosine'. Users are encouraged to explore
        different metrics, such as 'cosine' and 'jaccard'. The 'hamming' and 'jaccard' distances are also available
        for string vectors. Accepted metrics include NMSLib metrics and sklearn metrics. Some examples are:
        -'sqeuclidean'
        -'euclidean'
        -'l1'
        -'lp' - requires setting the parameter ``p``
        -'cosine'
        -'angular'
        -'negdotprod'
        -'levenshtein'
        -'hamming'
        -'jaccard'
        -'jansen-shan'

    p: int or float (optional, default 11/16 )
        P for the Lp metric, when `metric='lp'`.  Can be fractional. The default 11/16 approximates
        an astroid norm with some computational efficiency (2^n bases are less painstakinly slow to compute).
        See https://en.wikipedia.org/wiki/Lp_space for some context.

    transitions: bool (optional, default False)
        Whether to estimate the diffusion transitions graph. If `True`, maps a basis encoding neighborhood
         transitions probability during eigendecomposition. If 'False' (default), maps the diffusion kernel.

    alpha : int or float (optional, default 1)
        Alpha in the diffusion maps literature. Controls how much the results are biased by data distribution.
            Defaults to 1, which is suitable for normalized data.

    kernel_use: str (optional, default 'decay_adaptive')
        Which type of kernel to use in the diffusion approach. There are four implemented, considering the adaptive 
        decay and the neighborhood expansion, written as 'simple', 'decay', 'simple_adaptive' and 'decay_adaptive'.

        - The first, 'simple', is a locally-adaptive kernel similar to that proposed by Nadler et al.
        (https://doi.org/10.1016/j.acha.2005.07.004) and implemented in Setty et al. 
        (https://doi.org/10.1038/s41587-019-0068-4).

        - The 'decay' option applies an adaptive decay rate, but no neighborhood expansion.

        Those, followed by '_adaptive', apply the neighborhood expansion process. The default and recommended is 'decay_adaptive'.
        The neighborhood expansion can impact runtime, although this is not usually expressive for datasets under 10e6 samples.

    transitions: bool (optional, default False)
        Whether to decompose the transition graph when fitting the diffusion basis.
    n_jobs : int
        Number of threads to use in calculations. Defaults to all but one.
    verbose : bool (optional, default False)
        Controls verbosity.
    cache: bool (optional, default True)
        Whether to cache nearest-neighbors (before fit) and to store diffusion matrices after mapping (before transform).

    """""""""

    def fit(self, data):
        """
        Learn topological distances with diffusion harmonics and continuous metrics. Computes affinity operators
        that approximate the Laplace-Beltrami operator

        Parameters
        ----------
        data:
            High-dimensional data matrix. Currently, supports only data from similar type (i.e. all bool, all float)

        Returns
        -------

        TopoGraph instance with several slots, populated as per user settings.
            If `basis=diffusion`, populates `TopoGraph.MSDiffMap` with a multiscale diffusion mapping of data, and
                `TopoGraph.DiffBasis` with a fitted `topo.tpgraph.diff.Diffusor()` class containing diffusion metrics
                and transition probabilities, respectively stored in TopoGraph.DiffBasis.K and TopoGraph.DiffBasis.T

            If `basis=continuous`, populates `TopoGraph.CLapMap` with a continous Laplacian Eigenmapping of data, and
                `TopoGraph.ContBasis` with a continuous-k-nearest-neighbors model, containing continuous metrics and
                adjacency, respectively stored in `TopoGraph.ContBasis.K` and `TopoGraph.ContBasis.A`.

        """
        self.N = data.shape[0]
        self.M = data.shape[1]
        if self.random_state is None:
            self.random_state = random.RandomState()
        print('Building topological basis...')
        if self.basis == 'diffusion':
            start = time.time()
            self.DiffBasis = Diffusor(n_components=self.n_eigs,
                                      n_neighbors=self.base_knn,
                                      alpha=self.alpha,
                                      n_jobs=self.n_jobs,
                                      ann=self.ann,
                                      metric=self.base_metric,
                                      p=self.p,
                                      M=self.M,
                                      efC=self.efC,
                                      efS=self.efS,
                                      kernel_use=self.kernel_use,
                                      norm=self.norm,
                                      transitions=self.transitions,
                                      eigengap=self.eigengap,
                                      verbose=self.verbose,
                                      plot_spectrum=self.plot_spectrum,
                                      cache=self.cache_base)
            self.MSDiffMap = self.DiffBasis.fit_transform(data)
            self.DLapMap = spt.spectral_layout(
                data,
                self.DiffBasis.T,
                self.n_eigs,
                self.random_state,
                metric="precomputed",
            )
            expansion = 10.0 / np.abs(self.DLapMap).max()
            self.DLapMap = (self.DLapMap * expansion).astype(
                np.float32
            ) + self.random_state.normal(
                scale=0.0001, size=[self.DiffBasis.T.shape[0], self.n_eigs]
            ).astype(
                np.float32
            )
            end = time.time()
            print('Topological basis fitted with diffusion mappings in = %f (sec)' % (end - start))

        elif self.basis == 'continuous':
            start = time.time()
            self.ContBasis = cknn_graph(data,
                                        n_neighbors=self.base_knn,
                                        delta=self.delta,
                                        metric=self.base_metric,
                                        t=self.t,
                                        include_self=True,
                                        is_sparse=True,
                                        return_instance=True)

            self.CLapMap = spt.spectral_layout(
                data,
                self.ContBasis.K,
                self.n_eigs,
                self.random_state,
                metric="precomputed",
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
            print('Topological basis fitted with continuous mappings in = %f (sec)' % (end - start))

        return self

    def transform(self):
        """

        Learns new affinity, topological operators from chosen basis.

        Returns
        -------
        If `graph` is 'dgraph' or 'diffcknn', returns  a tuple containing the kernel and the transition matrices
            i.e. kgraph, tgraph = tg.transform(), and if `cache=True`, writes the topological graph to
            `TopoGraph.DiffGraph` or `TopoGraph.DiffCknnGraph'
        If `graph` is 'cknn' or 'cdiff', returns a topologically weighted k-nearest-neighbors graph, and if
            `cache=True`, writes the topological graph to `TopoGraph.CknnGraph` or `TopoGraph.CDiffGraph'.


        """
        print('Building diffusion graph...')
        start = time.time()
        if self.basis == 'continuous':
            use_basis = self.CLapMap
        elif self.basis == 'diffusion':
            use_basis = self.MSDiffMap


        if self.graph == 'dgraph' or self.graph == 'cdiff':
            DiffGraph = Diffusor(n_neighbors=self.graph_knn,
                                 alpha=self.alpha,
                                 n_jobs=self.n_jobs,
                                 ann=self.ann,
                                 metric=self.graph_metric,
                                 p=self.p,
                                 M=self.M,
                                 efC=self.efC,
                                 efS=self.efS,
                                 kernel_use='simple',
                                 norm=self.norm,
                                 transitions=self.transitions,
                                 eigengap=self.eigengap,
                                 verbose=self.verbose,
                                 plot_spectrum=self.plot_spectrum,
                                 cache=False
                                 ).fit(use_basis)
            if self.cache_graph:
                self.DiffGraph = DiffGraph
            if self.graph == 'cdiff':
                print('Building 2-nd order, continuous version of the diffusion graph...')
                CDiffGraph = cknn_graph(DiffGraph.K,
                                        n_neighbors=self.graph_knn,
                                        delta=self.delta,
                                        metric='precomputed',
                                        t=self.t,
                                        include_self=True,
                                        is_sparse=True)
                if self.cache_graph:
                    self.CDiffGraph = CDiffGraph
        if self.graph == 'cknn' or self.graph == 'diffcknn':
            CknnGraph = cknn_graph(use_basis,
                                   n_neighbors=self.graph_knn,
                                   delta=self.delta,
                                   metric=self.graph_metric,
                                   t=self.t,
                                   include_self=True,
                                   is_sparse=True)
            if self.cache_graph:
                self.CknnGraph = CknnGraph
            if self.graph == 'diffcknn':
                print('Building 2-nd order, diffuse version of the diffusion graph...')
                DiffCknnGraph = Diffusor(n_neighbors=self.graph_knn,
                                         alpha=self.alpha,
                                         n_jobs=self.n_jobs,
                                         ann=self.ann,
                                         metric=self.metric,
                                         p=self.p,
                                         M=self.M,
                                         efC=self.efC,
                                         efS=self.efS,
                                         kernel_use='simple',
                                         norm=self.norm,
                                         transitions=self.transitions,
                                         eigengap=self.eigengap,
                                         verbose=self.verbose,
                                         plot_spectrum=self.plot_spectrum,
                                         cache=self.cache_graph
                                         ).fit(CknnGraph)
                if self.cache_graph:
                    self.DiffCknnGraph = DiffCknnGraph
        end = time.time()
        print('Topological graphs extracted in = %f (sec)' % (end - start))
        if self.graph == 'dgraph':
            return DiffGraph
        elif self.graph == 'cdiff':
            return CDiffGraph
        elif self.graph == 'cknn':
            return CknnGraph
        elif self.graph == 'diffcknn':
            return DiffCknnGraph
        else:
            return self













