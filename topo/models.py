# TopOMetry models API

import numpy as np
import time
from sklearn.cluster import AffinityPropagation
from sklearn.manifold import SpectralEmbedding
from topo.base import ann as nn
from topo.tpgraph.diffusion import Diffusor
from topo.tpgraph.cknn import CkNearestNeighbors, cknn_graph
from topo.spectral import spectral as spt
from topo.tpgraph.GL import InvLaplGraph
from topo.layouts import uni, pac
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
                 graph_knn=20,
                 n_eigs=100,
                 basis='diffusion',
                 graph='dgraph',
                 kernel_use='decay_adaptive',
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
        self.DiffCknnGraph = None
        self.CknnGraph = None
        self.CDiffGraph = None
        self.DiffGraph = None
        self.LapGraph = None
        self.clusters = None
        self.computed_LapGraph = False
        self.ContBasis = None
        self.CLapMap = None
        self.DLapMap = None
        self.random_state = random_state

    def __repr__(self):
        base_print = "TopoGraph object with:"
        if self.DiffBasis is None:
            print("TopoGraph object without any fitted model")
        if self.DiffBasis is not None:
            print(base_print + " \n Diffusion basis fitted on %f samples and %f observations" % (
                        self.DiffBasis.N.shape[0] / self.DiffBasis.N.shape[1]))
        if self.ContBasis is not None:
            print(base_print + " \n Continuous basis fitted on %f samples and %f observations" % (
                        self.DiffBasis.N.shape[0] / self.DiffBasis.N.shape[1]))

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
                metric="precomputed",
            )
            expansion = 10.0 / np.abs(self.DLapMap).max()
            self.DLapMap = (self.DLapMap * expansion).astype(
                np.float32
            ) + self.random_state.normal(
                scale=0.0001, size=[self.ContBasis.K.shape[0], self.n_eigs]
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
                                 kernel_use='simple_adaptive',
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
                                         kernel_use='simple_adaptive',
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

    def coarse(self, edges,
               reductionTarget='edges',
               actionSwitch='both',
               numSamplesS='all',
               qOverS=1.0 / 8,
               minProbPerActionD=1.0 / 4,
               minTargetItems=1024,
               plotError=False,
               reproject=False
               ):
        """

        Parameters
        ----------
        edges: X
            Target basis to reduce.
        reductionTarget: str, 'edges' or 'nodes', default 'edges'
            Target item to reduce.
        actionSwitch: str, 'both' or 'delete', default 'both'
            Choosing 'delete' does not allow contraction.
        numSamplesS: int or 'all', default 'all'
            Perturbed edges per sampled edges. Setting to 0 gives q=1 per round using the single-edge method.
        qOverS: float, from 0.0 to 1.0, default 1.0 / 8
            Perturbed edges per sampled edges. Setting to 0 gives q=1 per round using the single-edge method.
        minProbPerActionD: float, from 0.0 to 1.0, default 1.0 / 4
            Minimum expected (target items removed)/(num actions taken).
        minTargetItems: int or 'all', default 1024
            Minimum expected (target items removed)/(num actions taken).
        plotError: bool, default False
            Decide whether or not to compute the hyperbolic alignment of the output of the original eigenvectors.
        reproject: bool, default False
            Whether to reproject the learned inverse laplacian back to the original dimensional space.
            Use this to get approximate solutions to your Lx=b problems.


        Returns
        -------
        reducedLaplacian
            Reduced node-weighted laplacian of size $\tilde{V} \times \tilde{V}$

        """

        if not self.computed_LapGraph:
            self.LapGraph = InvLaplGraph(edges,
                                         reductionTarget=reductionTarget,
                                         actionSwitch=actionSwitch,
                                         numSamplesS=numSamplesS,
                                         qOverS=qOverS,
                                         minProbPerActionD=minProbPerActionD,
                                         minTargetItems=minTargetItems,
                                         plotError=plotError,
                                         reproject=reproject)
            self.computed_LapGraph = True

        coarse = self.LapGraph.nodeWeightedInverseLaplacian
        return coarse

    def project(self, data):
        """

        Parameters
        ----------
        data


        Returns
        -------
        GLGraph.reducedLaplacianOriginalDimension
            The reduced node-weighted laplacian appropriately projected back to the original dimensions.
            Use this to get approximate solutions to your Lx=b problems.

        """

        if self.computed_LapGraph is False:
            self.LapGraph = InvLaplGraph(data)

        projection = self.LapGraph.reducedLaplacianProjected(data)

        return projection

    def affinity_clustering(self,
                            graph,
                            damping=1,
                            max_iter=500,
                            convergence_iter=30,
                            preference=None):

        """

        Performs affinity propagantion clustering on the selected topological graph.

        Parameters
        ----------
            graph:
                Selected graph representation to propagate affinities of.

            damping: float, default=0.5
                Damping factor (between 0.5 and 1) is the extent to which the current value is maintained relative to
                incoming values (weighted 1 - damping). This in order to avoid numerical oscillations when updating
                these values (messages).

            max_iter: int, default=200
                Maximum number of iterations.

            convergence_iter: int, default=15
                Number of iterations with no change in the number of estimated clusters that stops the convergence.

            preference: array-like of shape (n_samples,) or float, default=None
                Preferences for each point - points with larger values of preferences are more likely to be
                chosen as exemplars. The number of exemplars, ie of clusters, is influenced by the input preferences
                value. If the preferences are not passed as arguments, they will be set to the median of
                the input similarities.

        Returns
        -------
            labels: ndarray of shape (n_samples,)
                The labels of learned the clusters.

       """
        self.clusters = AffinityPropagation(damping=damping,
                                            max_iter=max_iter,
                                            convergence_iter=convergence_iter,
                                            preference=preference).fit_predict(graph)

        return self.clusters


    def MAP(self, data, graph,
            dims=2,
            min_dist=0.3,
            spread=1.2,
            initial_alpha=1,
            n_epochs=500,
            metric='cosine',
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
        [UMAP](https://umap-learn.readthedocs.io/en/latest/index.html). Perform a fuzzy simplicial set embedding, using a
        specified initialisation method and then minimizing the fuzzy set cross entropy between the 1-skeletons of the high
        and low dimensional fuzzy simplicial sets. The fuzzy simplicial set embedding was proposed and implemented by
        Leland McInnes in UMAP (see `umap-learn <https://github.com/lmcinnes/umap>`). Here we're using it only for the
        projection (layout optimization) by minimizing the cross-entropy between a phenotypic map (i.e. data, MSDiffMap,
        CknnL) and its graph topological representation (i.e. a continuous nearest-neighbors or a diffusion
        kernel, or their transition matrices).
    
    
        Parameters
        ----------
        data: array of shape (n_samples, n_features)
            The source data to be embedded by UMAP.
        graph: sparse matrix
            The 1-skeleton of the high dimensional fuzzy simplicial set as
            represented by a graph for which we require a sparse matrix for the
            (weighted) adjacency matrix.
        n_components: int
            The dimensionality of the euclidean space into which to embed the data.
        initial_alpha: float
            Initial learning rate for the SGD.
        a: float
            Parameter of differentiable approximation of right adjoint functor
        b: float
            Parameter of differentiable approximation of right adjoint functor
        gamma: float
            Weight to apply to negative samples.
        negative_sample_rate: int (optional, default 5)
            The number of negative samples to select per positive sample
            in the optimization process. Increasing this value will result
            in greater repulsive force being applied, greater optimization
            cost, but slightly more accuracy.
        n_epochs: int (optional, default 0)
            The number of training epochs to be used in optimizing the
            low dimensional embedding. Larger values result in more accurate
            embeddings. If 0 is specified a value will be selected based on
            the size of the input dataset (200 for large datasets, 500 for small).
        init: string
            How to initialize the low dimensional embedding. Options are:
                * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
                * 'random': assign initial embedding positions at random.
                * A numpy array of initial embedding positions.
        random_state: numpy RandomState or equivalent
            A state capable being used as a numpy random state.
        metric: string or callable
            The metric used to measure distance in high dimensional space; used if
            multiple connected components need to be layed out.
        metric_kwds: dict
            Key word arguments to be passed to the metric function; used if
            multiple connected components need to be layed out.
        densmap: bool
            Whether to use the density-augmented objective function to optimize
            the embedding according to the densMAP algorithm.
        densmap_kwds: dict
            Key word arguments to be used by the densMAP optimization.
        output_dens: bool
            Whether to output local radii in the original data and the embedding.
        output_metric: function
            Function returning the distance between two points in embedding space and
            the gradient of the distance wrt the first argument.
        output_metric_kwds: dict
            Key word arguments to be passed to the output_metric function.
        euclidean_output: bool
            Whether to use the faster code specialised for euclidean output metrics
        parallel: bool (optional, default False)
            Whether to run the computation using numba parallel.
            Running in parallel is non-deterministic, and is not used
            if a random seed has been set, to ensure reproducibility.
        return_init: bool , (optional, default False)
            Whether to also return the multicomponent spectral initialization.
        verbose: bool (optional, default False)
            Whether to report information on the current progress of the algorithm.
        Returns
        -------
        embedding: array of shape (n_samples, n_components)
            The optimized of ``graph`` into an ``n_components`` dimensional
            euclidean space.
        aux_data: dict
            Auxiliary dictionary output returned with the embedding.
            ``aux_data['Y_init']``: array of shape (n_samples, n_components)
                The spectral initialization of ``graph`` into an ``n_components`` dimensional
                euclidean space.
    
            When densMAP extension is turned on, this dictionary includes local radii in the original
            data (``aux_data['rad_orig']``) and in the embedding (``aux_data['rad_emb']``).
    
    
        """""

        start = time.time()
        results = uni.fuzzy_embedding(data, graph,
                                      n_components=dims,
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
        return results
