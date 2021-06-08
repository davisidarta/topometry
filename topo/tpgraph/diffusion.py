#####################################
# Author: Davi Sidarta-Oliveira
# School of Medical Sciences,University of Campinas,Brazil
# contact: davisidarta[at]fcm[dot]unicamp[dot]com
# License: MIT
######################################
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, find, issparse, dia_matrix
from scipy.sparse.linalg import eigs, eigsh
from sklearn.base import TransformerMixin
from sklearn.neighbors import NearestNeighbors
from topo.base import ann
from topo.tpgraph import multiscale
import warnings
from scipy.sparse import (SparseEfficiencyWarning,csr_matrix, find, issparse)
warnings.simplefilter('ignore', SparseEfficiencyWarning)
import matplotlib.pyplot as plt

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

class Diffusor(TransformerMixin):
    """
    Sklearn-compatible estimator for using fast anisotropic diffusion with an adaptive neighborhood search algorithm. The
    Diffusion Maps algorithm was initially proposed by Coifman et al in 2005, and was augmented by the work of many.
    This implementation aggregates recent advances in diffusion harmonics, and innovates only by implementing an
    adaptively decaying kernel (the rate of decay is dependent on neighborhood density)
    and an adaptive neighborhood estimation approach.

    Parameters
    ----------
    n_eigs : int (optional, default 50)
        Number of diffusion components to compute. This number can be iterated to get different views
        from data at distinct spectral resolution.

    use_eigs : int or str (optional, default 'knee')
        Number of eigenvectors to use. If 'max', expands to the maximum number of positive eigenvalues
        (reach of numerical precision), else to the maximum amount of computed components.
        If 'knee', uses Kneedle to find an optimal cutoff point, and expands it by ``expansion``.
        If 'comp_gap', tries to find a discrete eigengap from the computation process.


    n_neighbors : int (optional, default 10)
        Number of k-nearest-neighbors to compute. The adaptive kernel will normalize distances by each cell
        distance of its median neighbor. Nonetheless, this hyperparameter remains as an user input regarding
        the minimal sample neighborhood resolution that drives the computation of the diffusion metrics. For
        practical purposes, the minimum amount of samples one would expect to constitute a neighborhood of its
        own. Increasing `k` can generate more globally-comprehensive metrics and maps, to a certain extend,
        however at the expense of fine-grained resolution. More generally, consider this a calculus
        discretization threshold.

    backend : str (optional, default 'hnwslib')
        Which backend to use to compute nearest-neighbors. Options for fast, approximate nearest-neighbors
        are 'hnwslib' (default) and 'nmslib'. For exact nearest-neighbors, use 'sklearn'.

    metric : str (optional, default 'cosine')
        Distance metric for building an approximate kNN graph. Defaults to
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

    p : int or float (optional, default 11/16 )
        P for the Lp metric, when ``metric='lp'``.  Can be fractional. The default 11/16 approximates
        an astroid norm with some computational efficiency (2^n bases are less painstakinly slow to compute).
        See https://en.wikipedia.org/wiki/Lp_space for some context.

    transitions : bool (optional, default False)
        Whether to estimate the diffusion transitions graph. If `True`, maps a basis encoding neighborhood
         transitions probability during eigendecomposition. If 'False' (default), maps the diffusion kernel.

    alpha : int or float (optional, default 1)
        Alpha in the diffusion maps literature. Controls how much the results are biased by data distribution.
            Defaults to 1, which is suitable for normalized data.

    kernel_use : str (optional, default 'decay_adaptive')
        Which type of kernel to use. There are four implemented, considering the adaptive decay and the
        neighborhood expansion, written as 'simple', 'decay', 'simple_adaptive' and 'decay_adaptive'. The first, 'simple'
        , is a locally-adaptive kernel similar to that proposed by Nadler et al.(https://doi.org/10.1016/j.acha.2005.07.004)
        and implemented in Setty et al. (https://doi.org/10.1038/s41587-019-0068-4). The 'decay' option applies an
        adaptive decay rate, but no neighborhood expansion. Those, followed by '_adaptive', apply the neighborhood expansion process.
         The default and recommended is 'decay_adaptive'.
        The neighborhood expansion can impact runtime, although this is not usually expressive for datasets under 10e6 samples.

    transitions : bool (optional, default False)
        Whether to decompose the transition graph when transforming.
    norm : bool (optional, default True)
        Whether to normalize the kernel transition probabilities to approximate the LPO.
    eigengap : bool (optional, default True)
        Whether to expand the eigendecomposition a bit and stop if eigenvalues sign shift (limit of float64). Used
        to guarantee numerical stability.
    n_jobs : int (optional, default 4)
        Number of threads to use in calculations. Defaults to 4 for safety, but performance
        scales dramatically when using more threads.
    plot_spectrum : bool (optional, default False)
        Whether to plot the spectrum decay analysis.
    verbose : bool (optional, default False)
        Controls verbosity.
    cache : bool (optional, default True)
        Whether to cache nearest-neighbors (before fit) and to store diffusion matrices after mapping (before transform).

    Example
    -------------

    import numpy as np
    from sklearn.datasets import load_digits
    from scipy.sparse import csr_matrix
    from topo.tpgraph.diffusion import Diffusor

    digits = load_digits()
    data = csr_matrix(digits)

    diff = Diffusor().fit(data)

    msdiffmap = diff.transform(data)

    """

    def __init__(self,
                 n_neighbors=10,
                 n_components=50,
                 use_eigs='max',
                 metric='cosine',
                 kernel_use='decay_adaptive',
                 eigengap=True,
                 plot_spectrum=False,
                 verbose=False,
                 cache=False,
                 alpha=1,
                 n_jobs=10,
                 backend='hnswlib',
                 p=None,
                 M=15,
                 efC=50,
                 efS=50,
                 norm=False,
                 transitions=True
                 ):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.use_eigs = use_eigs
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.backend = backend
        self.metric = metric
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
        self.cache = cache
        self.kn = None
        self.scaled_eigs = None
        self.N = None
        self.M = None
        self.K = None
        self.T = None
        self.res = None

    def __repr__(self):
        if (self.N is not None) and (self.M is not None):
            msg = "Diffusor() instance with %i samples and %i observations" % (self.N, self.M) + " and:"
        else:
            msg = "Diffusor() instance object without any fitted data."
        if self.K is not None:
            msg = msg + " \n    Diffusion kernel fitted - Diffusor.K"
        if self.T is not None:
            msg = msg + " \n    Normalized diffusion transitions fitted - Diffusor.T"
        if self.res is not None:
            msg = msg + " \n    Multiscale diffusion maps fitted - Diffusor.res"
        return msg

    def fit(self, X):

        """
        Fits an adaptive anisotropic diffusion kernel to the data.

        Parameters
        ----------
        X :
            input data. Takes in numpy arrays and scipy csr sparse matrices.
        Use with sparse data for top performance. You can adjust a series of
        parameters that can make the process faster and more informational depending
        on your dataset.

        Returns
        -------
            Diffusor object with kernel Diffusor.K and the transition potencial Diffusor.T .

        """
        data = X
        start_time = time.time()
        self.N = data.shape[0]
        self.M = data.shape[1]
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
        if self.kernel_use not in ['simple', 'simple_adaptive', 'decay', 'decay_adaptive']:
            raise Exception('Kernel must be either \'simple\', \'simple_adaptive\', \'decay\' or \'decay_adaptive\'.')
        if self.backend == 'hnswlib' and self.metric not in ['euclidean', 'sqeuclidean', 'cosine', 'inner_product']:
            if self.verbose:
                print('Metric ' + str(self.metric) + ' not compatible with \'hnslib\' backend. Trying changing to \'nmslib\' backend.')
                self.backend = 'nmslib'
        if self.metric == 'lp' and self.backend != 'nmslib':
            print('Fractional norm distances are only available with `backend=\'nmslib\'`. Trying changing to \'nmslib\' backend.')
            self.backend = 'nmslib'
        if self.backend == 'nmslib':
            # Construct an approximate k-nearest-neighbors graph
            anbrs = ann.NMSlibTransformer(n_neighbors=self.n_neighbors,
                                          metric=self.metric,
                                          p=self.p,
                                          method='hnsw',
                                          n_jobs=self.n_jobs,
                                          M=self.M,
                                          efC=self.efC,
                                          efS=self.efS,
                                          verbose=self.verbose).fit(data)
            knn = anbrs.transform(data)
        elif self.backend == 'hnwslib':
            anbrs = ann.HNSWlibTransformer(n_neighbors=self.n_neighbors,
                                           metric=self.metric,
                                           n_jobs=self.n_jobs,
                                           M=self.M,
                                           efC=self.efC,
                                           efS=self.efS,
                                           verbose=False).fit(data)
            knn = anbrs.transform(data)
        else:
            # Construct a k-nearest-neighbors graph
            nbrs = NearestNeighbors(n_neighbors=int(self.n_neighbors), metric=self.metric, n_jobs=self.n_jobs).fit(
                data)
            knn = nbrs.kneighbors_graph(data, mode='distance')

        # X, y specific stds: Normalize by the distance of median nearest neighbor to account for neighborhood size.
        median_k = np.floor(self.n_neighbors / 2).astype(np.int)
        adap_sd = np.zeros(self.N)
        for i in np.arange(len(adap_sd)):
            adap_sd[i] = np.sort(knn.data[knn.indptr[i]: knn.indptr[i + 1]])[
                median_k - 1
                ]

        # Distance metrics
        x, y, dists = find(knn)  # k-nearest-neighbor distances

        if self.cache:
            self.dists = knn
            self.adap_sd = adap_sd

        # Neighborhood graph expansion
        # define decay as sample's pseudomedian k-nearest-neighbor
        pm = np.interp(adap_sd, (adap_sd.min(), adap_sd.max()), (2, self.n_neighbors))
        self.omega = pm

        # adaptive neighborhood size
        if self.kernel_use == 'simple_adaptive' or self.kernel_use == 'decay_adaptive':
            self.new_k = int(self.n_neighbors + (self.n_neighbors - pm.max()))
            # increase neighbor search:
            # Construct a new approximate k-nearest-neighbors graph with new k
            if self.backend == 'nmslib':
                anbrs_new = ann.NMSlibTransformer(n_neighbors=self.new_k,
                                              metric=self.metric,
                                              p=self.p,
                                              method='hnsw',
                                              n_jobs=self.n_jobs,
                                              M=self.M,
                                              efC=self.efC,
                                              efS=self.efS,
                                              verbose=self.verbose).fit(data)
                knn_new = anbrs_new.transform(data)
            elif self.backend == 'hnwslib':
                anbrs_new = ann.HNSWlibTransformer(n_neighbors=self.new_k,
                                               metric=self.metric,
                                               n_jobs=self.n_jobs,
                                               M=self.M,
                                               efC=self.efC,
                                               efS=self.efS,
                                               verbose=False).fit(data)
                knn_new = anbrs_new.transform(data)
            else:
                # Construct a k-nearest-neighbors graph
                anbrs_new = NearestNeighbors(n_neighbors=int(self.new_k), metric=self.metric, n_jobs=self.n_jobs).fit(
                    data)
                knn_new = anbrs_new.kneighbors_graph(data, mode='distance')

            x_new, y_new, dists_new = find(knn_new)

            # adaptive neighborhood size
            adap_nbr = np.zeros(self.N)
            for i in np.arange(len(adap_nbr)):
                adap_k = int(np.floor(pm[i]))
                adap_nbr[i] = np.sort(knn_new.data[knn_new.indptr[i]: knn_new.indptr[i + 1]])[
                    adap_k - 1
                ]

            pm_new = np.interp(adap_nbr, (adap_nbr.min(), adap_nbr.max()), (2, self.new_k))

            if self.cache:
                self.dists_new = knn_new
                self.adap_nbr_sd = adap_nbr
                self.omega_new = pm_new

        if self.kernel_use == 'simple':
            # X, y specific stds
            dists = dists / (adap_sd[x] + 1e-10)  # Normalize by the distance of median nearest neighbor
            W = csr_matrix((np.exp(-dists), (x, y)), shape=[self.N, self.N])

        if self.kernel_use == 'simple_adaptive':
            # X, y specific stds with neighborhood expansion
            dists = dists_new / (adap_nbr[x_new] + 1e-10)  # Normalize by normalized contribution to neighborhood size.
            W = csr_matrix((np.exp(-dists), (x_new, y_new)), shape=[self.N, self.N])

        if self.kernel_use == 'decay':
            # X, y specific stds, alpha-adaptive decay
            dists = (dists / (adap_sd[x] + 1e-10)) ** np.power(2, ((self.n_neighbors - pm[x]) / pm[x]))
            W = csr_matrix((np.exp(-dists), (x, y)), shape=[self.N, self.N])

        if self.kernel_use == 'decay_adaptive':
            # X, y specific stds, with neighborhood expansion
            dists = (dists_new / (adap_nbr[x_new] + 1e-10)) ** np.power(2, ((self.new_k - pm[x_new]) / pm[y_new]))
            W = csr_matrix((np.exp(-dists), (x_new, y_new)), shape=[self.N, self.N])

        # Kernel construction
        kernel = (W + W.T) / 2
        self.K = kernel
        self.K[(np.arange(self.K.shape[0]), np.arange(self.K.shape[0]))] = 0

        # handle nan, zeros
        self.K.data = np.where(np.isnan(self.K.data), 1, self.K.data)

        # Diffusion through Markov chain
        D = np.ravel(self.K.sum(axis=1))
        if self.alpha > 0:
            # L_alpha
            D[D != 0] = D[D != 0] ** (-self.alpha)
            mat = csr_matrix((D, (range(self.N), range(self.N))), shape=[self.N, self.N])
            kernel = mat.dot(self.K).dot(mat)
            D = np.ravel(kernel.sum(axis=1))

        D[D != 0] = 1 / D[D != 0]
        # Setting the diffusion operator
        if self.norm:
            self.K = kernel
            self.T = csr_matrix((D, (range(self.N), range(self.N))), shape=[self.N, self.N]).dot(self.K)
        else:
            self.T = csr_matrix((D, (range(self.N), range(self.N))), shape=[self.N, self.N]).dot(self.K)

        # Guarantee symmetry
        self.T = (self.T + self.T.T) / 2
        self.T[(np.arange(self.T.shape[0]), np.arange(self.T.shape[0]))] = 0

        end = time.time()
        if self.verbose:
            print('Diffusion time = %f (sec), per sample=%f (sec), per sample adjusted for thread number=%f (sec)' %
                  (end - start_time, float(end - start_time) / self.N, self.n_jobs * float(end - start_time) / self.N))

        return self


    def transform(self, X):
        """
        Fits the renormalized Laplacian approximating the Laplace Beltrami-Operator
        in a discrete eigendecomposition. Then multiscales the resulting components.
        Parameters
        ----------
        X :
            input data. Takes in numpy arrays and scipy csr sparse matrices.
        Use with sparse data for top performance. You can adjust a series of
        parameters that can make the process faster and more informational depending
        on your dataset.

        Returns
        -------

       ``Diffusor.res['MultiscaleComponents']]``

        """
        start_time = time.time()
        # Fit an optimal number of components based on the eigengap
        # Use user's  or default initial guess
        # initial eigen value decomposition
        if self.transitions:
            D, V = eigs(self.T, self.n_components, tol=1e-4, maxiter=(self.N // 10))
        else:
            D, V = eigs(self.K, self.n_components, tol=1e-4, maxiter=(self.N // 10))
        D = np.real(D)
        V = np.real(V)
        inds = np.argsort(D)[::-1]
        D = D[inds]
        V = V[:, inds]
        # Normalize by the first diffusion component
        for i in range(V.shape[1]):
            V[:, i] = V[:, i] / np.linalg.norm(V[:, i])
        vals = np.array(V)
        pos = np.sum(vals > 0, axis=0)
        residual = np.sum(vals < 0, axis=0)

        if self.eigengap and len(residual) < 1:
            #expand eigendecomposition
            target = self.n_components + 30
            while residual < 3:
                while target < 3 * self.n_components:
                    print('Eigengap not found for determined number of components. Expanding eigendecomposition to '
                          + str(target) + 'components.')
                    if self.transitions:
                        D, V = eigs(self.T, target, tol=1e-4, maxiter=(self.N // 10))
                    else:
                        D, V = eigs(self.K, target, tol=1e-4, maxiter=(self.N // 10))
                    D = np.real(D)
                    V = np.real(V)
                    inds = np.argsort(D)[::-1]
                    D = D[inds]
                    V = V[:, inds]
                    # Normalize by the first diffusion component
                    vals = np.array(V)
                    for i in range(V.shape[1]):
                        vals[:, i] = vals[:, i] / np.linalg.norm(vals[:, i])
                    pos = np.sum(vals > 0, axis=0)
                    target = int(target * 1.6)
                    residual = np.sum(vals < 0, axis=0)

                if residual < 1:
                    print('Could not find an eigengap! Consider increasing `n_neighbors` or `n_components` !'
                          ' Falling back to `eigengap=False`, will not attempt')
                    self.eigengap = False

        if self.eigengap:
            if len(residual) > 30:
                while len(residual) > 29:
                    self.n_eigs = len(pos) + 15
                    if self.transitions:
                       D, V = eigs(self.T, self.n_eigs, tol=1e-4, maxiter=self.N)
                    else:
                        D, V = eigs(self.K, self.n_eigs, tol=1e-4, maxiter=self.N)
                    D = np.real(D)
                    V = np.real(V)
                    inds = np.argsort(D)[::-1]
                    D = D[inds]
                    V = V[:, inds]
                    vals = np.array(V)
                    for i in range(V.shape[1]):
                        vals[:, i] = vals[:, i] / np.linalg.norm(vals[:, i])
                    pos = np.sum(vals > 0, axis=0)
                    target = int(target * 1.6)
                    residual = np.sum(vals < 0, axis=0)
                if residual < 1:
                    print('Could not find an eigengap! Consider increasing `n_neighbors` or `n_components` !'
                              ' Falling back to `eigengap=False`, will not attempt')
                    self.eigengap = False
        if not self.eigengap:
            if self.transitions:
                D, V = eigs(self.T, self.n_components, tol=1e-4, maxiter=self.N)
            else:
                D, V = eigs(self.K, self.n_components, tol=1e-4, maxiter=self.N)
            D = np.real(D)
            V = np.real(V)
            inds = np.argsort(D)[::-1]
            D = D[inds]
            V = V[:, inds]
        # Normalize by the first diffusion component
        for i in range(V.shape[1]):
            V[:, i] = V[:, i] / np.linalg.norm(V[:, i])

        if not self.cache:
            del self.K
            del self.T

        # Create the results dictionary
        self.res = {'EigenVectors': V, 'EigenValues': D}
        self.res['EigenVectors'] = pd.DataFrame(self.res['EigenVectors'])
        self.res["EigenValues"] = pd.Series(self.res["EigenValues"])

        self.res['MultiscaleComponents'], self.kn, self.scaled_eigs = multiscale.multiscale(self.res,
                                                                                            n_eigs=self.use_eigs,
                                                                                            verbose=self.verbose)

        end = time.time()
        if self.verbose:
            print('Multiscale decomposition time = %f (sec), per sample=%f (sec), per sample adjusted for thread number=%f (sec)' %
                  (end - start_time, float(end - start_time) / self.N, self.n_jobs * float(end - start_time) / self.N))
        if self.plot_spectrum:
            self.spectrum_plot()

        return self.res['MultiscaleComponents']

    def ind_dist_grad(self, data):
        """
        Utility function to get indices, distances and gradients from a multiscale diffusion map.

        Parameters
        ----------
        data :
            Input data matrix (numpy array, pandas df, csr_matrix).

        n_components: int (optional, default None)
            Numper of components to map to prior to learning.

        Returns
        -------
        A tuple containing neighborhood indices, distances, gradient and a knn graph.

        """

        start_time = time.time()
        # Fit an optimal number of components based on the eigengap
        # Use user's  or default initial guess
        multiplier = self.N // 10e4
        # initial eigen value decomposition
        if self.transitions:
            D, V = eigs(self.T, self.n_components, tol=1e-4, maxiter=self.N)
        else:
            D, V = eigs(self.K, self.n_components, tol=1e-4, maxiter=self.N)
        D = np.real(D)
        V = np.real(V)
        inds = np.argsort(D)[::-1]
        D = D[inds]
        V = V[:, inds]
        # Normalize by the first diffusion component
        for i in range(V.shape[1]):
            V[:, i] = V[:, i] / np.linalg.norm(V[:, i])
        vals = np.array(V)
        pos = np.sum(vals > 0, axis=0)
        residual = np.sum(vals < 0, axis=0)

        if self.eigengap and len(residual) < 1:
            #expand eigendecomposition
            target = self.n_components * multiplier
            while residual < 3:
                print('Eigengap not found for determined number of components. Expanding eigendecomposition to '
                      + str(target) + 'components.')
                if self.transitions:
                    D, V = eigs(self.T, target, tol=1e-4, maxiter=self.N)
                else:
                    D, V = eigs(self.K, target, tol=1e-4, maxiter=self.N)
                D = np.real(D)
                V = np.real(V)
                inds = np.argsort(D)[::-1]
                D = D[inds]
                V = V[:, inds]
                # Normalize by the first diffusion component
                for i in range(V.shape[1]):
                    V[:, i] = V[:, i] / np.linalg.norm(V[:, i])
                vals = np.array(V)
                residual = np.sum(vals < 0, axis=0)
                target = target * 2

            if len(residual) > 30:
                self.n_components = len(pos) + 15
                # adapted eigen value decomposition
                if self.transitions:
                    D, V = eigs(self.T, self.n_components, tol=1e-4, maxiter=self.N)
                else:
                    D, V = eigs(self.K, self.n_components, tol=1e-4, maxiter=self.N)
                D = np.real(D)
                V = np.real(V)
                inds = np.argsort(D)[::-1]
                D = D[inds]
                V = V[:, inds]
                # Normalize by the first diffusion component
                for i in range(V.shape[1]):
                    V[:, i] = V[:, i] / np.linalg.norm(V[:, i])


        # Create the results dictionary
        self.res = {'EigenVectors': V, 'EigenValues': D}
        self.res['EigenVectors'] = pd.DataFrame(self.res['EigenVectors'])
        if not issparse(data):
            self.res['EigenValues'] = pd.Series(self.res['EigenValues'])
        self.res["EigenValues"] = pd.Series(self.res["EigenValues"])

        self.res['MultiscaleComponents'], self.kn, self.scaled_eigs = multiscale.multiscale(self.res,
                                                                                            n_eigs=self.use_eigs,
                                                                                            verbose=self.verbose)

        if self.backend == 'nmslib':
            # Construct an approximate k-nearest-neighbors graph
            anbrs = ann.NMSlibTransformer(n_neighbors=self.n_neighbors,
                                          metric=self.metric,
                                          p=self.p,
                                          method='hnsw',
                                          n_jobs=self.n_jobs,
                                          M=self.M,
                                          efC=self.efC,
                                          efS=self.efS,
                                          verbose=self.verbose).fit(self.res['MultiscaleComponents'])
            ind, dists, grad, graph = anbrs.ind_dist_grad(self.res['MultiscaleComponents'])
        elif self.backend == 'hnwslib':
            anbrs = ann.HNSWlibTransformer(n_neighbors=self.n_neighbors,
                                           metric=self.metric,
                                           n_jobs=self.n_jobs,
                                           M=self.M,
                                           efC=self.efC,
                                           efS=self.efS,
                                           verbose=False).fit(self.res['MultiscaleComponents'])
            ind, dists, grad, graph = anbrs.ind_dist_grad(self.res['MultiscaleComponents'])
        else:
            # Construct a k-nearest-neighbors graph
            nbrs = NearestNeighbors(n_neighbors=int(self.n_neighbors), metric=self.metric, n_jobs=self.n_jobs).fit(
                data)
            dists, ind = nbrs.kneighbors(data, mode='distance')
        end = time.time()
        print('Diffusion time = %f (sec), per sample=%f (sec), per sample adjusted for thread number=%f (sec)' %
              (end - start_time, float(end - start_time) / self.N, self.n_jobs * float(end - start_time) / self.N))
        if self.plot_spectrum:
            self.spectrum_plot()

        return ind, dists, grad, graph

    def res_dict(self):
        """
        Returns
        -------
            Dictionary containing normalized and multiscaled Diffusion Components
            (Diffusor.res['StructureComponents']), their eigenvalues['EigenValues'] and
            non - multiscaled components(['EigenVectors']).
        """

        return self.res

    def rescale(self, n_eigs=None):
        """
        Re-scale the multiscale procedure to a new number of components.

        Parameters
        ----------
        self : Diffusor object.

        n_eigs : int. Number of diffusion components to multiscale.

        Returns
        -------

        np.ndarray containing the new multiscaled basis.

        """
        if n_eigs is None:
            n_eigs = self.n_components

        mms, self.kn, self.scaled_eigs = multiscale.multiscale(self.res,
                                                                n_eigs=n_eigs,
                                                                verbose=self.verbose)

        self.res['MultiscaleComponents'] = mms
        return mms


    def spectrum_plot(self, bla=None):
        """
        Plot the decay spectra.

        Parameters
        ----------
        self : Diffusor object.

        bla : Here only for autodoc's sake.

        Returns
        -------

        A nice plot of the diffusion spectra.

        """

        if self.kn is None:
            msc, self.kn, self.scaled_eigs = multiscale.multiscale(self.res,
                                                 n_eigs=self.use_eigs,
                                                 verbose=self.verbose)
        if not isinstance(self.kn.knee, int):
            ax1 = plt.subplot(1, 1, 1)
            ax1.set_title('Spectrum decay and eigengap (%i)' % int(self.scaled_eigs))
            ax1.plot(self.kn.x, self.kn.y, 'b', label='data')
            ax1.set_ylabel('Eigenvalues')
            ax1.set_xlabel('Eigenvectors')
            ax1.vlines(
                self.scaled_eigs, plt.ylim()[0], plt.ylim()[1], linestyles="--", label='Multiscaled eigs'
            )
            ax1.legend(loc='best')
            plt.tight_layout()
            plt.show()
        else:
            ax1 = plt.subplot(2, 1, 1)
            ax1.set_title('Spectrum decay and \'knee\' (%i)' % int(self.kn.knee))
            ax1.plot(self.kn.x, self.kn.y, 'b', label='data')
            ax1.set_ylabel('Eigenvalues')
            ax1.set_xlabel('Eigenvectors')
            ax1.vlines(
                self.kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles="--", label='Knee'
            )
            ax1.legend(loc='best')

            ax2 = plt.subplot(2, 1, 2)
            ax2.set_title('Curve analysis')
            ax2.plot(self.kn.x_normalized, self.kn.y_normalized, "b", label="normalized")
            ax2.plot(self.kn.x_difference, self.kn.y_difference, "r", label="differential")
            ax2.set_xticks(
                np.arange(self.kn.x_normalized.min(), self.kn.x_normalized.max() + 0.1, 0.1)
            )
            ax2.set_yticks(
                np.arange(self.kn.y_difference.min(), self.kn.y_normalized.max() + 0.1, 0.1)
            )

            ax2.vlines(
                self.kn.norm_knee,
                plt.ylim()[0],
                plt.ylim()[1],
                linestyles="--",
                label="Knee",
            )
            ax2.legend(loc="best")
            plt.tight_layout()
            plt.show()
        return plt



