#####################################
# Author: Davi Sidarta-Oliveira
# School of Medical Sciences,University of Campinas,Brazil
# contact: davisidarta[at]fcm[dot]unicamp[dot]com
# License: MIT
######################################
import time
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import (SparseEfficiencyWarning,
                          csr_matrix, find)
from scipy.sparse.linalg import eigsh
from sklearn.base import TransformerMixin
from topo.base.ann import kNN
from topo.tpgraph import multiscale as ms
from topo.spectral import diffusion_operator

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


class Diffusor(TransformerMixin):
    """
    Sklearn-compatible estimator for using adaptive multiscale diffusion harmonics and maps. The
    Diffusion Maps algorithm was initially proposed by Coifman et al in 2005, and was augmented by the work of many.
    This implementation aggregates several advances in diffusion harmonics and introduces a novel adaptively-decaying
    adaptive kernel. See the TopOMetry manuscript for further details.

    Parameters
    ----------
    n_eigs : int (optional, default 50)
        Number of diffusion components to compute. This number can be iterated to get different views
        from data at distinct spectral resolution.

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
        Defaults to 1 (no bias).

    tol : float (optional, default 1e-4)
        Tolerance during eigendecomposition. Set to 0 for exact (bweare, may be computationally very intensive!)

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

    norm : bool (optional, default False)
        Whether to renormalize the kernel transition probabilities.

    eigen_expansion : bool (optional, default False)
        Whether to expand the eigendecomposition and stop near a discrete eigengap (bit limit).

    n_jobs : int (optional, default 4)
        Number of threads to use in calculations. Defaults to -1 (all but one). Escalates dramatically
         when using more threads.

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
                 n_eigs=50,
                 metric='cosine',
                 kernel_use='simple',
                 t=5,
                 multiscale=True,
                 plot_spectrum=False,
                 verbose=False,
                 cache=False,
                 alpha=1,
                 tol=1e-4,
                 n_jobs=-1,
                 backend='nmslib',
                 p=None,
                 M=15,
                 efC=50,
                 efS=50,
                 ):
        self.t = t
        self.multiscale = multiscale
        self.n_neighbors = n_neighbors
        self.n_eigs = n_eigs 
        self.use_eigs = None # Deprecated
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.backend = backend
        self.metric = metric
        self.p = p
        self.M = M
        self.efC = efC
        self.efS = efS
        self.kernel_use = kernel_use
        self.verbose = verbose
        self.plot_spectrum = plot_spectrum
        self.tol = tol
        self.cache = cache
        self.omega = None
        self.omega_new = None
        self.kn = None
        self.scaled_eigs = None
        self.N = None
        self.M = None
        self.K = None
        self.P = None
        self.res = None
        self._D_left = None

    def __repr__(self):
        if self.metric != 'precomputed':
            if (self.N is not None) and (self.M is not None):
                msg = "Diffusor() instance with %i samples and %i observations" % (
                    self.N, self.M) + " and:"
            else:
                msg = "Diffusor() instance object without any fitted data."
        else:
            if (self.N is not None) and (self.M is not None):
                msg = "Diffusor() instance with %i samples" % (self.N) + " and:"
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
        if self.n_jobs == -1:
            from joblib import cpu_count
            self.n_jobs = cpu_count()

        start_time = time.time()
        self.N = X.shape[0]
        self.M = X.shape[1]
        if self.backend == 'hnswlib':
            if not _have_hnswlib:
                if _have_nmslib:
                    self.backend == 'nmslib'
                elif _have_annoy:
                    self.backend == 'annoy'
                elif _have_faiss:
                    self.backend == 'faiss'
                else:
                    self.backend == 'sklearn'
        if self.backend == 'nmslib':
            if not _have_nmslib:
                if _have_hnswlib:
                    self.backend == 'hnswlib'
                elif _have_annoy:
                    self.backend == 'annoy'
                elif _have_faiss:
                    self.backend == 'faiss'
                else:
                    self.backend == 'sklearn'
        if self.backend == 'annoy':
            if not _have_annoy:
                if _have_nmslib:
                    self.backend == 'nmslib'
                elif _have_hnswlib:
                    self.backend == 'hnswlib'
                elif _have_faiss:
                    self.backend == 'faiss'
                else:
                    self.backend == 'sklearn'
        if self.backend == 'faiss':
            if not _have_faiss:
                if _have_nmslib:
                    self.backend == 'nmslib'
                elif _have_hnswlib:
                    self.backend == 'hnswlib'
                elif _have_annoy:
                    self.backend == 'annoy'
                else:
                    self.backend == 'sklearn'

        if self.kernel_use not in ['simple', 'simple_adaptive', 'decay', 'decay_adaptive']:
            raise Exception(
                'Kernel must be either \'simple\', \'simple_adaptive\', \'decay\' or \'decay_adaptive\'.')
        if self.backend == 'hnswlib' and self.metric not in ['euclidean', 'sqeuclidean', 'cosine', 'inner_product']:
            if self.verbose:
                print('Metric ' + str(self.metric) +
                      ' not compatible with \'hnslib\' backend. Trying changing to \'nmslib\' backend.')
                self.backend = 'nmslib'
        if self.metric == 'lp' and self.backend != 'nmslib':
            print('Fractional norm distances are only available with `backend=\'nmslib\'`. Trying changing to \'nmslib\' backend.')
            self.backend = 'nmslib'

        if self.metric != 'precomputed':
            knn = kNN(X, n_neighbors=self.n_neighbors,
                      metric=self.metric,
                      n_jobs=self.n_jobs,
                      backend=self.backend,
                      M=self.M,
                      efC=self.efC,
                      efS=self.efS,
                      verbose=self.verbose)

        elif self.metric == 'precomputed':
            if self.verbose:
                print(
                    'Using precomputed distance matrix. We are not checking for symmetry!!!')
            if self.kernel_use == 'simple_adaptive':
                self.kernel_use = 'simple'
            if self.kernel_use == 'decay_adaptive':
                self.kernel_use = 'decay'
            if not isinstance(X, csr_matrix):
                knn = csr_matrix(X)
            else:
                knn = X

        # X, y specific stds: Normalize by the distance of median nearest neighbor to account for neighborhood size.
        median_k = np.floor(self.n_neighbors / 2).astype(int)
        adap_sd = np.zeros(self.N)
        for i in np.arange(len(adap_sd)):
            adap_sd[i] = np.sort(knn.data[knn.indptr[i]: knn.indptr[i + 1]])[
                median_k - 1
            ]

        # Distance metrics
        x, y, dists = find(knn)  # k-nearest-neighbor distances
        self.adap_sd = adap_sd

        # Neighborhood graph expansion
        # define decay as sample's pseudomedian k-nearest-neighbor
        pm = np.interp(adap_sd, (adap_sd.min(), adap_sd.max()),
                       (2, self.n_neighbors))
        self.omega = pm

        # adaptive neighborhood size
        if self.kernel_use == 'simple_adaptive' or self.kernel_use == 'decay_adaptive':
            self.new_k = int(self.n_neighbors + (self.n_neighbors - pm.max()))
            # increase neighbor search:
            # Construct a new approximate k-nearest-neighbors graph with new k
            knn_new = kNN(X, n_neighbors=self.new_k,
                          metric=self.metric,
                          n_jobs=self.n_jobs,
                          backend=self.backend,
                          M=self.M,
                          efC=self.efC,
                          efS=self.efS,
                          verbose=self.verbose)
            x_new, y_new, dists_new = find(knn_new)

            # adaptive neighborhood size
            adap_nbr = np.zeros(self.N)
            for i in np.arange(len(adap_nbr)):
                adap_k = int(np.floor(pm[i]))
                adap_nbr[i] = np.sort(knn_new.data[knn_new.indptr[i]: knn_new.indptr[i + 1]])[
                    adap_k - 1
                ]

            pm_new = np.interp(
                adap_nbr, (adap_nbr.min(), adap_nbr.max()), (2, self.new_k))

            self.adap_nbr_sd = adap_nbr
            self.omega_new = pm_new

        if self.kernel_use == 'simple':
            # X, y specific stds
            # Normalize by the distance of median nearest neighbor
            dists = dists / (adap_sd[x] + 1e-10)
            W = csr_matrix((np.exp(-dists), (x, y)), shape=[self.N, self.N])
        if self.kernel_use == 'simple_adaptive':
            # X, y specific stds with neighborhood expansion
            # Normalize by normalized contribution to neighborhood size.
            dists = dists_new / (adap_nbr[x_new] + 1e-10)
            W = csr_matrix((np.exp(-dists), (x_new, y_new)),
                           shape=[self.N, self.N])
        if self.kernel_use == 'decay':
            # X, y specific stds, alpha-adaptive decay
            dists = (dists / (adap_sd[x] + 1e-10)) ** np.power(2,
                                                               ((self.n_neighbors - pm[x]) / pm[x]))
            W = csr_matrix((np.exp(-dists), (x, y)), shape=[self.N, self.N])
        if self.kernel_use == 'decay_adaptive':
            # X, y specific stds, with neighborhood expansion
            dists = (dists_new / (adap_nbr[x_new] + 1e-10)) ** np.power(
                2, ((self.new_k - pm[x_new]) / pm[y_new]))
            W = csr_matrix((np.exp(-dists), (x_new, y_new)),
                           shape=[self.N, self.N])

        # Kernel symmetrization
        W = (W + W.T) / 2
        self.K = W
        self.K[(np.arange(self.K.shape[0]), np.arange(self.K.shape[0]))] = 0
        # handle nan, zeros
        self.K.data = np.where(np.isnan(self.K.data), 1, self.K.data)

        # Anisotropic diffusion. Here we'll use the symmetrized version, so we'll need to convert it back later
        if self.alpha > 0:
            self.P, self._D_left = diffusion_operator(
                self.K, self.alpha, return_D_inv_sqrt=True)
        else:
            # if no anisotropy is added, there's no need to convert it later
            D = np.ravel(self.K.sum(axis=1))
            D[D != 0] = 1 / D[D != 0]
            Dreg = csr_matrix(
                (D, (range(self.N), range(self.N))), shape=[self.N, self.N])
            self.P = Dreg.dot(self.K)

        end = time.time()
        if self.verbose:
            print('Diffusion operator building time = %f (sec), per sample=%f (sec), per sample adjusted for thread number=%f (sec)' %
                  (end - start_time, float(end - start_time) / self.N, self.n_jobs * float(end - start_time) / self.N))

        return self

    def T(self):
        return self.P

    def transform(self, X=None):
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

       ``Diffusor.res['DiffusionMaps']]``

        """
        start_time = time.time()
        if self.t is None or self.t < 1:
            self.multiscale = True

        evals, evecs = eigsh(self.P, self.n_eigs, tol=self.tol, maxiter=self.N)
        evals = np.real(evals)
        evecs = np.real(evecs)
        # If anisotropy was used, we'll need to convert the eigenvectors back
        if self.alpha > 0:
            evecs = self._D_left.dot(evecs)
        inds = np.argsort(evals)[::-1]
        evals = evals[inds]
        evecs = evecs[:, inds]
        # Normalize
        for i in range(evecs.shape[1]):
            evecs[:, i] = evecs[:, i] / np.linalg.norm(evecs[:, i])
        # Create the results dictionary
        self.res = {'EigenVectors': evecs, 'EigenValues': evals}
        self.res['EigenVectors'] = pd.DataFrame(self.res['EigenVectors'])
        self.res["EigenValues"] = pd.Series(self.res["EigenValues"])
        if self.multiscale:
            self.res['DiffusionMaps'], self.scaled_eigs = ms.multiscale(
                self.res, n_eigs='max', verbose=self.verbose)
        else:
            # Guarantee t is an integer
            if self.t is None:
                self.t = 1
            else:
                if self.t < 1:
                    self.t = 1
            if not isinstance(self.t, int):
                self.t = int(self.t)
                if not isinstance(self.t, int):
                    self.t = 1
            # Diffuse through t steps
            evals = np.power(evals, self.t)
            self.res['DiffusionMaps'] = evecs * evals

        end = time.time()

        if not self.cache:
            self.K = None
            self.T = None
            import gc
            gc.collect()

        if self.verbose:
            print('Eigendecomposition time = %f (sec), per sample=%f (sec), per sample adjusted for thread number=%f (sec)' %
                  (end - start_time, float(end - start_time) / self.N, self.n_jobs * float(end - start_time) / self.N))

        if self.plot_spectrum:
            self.spectrum_plot()
        return self.res['DiffusionMaps']

    def ind_dist_grad(self, data):
        """
        Depracated from 0.1.1.0 onwards.
        """

    def res_dict(self):
        """
        Returns
        -------
            Dictionary containing normalized and potentially multiscaled Diffusion Components
            (['DiffusionMaps']), and original eigenvalues (['EigenValues']) and
            eigenvectors (['EigenVectors']) from the eigendecomposition.
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
            n_eigs = self.n_eigs

        mms, self.scaled_eigs = ms.multiscale(self.res, n_eigs=n_eigs,
                                                               verbose=self.verbose)

        self.res['DiffusionMaps'] = mms
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
        import matplotlib.pyplot as plt
        evals = np.array(self.res['EigenValues'])
        if self.multiscale:
            label = 'Multiscaled eigenvalues'
        else:
            label = 'Eigengap'
        ax1 = plt.subplot(1, 1, 1)
        ax1.set_title('Spectrum decay and eigengap (%i)' %
                      int(self.scaled_eigs))
        ax1.plot(range(0, len(evals)), evals, 'b')
        ax1.set_ylabel('Eigenvalues')
        ax1.set_xlabel('Eigenvectors')
        ax1.vlines(
            self.scaled_eigs, plt.ylim()[0], plt.ylim()[1], linestyles="--", label=label
        )
        ax1.legend(loc='best')
        plt.tight_layout()
        return plt.show()
