## Algorithms intrinsic dimensionality estimation
import numpy as np
from scipy.sparse.linalg import eigsh
from topo.spectral import diffusion_operator
from topo.base.ann import kNN
from sklearn.base import BaseEstimator, TransformerMixin
from topo.utils._utils import get_indices_distances_from_sparse_matrix


class IntrinsicDim(BaseEstimator, TransformerMixin):
    """
    Scikit-learn flavored class for estimating the intrinsic dimensionalities of high-dimensional data.
    This class iterates over a range of possible values of k-nearest-neighbors to consider in calculations
    using two different methods: the Farahmand-Szepesvári-Audibert (FSA) dimension estimator and the Maximum Likelihood Estimator (MLE).

    Parameters
    ----------
    methods : list of str, (default ['fsa'])
        The dimensionality estimation methods to use. Current options are
        'fsa' () and 'mle'().

    k : int, range or list of ints, (default [10, 20, 50, 75, 100])
        The number of nearest neighbors to use for the dimensionality estimation methods.
        If a single value of `k` is provided, then the result dictionary will have
        keys corresponding to the methods, and values corresponding to the
        dimensionality estimates.
        If multiple values of `k` are provided, then the result dictionary will have
        keys corresponding to the number of k, and values corresponding to other dictionaries,
        which have keys corresponding to the methods, and values corresponding to the
        dimensionality estimates.

    metric : str (default 'euclidean')
        The metric to use when calculating distance between instances in a feature array.

    backend : str (optional, default 'nmslib').
        Which backend to use for k-nearest-neighbor computations. Defaults to 'nmslib'.
        Options are 'nmslib', 'hnswlib', 'faiss', 'annoy' and 'sklearn'.   

    n_jobs : int (optional, default 1).
        The number of jobs to use for parallel computations. If -1, all CPUs are used.
        Parallellization (multiprocessing) is ***highly*** recommended whenever possible.

    plot : bool (optional, default True).
        Whether to plot the results when using the `fit()` method.
        
    random_state : int or numpy.random.RandomState() (optional, default None).
        A pseudo random number generator. Used for generating colors for plotting.
    
    **kwargs : keyword arguments
        Additional keyword arguments to pass to the backend kNN estimator.


    Properties
    ----------
    local_id, global_id : dictionaries containing local and global dimensionality estimates, respectivelly.
    
        Their structure depends on the value of the `k` parameter:

        * If a single value of `k` is provided, then the dictionaries will have
        keys corresponding to the methods, and values corresponding to the
        dimensionality estimates. 

        * If multiple values of `k` are provided, then the dictionaries will have
        keys corresponding to the number of k, and values corresponding to other dictionaries,
        which have keys corresponding to the methods, and values corresponding to the
        dimensionality estimates. 

    """

    def __init__(self,
                methods=['fsa','mle'],
                k=[10, 20, 50, 75, 100],
                backend='hnswlib',
                metric='euclidean',
                n_jobs=-1,
                plot=True,
                random_state=None,
                **kwargs):
        if isinstance(methods, str):
            methods = [methods]
        if isinstance(k, list):
            n_k = len(k)
            use_k = k
        elif isinstance(k, int):
            n_k = 1
            use_k = k
        elif isinstance(k, range):
            n_k = len(k)
            use_k = k
        self.methods = methods
        self.use_k = use_k
        self.n_k = n_k
        self.backend = backend
        self.metric = metric
        self.n_jobs = n_jobs
        self.plot = plot
        self.kwargs = kwargs
        self.random_state = random_state
        self.local_id = {}
        self.global_id = {}

    def __repr__(self):
        msg = 'IntrinsicDim estimator'
        msg = msg + '\nMethods: ' + str(self.methods)
        msg = msg + '\nk: ' + str(self.use_k)
        return msg

    def _parse_random_state(self):
        if self.random_state is None:
            self.random_state = np.random.RandomState()
        elif isinstance(self.random_state, np.random.RandomState):
            pass
        elif isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        else:
            print('RandomState error! No random state was defined!')

    def _compute_id(self, X):
        self.local_id['fsa'] = {}
        self.local_id['mle'] = {}
        self.global_id['fsa'] = {}
        self.global_id['mle'] = {}
        if self.n_k == 1:
            knn = kNN(X, n_jobs=self.n_jobs, n_neighbors=self.use_k, metric=self.metric, backend=self.backend)
            for method in self.methods:
                if method not in ['fsa', 'mle']:
                    raise ValueError('Invalid method. Valid methods are: fsa, mle.')
                if method == 'fsa':
                    self.local_id['fsa'][str(self.use_k)] = fsa_local(knn, self.use_k)
                    self.global_id['fsa'][str(self.use_k)] = fsa_global(knn, id_local=self.local_id['fsa'][str(self.use_k)])
                elif method == 'mle':
                    self.local_id['mle'][str(self.use_k)] = mle_local(knn, self.use_k)
                    self.global_id['mle'][str(self.use_k)] = mle_global(knn, id_local=self.local_id['mle'][str(self.use_k)])

        else:
            for k in self.use_k:
                knn = kNN(X, n_jobs=self.n_jobs, n_neighbors=k, metric=self.metric, backend=self.backend)
                for method in self.methods:
                    if method not in ['fsa', 'mle']:
                        raise ValueError('Invalid method. Valid methods are: fsa, mle.')
                    if method == 'fsa':
                        self.local_id['fsa'][str(k)] = fsa_local(knn, k)
                        self.global_id['fsa'][str(k)] = fsa_global(knn, id_local=self.local_id['fsa'][str(k)])
                    elif method == 'mle':
                        self.local_id['mle'][str(k)] = mle_local(knn, k)
                        self.global_id['mle'][str(k)] = mle_global(knn, id_local=self.local_id['mle'][str(k)])

    def plot_id(self, bins=30, figsize=(6, 8), titlesize=22, labelsize=16, legendsize=10):
        self._parse_random_state()
        colors = []
        from random import randint
        from matplotlib import pyplot as plt
        for i in range(4):
            colors.append('#%06X' % randint(0, 0xFFFFFF))
        if len(self.methods) == 1:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            method = self.methods[0]
            for key in self.local_id[method].keys():
                i=0
                x = self.local_id[method][key]
                # Make a multiple-histogram of data-sets with different length.
                label = 'k = ' + key + '    ( estim.i.d. = ' + str(int(self.global_id[method][key])) + ' )'
                n, bins, patches  = ax.hist(x, bins=30, histtype='step', stacked=True, density=True, log=False, label=label)
                sigma = np.std(x)
                mu = np.mean(x)
                y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
                    np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
                i= i+1
            ax.set_title(method.upper(), fontsize=titlesize, pad=10)
            ax.legend(prop={'size': 12}, fontsize=legendsize)
            ax.set_xlabel('Estimated intrinsic dimension', fontsize=labelsize)
            ax.set_ylabel('Frequency', fontsize=labelsize)

        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
            for key in self.local_id['fsa'].keys():
                i=0
                x = self.local_id['fsa'][key]
                # Make a multiple-histogram of data-sets with different length.
                label = 'k = ' + key + '    ( estim.i.d. = ' + str(int(self.global_id['fsa'][key])) + ' )'
                n, bins, patches  = ax1.hist(x, bins=30, histtype='step', stacked=True, density=True, log=False, label=label)
                sigma = np.std(x)
                mu = np.mean(x)
                y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
                    np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
                i= i+1
            ax1.set_title('FSA', fontsize=titlesize, pad=10)
            ax1.legend(prop={'size': 12}, fontsize=legendsize)
            ax1.set_xlabel('Estimated intrinsic dimension', fontsize=labelsize)
            ax1.set_ylabel('Frequency', fontsize=labelsize)
            
            for key in self.local_id['mle'].keys():
                i=0
                x = self.local_id['mle'][key]
                # Make a multiple-histogram of data-sets with different length.
                label = 'k = ' + key + '    ( estim.i.d. = ' + str(int(self.global_id['mle'][key])) + ' )'
                n, bins, patches  = ax2.hist(x, bins=30, histtype='step', stacked=True, density=True, log=False, label=label)
                sigma = np.std(x)
                mu = np.mean(x)
                y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
                    np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
                i= i+1
            ax2.set_title('MLE', fontsize=titlesize, pad=10)
            ax2.legend(prop={'size': 12}, fontsize=legendsize)
            ax2.set_xlabel('Estimated intrinsic dimension', fontsize=labelsize)
            ax2.set_ylabel('Frequency', fontsize=labelsize)

        fig.tight_layout()
        plt.show()

    def fit(self, X, **kwargs):
        """
        Estimates the intrinsic dimensionalities of the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The set of points to compute the kernel matrix for. Accepts np.ndarrays and scipy.sparse matrices.
            If precomputed, assumed to be a square symmetric semidefinite matrix of k-nearest-neighbors, with
            its k higher or equal the highest value of k used to estimate the intrinsic dimensionalities.

        **kwargs: keyword arguments
            Additional keyword arguments to pass to the plotting function `IntrinsicDim.plot_id()`.

        Returns
        -------
        Populates the `local` and `global` properties of the class.

        Shows a plot of the results if `plot=True`.
        """
        self._compute_id(X)
        if self.plot:
            self.plot_id(**kwargs)

    def transform(self, X=None):
        """
        Does nothing. Here for compability with scikit-learn only.
        """
        print('Dummy function. Does nothing. Here for compability with scikit-learn only.')
        return self


def _get_dist_to_k_nearest_neighbor(K, n_neighbors=10):
    dist_to_k = np.zeros(K.shape[0])
    for i in np.arange(len(dist_to_k)):
        dist_to_k[i] = np.sort(
            K.data[K.indptr[i]: K.indptr[i + 1]])[n_neighbors - 1]
    return dist_to_k

def _get_dist_to_median_nearest_neighbor(K, n_neighbors=10):
    median_k = np.floor(n_neighbors/2).astype(int)
    dist_to_median_k = np.zeros(K.shape[0])
    for i in np.arange(len(dist_to_median_k)):
        dist_to_median_k[i] = np.sort(
            K.data[K.indptr[i]: K.indptr[i + 1]])[median_k - 1]
    return dist_to_median_k

def fsa_local(K, n_neighbors=10):
    """
    Measure local dimensionality using the Farahmand-Szepesvári-Audibert (FSA) dimension estimator
    
    Parameters
    ----------
    K: sparse matrix
        Sparse matrix of distances between points

    n_neighbors: int
        Number of neighbors to consider for the kNN graph. 
        Note this is actually half the number of neighbors used in the FSA estimator, for efficiency.

    Returns
    -------
    local_dim: array
        Local dimensionality estimate for each point
    """
    dist_to_k = _get_dist_to_k_nearest_neighbor(K, n_neighbors=n_neighbors)
    dist_to_median_k = _get_dist_to_median_nearest_neighbor(K, n_neighbors=n_neighbors)
    d = - np.log(2) / np.log(dist_to_median_k / dist_to_k)
    return d


def fsa_global(K, id_local=None, **kwargs):
    from statistics import median
    if id_local is None:
        dims = fsa_local(K, **kwargs)
    else:
        dims = id_local
    return median(np.abs(dims)) / np.log(2)


def mle_local(K, n_neighbors=10, k1=1):
    """Maximum likelihood estimator af intrinsic dimension (Levina-Bickel)"""
    inds, dists = get_indices_distances_from_sparse_matrix(K, n_neighbors)
    norm_dists = dists / dists[:, -1:]
    dims = -1./ np.nanmean(np.log(norm_dists[:, k1:-1]), axis=1)
    return dims

def mle_global(K, id_local=None, n_neighbors=15, k1=1):
    if id_local is None:
        id_local, _, _ = mle_local(K, n_neighbors, k1)
    return 1.0 / np.mean(1.0 / id_local)




# EXPERIMENTAL
def local_eigengap_experimental(X, max_n_components=30, n_neighbors=30, metric='cosine', verbose=False, **kwargs):
    # Construct neighborhood graph
    graph = kNN(X, n_neighbors=n_neighbors, metric=metric, **kwargs)
    # Compute graph Laplacian
    diff_op = diffusion_operator(graph)
    if verbose:
        print('Started evaluating locally...')
    # Compute local intrinsic dimensionality
    max_n_components = 50  # number of eigenvalues to compute
    local_dim = np.zeros((graph.shape[0], max_n_components))
    for i in range(graph.shape[0]):
        indices = graph.indices[graph.indptr[i]:graph.indptr[i+1]]
        P = diff_op[indices][:, indices]
        evals, evecs = eigsh(P, k=max_n_components, which='LM')
        max_eigs = int(np.sum(evals > 0, axis=0))
        first_diff = np.diff(evals)
        eigengap = np.argmax(first_diff) + 1
        if max_eigs == len(evals):
            # Could not find a discrete eigengap crossing 0
            use_eigengap = eigengap
        else:
            # Found a discrete eigengap crossing 0
            use_eigengap = max_eigs
        local_dim[i] = use_eigengap
    return local_dim

   

