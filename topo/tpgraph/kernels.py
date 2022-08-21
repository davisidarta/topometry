# Implementing different kernels
#####################################
# Author: Davi Sidarta-Oliveira
# School of Medical Sciences,University of Campinas,Brazil
# contact: davisidarta[at]fcm[dot]unicamp[dot]com
# License: MIT
######################################
import numpy as np
from scipy.sparse import find, csr_matrix
from sklearn.base import TransformerMixin
from topo.base.ann import kNN
from topo.base.dists import \
    (euclidean,
     standardised_euclidean,
     cosine,
     correlation,
     bray_curtis,
     canberra,
     chebyshev,
     manhattan,
     mahalanobis,
     minkowski,
     dice,
     hamming,
     jaccard,
     kulsinski,
     rogers_tanimoto,
     russellrao,
     sokal_michener,
     sokal_sneath,
     yule,
     matrix_pairwise_distance
     )


def _get_metric_function(metric):
    if metric == 'euclidean':
        metric_fun = euclidean
    elif metric == 'standardised_euclidean':
        metric_fun = standardised_euclidean
    elif metric == 'cosine':
        metric_fun = cosine
    elif metric == 'correlation':
        metric_fun = correlation
    elif metric == 'bray_curtis':
        metric_fun = bray_curtis
    elif metric == 'canberra':
        metric_fun = canberra
    elif metric == 'chebyshev':
        metric_fun = chebyshev
    elif metric == 'manhattan':
        metric_fun = manhattan
    elif metric == 'mahalanobis':
        metric_fun = mahalanobis
    elif metric == 'minkowski':
        metric_fun = minkowski
    elif metric == 'dice':
        metric_fun = dice
    elif metric == 'hamming':
        metric_fun = hamming
    elif metric == 'jaccard':
        metric_fun = jaccard
    elif metric == 'kulsinski':
        metric_fun = kulsinski
    elif metric == 'rogers_tanimoto':
        metric_fun = rogers_tanimoto
    elif metric == 'russellrao':
        metric_fun = russellrao
    elif metric == 'sokal_michener':
        metric_fun = sokal_michener
    elif metric == 'sokal_sneath':
        metric_fun = sokal_sneath
    elif metric == 'yule':
        metric_fun = yule
    return metric_fun

def _adap_bw(K, n_neighbors):
    median_k = np.floor(n_neighbors/2).astype(int)
    adap_sd = np.zeros(K.shape[0])
    for i in np.arange(len(adap_sd)):
        adap_sd[i] = np.sort(K.data[K.indptr[i]: K.indptr[i + 1]])[median_k - 1]
    return adap_sd

def compute_kernel(X, metric='cosine',
                     n_neighbors=10, pairwise=False, sigma=None, adaptive_bw=True,
                 expand_nbr_search=False, alpha_decaying=False, return_densities=False, symmetrize=True,
                 backend='nmslib', n_jobs=-1, **kwargs):
    """
    Compute a kernel matrix from a set of points.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features).
        The set of points to compute the kernel matrix for. Accepts np.ndarrays and scipy.sparse matrices.
        If precomputed, assumed to be a square symmetric semidefinite matrix.

    metric : string, optional (default: 'cosine').
        The metric to use when computing the kernel matrix.
        Possible values are: 'cosine', 'euclidean' and others. Accepts precomputed distances.

    n_neighbors : int, optional (default 10).
        The number of neighbors to use when computing the kernel matrix. Ignored if `pairwise` set to `True`.

    pairwise : bool, optional (default False).
        Whether to compute the kernel using dense pairwise distances. 
        If set to `True`, the `n_neighbors` and `backend` parameters are ignored.
        Uses `numba` for computations if available. If not, uses `sklearn`.

    sigma : float, optional (default None).
        Scaling factor if using fixed bandwidth kernel (only used if `adaptive_bw` is set to `False`).

    adaptive_bw : bool, optional (default True).
        Whether to use an adaptive bandwidth based on the distance to median k-nearest-neighbor.

    expand_nbr_search : bool, optional (default False).
        Whether to expand the neighborhood search (mitigates a choice of too small a number of k-neighbors).

    alpha_decaying : bool, optional (default False).
        Whether to use an adaptively decaying kernel.

    return_densities : bool, optional (default False).
        Whether to return the bandwidth metrics as a dictinary. If set to `True`, the function
        returns a tuple containing the kernel matrix and a dictionary containing the
        bandwidth metric.

    symmetrize : bool, optional (default True).
        Whether to symmetrize the kernel matrix after normalizations.

    backend : str, optional (default 'nmslib').
        Which backend to use for neighborhood computations. Defaults to 'nmslib'.
        Options are 'nmslib', 'hnswlib', 'faiss', 'annoy' and 'sklearn'. 

    **kwargs : dict, optional
        Additional arguments to be passed to the neighborhood backend.

    Returns
    -------
    K : array-like, shape (n_samples, n_samples)
        The kernel matrix.
    densities : dict, optional
        A dictionary containing the bandwidth metrics.

    """
    if n_jobs == -1:
        from joblib import cpu_count
        n_jobs = cpu_count()
    k = n_neighbors
    if metric == 'precomputed':
        K = X
        expand_nbr_search = False
    else:
        if pairwise:
            metric_fun = _get_metric_function(metric)
            K = matrix_pairwise_distance(X, metric_fun)
        else:
            K = kNN(X, metric=metric, n_neighbors=k, backend=backend, n_jobs=n_jobs, **kwargs)
    if adaptive_bw:
        adap_sd = _adap_bw(K, k)
        # Get an indirect measure of the local density
        pm = np.interp(adap_sd, (adap_sd.min(), adap_sd.max()), (2, k))
        if expand_nbr_search:
            new_k = int(k + (k - pm.max()))
            K = kNN(X, metric=metric, k=new_k, backend=backend, n_jobs=n_jobs, **kwargs)
            adap_sd_new = _adap_bw(K, new_k)
            # Get an indirect measure of the local density
            pm_new = np.interp(adap_sd_new, (adap_sd_new.min(), adap_sd_new.max()), (2, new_k))
    x, y, dists = find(K)
    # Normalize distances
    if adaptive_bw:
        # Alpha decaying: the kernel adaptively decays depending on neighborhood density
        if alpha_decaying:
            if expand_nbr_search:
                dists = (dists / (adap_sd_new[x] + 1e-10)) ** np.power(2,((new_k - pm_new[x]) / pm_new[x]))
            else:
                dists = (dists / (adap_sd[x] + 1e-10)) ** np.power(2,((k - pm[x]) / pm[x]))
        else:
            if expand_nbr_search:
                dists = (dists / (adap_sd_new[x] + 1e-10)) ** 2
            else:
                dists = (dists / (adap_sd[x] + 1e-10)) ** 2
        kernel = csr_matrix((np.exp(-dists), (x, y)), shape=K.shape)
    else:
        if sigma is not None:
            if sigma == 0:
                sigma = 1e-10
            dists = (dists / sigma) ** 2
        kernel = csr_matrix((np.exp(-dists), (x, y)), shape=K.shape)
    if symmetrize:
        kernel = (kernel + kernel.T) / 2
        kernel[(np.arange(kernel.shape[0]), np.arange(kernel.shape[0]))] = 0
        # handle nan, zeros
        kernel.data = np.where(np.isnan(kernel.data), 1, kernel.data)
    if not return_densities:
        return kernel
    else:
        if not adaptive_bw:
            return kernel, None
        else:
            dens_dict = {}
            dens_dict['omega'] = pm
            dens_dict['adaptive_bw'] = adap_sd
            if expand_nbr_search:
                dens_dict['omega_new'] = pm_new
                dens_dict['adaptive_bw_new'] = adap_sd_new
            return kernel, dens_dict


class Kernel(TransformerMixin):
    """
    Compute a kernel matrix from a set of points.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The set of points to compute the kernel matrix for. Accepts np.ndarrays and scipy.sparse matrices.
        If precomputed, assumed to be a square symmetric semidefinite matrix.

    metric : string, optional
        The metric to use when computing the kernel matrix.
        Possible values are: 'cosine', 'euclidean' and others. Accepts precomputed distances.

    n_neighbors : int, optional (default 10).
        The number of neighbors to use when computing the kernel matrix. Ignored if `pairwise` set to `True`.

    pairwise : bool, optional (default False).
        Whether to compute the kernel using dense pairwise distances. 
        If set to `True`, the `n_neighbors` and `backend` parameters are ignored.
        Uses `numba` for computations if available. If not, uses `sklearn`.

    sigma : float, optional
        Scaling factor if using fixed bandwidth kernel (only used if `adaptive_bw` is set to `False`).

    adaptive_bw : bool, optional
        Whether to use an adaptive bandwidth based on the distance to median k-nearest-neighbor.

    expand_nbr_search : bool, optional
        Whether to expand the neighborhood search (mitigates a choice of too small a number of k-neighbors).

    alpha_decaying : bool, optional
        Whether to use an adaptively decaying kernel.

    return_densities : bool, optional
        Whether to return the bandwidth metrics as a dictinary. If set to `True`, the function
        returns a tuple containing the kernel matrix and a dictionary containing the
        bandwidth metric.

    symmetrize : bool, optional
        Whether to symmetrize the kernel matrix after normalizations.

    backend : str, optional
        Which backend to use for k-nearest-neighbor computations. Defaults to 'nmslib'.
        Options are 'nmslib', 'hnswlib', 'faiss', 'annoy' and 'sklearn'.    

    """

    def __init__(self,
                metric='cosine',
                n_neighbors=10,
                pairwise=False,
                sigma=None,
                adaptive_bw=True,
                expand_nbr_search=False,
                alpha_decaying=False,
                return_densities=False,
                symmetrize=True,
                backend='nmslib',
                n_jobs=-1
                ):
        self.n_neighbors = n_neighbors
        self.pairwise = pairwise
        self.n_jobs = n_jobs
        self.backend = backend
        self.metric = metric
        self.sigma = sigma
        self.adaptive_bw = adaptive_bw
        self.expand_nbr_search = expand_nbr_search
        self.alpha_decaying = alpha_decaying
        self.return_densities = return_densities
        self.symmetrize = symmetrize

    def __repr__(self):
        if self.metric != 'precomputed':
            if (self.N is not None) and (self.M is not None):
                msg = "Kernel() instance with %i samples and %i observations" % (
                    self.N, self.M) + " and:"
            else:
                msg = "Kernel() instance object without any fitted data."
        else:
            if (self.N is not None) and (self.M is not None):
                msg = "Kernel() instance with %i samples" % (self.N) + " and:"
            else:
                msg = "Kernel() instance object without any fitted data."
        if self.knn is not None:
            msg = msg + " \n    %i-nearest-neighbors fitted. " % self.knn
        if self.pdists is not None:
            msg = msg + " \n    Pairwise distances fitted."
        if self.K is not None:
            if not self.adaptive_bw:
                kernel_msg = " \n fixed bandwidth kernel with sigma = %.2f" % self.sigma
            else:
                kernel_msg = " \n adaptive bandwidth "
                if self.alpha_decaying:
                    kernel_msg = kernel_msg + "with alpha decaying kernel"
                if self.expand_nbr_search:
                    kernel_msg = kernel_msg + ", with expanded neighborhood search."
            msg = msg + kernel_msg
        return msg

    def fit(self, X, **kwargs):
        """
        Fits the kernel matrix to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features).
            Input data. Takes in numpy arrays and scipy csr sparse matrices.
            Use with sparse data for top performance. You can adjust a series of
            parameters that can make the process faster and more informational depending
            on your dataset.

        **kwargs : dict, optional
            Additional arguments to be passed to the k-nearest-neighbor backend.

        Returns
        -------
            The kernel matrix.

        """

        self.K = compute_kernel(X, metric=self.metric,
                     n_neighbors=self.n_neighbors, pairwise=self.pairwise, sigma=self.sigma, adaptive_bw=self.adaptive_bw,
                 expand_nbr_search=self.expand_nbr_search, alpha_decaying=self.alpha_decaying, return_densities=self.return_densities, symmetrize=self.symmetrize,
                 backend=self.backend, n_jobs=self.n_jobs, **kwargs)
        return self.K
    
    def transform(self, X):
        """
        Returns the kernel matrix. Here for compability with scikit-learn only.
        """
        return self.K

    def fit_transform(self, X, **kwargs):
        """
        Fits the kernel matrix to the data and returns the kernel matrix.
        Here for compability with scikit-learn only.
        """
        self.fit(X, **kwargs)
        return self.K
