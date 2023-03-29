## Algorithms intrinsic dimensionality estimation
# The TwoNN implementation is adapted from Francesco Mottes (https://github.com/fmottes/TWO-NN/) and is licensed under the MIT license.
# The cmFSA and maximum-likelihood implementations are adapted from Zsigmond Benkő (https://github.com/phrenico/cmfsapy/) and is also licensed under the MIT license.

import numpy as np
import numba
from scipy.sparse import issparse, csr_matrix
try:
    from matplotlib import pyplot as plt
    _HAVE_MATPLOTLIB = True
except:
    _HAVE_MATPLOTLIB = False
import multiprocessing as mp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from topo.base.ann import kNN
from topo.utils._utils import get_indices_distances_from_sparse_matrix
from sklearn.utils.validation import check_array
from sklearn.linear_model import LinearRegression

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

def estimate_local_dim_fsa(K, n_neighbors=10):
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


# From https://github.com/phrenico/cmfsapy/
def ml_estimator(normed_dists):
    return -1./ np.nanmean(np.log(normed_dists), axis=1)

# From https://github.com/phrenico/cmfsapy/
def ml_dims(K, n_neighbors=10):
    """Maximum likelihood estimator af intrinsic dimension (Levina-Bickel)"""
    norm_dists = dists / _get_dist_to_k_nearest_neighbor(K, n_neighbors=n_neighbors)
    dims = ml_estimator(norm_dists.toarray()[:, 1:-1])
    return dims

# From https://github.com/phrenico/cmfsapy/
def szepes_ml(local_d):
    from scipy.stats import hmean
    """maximum likelihood estimator from local FSA estimates (for k=1)
    :param numpy.ndarray of float local_d: local FSA estimates
    :return: global ML-FSA estimate
    """
    return  hmean(local_d) / np.log(2)

def local_eigengap_experimental(K, max_n_components=30, verbose=False):
    from scipy.sparse.linalg import eigsh
    from topo.spectral import diffusion_operator
    from topo.base.ann import kNN
    # Construct neighborhood graph
    graph = kNN(X, n_neighbors=30)
    # Compute graph Laplacian
    diff_op = diffusion_operator(graph)
    if verbose:
        print('Started evaluating locally...')
    # Compute local intrinsic dimensionality
    n_components = 50  # number of eigenvalues to compute
    local_dim = np.zeros((X.shape[0], n_components))
    for i in range(X.shape[0]):
        indices = graph.indices[graph.indptr[i]:graph.indptr[i+1]]
        P = diff_op[indices][:, indices]
        evals, evecs = eigsh(diff_op, k=n_components, which='LM')
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




def _estimate_dim_two_nn(K, discard_fraction = 0.1):
    N = np.shape(K)[0]
    r1, r2 = K[:, 1], K[:, 2]
    _mu = r2 / r1
    # discard the largest distances
    mu = _mu[np.argsort(_mu)[: int(N * (1 - discard_fraction))]]
    # Empirical cumulate
    Femp = np.arange(int(N * (1 - discard_fraction))) / N
    # Fit line
    lr = LinearRegression(fit_intercept=False)
    lr.fit(np.log(mu).reshape(-1, 1), -np.log(1 - Femp).reshape(-1, 1))
    d = lr.coef_[0][0]  # extract slope
    return d
    
def estimate_dim_twoNN(X, discard_fraction = 0.1,
                precomputed_knn = False,
                n_neighbors = 30,
                metric='euclidean',
                n_jobs=1,
                backend='nmslib',
                **kwargs):
    """
    Intrinsic dimension estimation using the TwoNN algorithm. 

    Parameters
    ----------  
    discard_fraction: float 
        Fraction (between 0 and 1) of largest distances to discard (heuristic from the paper)

    precomputed_knn: bool
        Whether data is a precomputed distance matrix

    n_neighbors: int
        Number of neighbors to consider for the kNN graph

    metric: str
        Distance metric to use for the kNN graph
    
    n_jobs: int
        Number of jobs to use for the kNN graph

    backend: str
        Backend to use for the kNN graph (options are 'nmslib', 'hnswlib', 'annoy', 'faiss' and 'sklearn')

    kwargs: dict
        Additional keyword arguments to pass to the kNN estimator backend

    Returns
    -------
    A tuple with the following elements:

    global_dimension : float
        Estimated global intrinsic dimension of the data

    local_dimension : float
        Estimated local intrinsic dimension of each sample

    """
    if not precomputed_knn:
        K = kNN(X, n_neighbors=n_neighbors,
                    metric=metric,
                    n_jobs=n_jobs,
                    backend=backend,
                    return_instance=False,
                    verbose=False, **kwargs)
    else:
        K = X

    global_dim = _estimate_dim_two_nn(K, discard_fraction=discard_fraction)

    inds, dists = get_indices_distances_from_sparse_matrix(K)
    # if n_jobs > 1:
    #     pool = mp.Pool(n_jobs)
    #     results = pool.map(_estimate_dim_two_nn, [K[i, :] for i in inds])
    #     pool.close()
    #     dimension_pw_ = np.array([r for r in results])
    # else:
    local_dim = np.array(
        [_estimate_dim_two_nn(K[i, :], discard_fraction=discard_fraction) for i in inds]
    )
    return global_dim, local_dim
    
