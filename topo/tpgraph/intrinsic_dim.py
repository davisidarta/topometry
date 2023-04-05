## Algorithms intrinsic dimensionality estimation
# The cmFSA and maximum-likelihood implementations are adapted from Zsigmond Benkő (https://github.com/phrenico/cmfsapy/) and is licensed under the MIT license.
# The TwoNN implementation imports scikit-dimension (https://scikit-dimension.readthedocs.io/en/latest/index.html)
#
import numpy as np
from scipy.sparse.linalg import eigsh
from topo.spectral import diffusion_operator
from topo.base.ann import kNN
from scipy.spatial import cKDTree
from topo.utils._utils import get_indices_distances_from_sparse_matrix


def get_dists_inds_ck(X, k, boxsize):
    """computes the kNN distances and indices
    :param numpy.ndarray X:  2D array with data shape: (ndata, n_vars)
    :param int k: neighborhood size
    :param float boxsize:  circular boundary condition to [0, boxsice] interval for all input dimensions if not None.
    :return: KNN distances and indices
    """
    tree = cKDTree(X, boxsize=boxsize)
    dists, inds = tree.query(X, k + 1, n_jobs=-1)
    return dists, inds

def szepesvari_dimensionality(dists):
    """Compute szepesvari dimensions from kNN distances
    :param dists:
    :return:
    """
    n = dists.shape[1]
    lower_k = np.arange(np.ceil(n / 2)).astype(int)
    upper_k = np.arange(n)[::2]
    d = - np.log(2) / np.log(dists[:, lower_k] / dists[:, upper_k])
    return d

def fsa(X, k, boxsize=None):
    """Measure local Szepesvari-Farahmand dimension, distances are computed by the cKDTree algoritm
    :param arraylike X: data series [n x dim] shape
    :param k: maximum k value
    :param boxsize: apply d-toroidal distance computation with edge-size =boxsize, see ckdtree class for more
    :return: local estimates, distances, indicees
    """
    dists, inds = get_dists_inds_ck(X, 2*k, boxsize)
    dims = szepesvari_dimensionality(dists)
    return dims, dists, inds

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
def ml_dims(dists, n_neighbors=10):
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

   

