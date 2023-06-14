
#####################################
# Author: Davi Sidarta-Oliveira
# School of Medical Sciences,University of Campinas,Brazil
# contact: davisidarta[at]fcm[dot]unicamp[dot]com
# License: MIT
######################################
# This is a highly performant python implementation of the Ck-NN graph construction algorithm
# CkNN was developed by Tyrus Berry and Timothy Sauer [http://dx.doi.org/10.3934/fods.2019001]

import numpy as np
from scipy.sparse import csr_matrix, find
from sklearn.base import TransformerMixin
from topo.base.ann import kNN
from topo.spectral._spectral import graph_laplacian

def cknn_graph(X, n_neighbors=10,
               delta=1.0,
               metric='euclidean',
               weighted=False,
               include_self=False,
               return_densities=False,
               backend='nmslib',
               n_jobs=1,
               verbose=False,
               **kwargs):
    """
    Function-oriented implementation of Continuous k-Nearest-Neighbors (CkNN).
    An efficient implementation of [CkNN](https://arxiv.org/pdf/1606.02353.pdf).
    CkNN is the only unweighted graph construction that 
    can be used to approximate the Laplace-Beltrami Operator via the unnormalized graph Laplacian.
    It can also be used to wield an weighted affinity matrix with locality-sensitive weights.

    Parameters
    ----------

    n_neighbors : int (optional, default=5).
        Number of neighbors to compute. The actual number of k-nearest neighbors
        to be used in the CkNN normalization is half of it.

    delta : float (optional, default=1.0).
        A parameter to decide the radius for each points. The combination
        radius increases in proportion to this parameter. This should be tunned.

    metric : str (optional, default='euclidean').
        The metric of each points. This parameter depends on the parameter
        `metric` of scipy.spatial.distance.pdist.

    weighted : bool (optional, default=False).
        If True, the CkNN graph is weighted (i.e. an affinity matrix). If False, the CkNN graph is unweighted (i.e. the proper adjacency matrix).
        If None, will return a tuple of the adjacency matrix (unweighted) and the affinity matrix (weighted).

    return_densities : bool (optional, default=False).
        If True, will return the distance to the k-nearest-neighbor of each points.

    include_self : bool (optional, default=True).
            All diagonal elements are 1.0 if this parameter is True.

    backend : str 'hnwslib', 'nmslib' or 'sklearn' (optional, default 'nmslib').
        Which backend to use to compute nearest-neighbors. Options for fast, approximate nearest-neighbors
        are 'hnwslib'  and 'nmslib' (default). For exact nearest-neighbors, use 'sklearn'.

        * If using 'nmslib', a sparse
        csr_matrix input is expected. If using 'hnwslib' or 'sklearn', a dense array is expected.
        * I strongly recommend you use 'hnswlib' if handling with somewhat dense, array-shaped data. If the data
        is relatively sparse, you should use 'nmslib', which operates on sparse matrices by default on
        TopOMetry and will automatically convert the input array to csr_matrix for performance.

    n_jobs : int (optional, default 1).
        The number of jobs to use in the k-nearest-neighbors computation. Defaults to one (I highly recommend you use all available).

    verbose : bool (optional, default False).
        If True, print progress messages.

    kwargs : dict (optional, default {}).
        Additional parameters to pass to the k-nearest-neighbors backend.


    """
    N = X.shape[0]
    if metric == 'precomputed':
        knn = X.copy()
    else:
        # Fit kNN - we'll use a smaller number of k-neighbors for the CkNN normalization
        knn = kNN(X, n_neighbors=n_neighbors,
                  metric=metric,
                  n_jobs=n_jobs,
                  backend=backend,
                  verbose=verbose,
                  **kwargs)

    median_k = np.floor(n_neighbors / 2).astype(int)
    adap_sd = np.zeros(N)
    for i in np.arange(len(adap_sd)):
        adap_sd[i] = np.sort(knn.data[knn.indptr[i]: knn.indptr[i + 1]])[
            median_k - 1
        ]

    x, y, dists = find(knn)
    # The CkNN normalization
    # prevent division by zero
    cknn_norm = delta * np.sqrt(adap_sd.dot(adap_sd.T)) + 1e-12
    A = csr_matrix(((dists / cknn_norm), (x, y)),
                   shape=[N, N])
    dd = np.arange(N)
    if include_self:
        A[dd, dd] = True
    else:
        A[dd, dd] = False
    if weighted is None:
        if return_densities:
            return A.astype(int), A.astype(np.float), adap_sd
        else:
            return A.astype(int), A.astype(np.float)
    else:
        if weighted:
            if return_densities:
                return A.astype(np.float), adap_sd
            else:
                return A.astype(np.float)
        else:
            if return_densities:
                return A.astype(int), adap_sd
            else:
                return A.astype(int)


