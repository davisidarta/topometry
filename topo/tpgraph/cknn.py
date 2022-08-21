
#####################################
# Author: Davi Sidarta-Oliveira
# School of Medical Sciences,University of Campinas,Brazil
# contact: davisidarta[at]fcm[dot]unicamp[dot]com
# License: MIT
######################################
# This is a python implementation of the Ck-NN graph construction algorithm
# CkNN was developed by Tyrus Berry and Timothy Sauer [http://dx.doi.org/10.3934/fods.2019001]

import numpy as np
from scipy.sparse import csr_matrix, find
from topo.base.ann import kNN


def cknn_graph(X, n_neighbors=10,
               delta=1.0,
               metric='euclidean',
               weighted=False,
               include_self=False,
               backend='nmslib',
               n_jobs=1,
               M=15,
               efC=50,
               efS=50,
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

    M : int (optional, default 15).
        A neighborhood search parameter. Defines the maximum number of neighbors in the zero and above-zero layers
        during HSNW (Hierarchical Navigable Small World Graph). However, the actual default maximum number
        of neighbors for the zero layer is 2*M.  A reasonable range for this parameter
        is 5-100. For more information on HSNW, please check its manuscript(https://arxiv.org/abs/1603.09320).
        HSNW is implemented in python via NMSlib (https://github.com/nmslib/nmslib) and HNWSlib
        (https://github.com/nmslib/hnswlib).

    efC : int (optional, default 50).
        A neighborhood search parameter. Increasing this value improves the quality of a constructed graph
        and leads to higher accuracy of search. However this also leads to longer indexing times.
        A reasonable range for this parameter is 50-2000.

    efS : int (optional, default 50).
        A neighborhood search parameter. Similarly to efC, increasing this value improves recall at the
        expense of longer retrieval time. A reasonable range for this parameter is 100-2000.
    
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
                    M=M,
                    efC=efC,
                    efS=efS,
                    verbose=verbose,
                    *kwargs)
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
    if weighted:
        return A.astype(np.float)
    else:
        return A.astype(np.int)

class CkNN(object):
    """
    Object-oriented implementation of Continuous k-Nearest-Neighbors (CkNN).
    This is an efficient implementation of [CkNN](https://arxiv.org/pdf/1606.02353.pdf).
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

    M : int (optional, default 15).
        A neighborhood search parameter. Defines the maximum number of neighbors in the zero and above-zero layers
        during HSNW (Hierarchical Navigable Small World Graph). However, the actual default maximum number
        of neighbors for the zero layer is 2*M.  A reasonable range for this parameter
        is 5-100. For more information on HSNW, please check its manuscript(https://arxiv.org/abs/1603.09320).
        HSNW is implemented in python via NMSlib (https://github.com/nmslib/nmslib) and HNWSlib
        (https://github.com/nmslib/hnswlib).

    efC : int (optional, default 50).
        A neighborhood search parameter. Increasing this value improves the quality of a constructed graph
        and leads to higher accuracy of search. However this also leads to longer indexing times.
        A reasonable range for this parameter is 50-2000.

    efS : int (optional, default 50).
        A neighborhood search parameter. Similarly to efC, increasing this value improves recall at the
        expense of longer retrieval time. A reasonable range for this parameter is 100-2000.
    
    n_jobs : int (optional, default -1).
        The number of jobs to use in the k-nearest-neighbors computation. Defaults to all available CPUs.

    verbose : bool (optional, default False).
        If True, print progress messages.
    
    """

    def __repr__(self):
        if (self.n is not None):
            msg = "CkNearestNeighbors() object with %i fitted samples" % (self.n)
        else:
            msg = "CkNearestNeighbors() object object without any fitted data."
        return msg

    def __init__(self,
                 n_neighbors=10,
                 delta=1.0,
                 metric='euclidean',
                 weighted=True,
                 include_self=False,
                 backend='nmslib',
                 M=15,
                 efC=50,
                 efS=50,
                 n_jobs=-1,
                 verbose=False):
        self.n_neighbors = n_neighbors
        self.delta = delta
        self.metric = metric
        self.include_self = include_self
        self.weighted = weighted
        self.backend = backend
        self.M = M
        self.efC = efC
        self.efS = efS
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.A = None

    def fit(self, X, **kwargs):
        """
        Compute the CkNN graph.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data matrix.

        kwargs : dict
            Additional keyword arguments passed to the k-nearest-neighbors backend.

        Returns
        -------
        Adjacency matrix of the CkNN graph. Weighted or unweighted, depending on the `weighted` parameter.

        """
        cknn_graph(X, n_neighbors=self.n_neighbors,
                    delta=self.delta,
                    metric=self.metric,
                    weighted=self.weighted,
                    include_self=self.include_self,
                    backend=self.backend,
                    n_jobs=self.n_jobs,
                    M=self.M,
                    efC=self.efC,
                    efS=self.efS,
                    verbose=self.verbose,
                    *kwargs)

        return self.A

    def transform(self, X):
        """
        Transform the data into the CkNN graph. Here for consistency with the scikit-learn API.
        Will merely return the CkNN graph stored at CkNN.A.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data matrix. 

        Returns
        -------
        Adjacency matrix of the CkNN graph. Weighted or unweighted, depending on the `weighted` parameter.
        """
        return self.A

    def fit_transform(self, X, **kwargs):
        """
        Fit the CkNN graph and transform the data. Here for consistency with the scikit-learn API.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data matrix. 

        Returns
        -------
        Adjacency matrix of the CkNN graph. Weighted or unweighted, depending on the `weighted` parameter.
        """
        self.fit(X, *kwargs)
        return self.A
