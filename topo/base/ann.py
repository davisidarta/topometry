#####################################
# Wrappers for approximate nearest neighbor search
# Author: Davi Sidarta-Oliveira
# School of Medical Sciences,University of Campinas,Brazil
# contact: davisidarta@fcm.unicamp.br
######################################

import time
import numpy as np
from warnings import warn
from scipy.sparse import csr_matrix, issparse
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.neighbors import NearestNeighbors
from joblib import cpu_count


def _is_installed(module):
    from importlib.util import find_spec
    return find_spec(module) is not None


def resolve_backend(backend):
    """
    Return the neighbor-search backend that will actually be used when `backend` is asked for:
    `backend` itself if its library is installed, otherwise the first available of 'hnswlib',
    'nmslib' and 'sklearn' (exact search, always available). Falling back emits a warning.
    """
    if backend == 'sklearn':
        return backend
    if backend not in ('hnswlib', 'nmslib'):
        warn("Neighbor-search backend '%s' is not supported. Using scikit-learn's exact search." % backend)
        return 'sklearn'
    other = 'nmslib' if backend == 'hnswlib' else 'hnswlib'
    if _is_installed(backend):
        return backend
    if _is_installed(other):
        warn("'%s' is not installed. Using '%s' for neighbor search." % (backend, other))
        return other
    warn("No approximate nearest-neighbor library ('hnswlib' or 'nmslib') is installed. "
         "Using scikit-learn's exact search.")
    return 'sklearn'


def kNN(X, Y=None,
        n_neighbors=5,
        metric='euclidean',
        n_jobs=-1,
        backend='hnswlib',
        low_memory=True,
        M=60,
        p=11/16,
        efC=200,
        efS=200,
        n_trees=50,
        return_instance=False,
        verbose=False, **kwargs):

    """
    General function for computing k-nearest-neighbors graphs using NMSlib, HNSWlib or scikit-learn.

    Parameters
    ----------
    X : np.ndarray or scipy.sparse.csr_matrix.
        Input data.

    n_neighbors : int (optional, default 30)
        number of nearest-neighbors to look for. In practice,
        this should be considered the average neighborhood size and thus vary depending
        on your number of features, samples and data intrinsic dimensionality. Reasonable values
        range from 5 to 100. Smaller values tend to lead to increased graph structure
        resolution, but users should beware that a too low value may render granulated and vaguely
        defined neighborhoods that arise as an artifact of downsampling. Defaults to 30. Larger
        values can slightly increase computational time.

    backend : str (optional, default 'hnswlib').
        Which backend to use for neighborhood search. Options are 'nmslib', 'hnswlib'
        and 'sklearn'. If the library of the requested backend is not installed, the first
        available of 'hnswlib', 'nmslib' and 'sklearn' is used instead, with a warning.

    metric : str (optional, default 'cosine').
        Accepted metrics. Defaults to 'cosine'. Accepted metrics include:
        -'sqeuclidean'
        -'euclidean'
        -'l1'
        -'lp' - requires setting the parameter `p` - equivalent to minkowski distance
        -'cosine'
        -'angular'
        -'negdotprod'
        -'levenshtein'
        -'hamming'
        -'jaccard'
        -'jansen-shan'

    n_jobs : int (optional, default 1).
        Number of threads to be used in computation. Defaults to 1. Set to -1 to use all available CPUs.
        Most algorithms are highly scalable to multithreading.

    M : int (optional, default 30).
        defines the maximum number of neighbors in the zero and above-zero layers during HSNW
        (Hierarchical Navigable Small World Graph). However, the actual default maximum number
        of neighbors for the zero layer is 2*M.  A reasonable range for this parameter
        is 5-100. For more information on HSNW, please check https://arxiv.org/abs/1603.09320.
        HSNW is implemented in python via NMSlib. Please check more about NMSlib at https://github.com/nmslib/nmslib.

    efC : int (optional, default 100).
        A 'hnsw' parameter. Increasing this value improves the quality of a constructed graph
        and leads to higher accuracy of search. However this also leads to longer indexing times.
        A reasonable range for this parameter is 50-2000.

    efS : int (optional, default 100).
        A 'hnsw' parameter. Similarly to efC, increasing this value improves recall at the
        expense of longer retrieval time. A reasonable range for this parameter is 100-2000.
    
    symmetrize : bool (optional, default True).
        Whether to symmetrize the output of approximate nearest neighbors search. The default is True
        and uses additive symmetrization, i.e. knn = ( knn + knn.T ) / 2 .

    **kwargs : dict (optional, default {}).
        Additional parameters to be passed to the backend approximate nearest-neighbors library.
        Use only parameters known to the desired backend library.
         
    Returns
    -------

    A scipy.sparse.csr_matrix containing k-nearest-neighbor distances. Each row holds the point
    itself (at distance 0) and its `n_neighbors` nearest neighbors, whichever the backend.

    """
    if n_jobs == -1:
        from joblib import cpu_count
        n_jobs = cpu_count()
    if Y is not None:
        if backend in ['nmslib', 'hnswlib']:
            warn("Only the 'sklearn' backend supports Y. Falling back to 'sklearn'...")
            backend = 'sklearn'
    backend = resolve_backend(backend)
    if backend == 'nmslib':
        # nmslib handles dense input natively (DataType.DENSE_VECTOR), so dense arrays are
        # passed through rather than sparsified; NMSlibTransformer picks the matching
        # space and data type from the input it is given.
        nbrs = NMSlibTransformer(n_neighbors=n_neighbors,
                                      metric=metric,
                                      p=p,
                                      method='hnsw',
                                      n_jobs=n_jobs,
                                      M=M,
                                      efC=efC,
                                      efS=efS,
                                      dense=isinstance(X, np.ndarray),
                                      verbose=verbose).fit(X)
    elif backend == 'hnswlib':
        if issparse(X):
            warn("hnswlib does not support sparse matrices. Converting to array...")
            X = X.toarray()
        nbrs = HNSWlibTransformer(n_neighbors=n_neighbors,
                                       metric=metric,
                                       n_jobs=n_jobs,
                                       M=M,
                                       efC=efC,
                                       efS=efS,
                                       verbose=False).fit(X)

    if backend != 'sklearn':
        if Y is None:
            knn = nbrs.transform(X)
        else:
            knn = nbrs.transform(Y)

    if backend == 'sklearn':
        # Construct a k-nearest-neighbors graph. Queried with the indexed data, each point comes
        # back as its own first neighbor, so one extra is asked for - as the other backends do -
        # to return `n_neighbors` neighbors besides the point itself.
        k = int(n_neighbors) if Y is not None else min(int(n_neighbors) + 1, X.shape[0])
        nbrs = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=n_jobs, **kwargs).fit(X)
        if Y is None:
            knn = nbrs.kneighbors_graph(X, mode='distance')
        else:
            knn = nbrs.kneighbors_graph(Y, mode='distance')
    if metric == 'cosine':
        # Every backend already returns cosine *distances* (d = 1 - cos), so the graph is
        # kept as it comes. Rounding can leave d marginally outside [0, 2].
        np.clip(knn.data, 0.0, 2.0, out=knn.data)
    if Y is None and isinstance(metric, str) and metric not in ('negdotprod', 'inner_product'):
        # A point is at distance zero from itself. Backends return that distance as rounding
        # noise (exactly 0 for some rows, ~1e-7 for others), which the kernels built on this
        # graph would turn into a self-loop of weight ~1 for an arbitrary subset of points.
        rows = np.repeat(np.arange(knn.shape[0]), np.diff(knn.indptr))
        knn.data[knn.indices == rows] = 0.0
    if return_instance:
        return nbrs, knn
    else:
        return knn


def _recall_against_exact_search(data, ann_indices, k, metric, verbose=False):
    """Mean fraction of each point's `k` exact nearest neighbors found by an approximate search."""
    start = time.time()
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric, algorithm='brute').fit(data)
    exact = nbrs.kneighbors(data, return_distance=False)
    end = time.time()
    if verbose:
        print('brute-force gold-standart kNN time total=%f (sec), per query=%f (sec)' %
              (end - start, float(end - start) / data.shape[0]))
    return float(np.mean([len(set(exact[i]).intersection(ann_indices[i])) / k for i in range(data.shape[0])]))


class NMSlibTransformer(BaseEstimator, TransformerMixin):
    """
    Wrapper for using nmslib as sklearn's KNeighborsTransformer. This implements
    an escalable approximate k-nearest-neighbors graph on spaces defined by nmslib.
    Read more about nmslib and its various available metrics at
    https://github.com/nmslib/nmslib.
    Calling 'nn <- NMSlibTransformer()' initializes the class with default
     neighbour search parameters.

    Parameters
    ----------
    n_neighbors : int (optional, default 30)
        number of nearest-neighbors to look for. In practice,
        this should be considered the average neighborhood size and thus vary depending
        on your number of features, samples and data intrinsic dimensionality. Reasonable values
        range from 5 to 100. Smaller values tend to lead to increased graph structure
        resolution, but users should beware that a too low value may render granulated and vaguely
        defined neighborhoods that arise as an artifact of downsampling. Defaults to 30. Larger
        values can slightly increase computational time.

    metric : str (optional, default 'cosine').
        Accepted NMSLIB metrics. Defaults to 'cosine'. Accepted metrics include:
        * 'sqeuclidean'
        * 'euclidean'
        * 'l1'
        * 'lp' - requires setting the parameter `p` - equivalent to minkowski distance
        * 'cosine'
        * 'angular'
        * 'negdotprod'
        * 'levenshtein'
        * 'hamming'
        * 'jaccard'
        * 'jansen-shan'

    method : str (optional, default 'hsnw').
        approximate-neighbor search method. Available methods include:
                -'hnsw' : a Hierarchical Navigable Small World Graph.
                -'sw-graph' : a Small World Graph.
                -'vp-tree' : a Vantage-Point tree with a pruning rule adaptable to non-metric distances.
                -'napp' : a Neighborhood APProximation index.
                -'simple_invindx' : a vanilla, uncompressed, inverted index, which has no parameters.
                -'brute_force' : a brute-force search, which has no parameters.
        'hnsw' is usually the fastest method, followed by 'sw-graph' and 'vp-tree'.

    n_jobs : int (optional, default -1).
        number of threads to be used in computation. Defaults to -1 (all but one). The algorithm is highly
        scalable to multi-threading.

    M : int (optional, default 30).
        defines the maximum number of neighbors in the zero and above-zero layers during HSNW
        (Hierarchical Navigable Small World Graph). However, the actual default maximum number
        of neighbors for the zero layer is 2*M.  A reasonable range for this parameter
        is 5-100. For more information on HSNW, please check https://arxiv.org/abs/1603.09320.
        HSNW is implemented in python via NMSlib. Please check more about NMSlib at https://github.com/nmslib/nmslib.

    efC : int (optional, default 100).
        A 'hnsw' parameter. Increasing this value improves the quality of a constructed graph
        and leads to higher accuracy of search. However this also leads to longer indexing times.
        A reasonable range for this parameter is 50-2000.

    efS : int (optional, default 100).
        A 'hnsw' parameter. Similarly to efC, increasing this value improves recall at the
        expense of longer retrieval time. A reasonable range for this parameter is 100-2000.

    dense : bool (optional, default False).
        Whether to force the algorithm to use dense data, such as np.ndarrays and pandas DataFrames.

    Returns
    ---------
    Class for really fast approximate-nearest-neighbors search.


    Example
    -------------
    import numpy as np
    from sklearn.datasets import load_digits
    from scipy.sparse import csr_matrix
    from topo.base.ann import NMSlibTransformer
    #
    # Load the MNIST digits data, convert to sparse for speed
    digits = load_digits()
    data = csr_matrix(digits)
    #
    # Start class with parameters
    nn = NMSlibTransformer()
    nn = nn.fit(data)
    #
    # Obtain kNN graph
    knn = nn.transform(data)
    #
    # Obtain kNN indices, distances and the kNN graph
    ind, dist, graph = nn.ind_dist_grad(data)
    #
    # Test for recall efficiency during approximate nearest neighbors search
    test = nn.test_efficiency(data)
    """

    def __init__(self,
                 n_neighbors=15,
                 metric='cosine',
                 method='hnsw',
                 n_jobs=-1,
                 p=None,
                 M=60,
                 efC=200,
                 efS=200,
                 dense=False,
                 verbose=False
                 ):

        self.n_neighbors = n_neighbors
        self.method = method
        self.metric = metric
        self.n_jobs = n_jobs
        self.p = p
        self.M = M
        self.efC = efC
        self.efS = efS
        self.space = self.metric
        self.dense = dense
        self.verbose = verbose

    def fit(self, data):
        try:
            import nmslib
        except ImportError:
            raise ImportError("NMSlib is required for this transformer. Please install it with `pip install nmslib`.")

        if self.n_jobs == -1:
            self.n_jobs = cpu_count()

        # NOTE: the space name must match the index's data type, so it is chosen
        # alongside it below (see `use_sparse_index`), not up front.
        sparse_spaces = {
            'sqeuclidean': 'l2_sparse',
            'euclidean': 'l2_sparse',
            'cosine': 'cosinesimil_sparse_fast',
            'lp': 'lp_sparse',
            'l1_sparse': 'l1_sparse',
            'linf_sparse': 'linf_sparse',
            'angular_sparse': 'angulardist_sparse_fast',
            'negdotprod_sparse': 'negdotprod_sparse_fast',
            'jaccard_sparse': 'jaccard_sparse',
            'bit_jaccard': 'bit_jaccard',
            'bit_hamming': 'bit_hamming',
            'levenshtein': 'leven',
            'normleven': 'normleven'
        }
        start = time.time()
        # see more metrics in the manual
        # https://github.com/nmslib/nmslib/tree/master/manual
        if self.metric == 'lp' and self.p < 1:
            print('Fractional L norms are slower to compute. Computations are faster for fractions'
                  ' of the form \'1/2ek\', where k is a small integer (i.g. 0.5, 0.25) ')
        if self.dense:
            # A dense index cannot consume sparse rows: each row's stored values would be
            # read as the whole vector, so rows with differing nnz give differing lengths.
            if issparse(data):
                if self.verbose:
                    print('Dense index requested for sparse input. Densifying...')
                data = data.toarray()
        else:
            if issparse(data) == True:
                if self.verbose:
                    print('Sparse input. Proceding without converting...')
                if isinstance(data, np.ndarray):
                    data = csr_matrix(data)
            if issparse(data) == False:
                if self.verbose:
                    print('Input data is ' + str(type(data)) + ' .Converting input to sparse...')
                import pandas as pd
                if isinstance(data, pd.DataFrame):
                    data = csr_matrix(data.values.T)

        index_time_params = {'M': self.M, 'indexThreadQty': self.n_jobs, 'efConstruction': self.efC, 'post': 2}

        use_sparse_index = issparse(data) and (not self.dense) and (not isinstance(data, np.ndarray))
        # Queries have to be handed over in the index's own data type (see `_match_index_type`).
        self.sparse_index_ = use_sparse_index
        if use_sparse_index:
            self.space = sparse_spaces[self.metric]
            if self.metric not in ['levenshtein', 'normleven', 'jansen-shan']:
                if self.metric == 'lp':
                    self.nmslib_ = nmslib.init(method=self.method,
                                               space=self.space,
                                               space_params={'p': self.p},
                                               data_type=nmslib.DataType.SPARSE_VECTOR)
                else:
                    self.nmslib_ = nmslib.init(method=self.method,
                                               space=self.space,
                                               data_type=nmslib.DataType.SPARSE_VECTOR)
            else:
                print('Metric ' + self.metric + 'available for string data only. Trying to compute distances...')
                data = data.toarray()
                self.sparse_index_ = None  # string index: queries are passed through untouched
                self.nmslib_ = nmslib.init(method=self.method,
                                           space=self.space,
                                           data_type=nmslib.DataType.OBJECT_AS_STRING)
        else:
            self.space = {
                'sqeuclidean': 'l2',
                'euclidean': 'l2',
                'cosine': 'cosinesimil',
                'lp': 'lp',
                'l1': 'l1',
                'linf': 'linf',
                'angular': 'angulardist',
                'negdotprod': 'negdotprod',
                'levenshtein': 'leven',
                'jaccard_sparse': 'jaccard_sparse',
                'bit_jaccard': 'bit_jaccard',
                'bit_hamming': 'bit_hamming',
                'jansen-shan': 'jsmetrfastapprox'
            }[self.metric]
            if self.metric == 'lp':
                self.nmslib_ = nmslib.init(method=self.method,
                                           space=self.space,
                                           space_params={'p': self.p},
                                           data_type=nmslib.DataType.DENSE_VECTOR)

            else:
                self.nmslib_ = nmslib.init(method=self.method,
                                           space=self.space,
                                           data_type=nmslib.DataType.DENSE_VECTOR)

        self.nmslib_.addDataPointBatch(data)
        self.nmslib_.createIndex(index_time_params)
        end = time.time()
        if self.verbose:
            print('Index-time parameters', 'M:', self.M, 'n_threads:', self.n_jobs, 'efConstruction:', self.efC,
                  'post:0')
            print('Indexing time = %f (sec)' % (end - start))

        return self

    def _match_index_type(self, data):
        """
        Return `data` in the data type of the fitted index. `fit` may densify its input
        (`dense=True`) to build a dense index, which then cannot consume the sparse rows
        the caller still holds - and a sparse index cannot consume dense ones.
        """
        sparse_index = getattr(self, 'sparse_index_', None)
        if sparse_index is None:
            return data
        if sparse_index:
            return data if issparse(data) else csr_matrix(data)
        return data.toarray() if issparse(data) else data

    def _to_metric_units(self, distances):
        """
        Return query distances in the units of `self.metric`. nmslib's optimized HNSW index,
        which is what a dense 'l2' space gets, reports *squared* L2 distances; every other
        'l2' index (sparse, or built with another method) reports plain L2.
        """
        squared = (self.method == 'hnsw') and (self.space == 'l2')
        if self.metric == 'euclidean' and squared:
            return np.sqrt(distances)
        if self.metric == 'sqeuclidean' and not squared:
            return distances ** 2
        return distances

    def transform(self, data):
        start = time.time()
        data = self._match_index_type(data)
        n_samples_transform = data.shape[0]
        query_time_params = {'efSearch': self.efS}
        if self.verbose:
            print('Query-time parameter efSearch:', self.efS)
        self.nmslib_.setQueryTimeParams(query_time_params)

        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        k = self.n_neighbors + 1

        results = self.nmslib_.knnQueryBatch(data, k=k,
                                             num_threads=self.n_jobs)

        indices, distances = zip(*results)
        indices, distances = np.vstack(indices), np.vstack(distances)

        query_qty = data.shape[0]

        distances = self._to_metric_units(distances)

        indptr = np.arange(0, n_samples_transform * k + 1, k)
        kneighbors_graph = csr_matrix((distances.ravel(), indices.ravel(),
                                       indptr), shape=(n_samples_transform,
                                                       n_samples_transform))
        end = time.time()
        if self.verbose:
            print('Search time =%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' %
                  (end - start, float(end - start) / query_qty, self.n_jobs * float(end - start) / query_qty))

        return kneighbors_graph

    def ind_dist_grad(self, data, return_grad=False, return_graph=True):
        """
        Query the index and return neighbor indices and distances, and optionally the
        neighborhood graph.

        `return_grad` is kept for backwards compatibility only. Distance gradients were
        never computed from the data (they were derived from the indices), so asking for
        them now raises instead of returning meaningless values.
        """
        if return_grad:
            raise NotImplementedError('Distance gradients are not available. Call with `return_grad=False`.')
        start = time.time()
        data = self._match_index_type(data)
        n_samples_transform = data.shape[0]
        query_time_params = {'efSearch': self.efS}
        if self.verbose:
            print('Query-time parameter efSearch:', self.efS)
        self.nmslib_.setQueryTimeParams(query_time_params)
        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        k = self.n_neighbors + 1
        results = self.nmslib_.knnQueryBatch(data, k=k,
                                             num_threads=self.n_jobs)
        indices, distances = zip(*results)
        indices, distances = np.vstack(indices), np.vstack(distances)

        query_qty = data.shape[0]

        distances = self._to_metric_units(distances)

        end = time.time()

        if self.verbose:
            print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' %
                  (end - start, float(end - start) / query_qty, self.n_jobs * float(end - start) / query_qty))

        if return_graph:
            indptr = np.arange(0, n_samples_transform * k + 1, k)
            kneighbors_graph = csr_matrix((distances.ravel(), indices.ravel(),
                                           indptr), shape=(n_samples_transform,
                                                           n_samples_transform))
            return indices, distances, kneighbors_graph
        return indices, distances

    def test_efficiency(self, data, data_use=0.1):
        """
        Print and return the recall of the approximate search against scikit-learn's exact one.
        `data_use` is unused and kept for backwards compatibility.
        """
        query_qty = data.shape[0]
        query_time_params = {'efSearch': self.efS}
        if self.verbose:
            print('Setting query-time parameters', query_time_params)
        self.nmslib_.setQueryTimeParams(query_time_params)

        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        k = self.n_neighbors + 1
        start = time.time()
        ann_results = self.nmslib_.knnQueryBatch(self._match_index_type(data), k=k,
                                                 num_threads=self.n_jobs)
        end = time.time()
        if self.verbose:
            print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' %
                  (end - start, float(end - start) / query_qty, self.n_jobs * float(end - start) / query_qty))

        recall = _recall_against_exact_search(data, [res[0] for res in ann_results], k, self.metric,
                                              verbose=self.verbose)
        print('kNN recall %f' % recall)
        return recall

    def update_search(self, n_neighbors):
        """
        Updates number of neighbors for kNN distance computation.
        Parameters
        -----------
        n_neighbors: New number of neighbors to look for.

        """
        self.n_neighbors = n_neighbors
        return print('Updated neighbor search.')

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def grid_search(X, n_neighbors=15, metric='euclidean', nmslib_params=None,
                hnswlib_params=None, n_jobs=-1, verbose=False):
    """Evaluate approximate kNN graph quality for NMSlib and HNSWlib.

    Parameters
    ----------
    X : array-like or sparse matrix
        Input data used to build the neighborhood graph.
    n_neighbors : int, optional (default=15)
        Number of neighbors to retrieve.
    metric : str, optional (default='euclidean')
        Distance metric used for neighbor search.
    nmslib_params : dict, optional
        Parameter grid for :class:`NMSlibTransformer`.
    hnswlib_params : dict, optional
        Parameter grid for :class:`HNSWlibTransformer`.
    n_jobs : int, optional (default=-1)
        Number of parallel jobs for the transformers.
    verbose : bool, optional (default=False)
        If True, print recall and timing for each parameter combination.

    Returns
    -------
    results : dict
        Mapping of backend names to lists of dictionaries containing
        parameter settings, recall and execution time.
    """
    from sklearn.model_selection import ParameterGrid

    # Compute ground-truth neighbors using exact search
    gt = NearestNeighbors(n_neighbors=n_neighbors, metric=metric,
                          algorithm='brute').fit(X)
    true_ind = gt.kneighbors(X, return_distance=False)

    results = {}

    # Evaluate NMSlibTransformer
    grids = ParameterGrid(nmslib_params) if nmslib_params else [{}]
    results['nmslib'] = []
    for params in grids:
        model = NMSlibTransformer(n_neighbors=n_neighbors, metric=metric,
                                  n_jobs=n_jobs, **params)
        start = time.time()
        model.fit(X)
        ind, _ = model.ind_dist_grad(X, return_grad=False, return_graph=False)
        elapsed = time.time() - start
        ind = ind[:, 1:]
        recall = np.mean([
            np.intersect1d(true_ind[i], ind[i]).size / n_neighbors
            for i in range(X.shape[0])
        ])
        results['nmslib'].append({'params': params, 'recall': recall, 'time': elapsed})

    # Evaluate HNSWlibTransformer
    grids = ParameterGrid(hnswlib_params) if hnswlib_params else [{}]
    results['hnswlib'] = []
    for params in grids:
        model = HNSWlibTransformer(n_neighbors=n_neighbors, metric=metric,
                                   n_jobs=n_jobs, **params)
        start = time.time()
        model.fit(X)
        ind, _ = model.ind_dist_grad(X, return_grad=False, return_graph=False)
        elapsed = time.time() - start
        ind = ind[:, 1:]
        recall = np.mean([
            np.intersect1d(true_ind[i], ind[i]).size / n_neighbors
            for i in range(X.shape[0])
        ])
        results['hnswlib'].append({'params': params, 'recall': recall, 'time': elapsed})

    if verbose:
        for backend, res in results.items():
            for r in res:
                print(f"{backend}: params={r['params']}, recall={r['recall']:.3f}, time={r['time']:.3f}s")

    return results


class HNSWlibTransformer(TransformerMixin, BaseEstimator):
    """
    Wrapper for using HNSWlib as sklearn's KNeighborsTransformer. This implements
    an escalable approximate k-nearest-neighbors graph on spaces defined by hnwslib.
    Read more about hnwslib  at
    https://github.com/nmslib/hnswlib
    Calling 'nn <- HNSWlibTransformer()' initializes the class with
     neighbour search parameters.
    Parameters
    ----------
    n_neighbors : int (optional, default 30)
        number of nearest-neighbors to look for. In practice,
        this should be considered the average neighborhood size and thus vary depending
        on your number of features, samples and data intrinsic dimensionality. Reasonable values
        range from 5 to 100. Smaller values tend to lead to increased graph structure
        resolution, but users should beware that a too low value may render granulated and vaguely
        defined neighborhoods that arise as an artifact of downsampling. Defaults to 30. Larger
        values can slightly increase computational time.

    metric : str (optional, default 'cosine')
        accepted NMSLIB metrics. Defaults to 'cosine'. Accepted metrics include:
        * 'sqeuclidean' and 'euclidean'
        * 'inner_product'
        * 'cosine'
        For additional metrics, use the NMSLib backend.

    n_jobs : int (optional, default -1)
        number of threads to be used in computation. Defaults to -1 (all but one). The algorithm is highly
        scalable to multi-threading.

    M : int (optional, default 30)
        defines the maximum number of neighbors in the zero and above-zero layers during HSNW
        (Hierarchical Navigable Small World Graph). However, the actual default maximum number
        of neighbors for the zero layer is 2*M.  A reasonable range for this parameter
        is 5-100. For more information on HSNW, please check https://arxiv.org/abs/1603.09320.

    efC : int (optional, default 100)
        A 'hnsw' parameter. Increasing this value improves the quality of a constructed graph
        and leads to higher accuracy of search. However this also leads to longer indexing times.
        A reasonable range for this parameter is 50-2000.

    efS : int (optional, default 100)
        A 'hnsw' parameter. Similarly to efC, increasing this value improves recall at the
        expense of longer retrieval time. A reasonable range for this parameter is 100-2000.

    Returns
    ---------
    Class for really fast approximate-nearest-neighbors search.


    """

    def __init__(self,
                 n_neighbors=30,
                 metric='cosine',
                 n_jobs=-1,
                 M=60,
                 efC=200,
                 efS=200,
                 verbose=False
                 ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_jobs = n_jobs
        self.M = M
        self.efC = efC
        self.efS = efS
        self.space = metric
        self.verbose = verbose
        self.N = None
        self.m = None
        self.p = None

    def fit(self, data):
        try:
            import hnswlib
        except ImportError:
            raise ImportError("HNSWlib is required for this transformer. Please install it with `pip install hnswlib`.")
        if self.n_jobs == -1:
            self.n_jobs = cpu_count()

        self.N = data.shape[0]
        self.m = data.shape[1]
        data = self._as_dense(data)
        start = time.time()
        data_labels = np.arange(self.N) # indices
        self.space = {
                'sqeuclidean': 'l2',
                'euclidean': 'l2',
                'cosine': 'cosine',
                'inner_product': 'ip',
            }[self.metric]
        self.p = hnswlib.Index(space=self.space, dim=self.m)
        self.p.init_index(max_elements=self.N, ef_construction=self.efC, M=self.M)
        self.p.set_num_threads(self.n_jobs)
        #
        self.p.add_items(data, data_labels)
        #
        index_time_params = {'M': self.M, 'indexThreadQty': self.n_jobs, 'efConstruction': self.efC}
        #
        end = time.time()
        if self.verbose:
            print('Index-time parameters', 'M:', self.M, 'n_threads:', self.n_jobs, 'efConstruction:', self.efC,
                  'post:0')
            print('Indexing time = %f (sec)' % (end - start))
        return self

    @staticmethod
    def _as_dense(data):
        """hnswlib only takes dense arrays."""
        if isinstance(data, np.ndarray):
            return data
        if issparse(data):
            return data.toarray()
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return data.to_numpy()
        raise TypeError('Data should be a np.ndarray, a scipy sparse matrix or a pd.DataFrame!')

    def transform(self, data):
        start = time.time()
        data = self._as_dense(data)
        if self.verbose:
            print('Query-time parameter efSearch:', self.efS)
        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        k = self.n_neighbors + 1
        indices, distances = self.p.knn_query(data, k=k)
        query_qty = self.N
        if self.metric == 'euclidean':
            distances = np.sqrt(distances)
        indptr = np.arange(0, self.N * k + 1, k)
        kneighbors_graph = csr_matrix((distances.ravel(), indices.ravel(),
                                       indptr), shape=(self.N,
                                                       self.N))
        end = time.time()
        if self.verbose:
            print('Search time =%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' %
                  (end - start, float(end - start) / query_qty, self.n_jobs * float(end - start) / query_qty))
        return kneighbors_graph

    def ind_dist_grad(self, data, return_grad=False, return_graph=True):
        """
        Query the index and return neighbor indices and distances, and optionally the
        neighborhood graph.

        `return_grad` is kept for backwards compatibility only. Distance gradients were
        never computed from the data (they were derived from the indices), so asking for
        them now raises instead of returning meaningless values.
        """
        if return_grad:
            raise NotImplementedError('Distance gradients are not available. Call with `return_grad=False`.')
        start = time.time()
        data = self._as_dense(data)
        if self.verbose:
            print('Query-time parameter efSearch:', self.efS)
        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        k = self.n_neighbors + 1
        indices, distances = self.p.knn_query(data, k=k)
        query_qty = self.N
        if self.metric == 'euclidean':
            distances = np.sqrt(distances)

        end = time.time()

        if self.verbose:
            print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' %
                  (end - start, float(end - start) / query_qty, self.n_jobs * float(end - start) / query_qty))

        if return_graph:
            indptr = np.arange(0, self.N * k + 1, k)
            kneighbors_graph = csr_matrix((distances.ravel(), indices.ravel(),
                                           indptr), shape=(self.N,
                                                           self.N))
            return indices, distances, kneighbors_graph
        return indices, distances

    def test_efficiency(self, data, percent_use=0.1):
        """
        Print and return the recall of the approximate search against scikit-learn's exact one.
        `percent_use` is unused and kept for backwards compatibility.
        """
        data = self._as_dense(data)
        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        k = self.n_neighbors + 1
        indices, _ = self.p.knn_query(data, k=k)
        recall = _recall_against_exact_search(data, indices, k, self.metric, verbose=self.verbose)
        print('kNN recall %f' % recall)
        return recall

    def update_search(self, n_neighbors):
        """
        Updates number of neighbors for kNN distance computation.
        Parameters
        -----------
        n_neighbors: New number of neighbors to look for.

        """
        self.n_neighbors = n_neighbors
        return print('Updated neighbor search.')

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)



