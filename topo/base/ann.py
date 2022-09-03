#####################################
# Wrappers for approximate nearest neighbor search
# Author: Davi Sidarta-Oliveira
# School of Medical Sciences,University of Campinas,Brazil
# contact: davisidarta@fcm.unicamp.br
######################################

import time
import numpy as np
from scipy.sparse import csr_matrix, find, issparse
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from joblib import cpu_count


def kNN(X,
        n_neighbors=5,
        metric='euclidean',
        n_jobs=-1,
        backend='nmslib',
        low_memory=True,
        symmetrize=True,
        M=15,
        p=11/16,
        efC=50,
        efS=50,
        n_trees=50,
        return_instance=False,
        verbose=False, **kwargs):

    """
    General function for computing k-nearest-neighbors graphs using NMSlib, HNSWlib, PyNNDescent, ANNOY, FAISS or scikit-learn.

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

    backend : str (optional, default 'nmslib').
        Which backend to use for neighborhood search. Options are 'nmslib', 'hnswlib', 
        'pynndescent','annoy', 'faiss' and 'sklearn'.

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

    A scipy.sparse.csr_matrix containing k-nearest-neighbor distances.

    """
    if n_jobs == -1:
        from joblib import cpu_count
        n_jobs = cpu_count()
    if backend == 'nmslib':
        # Construct an approximate k-nearest-neighbors graph
        nbrs = NMSlibTransformer(n_neighbors=n_neighbors,
                                      metric=metric,
                                      p=p,
                                      method='hnsw',
                                      n_jobs=n_jobs,
                                      M=M,
                                      efC=efC,
                                      efS=efS,
                                      verbose=verbose).fit(X)
        knn = nbrs.transform(X)

    elif backend == 'hnswlib':
        nbrs = HNSWlibTransformer(n_neighbors=n_neighbors,
                                       metric=metric,
                                       n_jobs=n_jobs,
                                       M=M,
                                       efC=efC,
                                       efS=efS,
                                       verbose=False).fit(X)
        knn = nbrs.transform(X)

    elif backend == 'pynndescent':
        try:
            from pynndescent import PyNNDescentTransformer
        except ImportError:
            return(print("PyNNDescent is required to use `pynndescent` as a kNN backend. Please install it with `pip install pynndescent`. "))
        if metric == 'lp':
            metric = 'minkowski'
            metric_kwds={'p':p}
            nbrs = PyNNDescentTransformer(metric=metric,
                                          n_neighbors=n_neighbors,
                                          n_jobs=n_jobs,
                                          low_memory=low_memory,
                                          metric_kwds=metric_kwds,
                                          verbose=verbose, **kwargs).fit(X)
        else:
            nbrs = PyNNDescentTransformer(metric=metric,
                                          n_neighbors=n_neighbors,
                                          n_jobs=n_jobs,
                                          low_memory=low_memory,
                                          verbose=verbose, **kwargs).fit(X)
        knn = nbrs.transform(X)

    elif backend == 'annoy':
        nbrs = AnnoyTransformer(metric=metric,
                                n_neighbors=n_neighbors,
                                n_jobs=n_jobs,
                                n_trees=n_trees, **kwargs).fit(X)
        knn = nbrs.transform(X)


    elif backend == 'faiss':
        try:
            import faiss
        except ImportError:
            return(print("FAISS is required for using `faiss` as a kNN backend. Please install it with `pip install faiss`. "))

        nbrs = FAISSTransformer(metric=metric,
                                n_neighbors=n_neighbors,
                                n_jobs=n_jobs, **kwargs).fit(X)
        knn = nbrs.transform(X)

    else:
        backend = 'sklearn'
        if verbose:
            print('Falling back to sklearn nearest-neighbors!')

    if backend == 'sklearn':
        # Construct a k-nearest-neighbors graph
        nbrs = NearestNeighbors(n_neighbors=int(n_neighbors), metric=metric, n_jobs=n_jobs, **kwargs).fit(X)
        knn = nbrs.kneighbors_graph(X, mode='distance')

    if symmetrize:
        knn = ( knn + knn.T ) / 2
        if metric in ['angular', 'cosine']:
            # distances must be monotonically decreasing, needs to be inverted with angular metrics
            # otherwise, we'll have a similarity metric, not a distance metric
            knn.data = 1 - knn.data 
        knn[(np.arange(knn.shape[0]), np.arange(knn.shape[0]))] = 0
        knn.data = np.where(np.isnan(knn.data), 0, knn.data)
    if return_instance:
        return nbrs, knn
    else:
        return knn


class NMSlibTransformer(TransformerMixin, BaseEstimator):
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
    # Obtain kNN indices, distances and distance gradient
    ind, dist, grad = nn.ind_dist_grad(data)
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
                 M=15,
                 efC=50,
                 efS=50,
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
            return(print("MNMSlib is required for this function. Please install it with `pip install nmslib`. "))

        if self.n_jobs == -1:
            self.n_jobs = cpu_count()

        self.space = {
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
        }[self.metric]
        start = time.time()
        # see more metrics in the manual
        # https://github.com/nmslib/nmslib/tree/master/manual
        if self.metric == 'lp' and self.p < 1:
            print('Fractional L norms are slower to compute. Computations are faster for fractions'
                  ' of the form \'1/2ek\', where k is a small integer (i.g. 0.5, 0.25) ')
        if self.dense:
            self.nmslib_ = nmslib.init(method=self.method,
                                       space=self.space,
                                       data_type=nmslib.DataType.DENSE_VECTOR)

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

        if issparse(data) and (not self.dense) and (not isinstance(data, np.ndarray)):
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

    def transform(self, data):
        start = time.time()
        n_samples_transform = data.shape[0]
        query_time_params = {'efSearch': self.efS}
        if self.verbose:
            print('Query-time parameter efSearch:', self.efS)
        self.nmslib_.setQueryTimeParams(query_time_params)

        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        self.n_neighbors = self.n_neighbors + 1

        results = self.nmslib_.knnQueryBatch(data, k=self.n_neighbors,
                                             num_threads=self.n_jobs)

        indices, distances = zip(*results)
        indices, distances = np.vstack(indices), np.vstack(distances)

        query_qty = data.shape[0]

        if self.metric == 'sqeuclidean':
            distances **= 2

        indptr = np.arange(0, n_samples_transform * self.n_neighbors + 1,
                           self.n_neighbors)
        kneighbors_graph = csr_matrix((distances.ravel(), indices.ravel(),
                                       indptr), shape=(n_samples_transform,
                                                       n_samples_transform))
        end = time.time()
        if self.verbose:
            print('Search time =%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' %
                  (end - start, float(end - start) / query_qty, self.n_jobs * float(end - start) / query_qty))

        return kneighbors_graph

    def ind_dist_grad(self, data, return_grad=True, return_graph=True):
        start = time.time()
        n_samples_transform = data.shape[0]
        query_time_params = {'efSearch': self.efS}
        if self.verbose:
            print('Query-time parameter efSearch:', self.efS)
        self.nmslib_.setQueryTimeParams(query_time_params)
        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        self.n_neighbors = self.n_neighbors + 1
        results = self.nmslib_.knnQueryBatch(data, k=self.n_neighbors,
                                             num_threads=self.n_jobs)
        indices, distances = zip(*results)
        indices, distances = np.vstack(indices), np.vstack(distances)

        query_qty = data.shape[0]

        if self.metric == 'sqeuclidean':
            distances **= 2

        indptr = np.arange(0, n_samples_transform * self.n_neighbors + 1,
                           self.n_neighbors)
        kneighbors_graph = csr_matrix((distances.ravel(), indices.ravel(),
                                       indptr), shape=(n_samples_transform,
                                                       n_samples_transform))
        if return_grad:
            x, y, dists = find(kneighbors_graph)

            # Define gradients
            grad = []
            if self.metric not in ['sqeuclidean', 'euclidean', 'cosine', 'linf']:
                print('Gradient undefined for metric \'' + self.metric + '\'. Returning empty array.')

            if self.metric == 'cosine':
                norm_x = 0.0
                norm_y = 0.0
                for i in range(x.shape[0]):
                    norm_x += x[i] ** 2
                    norm_y += y[i] ** 2
                if norm_x == 0.0 and norm_y == 0.0:
                    grad = np.zeros(x.shape)
                elif norm_x == 0.0 or norm_y == 0.0:
                    grad = np.zeros(x.shape)
                else:
                    grad = -(x * dists - y * norm_x) / np.sqrt(norm_x ** 3 * norm_y)

            if self.metric == 'euclidean':
                grad = x - y / (1e-6 + np.sqrt(dists))

            if self.metric == 'sqeuclidean':
                grad = x - y / (1e-6 + dists)

            if self.metric == 'linf':
                result = 0.0
                max_i = 0
                for i in range(x.shape[0]):
                    v = np.abs(x[i] - y[i])
                    if v > result:
                        result = dists
                        max_i = i
                grad = np.zeros(x.shape)
                grad[max_i] = np.sign(x[max_i] - y[max_i])

        end = time.time()

        if self.verbose:
            print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' %
                  (end - start, float(end - start) / query_qty, self.n_jobs * float(end - start) / query_qty))

        if return_graph and return_grad:
            return indices, distances, grad, kneighbors_graph
        if return_graph and not return_grad:
            return indices, distances, kneighbors_graph
        if not return_graph and return_grad:
            return indices, distances, grad
        if not return_graph and not return_grad:
            return indices, distances

    def test_efficiency(self, data, data_use=0.1):
        """Test if NMSlibTransformer and KNeighborsTransformer give same results
        """
        self.data_use = data_use

        query_qty = data.shape[0]

        (dismiss, test) = train_test_split(data, test_size=self.data_use)
        query_time_params = {'efSearch': self.efS}
        if self.verbose:
            print('Setting query-time parameters', query_time_params)
        self.nmslib_.setQueryTimeParams(query_time_params)

        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        self.n_neighbors = self.n_neighbors + 1
        start = time.time()
        ann_results = self.nmslib_.knnQueryBatch(data, k=self.n_neighbors,
                                                 num_threads=self.n_jobs)
        end = time.time()
        if self.verbose:
            print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' %
                  (end - start, float(end - start) / query_qty, self.n_jobs * float(end - start) / query_qty))

        # Use sklearn for exact neighbor search
        start = time.time()
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors,
                                metric=self.metric,
                                algorithm='brute').fit(data)
        knn = nbrs.kneighbors(data)
        end = time.time()
        if self.verbose:
            print('brute-force gold-standart kNN time total=%f (sec), per query=%f (sec)' %
                  (end - start, float(end - start) / query_qty))

        recall = 0.0
        for i in range(0, query_qty):
            correct_set = set(knn[1][i])
            ret_set = set(ann_results[i][0])
            recall = recall + float(len(correct_set.intersection(ret_set))) / len(correct_set)
        recall = recall / query_qty
        print('kNN recall %f' % recall)

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
                 M=30,
                 efC=100,
                 efS=100,
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
            return(print("HNSWlib is required for this function. Please install it with `pip install hnswlib`. "))
        if self.n_jobs == -1:
            self.n_jobs = cpu_count()

        self.N = data.shape[0]
        self.m = data.shape[1]
        if not isinstance(data, np.ndarray):
            import pandas as pd
            if isinstance(data, csr_matrix):
                arr = np.zeros([self.N, self.m])
                data = data.toarray()
            if isinstance(data, pd.DataFrame):
                data = data.to_numpy()
            else:
                return print('Data should be a np.ndarray, sp.csr_matrix or pd.DataFrame!')
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

    def transform(self, data):
        start = time.time()
        if self.verbose:
            print('Query-time parameter efSearch:', self.efS)
        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        self.n_neighbors = self.n_neighbors + 1
        indices, distances = self.p.knn_query(data, k=self.n_neighbors)
        query_qty = self.N
        if self.metric == 'euclidean':
            distances = np.sqrt(distances)
        indptr = np.arange(0, self.N * self.n_neighbors + 1,
                           self.n_neighbors)
        kneighbors_graph = csr_matrix((distances.ravel(), indices.ravel(),
                                       indptr), shape=(self.N,
                                                       self.N))
        end = time.time()
        if self.verbose:
            print('Search time =%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' %
                  (end - start, float(end - start) / query_qty, self.n_jobs * float(end - start) / query_qty))
        return kneighbors_graph

    def ind_dist_grad(self, data, return_grad=True, return_graph=True):
        start = time.time()
        if self.verbose:
            print('Query-time parameter efSearch:', self.efS)
        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        self.n_neighbors = self.n_neighbors + 1
        indices, distances = self.p.knn_query(data, k=self.n_neighbors)
        query_qty = self.N
        if self.metric == 'euclidean':
            distances = np.sqrt(distances)
        indptr = np.arange(0, self.N * self.n_neighbors + 1,
                           self.n_neighbors)
        kneighbors_graph = csr_matrix((distances.ravel(), indices.ravel(),
                                       indptr), shape=(self.N,
                                                       self.N))
        if return_grad:
            x, y, dists = find(kneighbors_graph)

            # Define gradients
            grad = []
            if self.metric not in ['sqeuclidean', 'euclidean', 'cosine']:
                print('Gradient undefined for metric \'' + self.metric + '\'. Returning empty array.')

            if self.metric == 'cosine':
                norm_x = 0.0
                norm_y = 0.0
                for i in range(x.shape[0]):
                    norm_x += x[i] ** 2
                    norm_y += y[i] ** 2
                if norm_x == 0.0 and norm_y == 0.0:
                    grad = np.zeros(x.shape)
                elif norm_x == 0.0 or norm_y == 0.0:
                    grad = np.zeros(x.shape)
                else:
                    grad = -(x * dists - y * norm_x) / np.sqrt(norm_x ** 3 * norm_y)

            if self.metric == 'euclidean' or self.metric == 'sqeuclidean':
                grad = x - y / (1e-6 + dists)

        end = time.time()

        if self.verbose:
            print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' %
                  (end - start, float(end - start) / query_qty, self.n_jobs * float(end - start) / query_qty))

        if return_graph and return_grad:
            return indices, distances, grad, kneighbors_graph
        if return_graph and not return_grad:
            return indices, distances, kneighbors_graph
        if not return_graph and return_grad:
            return indices, distances, grad
        if not return_graph and not return_grad:
            return indices, distances

    def test_efficiency(self, data, percent_use=0.1):
        """Test if HNSWlibTransformer and KNeighborsTransformer give same results
        """
        self.data_use = percent_use

        query_qty = data.shape[0]

        (dismiss, test) = train_test_split(data, test_size=self.data_use)
        query_time_params = {'efSearch': self.efS}
        if self.verbose:
            print('Setting query-time parameters', query_time_params)
        self.nmslib_.setQueryTimeParams(query_time_params)

        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        self.n_neighbors = self.n_neighbors + 1
        start = time.time()
        ann_results = self.fit(data).transform(data)
        end = time.time()
        if self.verbose:
            print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' %
                  (end - start, float(end - start) / query_qty, self.n_jobs * float(end - start) / query_qty))

        # Use sklearn for exact neighbor search
        start = time.time()
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors,
                                metric=self.metric,
                                algorithm='brute').fit(data)
        knn = nbrs.kneighbors(data)
        end = time.time()
        if self.verbose:
            print('brute-force gold-standart kNN time total=%f (sec), per query=%f (sec)' %
                  (end - start, float(end - start) / query_qty))

        recall = 0.0
        for i in range(0, query_qty):
            correct_set = set(knn[1][i])
            ret_set = set(ann_results[i][0])
            recall = recall + float(len(correct_set.intersection(ret_set))) / len(correct_set)
        recall = recall / query_qty
        print('kNN recall %f' % recall)

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


class AnnoyTransformer(TransformerMixin, BaseEstimator):
    """
    Wrapper for using annoy.AnnoyIndex as sklearn's KNeighborsTransformer
    Read more about ANNOY at https://github.com/spotify/annoy

    Calling 'nn <- AnnoyTransformer()' initializes the class with default
     neighbour search parameters.

    Parameters
    ----------

    metric : str (optional, default 'euclidean').
        Accepted ANNOY metrics. Defaults to 'euclidean'. Accepted metrics include:
        * 'euclidean'
        * 'angular'
        * 'dot'
        * 'manhattan'
        * 'hamming'

    n_jobs : int (optional, default -1).
        Number of threads to be used in computation. Defaults to -1 (all but one).

    n_trees : int (optional, default 10).
        ANNOYS builds a forest of n_trees trees. More trees gives higher precision when querying,
        at a higher computational cost.

    """

    def __init__(self, n_neighbors=5, metric="euclidean", n_trees=10, n_jobs=-1):
        self.n_neighbors = n_neighbors
        self.n_trees = n_trees
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X):
        try:
            import annoy
        except ImportError:
            return(print("ANNOY is required to use `annoy` as a kNN backend. Please install it with `pip install annoy`. "))
        if self.n_jobs == -1:
            self.n_jobs = cpu_count()
        self.n_samples_fit_ = X.shape[0]
        self.annoy_ = annoy.AnnoyIndex(X.shape[1], metric=self.metric)
        for i, x in enumerate(X):
            self.annoy_.add_item(i, x.tolist())
        self.annoy_.build(self.n_trees, n_jobs=self.n_jobs)
        return self

    def transform(self, X):
        return self._transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X)._transform(X=None)

    def _transform(self, X):
        """As `transform`, but handles X is None for faster `fit_transform`."""

        n_samples_transform = self.n_samples_fit_ if X is None else X.shape[0]

        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        n_neighbors = self.n_neighbors + 1

        indices = np.empty((n_samples_transform, n_neighbors), dtype=int)
        distances = np.empty((n_samples_transform, n_neighbors))

        if X is None:
            for i in range(self.annoy_.get_n_items()):
                ind, dist = self.annoy_.get_nns_by_item(
                    i, n_neighbors, include_distances=True
                )

                indices[i], distances[i] = ind, dist
        else:
            for i, x in enumerate(X):
                indices[i], distances[i] = self.annoy_.get_nns_by_vector(
                    x.tolist(), n_neighbors, include_distances=True
                )

        indptr = np.arange(0, n_samples_transform * n_neighbors + 1, n_neighbors)
        kneighbors_graph = csr_matrix(
            (distances.ravel(), indices.ravel(), indptr),
            shape=(n_samples_transform, self.n_samples_fit_),
        )

        return kneighbors_graph


class FAISSTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        n_neighbors=5,
        *,
        metric="euclidean",
        index_key="",
        n_probe=128,
        n_jobs=-1,
        include_fwd=True,
        include_rev=False
    ):

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.index_key = index_key
        self.n_probe = n_probe
        self.n_jobs = n_jobs
        self.include_fwd = include_fwd
        self.include_rev = include_rev

    @property
    def _metric_info(self):
        try:
            import faiss
        except ImportError:
            return(print("FAISS is required for this function. Please install it with `pip install faiss`. "))

        L2_INFO = {"metric": faiss.METRIC_L2, "sqrt": True}
        METRIC_MAP = {
            "cosine": {
                "metric": faiss.METRIC_INNER_PRODUCT,
                "normalize": True,
                "negate": True,
            },
            "l1": {"metric": faiss.METRIC_L1},
            "cityblock": {"metric": faiss.METRIC_L1},
            "manhattan": {"metric": faiss.METRIC_L1},
            "l2": L2_INFO,
            "euclidean": L2_INFO,
            "sqeuclidean": {"metric": faiss.METRIC_L2},
            "canberra": {"metric": faiss.METRIC_Canberra},
            "braycurtis": {"metric": faiss.METRIC_BrayCurtis},
            "jensenshannon": {"metric": faiss.METRIC_JensenShannon},
        }
        return METRIC_MAP[self.metric]



    def mk_faiss_index(self, feats, inner_metric, index_key="", nprobe=128):
        try:
            import faiss
        except ImportError:
            return(print("FAISS is required for this function. Please install it with `pip install faiss`. "))
        import math
        size, dim = feats.shape
        if not index_key:
            if inner_metric == faiss.METRIC_INNER_PRODUCT:
                index = faiss.IndexFlatIP(dim)
            else:
                index = faiss.IndexFlatL2(dim)
        else:
            if index_key.find("HNSW") < 0:
                raise NotImplementedError(
                    "HNSW not implemented: returns distances insted of sims"
                )
            nlist = min(4096, 8 * round(math.sqrt(size)))
            if index_key == "IVF":
                quantizer = faiss.IndexFlatL2(dim)
                index = faiss.IndexIVFFlat(quantizer, dim, nlist, inner_metric)
            else:
                index = faiss.index_factory(dim, index_key, inner_metric)
            if index_key.find("Flat") < 0:
                assert not index.is_trained
            index.train(feats)
            index.nprobe = min(nprobe, nlist)
            assert index.is_trained
        index.add(feats)
        return index

    def fit(self, X, y=None):
        try:
            import faiss
        except ImportError:
            return (print("FAISS is required for this function. Please install it with `pip install faiss`. "))

        normalize = self._metric_info.get("normalize", False)
        X = self._validate_data(X, dtype=np.float32, copy=normalize)
        self.n_samples_fit_ = X.shape[0]
        if self.n_jobs == -1:
            n_jobs = cpu_count()
        else:
            n_jobs = self.n_jobs
        faiss.omp_set_num_threads(n_jobs)
        inner_metric = self._metric_info["metric"]
        if normalize:
            faiss.normalize_L2(X)
        self.faiss_ = self.mk_faiss_index(X, inner_metric, self.index_key, self.n_probe)
        return self

    def transform(self, X):
        try:
            import faiss
        except ImportError:
            return (print("FAISS is required for this function. Please install it with `pip install faiss`. "))
        normalize = self._metric_info.get("normalize", False)
        X = self._transform_checks(X, "faiss_", dtype=np.float32, copy=normalize)
        if normalize:
            faiss.normalize_L2(X)
        return self._transform(X)

    def _transform(self, X):
        try:
            import faiss
        except ImportError:
            return (print("FAISS is required for this function. Please install it with `pip install faiss`. "))
        n_samples_transform = self.n_samples_fit_ if X is None else X.shape[0]
        n_neighbors = self.n_neighbors + 1
        if X is None:
            sims, nbrs = self.faiss_.search(
                np.reshape(
                    faiss.vector_to_array(self.faiss_.xb),
                    (self.faiss_.ntotal, self.faiss_.d),
                ),
                k=n_neighbors,
            )
        else:
            sims, nbrs = self.faiss_.search(X, k=n_neighbors)
        dist_arr = np.array(sims, dtype=np.float32)
        if self._metric_info.get("sqrt", False):
            dist_arr = np.sqrt(dist_arr)
        if self._metric_info.get("negate", False):
            dist_arr = 1 - dist_arr
        del sims
        nbr_arr = np.array(nbrs, dtype=np.int32)
        del nbrs
        indptr = np.arange(0, n_samples_transform * n_neighbors + 1, n_neighbors)
        dist_arr = np.concatenate(
            [
                np.zeros(
                    (n_samples_transform, 1),
                    dtype=dist_arr.dtype
                ),
                dist_arr
            ], axis=1
        )
        nbr_arr = np.concatenate(
            [
                np.arange(n_samples_transform)[:, np.newaxis],
                nbr_arr
            ], axis=1
        )
        mat = csr_matrix(
            (dist_arr.ravel(), nbr_arr.ravel(), indptr),
            shape=(n_samples_transform, self.n_samples_fit_),
        )
        return postprocess_knn_csr(
            mat, include_fwd=self.include_fwd, include_rev=self.include_rev
        )

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y)._transform(X=None)


    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_estimators_pickle": "Cannot pickle FAISS index",
                "check_methods_subset_invariance": "Unable to reset FAISS internal RNG",
            },
            "requires_y": False,
            "preserves_dtype": [np.float32],
            # Could be made deterministic *if* we could reset FAISS's internal RNG
            "non_deterministic": True,
        }

def get_sparse_row(mat, idx):
    start_idx = mat.indptr[idx]
    end_idx = mat.indptr[idx + 1]
    return zip(mat.indices[start_idx:end_idx], mat.data[start_idx:end_idx])


def or_else_csrs(csr1, csr2):
    if csr1.shape != csr2.shape:
        raise ValueError("csr1 and csr2 must be the same shape")
    indptr = np.empty_like(csr1.indptr)
    indices = []
    data = []
    for row_idx in range(len(indptr) - 1):
        indptr[row_idx] = len(indices)
        csr1_it = iter(get_sparse_row(csr1, row_idx))
        csr2_it = iter(get_sparse_row(csr2, row_idx))
        cur_csr1 = next(csr1_it, None)
        cur_csr2 = next(csr2_it, None)
        while 1:
            if cur_csr1 is None and cur_csr2 is None:
                break
            elif cur_csr1 is None:
                cur_index, cur_datum = cur_csr2
            elif cur_csr2 is None:
                cur_index, cur_datum = cur_csr1
            elif cur_csr1[0] < cur_csr2[0]:
                cur_index, cur_datum = cur_csr1
                cur_csr1 = next(csr1_it, None)
            elif cur_csr2[0] < cur_csr1[0]:
                cur_index, cur_datum = cur_csr2
                cur_csr2 = next(csr2_it, None)
            else:
                cur_index, cur_datum = cur_csr1
                cur_csr1 = next(csr1_it, None)
                cur_csr2 = next(csr2_it, None)
            indices.append(cur_index)
            data.append(cur_datum)
    indptr[-1] = len(indices)
    return csr_matrix((data, indices, indptr), shape=csr1.shape)


def postprocess_knn_csr(knns, include_fwd=True, include_rev=False):
    if not include_fwd and not include_rev:
        raise ValueError("One of include_fwd or include_rev must be True")
    elif include_rev and not include_fwd:
        return knns.transpose(copy=False)
    elif not include_rev and include_fwd:
        return knns
    else:
        inv_knns = knns.transpose(copy=True)
        return or_else_csrs(knns, inv_knns)