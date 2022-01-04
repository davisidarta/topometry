#####################################
# NMSLIB approximate-nearest neighbors sklearn wrapper
# NMSLIB: https://github.com/nmslib/nmslib
# Wrapper author: Davi Sidarta-Oliveira
# School of Medical Sciences,University of Campinas,Brazil
# contact: davisidarta [at] fcm [dot] unicamp [dot] br
######################################

import time
import numpy as np
from scipy.sparse import csr_matrix, find, issparse
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


def kNN(X, n_neighbors=5, metric='euclidean', n_jobs=1, backend='nmslib', M=10, p=11/16, efC=50, efS=50, return_instance=False, verbose=False):
    """
    General class for computing k-nearest-neighbors graphs using NMSlib, HNSWlib or scikit-learn.

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

    metric : str (optional, default 'cosine').
        Accepted NMSLIB metrics. Defaults to 'cosine'. Accepted metrics include:
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
        number of threads to be used in computation. Defaults to 1. The algorithm is highly
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

    Returns
    -------

    A scipy.sparse.csr_matrix containing k-nearest-neighbor distances.

    """
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
    elif backend == 'hnwslib':
        nbrs = HNSWlibTransformer(n_neighbors=n_neighbors,
                                       metric=metric,
                                       n_jobs=n_jobs,
                                       M=M,
                                       efC=efC,
                                       efS=efS,
                                       verbose=False).fit(X)
        knn = nbrs.transform(X)
    else:
        # Construct a k-nearest-neighbors graph
        nbrs = NearestNeighbors(n_neighbors=int(n_neighbors), metric=metric, n_jobs=n_jobs).fit(X)
        knn = nbrs.kneighbors_graph(X, mode='distance')

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
    Calling 'nn <- NMSlibTransformer()' initializes the class with
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

    method : str (optional, default 'hsnw').
        approximate-neighbor search method. Available methods include:
                -'hnsw' : a Hierarchical Navigable Small World Graph.
                -'sw-graph' : a Small World Graph.
                -'vp-tree' : a Vantage-Point tree with a pruning rule adaptable to non-metric distances.
                -'napp' : a Neighborhood APProximation index.
                -'simple_invindx' : a vanilla, uncompressed, inverted index, which has no parameters.
                -'brute_force' : a brute-force search, which has no parameters.
        'hnsw' is usually the fastest method, followed by 'sw-graph' and 'vp-tree'.

    n_jobs : int (optional, default 1).
        number of threads to be used in computation. Defaults to 1. The algorithm is highly
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
                 n_jobs=10,
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



class HNSWlibTransformer(BaseEstimator):
    """
    Wrapper for using HNSWlib as sklearn's KNeighborsTransformer. This implements
    an escalable approximate k-nearest-neighbors graph on spaces defined by hnwslib.
    Read more about hnwslib  at
    https://github.com/nmslib/hnswlib
    Calling 'nn <- HNSWlibTransformer()' initializes the class with
     neighbour search parameters.
    Parameters
    ----------
    n_neighbors: int (optional, default 30)
        number of nearest-neighbors to look for. In practice,
        this should be considered the average neighborhood size and thus vary depending
        on your number of features, samples and data intrinsic dimensionality. Reasonable values
        range from 5 to 100. Smaller values tend to lead to increased graph structure
        resolution, but users should beware that a too low value may render granulated and vaguely
        defined neighborhoods that arise as an artifact of downsampling. Defaults to 30. Larger
        values can slightly increase computational time.
    metric: str (optional, default 'cosine')
        accepted NMSLIB metrics. Defaults to 'cosine'. Accepted metrics include:
        -'sqeuclidean' and 'euclidean'
        -'inner_product'
        -'cosine'
        For additional metrics, use the NMSLib backend.
    n_jobs: int (optional, default 1)
        number of threads to be used in computation. Defaults to 1. The algorithm is highly
        scalable to multi-threading.
    M: int (optional, default 30)
        defines the maximum number of neighbors in the zero and above-zero layers during HSNW
        (Hierarchical Navigable Small World Graph). However, the actual default maximum number
        of neighbors for the zero layer is 2*M.  A reasonable range for this parameter
        is 5-100. For more information on HSNW, please check https://arxiv.org/abs/1603.09320.
    efC: int (optional, default 100)
        A 'hnsw' parameter. Increasing this value improves the quality of a constructed graph
        and leads to higher accuracy of search. However this also leads to longer indexing times.
        A reasonable range for this parameter is 50-2000.
    efS: int (optional, default 100)
        A 'hnsw' parameter. Similarly to efC, increasing this value improves recall at the
        expense of longer retrieval time. A reasonable range for this parameter is 100-2000.

    Returns
    ---------
    Class for really fast approximate-nearest-neighbors search.

    Example
    -------------

    """

    def __init__(self,
                 n_neighbors=30,
                 metric='cosine',
                 n_jobs=10,
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