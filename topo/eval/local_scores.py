import numpy as np
from sklearn.utils import check_random_state
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, kendalltau
from scipy.sparse import csr_matrix, csgraph
from topo.utils._utils import get_landmark_indices
from topo.base.ann import kNN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

def geodesic_distance(adjacency, method='D', unweighted=False, directed=False, indices=None, n_jobs=-1, random_state=None):
    """
    Subsets the geodesic distance matrix to only include distances up to the k-th
    nearest neighbor distance for each point.

    Parameters:
    -----------
    knn_dists: scipy.sparse.csr_matrix
        Precomputed k-nearest-neighbors distances matrix.
    geodesic_dists: scipy.sparse.csr_matrix
        Geodesic distances matrix.

    Returns:
    --------
    subset_geodesics: scipy.sparse.csr_matrix
        Subsetted geodesic distances matrix.
    """
    n_points = knn_dists.shape[0]
    # Compute the maximum distance to consider for each point
    max_dist = knn_dists.max(axis=1).toarray()
    # Subset the geodesic distances matrix
    subset_geodesics = lil_matrix(geodesic_dists.shape, dtype=np.float32)
    for i in range(n_points):
        mask = (geodesic_dists[i,:] <= max_dist[i,:]).flatten()
        subset_geodesics[i,mask] = geodesic_dists[i,mask]
    subset_geodesics = subset_geodesics.tocsr()
    return subset_geodesics


def geodesic_distance(A, method='D', unweighted=False, directed=False, indices=None, n_jobs=-1, subset_to_knn=True, random_state=None):
    """
    Compute the geodesic distance matrix from an adjacency (or an affinity) matrix.
    The default behavior is to subset the geodesic distance matrix to only include distances up
    to the k-th nearest neighbor distance for each point. This is to ensure we are only assessing
    the performance of the embedding on the local structure of the data.

    Parameters
    ----------
    A : array-like, shape (n_vertices, n_vertices)
        Adjacency or affinity matrix of a graph.

    method : string, optional, default: 'D'
        Method to compute the shortest path.
        - 'D': Dijkstra's algorithm.
        - 'FW': Floyd-Warshall algorithm.
        - 'B': Bellman-Ford algorithm.
        - 'J': Johnson algorithm.
        - 'F': Floyd algorithm.

    unweighted : bool, optional, default: False
        If True, the adjacency matrix is considered as unweighted.

    directed : bool, optional, default: True
        If True, the adjacency matrix is considered as directed.
    
    indices : array-like, shape (n_indices, ), optional, default: None
        Indices of the vertices to compute the geodesic distance matrix.

    n_jobs : int, optional, default: 1
        The number of parallel jobs to use during search.
    
    Returns
    -------
    geodesic_distance : array-like, shape (n_vertices, n_vertices)

    """
    if n_jobs == 1:
        G = csgraph.shortest_path(adjacency, method=method,
                          unweighted=unweighted, directed=directed, indices=None)
        if indices is not None:
            G = G.T[indices].T
        # guarantee symmetry
        G = (G + G.T) / 2
        # zero diagonal
        G[(np.arange(G.shape[0]), np.arange(G.shape[0]))] = 0
    else:
        import multiprocessing as mp
        from functools import partial
        if n_jobs == -1:
            from joblib import cpu_count
            n_jobs = cpu_count()
        if not isinstance(n_jobs, int):
            n_jobs = 1
        if method == 'FW':
            raise ValueError('The Floyd-Warshall algorithm cannot be used with parallel computations.')
        if indices is None:
            indices = np.arange(A.shape[0])
        elif np.issubdtype(type(indices), np.integer):
            indices = np.array([indices])
        n = len(indices)
        local_function = partial(csgraph.shortest_path,
                                adjacency, method, directed, False, unweighted, False)
        if n_jobs == 1 or n == 1:
            try:
                G = csgraph.shortest_path(adjacency, method, directed, False,
                                                unweighted, False, indices)
            except csgraph.NegativeCycleError:
                raise ValueError(
                    "The shortest path computation could not be completed because a negative cycle is present.")
        else:
            try:
                with mp.Pool(n_jobs) as pool:
                    G = np.array(pool.map(local_function, indices))
            except csgraph.NegativeCycleError:
                pool.terminate()
                raise ValueError(
                    "The shortest path computation could not be completed because a negative cycle is present.")
        if n == 1:
            G = G.ravel()
        # guarantee symmetry
        G = (G + G.T) / 2
        #
        G[np.where(G == 0)] = np.inf
        # zero diagonal
        G[(np.arange(G.shape[0]), np.arange(G.shape[0]))] = 0
    return G


def knn_spearman_r(data_graph, embedding_graph, path_method='D', subsample_idx=None, unweighted=False, n_jobs=1):
    # data_graph is a (N,N) similarity matrix from the reference high-dimensional data
    # embedding_graph is a (N,N) similarity matrix from the lower dimensional embedding
    geodesic_dist = geodesic_distance(
        data_graph, method=path_method, unweighted=unweighted, n_jobs=n_jobs)
    if subsample_idx is not None:
        geodesic_dist = geodesic_dist[subsample_idx, :][:, subsample_idx]
    embedded_dist = geodesic_distance(
        embedding_graph, method=path_method, unweighted=unweighted, n_jobs=n_jobs)
    if subsample_idx is not None:
        embedded_dist = embedded_dist[subsample_idx, :][:, subsample_idx]
    res, _ = spearmanr(squareform(geodesic_dist), squareform(embedded_dist))
    return res


def knn_kendall_tau(data_graph, embedding_graph, path_method='D', subsample_idx=None, unweighted=False, n_jobs=1):
    geodesic_dist = geodesic_distance(
        data_graph, method=path_method, unweighted=unweighted, n_jobs=n_jobs)
    if subsample_idx is not None:
        geodesic_dist = geodesic_dist[subsample_idx, :][:, subsample_idx]
    embedded_dist = geodesic_distance(
        embedding_graph, method=path_method, unweighted=unweighted, n_jobs=n_jobs)
    if subsample_idx is not None:
        embedded_dist = embedded_dist[subsample_idx, :][:, subsample_idx]
    res, _ = kendalltau(squareform(geodesic_dist), squareform(embedded_dist))
    return res


def geodesic_correlation(data, emb, landmarks=None,
                        landmark_method='random',
                        metric='euclidean',
                        n_neighbors=3, n_jobs=-1,
                        cor_method='spearman', random_state=None, return_graphs=False , verbose=False, **kwargs):
    if random_state is None:
            random_state = np.random.RandomState()
    elif isinstance(random_state, np.random.RandomState):
        pass
    elif isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)
    else:
        print('RandomState error! No random state was defined!')
    if isinstance(data, csr_matrix):
        if data.shape[0] == data.shape[1]:
            DATA_IS_GRAPH = True
        else:
            DATA_IS_GRAPH = False
    else:
        if np.shape(data)[0] == np.shape(data)[1]:
            DATA_IS_GRAPH = True
        else:
            DATA_IS_GRAPH = False
    if isinstance(emb, csr_matrix):
        if emb.shape[0] == emb.shape[1]:
            EMB_IS_GRAPH = True
        else:
            EMB_IS_GRAPH = False
    else:
        if np.shape(emb)[0] == np.shape(emb)[1]:
            EMB_IS_GRAPH = True
        else:
            EMB_IS_GRAPH = False
    if not DATA_IS_GRAPH:
        data_graph = kNN(data, n_neighbors=n_neighbors,
                        metric=metric,
                        n_jobs=n_jobs,
                        return_instance=False,
                        verbose=False, **kwargs)
    else:
        data_graph = data.copy()
    if not EMB_IS_GRAPH:
        emb_graph = kNN(emb, n_neighbors=n_neighbors,
                        metric=metric,
                        n_jobs=n_jobs,
                        return_instance=False,
                        verbose=False, **kwargs)
    else:
        emb_graph = emb.copy()
    # Define landmarks if applicable
    if landmarks is not None:
        if isinstance(landmarks, int):
            landmark_indices = get_landmark_indices(
                data_graph, n_landmarks=landmarks, method=landmark_method, random_state=random_state)
            if landmark_indices.shape[0] == data.shape[0]:
                landmark_indices = None
        elif isinstance(landmarks, np.ndarray):
            landmark_indices = landmarks
        else:
            raise ValueError(
                '\'landmarks\' must be either an integer or a numpy array.')
    if landmarks is not None:
        data_graph = data_graph[landmark_indices, :][:, landmark_indices]
        emb_graph = emb_graph[landmark_indices, :][:, landmark_indices]
    base_geodesics = squareform(geodesic_distance(
        data_graph, directed=False, n_jobs=n_jobs))
    embedding_geodesics = squareform(geodesic_distance(
        emb_graph, directed=False, n_jobs=n_jobs))
    if cor_method == 'spearman':
        if verbose:
            print('Computing Spearman R...')
        results = spearmanr(
            base_geodesics, embedding_geodesics).correlation
    else:
        if verbose:
            print('Computing Kendall Tau for eigenbasis...')
        results = kendalltau(
            base_geodesics, embedding_geodesics).correlation
    if return_graphs:
        return results, data_graph, emb_graph
    return results

def trustworthiness(X, X_embedded, n_neighbors=3, metric='euclidean', X_is_distance=False, n_jobs=-1, backend='hnswlib', **kwargs):
    """
    Expresses to what extent the local structure is retained.
    Adapted from scikit-learn for increased computational efficiency.


    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
        If the metric is 'precomputed' X must be a square distance
        matrix. Otherwise it contains a sample per row.

    X_embedded : ndarray of shape (n_samples, n_components)
        Embedding of the training data in low-dimensional space.

    n_neighbors : int, default=5
        Number of neighbors k that will be considered.

    X_is_distance : bool, default=False
        Whether X is a distance matrix. If metric is 'precomputed', X may be a
        distance matrix, in which case X_is_distance must be True.

    metric : str or callable, default='euclidean'
        Which metric to use for computing pairwise distances between samples
        from the original input space. 


        
    Returns
    -------
    trustworthiness : float
        Trustworthiness of the low-dimensional embedding.
    """

    dist_X = pairwise_distances(X, metric=metric, n_jobs=n_jobs)
    if X_is_distance:
        dist_X = dist_X.copy()
    # we set the diagonal to np.inf to exclude the points themselves from
    # their own neighborhood
    np.fill_diagonal(dist_X, np.inf)
    ind_X = np.argsort(dist_X, axis=1)
    # `ind_X[i]` is the index of sorted distances between i and other samples
    
    if backend == 'sklearn':
        ind_X_embedded = NearestNeighbors(metric=metric, n_neighbors=n_neighbors, n_jobs=n_jobs).fit(
                X_embedded).kneighbors(return_distance=False)

    if backend == 'hnswlib':
        from topo.base.ann import HNSWlibTransformer
        nbrs_transformer = HNSWlibTransformer(metric=metric,
                                               n_neighbors=n_neighbors,
                                                 n_jobs=n_jobs, **kwargs).fit(X_embedded)
        ind_X_embedded, _ = nbrs_transformer.p.knn_query(X_embedded, k=n_neighbors+1)
        ind_X_embedded = ind_X_embedded[:, 1:]

    # We build an inverted index of neighbors in the input space: For sample i,
    # we define `inverted_index[i]` as the inverted index of sorted distances:
    # inverted_index[i][ind_X[i]] = np.arange(1, n_sample + 1)
    n_samples = X.shape[0]
    inverted_index = np.zeros((n_samples, n_samples), dtype=int)
    ordered_indices = np.arange(n_samples + 1)
    inverted_index[ordered_indices[:-1, np.newaxis],
                   ind_X] = ordered_indices[1:]
    ranks = inverted_index[ordered_indices[:-1, np.newaxis],
                           ind_X_embedded] - n_neighbors
    t = np.sum(ranks[ranks > 0])
    t = 1.0 - t * (2.0 / (n_samples * n_neighbors *
                          (2.0 * n_samples - 3.0 * n_neighbors - 1.0)))
    return t


