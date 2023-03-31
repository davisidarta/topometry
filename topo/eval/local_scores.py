import numpy as np
from sklearn.utils import check_random_state
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, kendalltau
from scipy.sparse.csgraph import shortest_path, NegativeCycleError
from topo.utils._utils import get_landmark_indices
from topo.base.ann import kNN


def geodesic_distance(adjacency, method='D', unweighted=False, directed=False, indices=None, n_jobs=1, random_state=None):
    """
    Compute the geodesic distance matrix from an adjacency matrix.
    Parameters
    ----------
    adjacency : array-like, shape (n_vertices, n_vertices)
        Adjacency matrix of a graph.

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
        G = shortest_path(adjacency, method=method,
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
            raise ValueError(
                'The Floyd-Warshall algorithm cannot be used with parallel computations.')
        if indices is None:
            indices = np.arange(adjacency.shape[0])
        elif np.issubdtype(type(indices), np.integer):
            indices = np.array([indices])
        n = len(indices)
        local_function = partial(shortest_path,
                                adjacency, method, directed, False, unweighted, False)
        if n_jobs == 1 or n == 1:
            try:
                res = shortest_path(adjacency, method, directed, False,
                                                unweighted, False, indices)
            except NegativeCycleError:
                raise ValueError(
                    "The shortest path computation could not be completed because a negative cycle is present.")
        else:
            try:
                with mp.Pool(n_jobs) as pool:
                    G = np.array(pool.map(local_function, indices))
            except NegativeCycleError:
                pool.terminate()
                raise ValueError(
                    "The shortest path computation could not be completed because a negative cycle is present.")
        if n == 1:
            G = G.ravel()
        # guarantee symmetry
        G = (G + G.T) / 2
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


def local_score(data, emb, n_neighbors=5, metric='cosine', n_jobs=-1, landmarks=None, landmark_method='random', random_state=None, data_is_graph=False, emb_is_graph=False, cor_method='spearman', **kwargs):
    random_state = check_random_state(random_state)
    if landmarks is not None:
        if isinstance(landmarks, int):
            landmarks_ = get_landmark_indices(
                emb, n_landmarks=landmarks, method=landmark_method, random_state=random_state)
        elif isinstance(landmarks, np.ndarray):
            landmarks_ = landmarks
        else:
            raise ValueError(
                '\'landmarks\' must be either an integer or a numpy array.')
    else:
        landmarks_ = None
    if data_is_graph:
        data_knn = data
    else:
        data_knn = kNN(data, n_neighbors=n_neighbors,
                       metric=metric, n_jobs=n_jobs, **kwargs)
    if emb_is_graph:
        emb_knn = emb
    else:
        emb_knn = kNN(emb, n_neighbors=n_neighbors,
                      metric='euclidean', n_jobs=n_jobs, **kwargs)
    if cor_method == 'kendall':
        local_scores = knn_kendall_tau(
            data_knn, emb_knn, subsample_idx=landmarks_, unweighted=False, n_jobs=n_jobs)
    elif cor_method == 'spearman':
        local_scores = knn_spearman_r(
            data_knn, emb_knn, subsample_idx=landmarks_, unweighted=False, n_jobs=n_jobs)
    return local_scores

