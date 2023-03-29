import numpy as np
from sklearn.utils import check_random_state
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, kendalltau
from scipy.sparse.csgraph import shortest_path, NegativeCycleError
from scipy.sparse import lil_matrix
from topo.utils._utils import get_landmark_indices
from topo.base.ann import kNN

def subset_geodesic_distances(knn_dists, geodesic_dists):
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
        G = shortest_path(A, method=method,
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
            indices = np.arange(A.shape[0])
        elif np.issubdtype(type(indices), np.integer):
            indices = np.array([indices])
        n = len(indices)
        local_function = partial(shortest_path,
                                A, method, directed, False, unweighted, False)
        if n_jobs == 1 or n == 1:
            try:
                res = shortest_path(A, method, directed, False,
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


def local_score(data, emb, n_neighbors=5, metric='euclidean', n_jobs=-1, landmarks=None, landmark_method='random', use_k_geodesics=False, random_state=None, data_is_graph=False, emb_is_graph=False, cor_method='spearman', path_method='D', indices=None, unweighted=False, **kwargs):
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
        
    data_geodesics = geodesic_distance(
        data_knn, method=path_method, unweighted=unweighted, indices=indices, n_jobs=n_jobs)
    emb_geodesics = geodesic_distance(
        emb_knn, method=path_method, unweighted=unweighted, indices=indices, n_jobs=n_jobs)
    if use_k_geodesics:
        data_k_geodesics = subset_geodesic_distances(data_knn, data_geodesics).toarray()
        emb_k_geodesics = subset_geodesic_distances(emb_knn, emb_geodesics).toarray()
        # Symmetrize
        data_k_geodesics = (data_k_geodesics + data_k_geodesics.T) / 2
        emb_k_geodesics = (emb_k_geodesics + emb_k_geodesics.T) / 2
        if cor_method == 'kendall':
            res, _ = kendalltau(squareform(data_k_geodesics), squareform(emb_k_geodesics))
        elif cor_method == 'spearman':
            res, _ = spearmanr(squareform(data_k_geodesics), squareform(emb_k_geodesics))
        else:
            raise ValueError('\'cor_method\' must be either \'kendall\' or \'spearman\'.')
    else:
        if cor_method == 'kendall':
            res, _ = kendalltau(squareform(data_geodesics), squareform(emb_geodesics))
        elif cor_method == 'spearman':
            res, _ = spearmanr(squareform(data_geodesics), squareform(emb_geodesics))
        else:
            raise ValueError('\'cor_method\' must be either \'kendall\' or \'spearman\'.')
    return res

