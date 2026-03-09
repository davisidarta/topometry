# Some other utility functions
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, issparse
from sklearn.utils import check_random_state
from sklearn.decomposition import TruncatedSVD


def get_landmark_indices(data, n_landmarks=1000, method='random', random_state=None, **kwargs):
    """
    Select landmark indices from data.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features) or sparse matrix
        Input data. For ``method='kmeans'``, must be a feature matrix (not a
        precomputed graph).
    n_landmarks : int, default 1000
        Number of landmarks to select.
    method : {'random', 'kmeans'}, default 'random'
        Landmark selection strategy.
        * ``'random'``: uniform random sample of row indices.
        * ``'kmeans'``: MiniBatchKMeans clustering; for each centroid the
          nearest actual data point is returned (so the result is always a
          valid index array into ``data``).
    random_state : int or numpy.random.RandomState, optional
        RNG seed / state.
    **kwargs
        Extra keyword arguments forwarded to ``MiniBatchKMeans``.

    Returns
    -------
    indices : ndarray of int, shape (n_landmarks,)
        Row indices of the selected landmarks.
    """
    random_state = check_random_state(random_state)
    if method == 'random':
        all_idx = np.arange(np.shape(data)[0])
        return random_state.choice(all_idx, size=n_landmarks, replace=False)
    elif method == 'kmeans':
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics import pairwise_distances_argmin
        data_arr = np.asarray(data.todense() if issparse(data) else data)
        kmeans = MiniBatchKMeans(
            n_clusters=n_landmarks, random_state=random_state, **kwargs
        ).fit(data_arr)
        # Return the index of the nearest actual data point to each centroid.
        indices = pairwise_distances_argmin(kmeans.cluster_centers_, data_arr)
        return indices
    else:
        raise ValueError("Unknown landmark selection method; use 'random' or 'kmeans'.")


def get_sparse_matrix_from_indices_distances(knn_indices, knn_dists, n_obs, n_neighbors):
    rows = np.zeros((n_obs * n_neighbors), dtype=int)
    cols = np.zeros((n_obs * n_neighbors), dtype=int)
    vals = np.zeros((n_obs * n_neighbors), dtype=float)
    for i in range(knn_indices.shape[0]):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            else:
                val = knn_dists[i, j]

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    result = coo_matrix((vals, (rows, cols)),
                        shape=(n_obs, n_obs))
    result.eliminate_zeros()
    return result.tocsr()


def get_indices_distances_from_sparse_matrix(X, n_neighbors):
    """
    Get the knn indices and distances for each point in a sparse k-nearest-neighbors matrix.

    Parameters
    ----------
    X : sparse matrix
        Input knn matrix to get indices and distances from.

    n_neighbors : int
        Number of neighbors to get.

    Returns
    -------
    knn_indices : ndarray of shape (n_obs, n_neighbors)
        The indices of the nearest neighbors for each point.

    knn_dists : ndarray of shape (n_obs, n_neighbors)
        The distances to the nearest neighbors for each point.
    """
    _knn_indices = np.zeros((X.shape[0], n_neighbors), dtype=int)
    _knn_dists = np.zeros(_knn_indices.shape, dtype=float)
    for row_id in range(X.shape[0]):
        # Find KNNs row-by-row
        row_data = X[row_id].data
        row_indices = X[row_id].indices
        if len(row_data) < n_neighbors:
            raise ValueError(
                "Some rows contain fewer than n_neighbors distances!"
            )
        row_nn_data_indices = np.argsort(row_data)[: n_neighbors]
        _knn_indices[row_id] = row_indices[row_nn_data_indices]
        _knn_dists[row_id] = row_data[row_nn_data_indices]
    return _knn_indices, _knn_dists
