# Some utility functions
import numpy as np
from scipy.sparse import coo_matrix

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