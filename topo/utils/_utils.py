# Some other utility functions
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.utils import check_random_state
from sklearn.decomposition import TruncatedSVD

def read_pkl(wd=None, filename='topograph.pkl'):
    try:
        import pickle
    except ImportError:
        return (print('Pickle is needed for loading the TopOGraph. Please install it with `pip3 install pickle`'))

    if wd is None:
        import os
        wd = os.getcwd()
    with open(wd + filename, 'rb') as input:
        TopOGraph = pickle.load(input)
    return TopOGraph

def get_landmark_indices(data, n_landmarks=1000, method='random', random_state=None, **kwargs):
    random_state = check_random_state(random_state)
    if method == 'random':
        landmarks_ = np.arange(np.shape(data)[0])
        return random_state.choice(landmarks_, size=n_landmarks, replace=False)
    elif method == 'kmeans':
        #raise ValueError('Not currently implemented')
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=n_landmarks,
                                 random_state=random_state, **kwargs).fit(data)
        return kmeans.cluster_centers_
    else:
        raise ValueError('Unknown landmark selection method')

def subsample_square_csr_to_indices(data, indices):
    import time
    print('Started subsampling')
    start = time.time()
    mask = np.isin(np.arange(data.shape[0]), indices) & np.isin(np.arange(data.shape[1]), indices)
    new = data[mask]
    data_sub = csr_matrix((new.data, new.indices, new.indptr), shape=(len(indices), len(indices)))
    print('Finished subsampling! Time taken: ', time.time() - start)
    return data_sub

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


<<<<<<< HEAD
=======

>>>>>>> master
def print_eval_results(evaluation_dict, n_top=3):
      for estimate in evaluation_dict.keys():
            if estimate == 'EigenbasisLocal':
                  estimate_str = ' local '
                  estimated_attr = ' eigenbases '
            elif estimate == 'ProjectionLocal':
                  estimate_str = ' local '
                  estimated_attr = ' projections '
            elif estimate == 'EigenbasisGlobal':
                  estimate_str = ' global '
                  estimated_attr = ' eigenbases '
            elif estimate == 'ProjectionGlobal':
                  estimate_str = ' global '
                  estimated_attr = ' projections '
            
            print('\n The top-' + str(n_top) + estimated_attr + 'which preserve' + estimate_str + 'information the most are: ')
            res = dict(sorted(evaluation_dict[estimate].items(), key = lambda x: x[1], reverse = True)[:3])
            i=1
            for key in res.keys():
                  print('   ' + str(i) + ' - ' + key + ': ' + str(res[key]))
                  i+=1


<<<<<<< HEAD
def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]

def get_eccentricity(emb, laplacian, H_emb=None):
    if H_emb is None:
        from topo.eval import RiemannMetric
        rmetric = RiemannMetric(emb, laplacian)
        H_emb = rmetric.get_dual_rmetric()
    N = np.shape(laplacian)[0]
    ecc_list = []
    for i in range(N):
        cov = H_emb[i, :, :]
        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        # Width and height are "full" widths, not radius
        width, height = np.sqrt(np.absolute(vals))
        if width > height:
            R = width / height
        else:
            R = height / width
        ecc = np.sqrt(np.abs(1 - R))
        ecc_list.append(ecc)
    return ecc_list
=======

>>>>>>> master
