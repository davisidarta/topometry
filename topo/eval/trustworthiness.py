# Wraps around scikit-learn trustworthness metric for manifold learning
import numpy as np
from scipy.sparse import csr_matrix, issparse
from topo.base.ann import kNN
from sklearn.manifold import trustworthiness as twt

def trustworthiness(X, X_embedded, n_neighbors=5, metric='euclidean', n_jobs=10, **kwargs):
    try:
        import hnswlib
        _have_hnswlib = True
    except ImportError:
        _have_hnswlib = False

    try:
        import nmslib
        _have_nmslib = True
    except ImportError:
        _have_nmslib = False

    if metric is not 'precomputed':
        if issparse(X) == True:
            if _have_nmslib:
                backend = 'nmslib'
                if isinstance(X, np.ndarray):
                    X = csr_matrix(X)
            else:
                backend = 'hnswlib'
                X = X.toarray()
        else:
            if _have_hnswlib:
                backend = 'hnswlib'
            else:
                backend = 'sklearn'

        knn_graph = kNN(X, n_neighbors=n_neighbors, metric=metric, n_jobs=n_jobs, backend=backend, **kwargs)
        return twt(knn_graph, X_embedded, n_neighbors=n_neighbors, metric=metric)

    else:
        return twt(X, X_embedded, n_neighbors=n_neighbors, metric='precomputed')
