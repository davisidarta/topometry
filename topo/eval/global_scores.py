import numpy as np
from scipy.sparse import csr_matrix, issparse
from topo.base.ann import HNSWlibTransformer, NMSlibTransformer
from topo.spectral._spectral import LapEigenmap
from sklearn.decomposition import TruncatedSVD

def global_loss_(X, Y):
    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)
    A = X.T @ (Y @ np.linalg.inv(Y.T @ Y))
    return np.mean(np.power(X.T - A @ Y.T, 2))


def global_score_pca(X, Y, n_dim=30):
    """
    Global score
    Input
    ------
    X: Instance matrix
    Y: Embedding
    """

    if issparse(X) == True:
        if isinstance(X, np.ndarray):
            X = csr_matrix(X)
    if issparse(X) == False:
        if isinstance(X, np.ndarray):
            X = csr_matrix(X)
        else:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                X = csr_matrix(X.values.T)

    Y_pca = TruncatedSVD(n_components=n_dim).fit_transform(X)
    gs_pca = global_loss_(X, Y_pca)
    gs_emb = global_loss_(X, Y)
    return np.exp(-(gs_emb - gs_pca) / gs_pca)



def global_score_laplacian(X, Y, k=5, metric='euclidean', n_jobs=10):
    """
    Global score

    Parameters
    ------
    X: Instance matrix
    Y: Embedding
    k: k-nearest neighbors
    n_jobs: number of jobs use when computing neighborhood graph for LE

    """
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

    if metric != 'precomputed':
        if issparse(X):
            if _have_nmslib:
                from topo.base.ann import NMSlibTransformer
                data_graph = NMSlibTransformer(n_neighbors=k, n_jobs=n_jobs, metric=metric).fit_transform(X)
            elif _have_hnswlib:
                from topo.base.ann import HNSWlibTransformer
                data = X.toarray()
                data_graph = HNSWlibTransformer(n_neighbors=k, n_jobs=n_jobs, metric=metric).fit_transform(X)
            else:
                from sklearn.neighbors import NearestNeighbors
                data_nbrs = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs,
                                             metric=metric).fit(X)
                data_graph = data_nbrs.kneighbors(X)
        else:
            if _have_hnswlib:
                from topo.base.ann import HNSWlibTransformer
                if issparse(X):
                    data = X.toarray()
                data_graph = HNSWlibTransformer(n_neighbors=k, n_jobs=n_jobs, metric=metric).fit_transform(X)
            elif _have_nmslib:
                from topo.base.ann import NMSlibTransformer
                data = csr_matrix(X)
                data_graph = NMSlibTransformer(n_neighbors=k, n_jobs=n_jobs, metric=metric).fit_transform(X)
            else:
                from sklearn.neighbors import NearestNeighbors
                data_nbrs = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs,
                                             metric=metric).fit(X)
                data_graph = data_nbrs.kneighbors(X)
    else:
        data_graph = X
    if not isinstance(data_graph, csr_matrix):
        data_graph = csr_matrix(data_graph)

    n_dims = Y.shape[1]

    Y_lap = LapEigenmap(W=data_graph,
        n_eigs=n_dims)
    Y_lap /= Y_lap.max()
    gs_lap = global_loss_(X, Y_lap)
    gs_emb = global_loss_(X, Y)
    return np.exp(-(gs_emb - gs_lap) / gs_lap)