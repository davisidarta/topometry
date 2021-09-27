import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA, TruncatedSVD

def global_loss_(X, Y):
    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)
    A = X.T @ (Y @ np.linalg.inv(Y.T @ Y))
    return np.mean(np.power(X.T - A @ Y.T, 2))


def global_score_pca(X, Y):
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

    n_dims = Y.shape[1]
    Y_pca = TruncatedSVD(n_components=n_dims).fit_transform(X)
    gs_pca = global_loss_(X, Y_pca)
    gs_emb = global_loss_(X, Y)
    return np.exp(-(gs_emb - gs_pca) / gs_pca)

def global_score_laplacian(X, Y, random_state=None):
    """
    Global score
    Input
    ------
    X: Instance matrix
    Y: Embedding
    """

    if issparse(X) == True:
        if isinstance(X, csr_matrix):
            X = X.toarray()
    if issparse(X) == False:
        if isinstance(X, np.ndarray):
            X = X
        else:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                X = np.array(X.values.T)
    n_dims = Y.shape[1]
    if random_state is None:
        random_state = np.random.RandomState()
    Y_lap = SpectralEmbedding(
        n_components=n_dims, random_state=random_state
    ).fit_transform(X)
    Y_lap /= Y_lap.max()
    gs_lap = global_loss_(X, Y_lap)
    gs_emb = global_loss_(X, Y)
    return np.exp(-(gs_emb - gs_lap) / gs_lap)