import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA, TruncatedSVD


def global_loss_(X, Y):
    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)
    A = X.T @ (Y @ np.linalg.inv(Y.T @ Y))
    GL = np.mean(np.power(X.T - A @ Y.T, 2))
    return GL


def global_score_pca(X, Y, Y_pca=None):
    """
    Compute the global score comparing an embedding to PCA.

    The score is defined as ``exp(-(L_emb - L_pca) / L_pca)`` where ``L``
    denotes the mean reconstruction error (global loss) of a linear
    projection.  A score of 1 means the embedding preserves as much global
    structure as PCA; scores below 1 indicate worse global preservation.
    The result is clipped to [0, 1].

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features) or sparse matrix
        Input feature matrix.
    Y : array-like of shape (n_samples, n_components)
        Low-dimensional embedding to evaluate.
    Y_pca : array-like of shape (n_samples, n_components), optional
        Pre-computed PCA embedding. If None, computed from ``X``.

    Returns
    -------
    score : float in (0, 1]
        Global structure preservation score relative to PCA.
    """
    if Y_pca is None:
        n_dims = Y.shape[1]
        if issparse(X):
            Y_pca = TruncatedSVD(n_components=n_dims).fit_transform(X)
        else:
            Y_pca = PCA(n_components=n_dims).fit_transform(X)

    gs_pca = global_loss_(X, Y_pca)
    gs_emb = global_loss_(X, Y)
    GSP = np.exp(-(gs_emb - gs_pca) / gs_pca)
    if GSP > 1.0:
        GSP = 1.0
    return GSP


def global_score_laplacian(X, Y, k=10, data_is_graph=False, n_jobs=12, random_state=None):
    """
    Compute the global score comparing an embedding to a Laplacian Eigenmap baseline.

    The score is defined as ``exp(-(L_emb - L_lap) / L_lap)`` where ``L``
    denotes the mean reconstruction error (global loss) of a linear
    projection.  A score of 1 means the embedding preserves as much global
    structure as a Laplacian Eigenmap of the same dimension; scores below 1
    indicate worse global preservation.  The result is clipped to [0, 1].

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features) or sparse (n_samples, n_samples)
        Input feature matrix, or precomputed affinity graph if
        ``data_is_graph=True``.
    Y : array-like of shape (n_samples, n_components)
        Low-dimensional embedding to evaluate.
    k : int, default 10
        Number of neighbors used by ``SpectralEmbedding`` when
        ``data_is_graph=False``.
    data_is_graph : bool, default False
        If True, ``X`` is treated as a precomputed affinity graph.
    n_jobs : int, default 12
        Number of parallel jobs for ``SpectralEmbedding``.
    random_state : numpy.random.RandomState or int, optional
        Random state for ``SpectralEmbedding``.

    Returns
    -------
    score : float in (0, 1]
        Global structure preservation score relative to Laplacian Eigenmaps.
    """
    if issparse(X):
        X = X.toarray()
    elif not isinstance(X, np.ndarray):
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            X = np.array(X.values.T)

    n_dims = Y.shape[1]
    if random_state is None:
        random_state = np.random.RandomState()

    affinity = 'precomputed' if data_is_graph else 'nearest_neighbors'
    Y_lap = SpectralEmbedding(
        n_components=n_dims, n_neighbors=k, n_jobs=n_jobs,
        affinity=affinity, random_state=random_state,
    ).fit_transform(X)
    Y_lap /= Y_lap.max()

    gs_lap = global_loss_(X, Y_lap)
    gs_emb = global_loss_(X, Y)
    GSL = np.exp(-(gs_emb - gs_lap) / gs_lap)
    if GSL > 1.0:
        GSL = 1.0
    return GSL
