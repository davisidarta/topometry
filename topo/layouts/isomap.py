# Isomap implementation

import numpy as np
from sklearn.preprocessing import KernelCenterer
from topo.eval.local_scores import geodesic_distance
from topo.utils._utils import get_landmark_indices
from topo.base.ann import kNN
from topo.spectral.eigen import eigendecompose


def Isomap(X, n_components=2, n_neighbors=50, metric='cosine', landmarks=None,
           landmark_method='kmeans', eig_tol=0, n_jobs=1, **kwargs):
    """
    Isomap embedding of a dataset or precomputed graph.

    Computes geodesic distances on the kNN graph, applies double-centering
    to the squared distance matrix (classical MDS kernel), and returns the
    top eigenvectors scaled by the square root of the corresponding
    eigenvalues — the standard Isomap procedure.

    Parameters
    ----------
    X : array-like or sparse
        Input data matrix of shape (n_samples, n_features), or a precomputed
        kNN graph (sparse) when ``metric='precomputed'``.

    n_components : int, default 2
        Number of dimensions to embed into.

    n_neighbors : int, default 50
        Number of neighbors used to build the kNN graph (ignored when
        ``metric='precomputed'``).

    metric : str, default 'cosine'
        Distance metric for the kNN graph.  Pass ``'precomputed'`` if ``X``
        is already a sparse adjacency matrix.

    landmarks : int or array-like of int, optional
        If an ``int``, the number of landmarks to select (using
        ``landmark_method``). If an array, the explicit landmark indices.
        When set, only the landmark-to-landmark geodesic sub-matrix is used.

    landmark_method : {'kmeans', 'random'}, default 'kmeans'
        Method for landmark selection when ``landmarks`` is an integer.

    eig_tol : float, default 0
        Stopping tolerance for the eigendecomposition (passed to ARPACK).

    n_jobs : int, default 1
        Number of parallel jobs for kNN construction and shortest-path
        computation.

    **kwargs
        Additional keyword arguments forwarded to :func:`topo.base.ann.kNN`.

    Returns
    -------
    Y : ndarray of shape (n_samples, n_components)
        Isomap embedding coordinates.
    """
    if landmarks is not None:
        if isinstance(landmarks, np.ndarray):
            pass  # use as-is
        elif isinstance(landmarks, int):
            landmarks = get_landmark_indices(X, n_landmarks=landmarks,
                                             method=landmark_method)

    # Build kNN graph
    if metric != 'precomputed':
        K = kNN(X, metric=metric, n_neighbors=n_neighbors,
                n_jobs=n_jobs, **kwargs)
    else:
        K = X.copy()

    # Pairwise geodesic distances
    G = geodesic_distance(K, method='D', unweighted=False, directed=False,
                          indices=landmarks, n_jobs=n_jobs)
    if landmarks is not None:
        G = G.T[landmarks].T

    # Guarantee symmetry and zero diagonal
    G = (G + G.T) / 2
    G[(np.arange(G.shape[0]), np.arange(G.shape[0]))] = 0

    # Replace infinities (disconnected pairs) with a large finite distance
    finite_mask = np.isfinite(G)
    if not finite_mask.all():
        max_finite = G[finite_mask].max() if finite_mask.any() else 1.0
        G[~finite_mask] = max_finite * 2.0

    # Standard Isomap: double-center the squared geodesic distance matrix
    # (equivalent to the MDS kernel -½ H D² H)
    D2 = G ** 2
    K_matrix = KernelCenterer().fit_transform(-0.5 * D2)

    # Eigendecompose for the largest eigenpairs
    evals, evecs = eigendecompose(
        K_matrix, n_components=n_components,
        eigensolver='arpack', largest=True,
        eigen_tol=eig_tol, random_state=None, verbose=False,
    )

    # Keep only positive eigenvalues
    pos = evals > 0
    evals = evals[pos]
    evecs = evecs[:, pos]

    # Isomap coordinates: eigenvectors scaled by sqrt(eigenvalue)
    Y = evecs * np.sqrt(evals)
    return Y[:, :n_components]
