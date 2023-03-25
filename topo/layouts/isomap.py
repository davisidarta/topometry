# Efficient ISOMAP implementation

import numpy as np
from sklearn.preprocessing import KernelCenterer
from topo.eval.local_scores import geodesic_distance
from topo.utils._utils import get_landmark_indices
from topo.base.ann import kNN
from topo.spectral.eigen import eigendecompose

def Isomap(X, n_components=2,  n_neighbors=50, metric='cosine', landmarks=None, landmark_method='kmeans', eig_tol=0, n_jobs=1, **kwargs):
    """
    Isomap embedding of a graph. This is a highly efficient implementation can also operate with landmarks.

    Parameters
    ----------
    X : array-like or sparse
        The input data.
    
    n_components : int (optional, default 2).
        The number of componetns to embed into.
    
    n_neighbors : int (optional, default 5).
        The number of neighbors to use for the geodesic distance matrix.
    
    metric : str (optional, default 'euclidean').
        The metric to use for the geodesic distance matrix. Can be 'precomputed'

    landmarks : int or array of shape (n_samples,) (optional, default None).
        If passed as `int`, will obtain the number of landmarks. If passed as `np.ndarray`, will use the specified indexes in the array.
        Any value other than `None` will result in only the specified landmarks being used in the layout optimization, and will
        populate the Projector.landmarks_ slot.

    landmark_method : str (optional, default 'kmeans').
        The method to use for selecting landmarks. If `landmarks` is passed as an `int`, this will be used to select the landmarks.
        Can be either 'kmeans' or 'random'.

    eig_tol : float (optional, default 0).
        Stopping criterion for eigendecomposition of the pairwise geodesics matrix.

    n_jobs : int (optional, default 1).
        The number of jobs to use for the computation. 

    **kwargs : dict (optional, default {}).
        Additional keyword arguments to pass to the kNN function.

    Returns
    -------
    Y : array of shape (n_samples, n_components)
        The embedding vectors.
    """
    N = np.shape(X)[0]
    if landmarks is not None:
        if isinstance(landmarks, np.ndarray):
            landmarks = landmarks
        elif isinstance(landmarks, int):
            landmarks = get_landmark_indices(X, n_landmarks=landmarks, method=landmark_method)
    # kNN
    if metric != 'precomputed':
        K = kNN(X, metric=metric, n_neighbors=n_neighbors, symmetrize=True, n_jobs=n_jobs, **kwargs)
    else:
        K = X.copy()
    # Pairwise geodesics
    G = geodesic_distance(K, method='D', unweighted=False, directed=False, indices=landmarks, n_jobs=n_jobs)
    if landmarks is not None:
        G = G.T[landmarks].T
    # guarantee symmetry
    G = (G + G.T) / 2
    # zero diagonal 
    G[(np.arange(G.shape[0]), np.arange(G.shape[0]))] = 0
    G = np.exp(G ** 2)
    infinities = np.isinf(G)
    G[infinities] = 0
    # eigendecompose
    evals, evecs = eigendecompose(G, n_components=n_components+1, eigensolver='arpack', largest=True, eigen_tol=eig_tol, random_state=None, verbose=False)
    idxs = evals.argsort()[::-1]
    evals = evals[idxs]
    evecs = evecs[:, idxs]
    stack = []
    # embedding
    for i in range(1, n_components + 1):
        #stack.append(evecs[:, -i] * np.sqrt(evals[-i] + 1e-4))
        stack.append(evecs[:, -i] * evals[-i])
    Y = np.column_stack(tuple(stack))
    return Y