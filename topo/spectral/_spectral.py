#####################################
# Author: Davi Sidarta-Oliveira
# School of Medical Sciences,University of Campinas,Brazil
# contact: davisidarta[at]fcm[dot]unicamp[dot]com
# License: MIT
######################################
# Being clear about laplacian-type operators and spectral decompositions

from warnings import warn
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.utils import check_random_state, as_float_array

def _dense_degree(W):
    return np.diag(W.sum(axis=1))

def _dense_unnormalized_laplacian(W):
    D = _dense_degree(W)
    L = D - W
    # Return ndarray instead of np.matrix
    return L.A

def _dense_symmetric_normalized_laplacian(W):
    D = _dense_degree(W)
    L = D - W
    D_tilde = np.diag(1/np.sqrt(D.diagonal()))
    # Lsym = D^-1/2 @ L @ D^-1/2 = I - D^-1/2 @ W @ D^-1/2
    Lsym_norm = D_tilde @ (L @ D_tilde)
    return Lsym_norm

def _dense_normalized_random_walk_laplacian(W):
    D = _dense_degree(W)
    # Lr = D^-1@Lr = I - D^-1@W = I - P
    Lr = np.eye(*W.shape) - (np.diag(1/D.diagonal())@W)
    return Lr

def _dense_anisotropic_diffusion(W, alpha=1):
    # Note the resulting operator is not symmetric!
    D = _dense_degree(W)
    Da = np.diag(1/(D.diagonal()**alpha))
    Wa = Da @ (W @ Da)
    Dd = _dense_degree(Wa)
    Pa = (np.diag(1/Dd.diagonal())@Wa)
    return Pa

def _dense_anisotropic_diffusion_symmetric(W, alpha=1, return_D_inv_sqrt=False):
    if alpha < 0:
        alpha = 0
    D = _dense_degree(W)
    # Dinva is D^-alpha
    Dinva = np.diag(1/(D.diagonal()**alpha))
    Wa = Dinva @ (W @ Dinva)
    Da = _dense_degree(Wa)
    Dalpha_inv = np.diag(1/(Da.diagonal()))
    Pa = Dalpha_inv @ Wa
    # Now let's build the symmetrized version Pasym:
    D_right = _dense_degree(Pa)
    D_left = D_right.copy()
    D_right = np.sqrt(D_right.diagonal())
    D_left = np.diag(1/np.sqrt(D_left.diagonal()))
    Psym = D_right @ (Pa @ D_left)
    if return_D_inv_sqrt:
        return Psym, D_left
    else:
        return Psym

def _sparse_degree(W):
    N = np.shape(W)[0]
    D = np.ravel(W.sum(axis=1))
    return sparse.csr_matrix((D, (range(N), range(N))), shape=[N, N])

def degree(W):
    if sparse.issparse(W):
        return _sparse_degree(W)
    else:
        return _dense_degree(W)


def _sparse_unnormalized_laplacian(W):
    D = _sparse_degree(W)
    L = D - W
    return L

def _sparse_symmetrized_normalized_laplacian(W):
    D = _sparse_degree(W)
    L = D - W
    N = np.shape(W)[0]
    D_tilde = np.ravel(W.sum(axis=1))
    # D ^-1/2:
    D_tilde[D_tilde != 0] = 1 / np.sqrt(D_tilde[D_tilde != 0])
    Dinvs = sparse.csr_matrix((D_tilde, (range(N), range(N))), shape=[N, N])
    # Lsym = D^-1/2 @ L @ D^-1/2 = I - D^-1/2 @ W @ D^-1/2
    Lsym_norm = Dinvs.dot(L).dot(Dinvs)
    return Lsym_norm

def _sparse_normalized_random_walk_laplacian(W):
    N = np.shape(W)[0]
    D = np.ravel(W.sum(axis=1))
    # D ^-1/2:
    D[D != 0] = 1 / D[D != 0]
    Dinv = sparse.csr_matrix((D, (range(N), range(N))), shape=[N, N])
    I = sparse.identity(W.shape[0], dtype=np.float32)
    Lr = I - Dinv.dot(W)
    return Lr

def _sparse_anisotropic_diffusion(W, alpha=1, return_d_alpha_inv=False):
    # Note the resulting operator is not symmetric!
    N = np.shape(W)[0]
    D = np.ravel(W.sum(axis=1))
    D[D != 0] = D[D != 0] ** (-alpha)
    # Dinva is D^-alpha
    Dinva = sparse.csr_matrix((D, (range(N), range(N))), shape=[N, N])
    Wa = Dinva.dot(W).dot(Dinva)
    Da = np.ravel(Wa.sum(axis=1))
    Da[Da != 0] = 1 / Da[Da != 0]
    # Da is now D(alpha)^-1
    Pa = sparse.csr_matrix((Da, (range(N), range(N))), shape=[N, N]).dot(Wa)
    if return_d_alpha_inv:
        return Pa, Da
    else:
        return Pa

def _sparse_anisotropic_diffusion_symmetric(W, alpha=1, return_D_inv_sqrt=False):
    if alpha < 0:
        alpha = 0
    N = np.shape(W)[0]
    D = np.ravel(W.sum(axis=1))
    D[D != 0] = D[D != 0] ** (-alpha)
    # Dinva is D^-alpha
    Dinva = sparse.csr_matrix((D, (range(N), range(N))), shape=[N, N])
    Wa = Dinva.dot(W).dot(Dinva)
    Da = np.ravel(Wa.sum(axis=1))
    Da[Da != 0] = 1 / Da[Da != 0]
    Dalpha_inv = sparse.csr_matrix((Da, (range(N), range(N))), shape=[N, N])
    Pa = Dalpha_inv.dot(Wa)
    # Now let's build the symmetrized version Psym:
    D_right = np.ravel(W.sum(axis=1))
    D_left = D_right.copy()
    D_right[D_right != 0] = np.sqrt(D_right[D_right != 0])
    D_left[D_left != 0] = 1 / np.sqrt(D_left[D_left != 0])
    D_right = sparse.csr_matrix((D_right, (range(N), range(N))), shape=[N, N])
    D_left = sparse.csr_matrix((D_left, (range(N), range(N))), shape=[N, N])
    # Note the resulting operator is symmetric!
    Psym = D_right.dot(Pa).dot(D_left)
    if return_D_inv_sqrt:
        return Psym, D_left
    else:
        return Psym


def graph_laplacian(W, laplacian_type='random_walk'):
    """
    Compute the graph Laplacian, given a adjacency or affinity graph W. For a friendly reference,
    see this material from James Melville: https://jlmelville.github.io/smallvis/spectral.html
    
    Parameters
    ----------
    W : scipy.sparse.csr_matrix or np.ndarray
        The graph adjacency or affinity matrix. Assumed to be symmetric and with zero diagonal.
        No further symmetrization is performed, so make sure to symmetrize W if necessary (usually done additively with W = (W + W.T)/2 ).

    laplacian : str (optional, default 'random_walk').
        The type of laplacian to use. Can be 'unnormalized', 'normalized', or 'random_walk'.

    Returns
    -------
    L : scipy.sparse.csr_matrix
        The graph Laplacian.
    """
    if sparse.issparse(W):
        if laplacian_type == 'unnormalized':
            lap_fun = _sparse_unnormalized_laplacian
        elif laplacian_type == 'normalized':
            lap_fun = _sparse_symmetrized_normalized_laplacian
        elif laplacian_type == 'random_walk':
            lap_fun = _sparse_normalized_random_walk_laplacian
        else:
            raise ValueError('Unknown laplacian type: {}'.format(laplacian_type) + '. Should \
            be one of \"unnormalized\", \"normalized\", or \"random_walk\".')
    else:
        if laplacian_type == 'unnormalized':
            lap_fun = _dense_unnormalized_laplacian
        elif laplacian_type == 'normalized':
            lap_fun = _dense_symmetric_normalized_laplacian
        elif laplacian_type == 'random_walk':
            lap_fun = _dense_normalized_random_walk_laplacian
        else:
            raise ValueError('Unknown laplacian type: {}'.format(laplacian_type) + '. Should \
            be one of \"unnormalized\", \"normalized\", or \"random_walk\".')
    return lap_fun(W)


def LE(W, n_eigs=10, laplacian_type='random_walk', drop_first=True, weight=True, return_evals=False, eigen_tol=0, random_state=None):
    """
    Performs [Laplacian Eigenmaps](https://www2.imm.dtu.dk/projects/manifold/Papers/Laplacian.pdf), given a adjacency or affinity graph W.
    The graph W can be a sparse matrix or a dense matrix. It is assumed to be symmetric (no further symmetrization is performed, be sure it is),
    and with zero diagonal (all diagonal elements are 0). The eigenvectors associated with the smallest eigenvalues
    form a new orthonormal basis which represents the graph in the feature space and are useful for denoising and clustering.

    Parameters
    ----------
    W : scipy.sparse.csr_matrix or np.ndarray
        The graph adjacency or affinity matrix. Assumed to be symmetric and with zero diagonal.

    n_eigs : int (optional, default 10).
        The number of eigenvectors to compute.

    laplacian : str (optional, default 'random_walk').
        The type of laplacian to use. Can be 'unnormalized', 'symmetric', or 'random_walk'.

    drop_first : bool (optional, default True).
        Whether to drop the first eigenvector.

    weight : bool (optional, default True).
        Whether to weight the eigenvectors by the square root of the eigenvalues.

    return_evals : bool (optional, default False).
        Whether to return the eigenvalues. If True, returns a tuple of (eigenvectors, eigenvalues).

    eigen_tol : float (optional, default 0).
        The tolerance for the eigendecomposition.

    random_state : int (optional, default None).
        The random state for the eigendecomposition in scipy.sparse.linalg.lobpcg() if the data has more than
        a million samples.

    Returns
    -----------
    evecs : np.ndarray of shape (W.shape[0], n_eigs)
        The eigenvectors of the graph Laplacian, sorted by ascending eigenvalues.

    If return_evals:
        evecs, evals : tuple of ndarrays
        The eigenvectors and associated eigenvalues, sorted by ascending eigenvalues.

    """
    random_state = check_random_state(random_state)
    if n_eigs > np.shape(W)[0]:
        raise ValueError(
            'n_eigs must be less than or equal to the number of nodes.')
    # Compute graph Laplacian
    L = graph_laplacian(W, laplacian_type)
    if not sparse.issparse(L):
        L = sparse.csr_matrix(L)  # for ARPACK efficiency
    # Add one more eig if drop_first is True
    if drop_first:
        n_eigs = n_eigs + 1
    num_lanczos_vectors = max(2 * n_eigs + 2, int(np.sqrt(W.shape[0])))
    # Compute eigenvalues and eigenvectors
    try:
        if L.shape[0] < 1000000:
            evals, evecs = sparse.linalg.eigsh(
                L, k=n_eigs, which='SM', tol=eigen_tol, ncv=num_lanczos_vectors, maxiter=L.shape[0] * 5)
        else:
            evals, evecs = sparse.linalg.lobpcg(
                L, random_state.normal(size=(L.shape[0], n_eigs)), largest=False, tol=1e-8
            ) 
    except sparse.linalg.ArpackError:
            warn(
                "WARNING: spectral decomposition FAILED! This is likely due to too small an eigengap. Consider\n"
                "adding some noise or jitter to your data."
            )
            return None
    evals = np.real(evals)
    evecs = np.real(evecs)
    # Sort eigenvalues and eigenvectors in ascending order
    idx = evals.argsort()
    evals = evals[idx]
    evecs = evecs[:, idx]
    # Normalize
    for i in range(evecs.shape[1]):
        evecs[:, i] = evecs[:, i] / np.linalg.norm(evecs[:, i])
    evecs = evecs * np.sqrt(evals + 1e-12) # prevent division by zero
    # Return embedding and evals
    if drop_first:
        evecs = evecs[:, 1:]
        evals = evals[1:]
    if return_evals:
        return evecs, evals
    else:
        return evecs


def diffusion_operator(W, alpha=1.0, symmetric=True, return_D_inv_sqrt=False):
    """
    Computes the [diffusion operator](https://doi.org/10.1016/j.acha.2006.04.006).

    Parameters
    ----------
    W : scipy.sparse.csr_matrix or np.ndarray
        The graph adjacency or affinity matrix. Assumed to be symmetric and with zero diagonal.
        No further symmetrization is performed, so make sure to symmetrize W if necessary (usually done additively with W = (W + W.T)/2 ).

    alpha : float (optional, default 1.0).
        Anisotropy to apply. 'Alpha' in the diffusion maps literature.
    
    symmetric : bool (optional, default True).
        Whether to use a symmetric version of the diffusion operator. This is particularly useful to yield a symmetric operator
        when using anisotropy (alpha > 0), as the diffusion operator P would be assymetric otherwise, which can be problematic
        during matrix decomposition. Eigenvalues are the same of the assymetric version, and the eigenvectors of the original assymetric
        operator can be obtained by left multiplying by D_inv_sqrt (returned if `return_D_inv_sqrt` set to True).

    return_D_inv_sqrt : bool (optional, default False).
        Whether to return a tuple of diffusion operator P and inverse square root of the degree matrix.

    Returns
    -------
    P : scipy.sparse.csr_matrix
        The graph diffusion operator.


    """
    # Compute diffusion operator
    if sparse.issparse(W):
        if symmetric:
            P, D_left = _sparse_anisotropic_diffusion_symmetric(
                W, alpha, return_D_inv_sqrt=return_D_inv_sqrt)
        else:
            P = _sparse_anisotropic_diffusion(W, alpha)
    else:
        if symmetric:
            P, D_left = _dense_anisotropic_diffusion_symmetric(
                W, alpha, return_D_inv_sqrt=return_D_inv_sqrt)
        else:
            P = _dense_anisotropic_diffusion(W, alpha)
    if symmetric:
        if return_D_inv_sqrt:
            return P, D_left
        else:
            return P
    else:
        return P


def DM(W, n_eigs=10, alpha=1.0, return_evals=False, symmetric=True, eigen_tol=10e-4, t=None):
    """
    Performs [Diffusion Maps](https://doi.org/10.1016/j.acha.2006.04.006), given a adjacency or affinity graph W.
    The graph W can be a sparse matrix or a dense matrix. It is assumed to be symmetric (no further symmetrization is performed, be sure it is),
    and with zero diagonal (all diagonal elements are 0). The eigenvectors associated with the largest eigenvalues
    form a new orthonormal basis which represents the graph in the feature space and are useful for denoising and clustering.

    Parameters
    ----------

    W : scipy.sparse.csr_matrix or np.ndarray
        The graph adjacency or affinity matrix. Assumed to be symmetric and with zero diagonal.

    n_eigs : int (optional, default 10).
        The number of eigenvectors to return.

    alpha : float (optional, default 1.0).
        Anisotropy to be applied to the diffusion map. Refered to as alpha in the diffusion maps literature.

    return_evals : bool (optional, default False).
        Whether to return the eigenvalues. If True, returns a tuple of (eigenvectors, eigenvalues).

    eigen_tol : float (optional, default 0).
        The tolerance for the eigendecomposition in scipy.sparse.linalg.eigsh().

    t : int (optional, default 1).
        The number of steps to take in the diffusion map.

    Returns
    ----------
        * If return_evals is True :
            A tuple of scaled eigenvectors (the diffusion maps) and eigenvalues.
        * If return_evals is False :
            An array of scaled eigenvectors (the diffusion maps).

    """
    if t is not None:
        if not isinstance(t, int):
            raise ValueError('t must be `None` or an integer.')
    if n_eigs > np.shape(W)[0]:
        raise ValueError(
            'n_eigs must be less than or equal to the number of nodes.')
    # Compute diffusion operator
    if symmetric:
        P, D_left = diffusion_operator(
            W, alpha, symmetric, return_D_inv_sqrt=True)
    else:
        P = diffusion_operator(W, alpha, symmetric, return_D_inv_sqrt=False)
    # Compute eigenvalues and eigenvectors
    num_lanczos_vectors = max(2 * n_eigs + 2, int(np.sqrt(P.shape[0])))
    evals, evecs = sparse.linalg.eigsh(
        P, k=n_eigs, which='LM', tol=eigen_tol, ncv=num_lanczos_vectors, maxiter=P.shape[0] * 5)
    evecs = np.real(evecs)
    evals = np.real(evals)
    if symmetric:
        evecs = D_left.dot(evecs)
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    # Normalize
    for i in range(evecs.shape[1]):
        evecs[:, i] = evecs[:, i] / np.linalg.norm(evecs[:, i])
    if t is not None:
        if t > 1:
            evals = np.power(evals, t)
        diffmap = evecs * evals
    else:
        use_eigs = int(np.sum(evals > 0, axis=0))
        eigs_idx = list(range(1, int(use_eigs)))
        eig_vals = np.ravel(evals[eigs_idx])
        diffmap = evecs[:, eigs_idx] * (eig_vals / (1 - eig_vals))
    if return_evals:
        return diffmap, evals
    else:
        return diffmap


def _set_diag(laplacian, value, norm_laplacian):
    n_nodes = laplacian.shape[0]
    # We need all entries in the diagonal to values
    if not sparse.isspmatrix(laplacian):
        if norm_laplacian:
            laplacian.flat[:: n_nodes + 1] = value
    else:
        laplacian = laplacian.tocoo()
        if norm_laplacian:
            diag_idx = laplacian.row == laplacian.col
            laplacian.data[diag_idx] = value
        # If the matrix has a small number of diagonals (as in the
        # case of structured matrices coming from images), the
        # dia format might be best suited for matvec products:
        n_diags = np.unique(laplacian.row - laplacian.col).size
        if n_diags <= 7:
            # 3 or less outer diagonals on each side
            laplacian = laplacian.todia()
        else:
            # csr has the fastest matvec and is thus best suited to
            # arpack
            laplacian = laplacian.tocsr()
    return laplacian






def spectral_clustering(init, max_svd_restarts=30, n_iter_max=30, random_state=None, copy=True):
    """
    Search for a partition matrix (clustering) which is closest to the eigenvector embedding.

    Parameters
    ----------
    init : array-like of shape (n_samples, n_clusters)
        The embedding space of the samples.
    max_svd_restarts : int, default=30
        Maximum number of attempts to restart SVD if convergence fails
    n_iter_max : int, default=30
        Maximum number of iterations to attempt in rotation and partition
        matrix search if machine precision convergence is not reached
    random_state : int, RandomState instance, default=None
        Determines random number generation for rotation matrix initialization.
        Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.
    copy : bool, default=True
        Whether to copy vectors, or perform in-place normalization.


    Returns
    -------
    labels : array of integers, shape: n_samples
        The labels of the clusters.

    References
    ----------
    - Multiclass spectral clustering, 2003
      Stella X. Yu, Jianbo Shi
      https://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf
   
   
    Notes
    -----
    The eigenvector embedding is used to iteratively search for the
    closest discrete partition.  First, the eigenvector embedding is
    normalized to the space of partition matrices. An optimal discrete
    partition matrix closest to this normalized embedding multiplied by
    an initial rotation is calculated.  Fixing this discrete partition
    matrix, an optimal rotation matrix is calculated.  These two
    calculations are performed until convergence.  The discrete partition
    matrix is returned as the clustering solution.  Used in spectral
    clustering, this method tends to be faster and more robust to random
    initialization than k-means.
    """

    from scipy.sparse import csc_matrix
    from scipy.linalg import LinAlgError

    vectors = as_float_array(init, copy=copy)

    eps = np.finfo(float).eps
    n_samples, n_components = vectors.shape

    # Normalize the eigenvectors to an equal length of a vector of ones.
    # Reorient the eigenvectors to point in the negative direction with respect
    # to the first element.  This may have to do with constraining the
    # eigenvectors to lie in a specific quadrant to make the discretization
    # search easier.
    norm_ones = np.sqrt(n_samples)
    for i in range(vectors.shape[1]):
        vectors[:, i] = (vectors[:, i] / np.linalg.norm(vectors[:, i])) \
            * norm_ones
    if vectors[0, i] != 0:
        vectors[:, i] = -1 * vectors[:, i] * np.sign(vectors[0, i])

    # Normalize the rows of the eigenvectors.  Samples should lie on the unit
    # hypersphere centered at the origin.  This transforms the samples in the
    # embedding space to the space of partition matrices.
    vectors = vectors / np.sqrt((vectors ** 2).sum(axis=1))[:, np.newaxis]

    svd_restarts = 0
    has_converged = False

    # If there is an exception we try to randomize and rerun SVD again
    # do this max_svd_restarts times.

    while (svd_restarts < max_svd_restarts) and not has_converged:

        # Initialize first column of rotation matrix with a row of the
        # eigenvectors
        rotation = np.zeros((n_components, n_components))
        rotation[:, 0] = vectors[random_state.randint(n_samples), :].T

        # To initialize the rest of the rotation matrix, find the rows
        # of the eigenvectors that are as orthogonal to each other as
        # possible
        c = np.zeros(n_samples)
        for j in range(1, n_components):
            # Accumulate c to ensure row is as orthogonal as possible to
            # previous picks as well as current one
            c += np.abs(np.dot(vectors, rotation[:, j - 1]))
            rotation[:, j] = vectors[c.argmin(), :].T

        last_objective_value = 0.0
        n_iter = 0

        while not has_converged:
            n_iter += 1

            t_discrete = np.dot(vectors, rotation)

            labels = t_discrete.argmax(axis=1)
            vectors_discrete = csc_matrix(
                (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
                shape=(n_samples, n_components))

            t_svd = vectors_discrete.T * vectors

            try:
                U, S, Vh = np.linalg.svd(t_svd)
                svd_restarts += 1
            except LinAlgError:
                print("SVD did not converge, randomizing and trying again")
                break

            ncut_value = 2.0 * (n_samples - S.sum())
            if ((abs(ncut_value - last_objective_value) < eps) or
                    (n_iter > n_iter_max)):
                has_converged = True
            else:
                # otherwise calculate rotation and continue
                last_objective_value = ncut_value
                rotation = np.dot(Vh.T, U.T)

    if not has_converged:
        raise LinAlgError('SVD did not converge')
    return labels
