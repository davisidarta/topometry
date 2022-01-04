from warnings import warn
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import pairwise_distances
from sklearn.utils import as_float_array
from topo.base.dists import pairwise_special_metric, SPECIAL_METRICS
from topo.base.sparse import SPARSE_SPECIAL_METRICS, sparse_named_distances

#from fastlapmap.similarities import fuzzy_simplicial_set_ann, cknn_graph, diffusion_harmonics

def LapEigenmap(W,
                n_eigs=10,
                norm_laplacian=True,
                eigen_tol=10e-4,
                return_evals=False):
    """
    Performs [Laplacian Eigenmaps](https://www2.imm.dtu.dk/projects/manifold/Papers/Laplacian.pdf) on the input data.

    ----------
    Parameters
    ----------

    `W` : numpy.ndarray, pandas.DataFrame or scipy.sparse.csr_matrix.
         Affinity or adjacency matrix.

    `n_eigs` : int (optional, default 10).
         Number of eigenvectors to decompose the graph Laplacian into.

    `norm_laplacian` : bool (optional, default True).
        Whether to renormalize the graph Laplacian.

    `return_evals` : bool (optional, default False).
        Whether to also return the eigenvalues in a tuple of eigenvectors, eigenvalues. Defaults to False.


    ----------
    Returns
    ----------
        * If return_evals is True :
            A tuple of eigenvectors and eigenvalues.
        * If return_evals is False :
            An array of ranked eigenvectors.

    """

    if isinstance(W, sparse.csr_matrix):
        pass
    elif isinstance(W, np.ndarray):
        W = sparse.csr_matrix(W)
    elif isinstance(W, pd.DataFrame):
        W = W.to_numpy()
        W = sparse.csr_matrix(W)
    else:
        return print('Data should be a numpy.ndarray,pandas.DataFrame or'
                     'a scipy.sparse.csr_matrix for obtaining approximate nearest neighbors with \'nmslib\'.')

    laplacian, dd = sparse.csgraph.laplacian(W, normed=norm_laplacian, return_diag=True)
    laplacian = _set_diag(laplacian, 1, norm_laplacian)
    laplacian *= -1
    n_eigs = n_eigs + 1
    evals, evecs = sparse.linalg.eigsh(laplacian, k=n_eigs, which='LM', sigma=1.0, tol=eigen_tol)
    evecs = evecs.T[n_eigs::-1]
    if norm_laplacian:
        # recover u = D^-1/2 x from the eigenvector output x
        evecs = evecs / dd
    evecs = evecs[1:n_eigs].T

    if return_evals:
        return evecs, evals
    else:
        return evecs




def _set_diag(laplacian, value, norm_laplacian):
    """Set the diagonal of the laplacian matrix and convert it to a
    sparse format well suited for eigenvalue decomposition.
    Parameters
    ----------
    laplacian : {ndarray, sparse matrix}
        The graph laplacian.
    value : float
        The value of the diagonal.
    norm_laplacian : bool
        Whether the value of the diagonal should be changed or not.
    Returns
    -------
    laplacian : {array, sparse matrix}
        An array of matrix in a form that is well suited to fast
        eigenvalue decomposition, depending on the band width of the
        matrix.
    """
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

def spectral_decomposition(affinity_matrix, n_eigs, expand=False):
    N = np.shape(affinity_matrix)[0]
    D, V = sparse.linalg.eigsh(affinity_matrix, n_eigs, tol=1e-4, maxiter=(N // 10))
    D = np.real(D)
    V = np.real(V)
    inds = np.argsort(D)[::-1]
    D = D[inds]
    V = V[:, inds]
    # Normalize by the first diffusion component
    for i in range(V.shape[1]):
        V[:, i] = V[:, i] / np.linalg.norm(V[:, i])
    vals = np.array(V)
    pos = np.sum(vals > 0, axis=0)
    residual = np.sum(vals < 0, axis=0)

    if expand and len(residual) < 1:
        # expand eigendecomposition
        target = n_eigs + 30
        while residual < 3:
            while target < 3 * n_eigs:
                print('Eigengap not found for determined number of components. Expanding eigendecomposition to '
                      + str(target) + 'components.')
                D, V = sparse.linalg.eigsh(affinity_matrix, target, tol=1e-4, maxiter=(N // 10))
                D = np.real(D)
                V = np.real(V)
                inds = np.argsort(D)[::-1]
                D = D[inds]
                V = V[:, inds]
                # Normalize by the first diffusion component
                vals = np.array(V)
                for i in range(V.shape[1]):
                    vals[:, i] = vals[:, i] / np.linalg.norm(vals[:, i])
                pos = np.sum(vals > 0, axis=0)
                target = int(target * 1.6)
                residual = np.sum(vals < 0, axis=0)

            if residual < 1:
                print('Could not find an eigengap! Consider increasing `n_neighbors` or `n_eigs` !'
                      ' Falling back to `eigen_expansion=False`, will not attempt')
                expand = False
    if expand:
        if len(residual) > 30:
            target = n_eigs - 15
            while len(residual) > 29:
                D, V = sparse.linalg.eigsh(affinity_matrix, target, tol=1e-4, maxiter=(N // 10))
                D = np.real(D)
                V = np.real(V)
                inds = np.argsort(D)[::-1]
                D = D[inds]
                V = V[:, inds]
                vals = np.array(V)
                for i in range(V.shape[1]):
                    vals[:, i] = vals[:, i] / np.linalg.norm(vals[:, i])
                pos = np.sum(vals > 0, axis=0)
                residual = np.sum(vals < 0, axis=0)
                if len(residual) < 15:
                    break
                else:
                    target = pos - int(residual // 2)

            if len(residual) < 1:
                print('Could not find an eigengap! Consider increasing `n_neighbors` or `n_eigs` !'
                      ' Falling back to `eigen_expansion=False`, will not attempt eigendecomposition expansion.')
                expand = False

    if not expand:
        D, V = sparse.linalg.eigsh(affinity_matrix, n_eigs, tol=1e-4, maxiter=(N // 10))
        D = np.real(D)
        V = np.real(V)
        inds = np.argsort(D)[::-1]
        D = D[inds]
        V = V[:, inds]

    # Normalize by the first eigencomponent
    for i in range(V.shape[1]):
        V[:, i] = V[:, i] / np.linalg.norm(V[:, i])

    # Normalize eigenvalues
    D = D / D.max()

    return V, D



def component_layout(
        W,
        n_components,
        component_labels,
        dim,
        norm_laplacian=True,
        eigen_tol=10e-4
):
    """Provide a layout relating the separate connected components. This is done
    by taking the centroid of each component and then performing a spectral embedding
    of the centroids.
    Parameters
    ----------
    W: numpy.ndarray, pandas.DataFrame or scipy.sparse.csr_matrix.
         Affinity or adjacency matrix.
    n_components: int
        The number of distinct components to be layed out.
    component_labels: array of shape (n_samples)
        For each vertex in the graph the label of the component to
        which the vertex belongs.
    dim: int
        The chosen embedding dimension.

    Returns
    -------
    component_embedding: array of shape (n_components, dim)
        The ``dim``-dimensional embedding of the ``n_components``-many
        connected components.
    """

    # cannot compute centroids from precomputed distances
    # instead, compute centroid distances using linkage
    distance_matrix = np.zeros((n_components, n_components), dtype=np.float64)
    linkage = np.min

    for c_i in range(n_components):
        dm_i = W[component_labels == c_i]
        for c_j in range(c_i + 1, n_components):
            dist = linkage(dm_i[:, component_labels == c_j])
            distance_matrix[c_i, c_j] = dist
            distance_matrix[c_j, c_i] = dist

    affinity_matrix = np.exp(-(distance_matrix ** 2))

    component_embedding = LapEigenmap(W=affinity_matrix, n_eigs=dim, norm_laplacian=norm_laplacian, eigen_tol=eigen_tol)
    component_embedding /= component_embedding.max()

    return component_embedding


def multi_component_layout(
    graph,
    n_components,
    component_labels,
    dim,
    random_state
):
    """Specialised layout algorithm for dealing with graphs with many connected components.
    This will first find relative positions for the components by spectrally embedding
    their centroids, then spectrally embed each individual connected component positioning
    them according to the centroid embeddings. This provides a decent embedding of each
    component while placing the components in good relative positions to one another.
    Parameters
    ----------
    graph: sparse matrix
        The adjacency matrix of the graph to be embedded.
    n_components: int
        The number of distinct components to be layed out.
    component_labels: array of shape (n_samples)
        For each vertex in the graph the label of the component to
        which the vertex belongs.
    dim: int
        The chosen embedding dimension.

    Returns
    -------
    embedding: array of shape (n_samples, dim)
        The initial embedding of ``graph``.
    """

    result = np.empty((graph.shape[0], dim), dtype=np.float32)

    if n_components > 2 * dim:
        meta_embedding = component_layout(
            graph,
            n_components,
            component_labels,
            dim,
        )
    else:
        k = int(np.ceil(n_components / 2.0))
        base = np.hstack([np.eye(k), np.zeros((k, dim - k))])
        meta_embedding = np.vstack([base, -base])[:n_components]

    for label in range(n_components):
        component_graph = graph.tocsr()[component_labels == label, :].tocsc()
        component_graph = component_graph[:, component_labels == label].tocoo()

        distances = pairwise_distances([meta_embedding[label]], meta_embedding)
        data_range = distances[distances > 0.0].min() / 2.0

        if component_graph.shape[0] < 2 * dim:
            result[component_labels == label] = (
                random_state.uniform(
                    low=-data_range,
                    high=data_range,
                    size=(component_graph.shape[0], dim),
                )
                + meta_embedding[label]
            )
            continue

        diag_data = np.asarray(component_graph.sum(axis=0))
        # standard Laplacian
        # D = scipy.sparse.spdiags(diag_data, 0, graph.shape[0], graph.shape[0])
        # L = D - graph
        # Normalized Laplacian
        I = sparse.identity(component_graph.shape[0], dtype=np.float64)
        D = sparse.spdiags(
            1.0 / (np.sqrt(diag_data)+10e-6),
            0,
            component_graph.shape[0],
            component_graph.shape[0],
        )
        L = I - D * component_graph * D
        k = dim + 1
        num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(component_graph.shape[0])))
        try:
            eigenvalues, eigenvectors = sparse.linalg.eigsh(
                L,
                k,
                which="SM",
                ncv=num_lanczos_vectors,
                tol=1e-4,
                v0=np.ones(L.shape[0]),
                maxiter=graph.shape[0] * 5,
            )
            order = np.argsort(eigenvalues)[1:k]
            component_embedding = eigenvectors[:, order]
            expansion = data_range / np.max(np.abs(component_embedding))
            component_embedding *= expansion
            result[component_labels == label] = (
                component_embedding + meta_embedding[label]
            )
        except sparse.linalg.ArpackError:
            warn(
                "WARNING: spectral initialisation failed! The eigenvector solver\n"
                "failed. This is likely due to too small an eigengap. Consider\n"
                "adding some noise or jitter to your data.\n\n"
                "Falling back to random initialisation!"
            )
            result[component_labels == label] = (
                random_state.uniform(
                    low=-data_range,
                    high=data_range,
                    size=(component_graph.shape[0], dim),
                )
                + meta_embedding[label]
            )

    return result


def spectral_layout(graph, dim, random_state):
    """Given a graph compute the spectral embedding of the graph. This is
    simply the eigenvectors of the laplacian of the graph. Here we use the
    normalized laplacian.

    Parameters
    ----------
    graph: sparse matrix
        The (weighted) adjacency matrix of the graph as a sparse matrix.
    dim: int
        The dimension of the space into which to embed.
    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.

    Returns
    -------
    embedding: array of shape (n_vertices, dim)
        The spectral embedding of the graph.
    """
    n_components, labels = sparse.csgraph.connected_components(graph)

    if n_components > 1:
        return multi_component_layout(
            graph,
            n_components,
            labels,
            dim,
            random_state,
        )

    diag_data = np.asarray(graph.sum(axis=0))
    # standard Laplacian
    # D = scipy.sparse.spdiags(diag_data, 0, graph.shape[0], graph.shape[0])
    # L = D - graph
    # Normalized Laplacian
    I = sparse.identity(graph.shape[0], dtype=np.float64)
    D = sparse.spdiags(
        1.0 / (np.sqrt(diag_data)+10e-8), 0, graph.shape[0], graph.shape[0]
    )
    L = I - D * graph * D

    k = dim + 1
    num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(graph.shape[0])))
    try:
        if L.shape[0] < 2000000:
            eigenvalues, eigenvectors = sparse.linalg.eigsh(
                L,
                k,
                which="SM",
                ncv=num_lanczos_vectors,
                tol=1e-4,
                v0=np.ones(L.shape[0]),
                maxiter=graph.shape[0] * 5,
            )
        else:
            eigenvalues, eigenvectors = sparse.linalg.lobpcg(
                L, random_state.normal(size=(L.shape[0], k)), largest=False, tol=1e-8
            )
        order = np.argsort(eigenvalues)[1:k]
        return eigenvectors[:, order]
    except sparse.linalg.ArpackError:
        warn(
            "WARNING: spectral initialisation failed! The eigenvector solver\n"
            "failed. This is likely due to too small an eigengap. Consider\n"
            "adding some noise or jitter to your data.\n\n"
            "Falling back to random initialisation!"
        )
        return random_state.uniform(low=-10.0, high=10.0, size=(graph.shape[0], dim))


def spectral_clustering(init, max_svd_restarts=30, n_iter_max=30, random_state=None, copy=True):
    """Search for a partition matrix (clustering) which is closest to the
        eigenvector embedding.

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