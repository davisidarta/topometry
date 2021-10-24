from warnings import warn

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from scipy.sparse.linalg import eigs, eigsh
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import pairwise_distances
from sklearn.utils import as_float_array
from scipy.sparse.csgraph import laplacian as csgraph_laplacian

from topo.base.dists import pairwise_special_metric, SPECIAL_METRICS
from topo.base.sparse import SPARSE_SPECIAL_METRICS, sparse_named_distances


def spectral_decomposition(affinity_matrix, n_eigs, expand=False):
    N = np.shape(affinity_matrix)[0]
    D, V = eigsh(affinity_matrix, n_eigs, tol=1e-4, maxiter=(N // 10))
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
                D, V = eigsh(affinity_matrix, target, tol=1e-4, maxiter=(N // 10))
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
                D, V = eigsh(affinity_matrix, target, tol=1e-4, maxiter=(N // 10))
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
        D, V = eigsh(affinity_matrix, n_eigs, tol=1e-4, maxiter=(N // 10))
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


def LapEigenmap(affinity_matrix, n_eigs, norm_laplacian=True, expand=False, return_evals=True):
    laplacian = csgraph_laplacian(affinity_matrix, normed=norm_laplacian,
                                      return_diag=False)
    V, D = spectral_decomposition(laplacian, n_eigs=n_eigs, expand=expand)
    if return_evals:
        return V, D
    else:
        return V


def component_layout(
    data,
    n_components,
    component_labels,
    dim,
    random_state,
    metric="euclidean",
    metric_kwds={},
    n_jobs=10
):
    """Provide a layout relating the separate connected components. This is done
    by taking the centroid of each component and then performing a spectral embedding
    of the centroids.
    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data -- required so we can generate centroids for each
        connected component of the graph.
    is_dist: bool
        Whether if X is a distance or affinity matrix.
    n_components: int
        The number of distinct components to be layed out.
    component_labels: array of shape (n_samples)
        For each vertex in the graph the label of the component to
        which the vertex belongs.
    dim: int
        The chosen embedding dimension.
    metric: string or callable (optional, default 'euclidean')
        The metric used to measure distances among the source data points.
    metric_kwds: dict (optional, default {})
        Keyword arguments to be passed to the metric function.
        If metric is 'precomputed', 'linkage' keyword can be used to specify
        'average', 'complete', or 'single' linkage. Default is 'single'
    Returns
    -------
    component_embedding: array of shape (n_components, dim)
        The ``dim``-dimensional embedding of the ``n_components``-many
        connected components.
    """

    component_centroids = np.empty((n_components, data.shape[1]), dtype=np.float64)

    if metric == "precomputed":
        # cannot compute centroids from precomputed distances
        # instead, compute centroid distances using linkage
        distance_matrix = np.zeros((n_components, n_components), dtype=np.float64)
        linkage = metric_kwds.get("linkage", "single")
        if linkage == "average":
            linkage = np.mean
        elif linkage == "complete":
            linkage = np.max
        elif linkage == "single":
            linkage = np.min
        else:
            raise ValueError(
                "Unrecognized linkage '%s'. Please choose from "
                "'average', 'complete', or 'single'" % linkage
            )
        for c_i in range(n_components):
            dm_i = data[component_labels == c_i]
            for c_j in range(c_i + 1, n_components):
                dist = linkage(dm_i[:, component_labels == c_j])
                distance_matrix[c_i, c_j] = dist
                distance_matrix[c_j, c_i] = dist
    else:
        for label in range(n_components):
            component_centroids[label] = data[component_labels == label].mean(axis=0)

        if scipy.sparse.isspmatrix(component_centroids):
            warn(
                "Forcing component centroids to dense; if you are running out of "
                "memory then consider increasing n_neighbors."
            )
            component_centroids = component_centroids.toarray()

        if metric in SPECIAL_METRICS:
            distance_matrix = pairwise_special_metric(
                component_centroids, metric=metric
            )
        elif metric in SPARSE_SPECIAL_METRICS:
            distance_matrix = pairwise_special_metric(
                component_centroids, metric=SPARSE_SPECIAL_METRICS[metric]
            )
        else:
            if callable(
                metric
            ) and scipy.sparse.isspmatrix(data):
                function_to_name_mapping = {
                    v: k for k, v in sparse_named_distances.items()
                }
                try:
                    metric_name = function_to_name_mapping[metric]
                except KeyError:
                    raise NotImplementedError(
                        "Multicomponent layout for custom "
                        "sparse metrics is not implemented at "
                        "this time."
                    )
                distance_matrix = pairwise_distances(
                    component_centroids, metric=metric_name, **metric_kwds
                )
            else:
                distance_matrix = pairwise_distances(
                    component_centroids, metric=metric, **metric_kwds
                )

    affinity_matrix = np.exp(-(distance_matrix ** 2))

    component_embedding = SpectralEmbedding(
        n_components=dim, affinity="precomputed", random_state=random_state, n_jobs=n_jobs
    ).fit_transform(affinity_matrix)
    component_embedding /= component_embedding.max()

    return component_embedding


def multi_component_layout(
    data,
    graph,
    n_components,
    component_labels,
    dim,
    random_state,
    metric="euclidean",
    metric_kwds={},
    n_jobs=10
):
    """Specialised layout algorithm for dealing with graphs with many connected components.
    This will first find relative positions for the components by spectrally embedding
    their centroids, then spectrally embed each individual connected component positioning
    them according to the centroid embeddings. This provides a decent embedding of each
    component while placing the components in good relative positions to one another.
    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data -- required so we can generate centroids for each
        connected component of the graph.
    graph: sparse matrix
        The adjacency matrix of the graph to be embedded.
    n_components: int
        The number of distinct components to be layed out.
    component_labels: array of shape (n_samples)
        For each vertex in the graph the label of the component to
        which the vertex belongs.
    dim: int
        The chosen embedding dimension.
    metric: string or callable (optional, default 'euclidean')
        The metric used to measure distances among the source data points.
    metric_kwds: dict (optional, default {})
        Keyword arguments to be passed to the metric function.
    Returns
    -------
    embedding: array of shape (n_samples, dim)
        The initial embedding of ``graph``.
    """

    result = np.empty((graph.shape[0], dim), dtype=np.float32)

    if n_components > 2 * dim:
        meta_embedding = component_layout(
            data,
            n_components,
            component_labels,
            dim,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
            n_jobs=n_jobs
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
        I = scipy.sparse.identity(component_graph.shape[0], dtype=np.float64)
        D = scipy.sparse.spdiags(
            1.0 / (np.sqrt(diag_data)+10e-6),
            0,
            component_graph.shape[0],
            component_graph.shape[0],
        )
        L = I - D * component_graph * D

        k = dim + 1
        num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(component_graph.shape[0])))
        try:
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
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
        except scipy.sparse.linalg.ArpackError:
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


def spectral_layout(data, graph, dim, random_state, metric="euclidean", metric_kwds={}):
    """Given a graph compute the spectral embedding of the graph. This is
    simply the eigenvectors of the laplacian of the graph. Here we use the
    normalized laplacian.
    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data
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
    n_components, labels = scipy.sparse.csgraph.connected_components(graph)

    if n_components > 1:
        return multi_component_layout(
            data,
            graph,
            n_components,
            labels,
            dim,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )

    diag_data = np.asarray(graph.sum(axis=0))
    # standard Laplacian
    # D = scipy.sparse.spdiags(diag_data, 0, graph.shape[0], graph.shape[0])
    # L = D - graph
    # Normalized Laplacian
    I = scipy.sparse.identity(graph.shape[0], dtype=np.float64)
    D = scipy.sparse.spdiags(
        1.0 / (np.sqrt(diag_data)+10e-8), 0, graph.shape[0], graph.shape[0]
    )
    L = I - D * graph * D

    k = dim + 1
    num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(graph.shape[0])))
    try:
        if L.shape[0] < 2000000:
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                L,
                k,
                which="SM",
                ncv=num_lanczos_vectors,
                tol=1e-4,
                v0=np.ones(L.shape[0]),
                maxiter=graph.shape[0] * 5,
            )
        else:
            eigenvalues, eigenvectors = scipy.sparse.linalg.lobpcg(
                L, random_state.normal(size=(L.shape[0], k)), largest=False, tol=1e-8
            )
        order = np.argsort(eigenvalues)[1:k]
        return eigenvectors[:, order]
    except scipy.sparse.linalg.ArpackError:
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