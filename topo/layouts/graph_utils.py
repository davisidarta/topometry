# These are some graph learning functions implemented in UMAP, added here with modifications
# for better speed and computational efficiency.
# Originally implemented by Leland McInnes at https://github.com/lmcinnes/umap
# License: BSD 3 clause
#
# For more information on the original UMAP implementation, please see: https://umap-learn.readthedocs.io/
#
# BSD 3-Clause License
#
# Copyright (c) 2017, Leland McInnes
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors

from topo.base import ann
from topo.base import dists as dist
from topo.spectral import _spectral, umap_layouts
from topo.tpgraph import diffusion
from topo.utils import umap_utils

ts = umap_utils.ts
csr_unique = umap_utils.csr_unique
fast_knn_indices = umap_utils.fast_knn_indices


optimize_layout_euclidean = umap_layouts.optimize_layout_euclidean
optimize_layout_generic = umap_layouts.optimize_layout_generic
optimize_layout_inverse = umap_layouts.optimize_layout_inverse

try:
    import joblib
except ImportError:
    # sklearn.externals.joblib is deprecated in 0.21, will be removed in 0.23. Try installing joblib.
    from sklearn.externals import joblib

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf
INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


def fuzzy_simplicial_set_ann(
        X,
        n_neighbors=15,
        knn_indices=None,
        knn_dists=None,
        backend='hnswlib',
        metric='cosine',
        n_jobs=None,
        efC=50,
        efS=50,
        M=15,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        apply_set_operations=True,
        return_dists=False,
        verbose=False):
    """Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.
    Parameters
    ----------
    X : array of shape (n_samples, n_features).
        The data to be modelled as a fuzzy simplicial set.

    n_neighbors : int.
        The number of neighbors to use to approximate geodesic distance.
        Larger numbers induce more global estimates of the manifold that can
        miss finer detail, while smaller values will focus on fine manifold
        structure to the detriment of the larger picture.

    backend : str (optional, default 'hnwslib').
        Which backend to use to compute nearest-neighbors. Options for fast, approximate nearest-neighbors
        are 'hnwslib' (default) and 'nmslib'. For exact nearest-neighbors, use 'sklearn'.

    metric : str (optional, default 'cosine').
        Distance metric for building an approximate kNN graph. Defaults to
        'cosine'. Users are encouraged to explore different metrics, such as 'euclidean' and 'inner_product'.
        The 'hamming' and 'jaccard' distances are also available for string vectors.
         Accepted metrics include NMSLib*, HNSWlib** and sklearn metrics. Some examples are:

        -'sqeuclidean' (*, **)

        -'euclidean' (*, **)

        -'l1' (*)

        -'lp' - requires setting the parameter ``p`` (*) - similar to Minkowski

        -'cosine' (*, **)

        -'inner_product' (**)

        -'angular' (*)

        -'negdotprod' (*)

        -'levenshtein' (*)

        -'hamming' (*)

        -'jaccard' (*)

        -'jansen-shan' (*).

    n_jobs : int (optional, default 1).
        number of threads to be used in computation. Defaults to 1. The algorithm is highly
        scalable to multi-threading.

    M : int (optional, default 30).
        defines the maximum number of neighbors in the zero and above-zero layers during HSNW
        (Hierarchical Navigable Small World Graph). However, the actual default maximum number
        of neighbors for the zero layer is 2*M.  A reasonable range for this parameter
        is 5-100. For more information on HSNW, please check https://arxiv.org/abs/1603.09320.
        HSNW is implemented in python via NMSlib. Please check more about NMSlib at https://github.com/nmslib/nmslib.

    efC : int (optional, default 100).
        A 'hnsw' parameter. Increasing this value improves the quality of a constructed graph
        and leads to higher accuracy of search. However this also leads to longer indexing times.
        A reasonable range for this parameter is 50-2000.

    efS : int (optional, default 100).
        A 'hnsw' parameter. Similarly to efC, increasing this value improves recall at the
        expense of longer retrieval time. A reasonable range for this parameter is 50-2000.

    knn_indices : array of shape (n_samples, n_neighbors) (optional).
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the indices of the k-nearest neighbors as a row for
        each data point.

    knn_dists : array of shape (n_samples, n_neighbors) (optional).
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the distances of the k-nearest neighbors as a row for
        each data point.

    set_op_mix_ratio : float (optional, default 1.0).
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial sets. Both fuzzy set operations use the product t-norm.
        The value of this parameter should be between 0.0 and 1.0; a value of
        1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
        intersection.

    local_connectivity : int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    verbose : bool (optional, default False)
        Whether to report information on the current progress of the algorithm.

    return_dists : bool or None (optional, default none)
        Whether to return the pairwise distance associated with each edge.

    Returns
    -------
    fuzzy_simplicial_set : coo_matrix
        A fuzzy simplicial set represented as a sparse matrix. The (i,
        j) entry of the matrix represents the membership strength of the
        1-simplex between the ith and jth sample points.
    """
    if knn_indices is None or knn_dists is None:
        if verbose:
            print('Running fast approximate nearest neighbors with NMSLIB using HNSW...')
        if metric not in ['sqeuclidean',
                                 'euclidean',
                                 'l1',
                                 'cosine',
                                 'angular',
                                 'negdotprod',
                                 'levenshtein',
                                 'hamming',
                                 'jaccard',
                                 'jansen-shan']:
            print('Please input a metric compatible with NMSLIB when use_nmslib is set to True')
        knn_indices, knn_dists = approximate_n_neighbors(X,
                                                         n_neighbors=n_neighbors,
                                                         metric=metric,
                                                         backend=backend,
                                                         n_jobs=n_jobs,
                                                         efC=efC,
                                                         efS=efS,
                                                         M=M,
                                                         verbose=verbose)

    knn_dists = knn_dists.astype(np.float32)
    knn_dists = knn_dists

    sigmas, rhos = smooth_knn_dist(
        knn_dists, float(n_neighbors), local_connectivity=float(local_connectivity),
    )

    rows, cols, vals = compute_membership_strengths(
        knn_indices, knn_dists, sigmas, rhos
    )

    result = coo_matrix(
        (vals, (rows, cols)), shape=(X.shape[0], X.shape[0])
    )
    result.eliminate_zeros()

    if apply_set_operations:
        transpose = result.transpose()

        prod_matrix = result.multiply(transpose)

        result = (
                set_op_mix_ratio * (result + transpose - prod_matrix)
                + (1.0 - set_op_mix_ratio) * prod_matrix
        )

    result.eliminate_zeros()
    if return_dists:
        return result, sigmas, rhos, knn_dists
    else:
        return result, sigmas, rhos


def compute_diff_connectivities(
        data,
        n_components=100,
        n_neighbors=30,
        alpha=0.0,
        n_jobs=10,
        ann=True,
        ann_dist='cosine',
        M=30,
        efC=100,
        efS=100,
        knn_dist='euclidean',
        kernel_use='simple_adaptive',
        sensitivity=1,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        verbose=False

):
    """
        Sklearn estimator for using fast anisotropic diffusion with an anisotropic
        adaptive algorithm as proposed by Setty et al, 2018, and optimized by Sidarta-Oliveira, 2020.
        This procedure generates diffusion components that effectivelly carry the maximum amount of
        information regarding the data geometric structure (structure components).
        These structure components then undergo a fuzzy-union of simplicial sets. This step is
        from umap.fuzzy_simplicial_set [McInnes18]_. Given a set of data X, a neighborhood size, and a measure of distance
        compute the fuzzy simplicial set (here represented as a fuzzy graph in the form of a sparse matrix) associated to the data. This is done by
        locally approximating geodesic distance at each point, creating a fuzzy simplicial set for each such point, and then combining all the local
        fuzzy simplicial sets into a global one via a fuzzy union.

    Parameters
    ----------
    n_components : Number of diffusion components to compute. Defaults to 100. We suggest larger values if
                   analyzing more than 10,000 cells.

    n_neighbors : Number of k-nearest-neighbors to compute. The adaptive kernel will normalize distances by each cell
                  distance of its median neighbor.

    knn_dist : Distance metric for building kNN graph. Defaults to 'euclidean'. 

    ann : Boolean. Whether to use approximate nearest neighbors for graph construction. Defaults to True.

    alpha : Alpha in the diffusion maps literature. Controls how much the results are biased by data distribution.
            Defaults to 1, which is suitable for normalized data.

    n_jobs : Number of threads to use in calculations. Defaults to all but one.

    sensitivity : Sensitivity to select eigenvectors if diff_normalization is set to 'knee'. Useful when dealing wit

    :returns: Diffusion components ['EigenVectors'], associated eigenvalues ['EigenValues'] and suggested number of
             resulting components to use during Multiscaling.

    Example
    -------------

    import numpy as np
    from sklearn.datasets import load_digits
    from scipy.sparse import csr_matrix
    from topo.tpgraph.diffusion import Diffusor

    # Load the MNIST digits data, convert to sparse for speed
    digits = load_digits()
    data = csr_matrix(digits)

    # Fit the anisotropic diffusion process
    tpgraph = Diffusor().fit(data)

    # Find multiscale diffusion components
    mds = tpgraph.transform(data)



    """
    from scipy.sparse import csr_matrix, issparse
    if issparse(data) == True:
        if verbose:
            print('Sparse input. Proceding without converting...')
        if isinstance(data, np.ndarray):
            data = csr_matrix(data)
    if issparse(data) == False:
        if verbose:
            print('Converting input to sparse...')
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            data = csr_matrix(data.values.T)
    elif isinstance(data, np.ndarray):
        data = data

    diff = diffusion.Diffusor(
        n_components=n_components,
        n_neighbors=n_neighbors,
        alpha=alpha,
        n_jobs=n_jobs,
        ann=ann,
        ann_dist=ann_dist,
        M=M,
        efC=efC,
        efS=efS,
        knn_dist=knn_dist,
        kernel_use=kernel_use,
        sensitivity=sensitivity).fit(data)
    knn_indices, knn_distances, knn_grad, knn_graph = diff.ind_dist_grad(data, dense=False)

    if not issparse(knn_graph):
        knn_graph = csr_matrix(knn_graph)

    distances, connectivities, sigmas, rhos = fuzzy_simplicial_set_ann(
        knn_graph,
        n_neighbors,
        knn_indices=knn_indices,
        knn_dists=knn_distances,
        nmslib_metric=ann_dist,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
        verbose=verbose
    )
    if isinstance(connectivities, tuple):
        # In umap-learn 0.4, this returns (result, sigmas, rhos)
        connectivities = connectivities[0]
    if issparse(connectivities):
        if isinstance(connectivities, np.ndarray):
            connectivities = csr_matrix(connectivities)
        else:
            connectivities = connectivities.tocsr()

    return knn_graph, connectivities


def get_sparse_matrix_from_indices_distances_dbmap(knn_indices, knn_dists, n_obs, n_neighbors):
    rows = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    cols = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    vals = np.zeros((n_obs * n_neighbors), dtype=np.float64)

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


def approximate_n_neighbors(data,
                            n_neighbors=15,
                            metric='cosine',
                            backend='hnswlib',
                            n_jobs=10,
                            efC=50,
                            efS=50,
                            M=15,
                            p=11/16,
                            dense=False,
                            verbose=False
                            ):
    """
    Simple function using NMSlibTransformer from topodata.ann. This implements a very fast
    and scalable approximate k-nearest-neighbors graph on spaces defined by nmslib.
    Read more about nmslib and its various available metrics at
    https://github.com/nmslib/nmslib. Read more about dbMAP at
    https://github.com/davisidarta/dbMAP.


    Parameters
    ----------
    n_neighbors : number of nearest-neighbors to look for. In practice,
                     this should be considered the average neighborhood size and thus vary depending
                     on your number of features, samples and data intrinsic dimensionality. Reasonable values
                     range from 5 to 100. Smaller values tend to lead to increased graph structure
                     resolution, but users should beware that a too low value may render granulated and vaguely
                     defined neighborhoods that arise as an artifact of downsampling. Defaults to 15. Larger
                     values can slightly increase computational time.

    backend : str (optional, default 'hnwslib')
        Which backend to use to compute nearest-neighbors. Options for fast, approximate nearest-neighbors
        are 'hnwslib' (default) and 'nmslib'. For exact nearest-neighbors, use 'sklearn'.

    metric : str (optional, default 'cosine')
        Distance metric for building an approximate kNN graph. Defaults to
        'cosine'. Users are encouraged to explore different metrics, such as 'euclidean' and 'inner_product'.
        The 'hamming' and 'jaccard' distances are also available for string vectors.
         Accepted metrics include NMSLib*, HNSWlib** and sklearn metrics. Some examples are:

        -'sqeuclidean' (*, **)

        -'euclidean' (*, **)

        -'l1' (*)

        -'lp' - requires setting the parameter ``p`` (*) - similar to Minkowski

        -'cosine' (*, **)

        -'inner_product' (**)

        -'angular' (*)

        -'negdotprod' (*)

        -'levenshtein' (*)

        -'hamming' (*)

        -'jaccard' (*)

        -'jansen-shan' (*)

    p : int or float (optional, default 11/16 )
        P for the Lp metric, when ``metric='lp'``.  Can be fractional. The default 11/16 approximates
        an astroid norm with some computational efficiency (2^n bases are less painstakinly slow to compute).
        See https://en.wikipedia.org/wiki/Lp_space for some context.

    n_jobs : number of threads to be used in computation. Defaults to 10 (~5 cores).

    efC : increasing this value improves the quality of a constructed graph and leads to higher
             accuracy of search. However this also leads to longer indexing times. A reasonable
             range is 100-2000. Defaults to 100.

    efS : similarly to efC, improving this value improves recall at the expense of longer
             retrieval time. A reasonable range is 100-2000.

    M : defines the maximum number of neighbors in the zero and above-zero layers during HSNW
           (Hierarchical Navigable Small World Graph). However, the actual default maximum number
           of neighbors for the zero layer is 2*M. For more information on HSNW, please check
           https://arxiv.org/abs/1603.09320. HSNW is implemented in python via NMSLIB. Please check
           more about NMSLIB at https://github.com/nmslib/nmslib .

    Returns
    -------------
     k-nearest-neighbors indices and distances. Can be customized to also return
        return the k-nearest-neighbors graph and its gradient.

    Example
    -------------

    knn_indices, knn_dists = approximate_n_neighbors(data)


    """
    if backend == 'hnswlib' and metric not in ['euclidean', 'sqeuclidean', 'cosine', 'inner_product']:
        if verbose:
            print('Metric ' + str(
                metric) + ' not compatible with \'hnslib\' backend. Changing to \'nmslib\' backend.')
        backend = 'nmslib'
    if backend == 'nmslib':
        # Construct an approximate k-nearest-neighbors graph
        anbrs = ann.NMSlibTransformer(n_neighbors=n_neighbors,
                                      metric=metric,
                                      p=p,
                                      method='hnsw',
                                      n_jobs=n_jobs,
                                      M=M,
                                      efC=efC,
                                      efS=efS,
                                      dense=dense,
                                      verbose=verbose).fit(data)
        knn_inds, knn_distances, grad, knn_graph = anbrs.ind_dist_grad(data)
    elif backend == 'hnwslib':
        anbrs = ann.HNSWlibTransformer(n_neighbors=n_neighbors,
                                       metric=metric,
                                       n_jobs=n_jobs,
                                       M=M,
                                       efC=efC,
                                       efS=efS,
                                       verbose=verbose).fit(data)
        knn_inds, knn_distances, grad, knn_graph = anbrs.ind_dist_grad(data)
    else:
        # Construct a k-nearest-neighbors graph
        nbrs = NearestNeighbors(n_neighbors=int(n_neighbors), metric=metric, n_jobs=n_jobs).fit(
            data)
        knn_distances, knn_inds = nbrs.kneighbors(data)

    return knn_inds, knn_distances


def compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos):
    """Construct the membership strength data for the 1-skeleton of each local
    fuzzy simplicial set -- this is formed as a sparse matrix where each row is
    a local fuzzy simplicial set, with a membership strength for the
    1-simplex to each other data point.
    Parameters
    ----------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.
    knn_dists: array of shape (n_samples, n_neighbors)
        The distances to the ``n_neighbors`` closest points in the dataset.
    sigmas: array of shape(n_samples)
        The normalization factor derived from the metric tensor approximation.
    rhos: array of shape(n_samples)
        The local connectivity adjustment.
    Returns
    -------
    rows: array of shape (n_samples * n_neighbors)
        Row data for the resulting sparse matrix (coo format)
    cols: array of shape (n_samples * n_neighbors)
        Column data for the resulting sparse matrix (coo format)
    vals: array of shape (n_samples * n_neighbors)
        Entries for the resulting sparse matrix (coo format)
    """
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float32)

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    return rows, cols, vals


def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0, bandwidth=1.0):
    """Compute a continuous version of the distance to the kth nearest
    neighbor. That is, this is similar to knn-distance but allows continuous
    k values rather than requiring an integral k. In essence we are simply
    computing the distance such that the cardinality of fuzzy set we generate
    is k.
    Parameters
    ----------
    distances: array of shape (n_samples, n_neighbors)
        Distances to nearest neighbors for each samples. Each row should be a
        sorted list of distances to a given samples nearest neighbors.
    k: float
        The number of nearest neighbors to approximate for.
    n_iter: int (optional, default 64)
        We need to binary search for the correct distance value. This is the
        max number of iterations to use in such a search.
    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.
    bandwidth: float (optional, default 1)
        The target bandwidth of the kernel, larger values will produce
        larger return values.
    Returns
    -------
    knn_dist: array of shape (n_samples,)
        The distance to kth nearest neighbor, as suitably approximated.
    nn_dist: array of shape (n_samples,)
        The distance to the 1st nearest neighbor for each point.
    """
    target = np.log2(k) * bandwidth
    rho = np.zeros(distances.shape[0], dtype=np.float32)
    result = np.zeros(distances.shape[0], dtype=np.float32)

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        # TODO: This is very inefficient, but will do for now. FIXME
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho[i] += interpolation * (
                        non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0

            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        result[i] = mid

        # TODO: This is very inefficient, but will do for now. FIXME
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if result[i] < MIN_K_DIST_SCALE * mean_ith_distances:
                result[i] = MIN_K_DIST_SCALE * mean_ith_distances
        else:
            if result[i] < MIN_K_DIST_SCALE * mean_distances:
                result[i] = MIN_K_DIST_SCALE * mean_distances

    return result, rho


def get_igraph_from_adjacency(adjacency, directed=None):
    """Get igraph graph from adjacency matrix."""
    import igraph as ig

    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])  # this adds adjacency.shape[0] vertices
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es['weight'] = weights
    except:
        pass
    if g.vcount() != adjacency.shape[0]:
        print(
            f'The constructed graph has only {g.vcount()} nodes. '
            'Your adjacency matrix contained redundant nodes.'
        )
    return g

def make_epochs_per_sample(weights, n_epochs):
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.
    Parameters
    ----------
    weights: array of shape (n_1_simplices)
        The weights ofhow much we wish to sample each 1-simplex.
    n_epochs: int
        The total number of epochs we want to train for.
    Returns
    -------
    An array of number of epochs per sample, one for each 1-simplex.
    """
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / n_samples[n_samples > 0]
    return result


def simplicial_set_embedding(
    graph,
    n_components,
    initial_alpha,
    a,
    b,
    gamma,
    negative_sample_rate,
    n_epochs,
    init,
    random_state,
    metric,
    metric_kwds,
    densmap,
    densmap_kwds,
    output_dens,
    output_metric=dist.named_distances_with_gradients["euclidean"],
    output_metric_kwds={},
    euclidean_output=True,
    parallel=False,
    verbose=False,
):
    """Perform a fuzzy simplicial set embedding, using a specified
    initialisation method and then minimizing the fuzzy set cross entropy
    between the 1-skeletons of the high and low dimensional fuzzy simplicial
    sets.
    Parameters
    ----------
    graph: sparse matrix
        The 1-skeleton of the high dimensional fuzzy simplicial set as
        represented by a graph for which we require a sparse matrix for the
        (weighted) adjacency matrix.
    n_components: int
        The dimensionality of the euclidean space into which to embed the data.
    initial_alpha: float
        Initial learning rate for the SGD.
    a: float
        Parameter of differentiable approximation of right adjoint functor
    b: float
        Parameter of differentiable approximation of right adjoint functor
    gamma: float
        Weight to apply to negative samples.
    negative_sample_rate: int (optional, default 5)
        The number of negative samples to select per positive sample
        in the optimization process. Increasing this value will result
        in greater repulsive force being applied, greater optimization
        cost, but slightly more accuracy.
    n_epochs: int (optional, default 0)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If 0 is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).
    init: string
        How to initialize the low dimensional embedding. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * A numpy array of initial embedding positions.
    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.
    metric: string or callable
        The metric used to measure distance in high dimensional space; used if
        multiple connected components need to be layed out.
    metric_kwds: dict
        Key word arguments to be passed to the metric function; used if
        multiple connected components need to be layed out.
    densmap: bool
        Whether to use the density-augmented objective function to optimize
        the embedding according to the densMAP algorithm.
    densmap_kwds: dict
        Key word arguments to be used by the densMAP optimization.
    output_dens: bool
        Whether to output local radii in the original data and the embedding.
    output_metric: function
        Function returning the distance between two points in embedding space and
        the gradient of the distance wrt the first argument.
    output_metric_kwds: dict
        Key word arguments to be passed to the output_metric function.
    euclidean_output: bool
        Whether to use the faster code specialised for euclidean output metrics
    parallel: bool (optional, default False)
        Whether to run the computation using numba parallel.
        Running in parallel is non-deterministic, and is not used
        if a random seed has been set, to ensure reproducibility.
    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.
    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized of ``graph`` into an ``n_components`` dimensional
        euclidean space.
    aux_data: dict
        Auxiliary output returned with the embedding. When densMAP extension
        is turned on, this dictionary includes local radii in the original
        data (``rad_orig``) and in the embedding (``rad_emb``).
    """
    graph = graph.tocoo()
    graph.sum_duplicates()
    n_vertices = graph.shape[1]
    if (n_epochs <= 0) or (n_epochs is None):
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 1000
        else:
            n_epochs = 300

        # Use more epochs for densMAP
        if densmap:
            n_epochs += 200

    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()

    if isinstance(init, np.ndarray):
        initialisation = init
        embedding = init
    elif isinstance(init, str) and init == "random":
        embedding = random_state.uniform(
            low=-10.0, high=10.0, size=(graph.shape[0], n_components)
        ).astype(np.float32)
        initialisation = embedding
    elif isinstance(init, str) and init == "spectral":
        # We add a little noise to avoid local minima for optimization to come
        initialisation = _spectral.spectral_layout(
            graph,
            dim=n_components,
            random_state=random_state
        )
        expansion = 10.0 / np.abs(initialisation).max()
        embedding = (initialisation * expansion).astype(
            np.float32
        ) + random_state.normal(
            scale=0.0001, size=[graph.shape[0], n_components]
        ).astype(
            np.float32
        )
    else:
        init_data = np.array(init)
        if len(init_data.shape) == 2:
            if np.unique(init_data, axis=0).shape[0] < init_data.shape[0]:
                tree = KDTree(init_data)
                dist, ind = tree.query(init_data, k=2)
                nndist = np.mean(dist[:, 1])
                embedding = init_data + random_state.normal(
                    scale=0.001 * nndist, size=init_data.shape
                ).astype(np.float32)
            else:
                embedding = init_data

    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

    head = graph.row
    tail = graph.col
    weight = graph.data

    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

    aux_data = {}

    if densmap or output_dens:
        if verbose:
            print(ts() + " Computing original densities")

        dists = densmap_kwds["graph_dists"]

        mu_sum = np.zeros(n_vertices, dtype=np.float32)
        ro = np.zeros(n_vertices, dtype=np.float32)
        for i in range(len(head)):
            j = head[i]
            k = tail[i]

            D = dists[j, k] * dists[j, k]  # match sq-Euclidean used for embedding
            mu = graph.data[i]

            ro[j] += mu * D
            ro[k] += mu * D
            mu_sum[j] += mu
            mu_sum[k] += mu

        epsilon = 1e-8
        ro = np.log(epsilon + (ro / mu_sum))

        if densmap:
            R = (ro - np.mean(ro)) / np.std(ro)
            densmap_kwds["mu"] = graph.data
            densmap_kwds["mu_sum"] = mu_sum
            densmap_kwds["R"] = R

        if output_dens:
            aux_data["rad_orig"] = ro

    embedding = (
        10.0
        * (embedding - np.min(embedding, 0))
        / (np.max(embedding, 0) - np.min(embedding, 0))
    ).astype(np.float32, order="C")

    if euclidean_output:
        embedding = optimize_layout_euclidean(
            embedding,
            embedding,
            head,
            tail,
            n_epochs,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            initial_alpha,
            negative_sample_rate,
            parallel=parallel,
            verbose=verbose,
            densmap=densmap,
            densmap_kwds=densmap_kwds,
        )
    else:
        embedding = optimize_layout_generic(
            embedding,
            embedding,
            head,
            tail,
            n_epochs,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            initial_alpha,
            negative_sample_rate,
            output_metric,
            tuple(output_metric_kwds.values()),
            verbose=verbose,
        )

    if output_dens:
        if verbose:
            print(ts() + " Computing embedding densities")

        # Compute graph in embedding
        (knn_indices, knn_dists, rp_forest,) = fast_knn_indices(
            embedding,
            densmap_kwds["n_neighbors"],
            metric,
            {},
            False,
            random_state,
            verbose=verbose,
        )

        emb_graph, emb_sigmas, emb_rhos, emb_dists = fuzzy_simplicial_set_ann(
            embedding,
            densmap_kwds["n_neighbors"],
            random_state,
            metric,
            {},
            knn_indices,
            knn_dists,
            verbose=verbose,
            return_dists=True,
        )

        emb_graph = emb_graph.tocoo()
        emb_graph.sum_duplicates()
        emb_graph.eliminate_zeros()

        n_vertices = emb_graph.shape[1]

        mu_sum = np.zeros(n_vertices, dtype=np.float32)
        re = np.zeros(n_vertices, dtype=np.float32)

        head = emb_graph.row
        tail = emb_graph.col
        for i in range(len(head)):
            j = head[i]
            k = tail[i]

            D = emb_dists[j, k]
            mu = emb_graph.data[i]

            re[j] += mu * D
            re[k] += mu * D
            mu_sum[j] += mu
            mu_sum[k] += mu

        epsilon = 1e-8
        re = np.log(epsilon + (re / mu_sum))

        aux_data["rad_emb"] = re
    aux_data["initiasation"] = initialisation
    return embedding, aux_data



def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]
