# These are some graph learning functions implemented in UMAP, added here with modifications
# for better speed and computational efficiency.
# Originally implemented by Leland McInnes at https://github.com/lmcinnes/umap
# I've implemented a scikit-learn compatible version with minor improvements for speed and scalability here.
#
#
# Below are some graph learning functions implemented in UMAP, added here with modifications
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
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.base import TransformerMixin
from topo.base.ann import kNN
from topo.utils._utils import get_indices_distances_from_sparse_matrix

SMOOTH_K_TOLERANCE = 1e-6
MIN_K_DIST_SCALE = 1e-4
NPY_INFINITY = np.inf
INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

def fuzzy_simplicial_set(
        X,
        n_neighbors=15,
        metric='cosine',
        backend='nmslib',
        n_jobs=1,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        apply_set_operations=True,
        return_dists=False,
        verbose=False,
        **kwargs):
    """
    Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.

    Originally implemented by Leland McInnes at https://github.com/lmcinnes/umap
    under the BSD 3-Clause License.

    Parameters
    ----------
    X : array of shape (n_samples, n_features).
        The data to be modelled as a fuzzy simplicial set.

    n_neighbors : int.
        The number of neighbors to use to approximate geodesic distance.
        Larger numbers induce more global estimates of the manifold that can
        miss finer detail, while smaller values will focus on fine manifold
        structure to the detriment of the larger picture.

    backend : str (optional, default 'nmslib').
        Which backend to use for neighborhood search. Options are 'nmslib', 'hnswlib', 
        'pynndescent','annoy', 'faiss' and 'sklearn'.

    metric : str (optional, default 'cosine').
        Accepted metrics. Defaults to 'cosine'. Accepted metrics include:
        -'sqeuclidean'
        -'euclidean'
        -'l1'
        -'lp' - requires setting the parameter `p` - equivalent to minkowski distance
        -'cosine'
        -'angular'
        -'negdotprod'
        -'levenshtein'
        -'hamming'
        -'jaccard'
        -'jansen-shan'

    n_jobs : int (optional, default 1).
        Number of threads to be used in computation of nearest neighbors.  Set to -1 to use all available CPUs.

    knn_indices : array of shape (n_samples, n_neighbors) (optional).
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the indices of the k-nearest neighbors as a row for
        each data point. Ignored if metric is 'precomputed'.

    knn_dists : array of shape (n_samples, n_neighbors) (optional).
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the distances of the k-nearest neighbors as a row for
        each data point. Ignored if metric is 'precomputed'.

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

    **kwargs : dict (optional, default {}).
        Additional parameters to be passed to the backend approximate nearest-neighbors library.
        Use only parameters known to the desired backend library.


    Returns
    -------
    fuzzy_ss : coo_matrix
        A fuzzy simplicial set represented as a sparse matrix. The (i,
        j) entry of the matrix represents the membership strength of the
        1-simplex between the ith and jth sample points.

    sigmas: array of shape (n_samples,)
        The normalization factor derived from the metric tensor approximation. Equal
        to the distance 

    rhos: array of shape (n_samples,)
        The distance to the 1st nearest neighbor for each point.
    """
    if metric == 'precomputed':
        if not isinstance(X, csr_matrix):
            raise TypeError(
                'X should be a sparse csr_matrix if using precomputed distances.')
        knn_indices, knn_dists = get_indices_distances_from_sparse_matrix(
            X, n_neighbors)
    else:
        knn = kNN(X, n_neighbors=n_neighbors,
                  metric=metric,
                  n_jobs=n_jobs,
                  backend=backend,
<<<<<<< HEAD
                  low_memory=True,
=======
>>>>>>> master
                  return_instance=False,
                  verbose=verbose,
                  **kwargs)
        knn_indices, knn_dists = get_indices_distances_from_sparse_matrix(
            knn, n_neighbors)

    knn_dists = knn_dists.astype(np.float32)

    sigmas, rhos = smooth_knn_dist(
        knn_dists, float(n_neighbors), local_connectivity=float(local_connectivity),
    )

    rows, cols, vals = compute_membership_strengths(
        knn_indices, knn_dists, sigmas, rhos
    )

    fuzzy_ss = coo_matrix(
        (vals, (rows, cols)), shape=(X.shape[0], X.shape[0])
    )
    fuzzy_ss.eliminate_zeros()

    if apply_set_operations:
        transpose = fuzzy_ss.transpose()

        prod_matrix = fuzzy_ss.multiply(transpose)

        fuzzy_ss = (
            set_op_mix_ratio * (fuzzy_ss + transpose - prod_matrix)
            + (1.0 - set_op_mix_ratio) * prod_matrix
        )

    fuzzy_ss.eliminate_zeros()
    if return_dists:
        return fuzzy_ss, sigmas, rhos, knn_dists
    else:
        return fuzzy_ss, sigmas, rhos


def compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos):
    """Construct the membership strength data for the 1-skeleton of each local
    fuzzy simplicial set -- this is formed as a sparse matrix where each row is
    a local fuzzy simplicial set, with a membership strength for the
    1-simplex to each other data point.

    Originally implemented by Leland McInnes at https://github.com/lmcinnes/umap
    under the BSD 3-Clause License.

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

    Originally implemented by Leland McInnes at https://github.com/lmcinnes/umap
    under the BSD 3-Clause License.
    
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
    # Isn't there a way to do this without the loop?
    # Maybe with a sparse matrix? :)
    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0
        # TODO: Will remove interpolation computation, since it does nothing but slow us down
        #
        # (Leland): This is very inefficient, but will do for now.
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
        # Do we really need to iterate this for each sample?
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
