# NOTICE
# This is an adapted implementation of the TriMAP algorithm, developed by Ehsan Amid and Manfred Warmuth,
# as described in their manuscript 'TriMap: Large-scale Dimensionality Reduction Using Triplets'
# and available at https://arxiv.org/abs/1910.00204 . Their original implementation is available at
# https://github.com/eamid/trimap and is bound to the Apache License.
#
# I basically modified the implementation to take in a weighted similarity matrix
# (consisting of topological metrics in TopOMetry) instead of a raw data matrix of measurements.
#
# As stated by TriMAP Apache License, available at the end of this file is a copy of their license,
# and you are bounded to that license if you use TriMAP within TopOMetry for your work,
# in addition to TopOMetry's MIT license.
#

from sklearn.base import BaseEstimator
from annoy import AnnoyIndex
import time
import datetime
from topo.base.dists import *
import warnings

@numba.jit(nopython=False, fastmath=True)
def calculate_dist(x1, x2, metric):
    metric = named_distances[metric]
    return metric(x1, x2)

@numba.njit()
def rejection_sample(n_samples, max_int, rejects):
    """
    Samples "n_samples" integers from a given interval [0,max_int] while
    rejecting the values that are in the "rejects".
    """
    result = np.empty(n_samples, dtype=np.int32)
    for i in range(n_samples):
        reject_sample = True
        while reject_sample:
            j = np.random.randint(max_int)
            for k in range(i):
                if j == result[k]:
                    break
            for k in range(rejects.shape[0]):
                if j == rejects[k]:
                    break
            else:
                reject_sample = False
        result[i] = j
    return result

@numba.njit("i4[:,:](f4[:,:],i4[:,:],i4,i4)", parallel=True, nogil=True)
def sample_knn_triplets(P, nbrs, n_inliers, n_outliers):
    """
    Sample nearest neighbors triplets based on the similarity values given in P
    Input
    ------
    nbrs: Nearest neighbors indices for each point. The similarity values
        are given in matrix P. Row i corresponds to the i-th point.
    P: Matrix of pairwise similarities between each point and its neighbors
        given in matrix nbrs
    n_inliers: Number of inlier points
    n_outliers: Number of outlier points
    Output
    ------
    triplets: Sampled triplets
    """
    n, n_neighbors = nbrs.shape
    triplets = np.empty((n * n_inliers * n_outliers, 3), dtype=np.int32)
    for i in numba.prange(n):
        sort_indices = np.argsort(-P[i])
        for j in numba.prange(n_inliers):
            sim = nbrs[i][sort_indices[j + 1]]
            samples = rejection_sample(n_outliers, n, nbrs[i][sort_indices[: j+2]])
            for k in numba.prange(n_outliers):
                index = i * n_inliers * n_outliers + j * n_outliers + k
                out = samples[k]
                triplets[index][0] = i
                triplets[index][1] = sim
                triplets[index][2] = out
    return triplets

@numba.njit("f4[:,:](f4[:,:],i4,f4[:],i4)", parallel=True, nogil=True)
def sample_random_triplets(X, n_random):
    """
    Sample uniformly random triplets
    Input
    ------
    X: Instance matrix or pairwise distances
    n_random: Number of random triplets per point
    Output
    ------
    rand_triplets: Sampled triplets
    """
    n = X.shape[0]
    rand_triplets = np.empty((n * n_random, 4), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n_random):
            sim = np.random.choice(n)
            while sim == i:
                sim = np.random.choice(n)
            out = np.random.choice(n)
            while out == i or out == sim:
                out = np.random.choice(n)
            p_sim = X[i, sim]
            if p_sim < 1e-20:
                p_sim = 1e-20
            p_out = X[i, out]
            if p_out < 1e-20:
                p_out = 1e-20
            if p_sim < p_out:
                sim, out = out, sim
                p_sim, p_out = p_out, p_sim
            rand_triplets[i * n_random + j][0] = i
            rand_triplets[i * n_random + j][1] = sim
            rand_triplets[i * n_random + j][2] = out
            rand_triplets[i * n_random + j][3] = p_sim / p_out
    return rand_triplets

#
# @numba.njit("f4[:,:](f4[:,:],f4[:],i4[:,:])", parallel=True, nogil=True)
# def find_p(knn_distances, sig, nbrs):
#     """
#     Calculates the similarity matrix P
#     Input
#     ------
#     knn_distances: Matrix of pairwise knn distances
#     sig: Scaling factor for the distances
#     nbrs: Nearest neighbors
#     Output
#     ------
#     P: Pairwise similarity matrix
#     """
#     n, n_neighbors = knn_distances.shape
#     P = np.zeros((n, n_neighbors), dtype=np.float32)
#     for i in numba.prange(n):
#         for j in numba.prange(n_neighbors):
#             P[i][j] = np.exp(-knn_distances[i][j] ** 2 / sig[i] / sig[nbrs[i][j]])
#     return P


@numba.njit("f4[:](i4[:,:],f4[:,:],i4[:,:],f4[:],f4[:])", parallel=True, nogil=True)
def find_weights(triplets, P, nbrs, outlier_distances):
    """
    Calculates the weights for the sampled nearest neighbors triplets
    Input
    ------
    triplets: Sampled triplets
    P: Pairwise similarity matrix
    nbrs: Nearest neighbors
    outlier_distances: Matrix of pairwise outlier distances
    Output
    ------
    weights: Weights for the triplets
    """
    n_triplets = triplets.shape[0]
    weights = np.empty(n_triplets, dtype=np.float32)
    for t in numba.prange(n_triplets):
        i = triplets[t][0]
        sim = 0
        while nbrs[i][sim] != triplets[t][1]:
            sim += 1
        p_sim = P[i][sim]
        p_out = outlier_distances[t]
        if p_out < 1e-20:
            p_out = 1e-20
        weights[t] = p_sim / p_out
    return weights

#
# def generate_triplets(
#     X,
#     n_inliers,
#     n_outliers,
#     n_random,
#     distance="euclidean",
#     weight_adj=500.0,
#     verbose=True,
# ):
#     distance_dict = {"euclidean": 0, "manhattan": 1, "angular": 2, "hamming": 3}
#     distance_index = distance_dict[distance]
#     n, dim = X.shape
#     n_extra = min(n_inliers + 50, n)
#     tree = AnnoyIndex(dim, metric=distance)
#     for i in range(n):
#         tree.add_item(i, X[i, :])
#     tree.build(20)
#     nbrs = np.empty((n, n_extra), dtype=np.int32)
#     knn_distances = np.empty((n, n_extra), dtype=np.float32)
#     for i in range(n):
#         nbrs[i, :] = tree.get_nns_by_item(i, n_extra)
#         for j in range(n_extra):
#             knn_distances[i, j] = tree.get_distance(i, nbrs[i, j])
#     if verbose:
#         print("found nearest neighbors")
#     sig = np.maximum(np.mean(knn_distances[:, 3:6], axis=1), 1e-10)  # scale parameter
#     P = find_p(knn_distances, sig, nbrs)
#     triplets = sample_knn_triplets(P, nbrs, n_inliers, n_outliers)
#     n_triplets = triplets.shape[0]
#     outlier_distances = np.empty(n_triplets, dtype=np.float32)
#     for t in range(n_triplets):
#         outlier_distances[t] = calculate_dist(
#             X[triplets[t, 0], :], X[triplets[t, 2], :], distance_index
#         )
#     weights = find_weights(triplets, P, nbrs, outlier_distances, sig)
#     if n_random > 0:
#         rand_triplets = sample_random_triplets(X, n_random, sig, distance_index)
#         rand_weights = rand_triplets[:, -1]
#         rand_triplets = rand_triplets[:, :-1].astype(np.int32)
#         triplets = np.vstack((triplets, rand_triplets))
#         weights = np.hstack((weights, rand_weights))
#     weights[np.isnan(weights)] = 0.0
#     weights /= np.max(weights)
#     weights += 0.0001
#     if weight_adj:
#         if not isinstance(weight_adj, (int, float)):
#             weight_adj = 500.0
#         weights = np.log(1 + weight_adj * weights)
#         weights /= np.max(weights)
#     return (triplets, weights)


def generate_triplets_known_P(
    knn_nbrs,
    P,
    n_inliers,
    n_outliers,
    n_random,
    pairwise_dist_matrix,
    weight_adj=500.0,
    verbose=True,
):
    # check whether the first nn of each point is itself
    if knn_nbrs[0, 0] != 0:
        knn_nbrs = np.hstack(
            (np.array(range(knn_nbrs.shape[0]))[:, np.newaxis], knn_nbrs)
        ).astype(np.int32)
        knn_distances = np.hstack(
            (np.zeros((P.shape[0], 1)), P)
        ).astype(np.float32)
    triplets = sample_knn_triplets(P, knn_nbrs, n_inliers, n_outliers)
    n_triplets = triplets.shape[0]
    outlier_distances = np.empty(n_triplets, dtype=np.float32)
    for t in range(n_triplets):
        outlier_distances[t] = pairwise_dist_matrix[triplets[t, 0], triplets[t, 2]]
    weights = find_weights(triplets, P, knn_nbrs, outlier_distances)
    if n_random > 0:
        rand_triplets = sample_random_triplets(
                pairwise_dist_matrix, n_random
            )
        rand_weights = rand_triplets[:, -1]
        rand_triplets = rand_triplets[:, :-1].astype(np.int32)
        triplets = np.vstack((triplets, rand_triplets))
        weights = np.hstack((weights, rand_weights))
    weights[np.isnan(weights)] = 0.0
    weights /= np.max(weights)
    weights += 0.0001
    if weight_adj:
        if not isinstance(weight_adj, (int, float)):
            weight_adj = 500.0
        weights = np.log(1 + weight_adj * weights)
        weights /= np.max(weights)
    return (triplets, weights)


@numba.njit("void(f4[:,:],f4[:,:],f4[:,:],f4,i4,i4)", parallel=True, nogil=True)
def update_embedding(Y, grad, vel, lr, iter_num, opt_method):
    n, dim = Y.shape
    if opt_method == 0:  # sd
        for i in numba.prange(n):
            for d in numba.prange(dim):
                Y[i][d] -= lr * grad[i][d]
    elif opt_method == 1:  # momentum
        if iter_num > 250:
            gamma = 0.5
        else:
            gamma = 0.3
        for i in numba.prange(n):
            for d in numba.prange(dim):
                vel[i][d] = gamma * vel[i][d] - lr * grad[i][d]  # - 1e-5 * Y[i,d]
                Y[i][d] += vel[i][d]


@numba.njit("void(f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4,i4)", parallel=True, nogil=True)
def update_embedding_dbd(Y, grad, vel, gain, lr, iter_num):
    n, dim = Y.shape
    if iter_num > 250:
        gamma = 0.8  # moment parameter
    else:
        gamma = 0.5
    min_gain = 0.01
    for i in numba.prange(n):
        for d in numba.prange(dim):
            gain[i][d] = (
                (gain[i][d] + 0.2)
                if (np.sign(vel[i][d]) != np.sign(grad[i][d]))
                else np.maximum(gain[i][d] * 0.8, min_gain)
            )
            vel[i][d] = gamma * vel[i][d] - lr * gain[i][d] * grad[i][d]
            Y[i][d] += vel[i][d]


@numba.njit("f4[:,:](f4[:,:],i4,i4,i4[:,:],f4[:])", parallel=True, nogil=True)
def trimap_grad(Y, n_inliers, n_outliers, triplets, weights):
    n, dim = Y.shape
    n_triplets = triplets.shape[0]
    grad = np.zeros((n, dim), dtype=np.float32)
    y_ij = np.empty(dim, dtype=np.float32)
    y_ik = np.empty(dim, dtype=np.float32)
    n_viol = 0.0
    loss = 0.0
    n_knn_triplets = n * n_inliers * n_outliers
    for t in range(n_triplets):
        i = triplets[t, 0]
        j = triplets[t, 1]
        k = triplets[t, 2]
        if (t % n_outliers) == 0 or (
            t >= n_knn_triplets
        ):  # update y_ij, y_ik, d_ij, d_ik
            d_ij = 1.0
            d_ik = 1.0
            for d in range(dim):
                y_ij[d] = Y[i, d] - Y[j, d]
                y_ik[d] = Y[i, d] - Y[k, d]
                d_ij += y_ij[d] ** 2
                d_ik += y_ik[d] ** 2
        else:  # update y_ik and d_ik only
            d_ik = 1.0
            for d in range(dim):
                y_ik[d] = Y[i, d] - Y[k, d]
                d_ik += y_ik[d] ** 2
        if d_ij > d_ik:
            n_viol += 1.0
        loss += weights[t] * 1.0 / (1.0 + d_ik / d_ij)
        w = weights[t] / (d_ij + d_ik) ** 2
        for d in range(dim):
            gs = y_ij[d] * d_ik * w
            go = y_ik[d] * d_ij * w
            grad[i, d] += gs - go
            grad[j, d] -= gs
            grad[k, d] += go
    last = np.zeros((1, dim), dtype=np.float32)
    last[0] = loss
    last[1] = n_viol
    return np.vstack((grad, last))


def trimap(
    X,
    triplets,
    weights,
    n_dims,
    n_inliers,
    n_outliers,
    n_random,
    distance,
    lr,
    n_iters,
    Yinit,
    weight_adj,
    opt_method,
    verbose,
    return_seq,
):
    """
    Apply TriMap.
    """

    opt_method_dict = {"sd": 0, "momentum": 1, "dbd": 2}
    if verbose:
        t = time.time()
    n, dim = X.shape
    assert n_inliers < n - 1, "n_inliers must be less than (number of data points - 1)."
    if verbose:
        print("running TriMap on %d points with dimension %d" % (n, dim))
    if triplets is None:
        pairwise_dist_matrix = X
        pairwise_dist_matrix = pairwise_dist_matrix.astype(np.float32)
        n_extra = min(n_inliers + 50, n)
        knn_nbrs = np.zeros((n, n_extra), dtype=np.int32)
        knn_distances = np.zeros((n, n_extra), dtype=np.float32)
        for nn in range(n):
            bottom_k_indices = np.argpartition(
                pairwise_dist_matrix[nn, :], n_extra
            )[:n_extra]
            bottom_k_distances = pairwise_dist_matrix[nn, bottom_k_indices]
            sort_indices = np.argsort(bottom_k_distances)
            knn_nbrs[nn, :] = bottom_k_indices[sort_indices]
            knn_distances[nn, :] = bottom_k_distances[sort_indices]
        triplets, weights = generate_triplets_known_P(
            knn_nbrs,
            knn_distances,
            n_inliers,
            n_outliers,
            n_random,
            pairwise_dist_matrix,
            weight_adj,
            verbose,
        )
    else:
        if verbose:
            print("using stored triplets")

    if Yinit is None:
        Yinit = 'random'
    if Yinit is "random":
        Y = np.random.normal(size=[n, n_dims]).astype(np.float32) * 0.0001
    else:
        Y = Yinit.astype(np.float32)
    if return_seq:
        Y_all = np.zeros((n, n_dims, int(n_iters / 10 + 1)))
        Y_all[:, :, 0] = Yinit

    C = np.inf
    tol = 1e-7
    n_triplets = float(triplets.shape[0])
    lr = lr * n / n_triplets
    if verbose:
        print("running TriMap with " + opt_method)
    vel = np.zeros_like(Y, dtype=np.float32)
    if opt_method_dict[opt_method] == 2:
        gain = np.ones_like(Y, dtype=np.float32)

    for itr in range(n_iters):
        old_C = C
        if opt_method_dict[opt_method] == 0:
            grad = trimap_grad(Y, n_inliers, n_outliers, triplets, weights)
        else:
            if itr > 250:
                gamma = 0.5
            else:
                gamma = 0.3
            grad = trimap_grad(
                Y + gamma * vel, n_inliers, n_outliers, triplets, weights
            )
        C = grad[-1, 0]
        n_viol = grad[-1, 1]

        # update Y
        if opt_method_dict[opt_method] < 2:
            update_embedding(Y, grad, vel, lr, itr, opt_method_dict[opt_method])
        else:
            update_embedding_dbd(Y, grad, vel, gain, lr, itr)

        # update the learning rate
        if opt_method_dict[opt_method] < 2:
            if old_C > C + tol:
                lr = lr * 1.01
            else:
                lr = lr * 0.9
        if return_seq and (itr + 1) % 10 == 0:
            Y_all[:, :, int((itr + 1) / 10)] = Y
        if verbose:
            if (itr + 1) % 100 == 0:
                print(
                    "Iteration: %4d, Loss: %3.3f, Violated triplets: %0.4f"
                    % (itr + 1, C, n_viol / n_triplets * 100.0)
                )
    if verbose:
        elapsed = str(datetime.timedelta(seconds=time.time() - t))
        print("Elapsed time: %s" % (elapsed))
    if return_seq:
        return (Y_all, triplets, weights)
    else:
        return (Y, triplets, weights)


class TRIMAP(BaseEstimator):
    """
    Dimensionality Reduction Using Triplet Constraints
    Find a low-dimensional representation of the similarity matrix by satisfying the sampled
    triplet constraints from the high-dimensional features.
    Input
    ------
    n_dims: Number of dimensions of the embedding (default = 2)
    n_inliers: Number of inlier points for triplet constraints (default = 10)
    n_outliers: Number of outlier points for triplet constraints (default = 5)
    n_random: Number of random triplet constraints per point (default = 5)
    distance: Distance measure ('euclidean' (default), 'manhattan', 'angular',
    'hamming')
    lr: Learning rate (default = 1000.0)
    n_iters: Number of iterations (default = 400)
    opt_method: Optimization method ('sd': steepest descent,  'momentum': GD
    with momentum, 'dbd': GD with momentum delta-bar-delta (default))
    verbose: Print the progress report (default = True)
    weight_adj: Adjusting the weights using a non-linear transformation
    (default = 500.0)
    return_seq: Return the sequence of maps recorded every 10 iterations
    (default = False)
    """

    def __init__(
        self,
        n_dims=2,
        n_inliers=10,
        n_outliers=5,
        n_random=5,
        lr=1000.0,
        n_iters=400,
        triplets=None,
        weights=None,
        verbose=True,
        weight_adj=500.0,
        opt_method="dbd",
        return_seq=False,
    ):
        self.n_dims = n_dims
        self.n_inliers = n_inliers
        self.n_outliers = n_outliers
        self.n_random = n_random
        self.lr = lr
        self.n_iters = n_iters
        self.triplets = triplets
        self.weights = weights
        self.weight_adj = weight_adj
        self.opt_method = opt_method
        self.verbose = verbose
        self.return_seq = return_seq

        if self.n_dims < 2:
            raise ValueError("The number of output dimensions must be at least 2.")
        if self.n_inliers < 1:
            raise ValueError("The number of inliers must be a positive number.")
        if self.n_outliers < 1:
            raise ValueError("The number of outliers must be a positive number.")
        if self.n_random < 0:
            raise ValueError(
                "The number of random triplets must be a non-negative number."
            )
        if self.lr <= 0:
            raise ValueError("The learning rate must be a positive value.")
        if self.verbose:
            print(
                "TRIMAP(n_inliers={}, n_outliers={}, n_random={}, "
                "lr={}, n_iters={}, weight_adj={}, opt_method={}, verbose={}, return_seq={})".format(
                    n_inliers,
                    n_outliers,
                    n_random,
                    lr,
                    n_iters,
                    weight_adj,
                    opt_method,
                    verbose,
                    return_seq,
                )
            )

    def fit(self, X, init=None):
        """
        Runs the TriMap algorithm on the input data X
        Input
        ------
        X: Similarity matrix
        init: Initial solution
        """
        X = X.astype(np.float32)

        self.embedding_, self.triplets, self.weights = trimap(
            X,
            self.triplets,
            self.weights,
            self.n_dims,
            self.n_inliers,
            self.n_outliers,
            self.n_random,
            self.lr,
            self.n_iters,
            init,
            self.weight_adj,
            self.opt_method,
            self.verbose,
            self.return_seq,
        )
        return self

    def fit_transform(self, X, init=None):
        """
        Runs the TriMap algorithm on the input data X and returns the embedding
        Input
        ------
        X: Instance matrix
        init: Initial solution
        """
        self.fit(X, init)
        return self.embedding_

    def sample_triplets(self, X):
        """
        Samples and stores triplets
        Input
        ------
        X: Instance matrix
        """
        if self.verbose:
            print("pre-processing")
        X = X.astype(np.float32)
        n, dim = X.shape
        assert self.n_inliers < n - 1, "n_inliers must be less than (number of data points - 1)."
        if self.distance != "hamming":
                X -= np.min(X)
                X /= np.max(X)
                X -= np.mean(X, axis=0)
        pairwise_dist_matrix = X
        n_extra = min(self.n_inliers + 50, n)
        knn_nbrs = np.zeros((n, n_extra), dtype=np.int32)
        knn_distances = np.zeros((n, n_extra), dtype=np.float32)
        for nn in range(n):
            bottom_k_indices = np.argpartition(
                pairwise_dist_matrix[nn, :], n_extra
            )[:n_extra]
            bottom_k_distances = pairwise_dist_matrix[nn, bottom_k_indices]
            sort_indices = np.argsort(bottom_k_distances)
            knn_nbrs[nn, :] = bottom_k_indices[sort_indices]
            knn_distances[nn, :] = bottom_k_distances[sort_indices]
        self.triplets, self.weights = generate_triplets_known_P(
            knn_nbrs,
            knn_distances,
            self.n_inliers,
            self.n_outliers,
            self.n_random,
            pairwise_dist_matrix,
            self.weight_adj,
            self.verbose,
        )
        if self.verbose:
            print("sampled triplets")

        return self

    def del_triplets(self):
        """
        Deletes the stored triplets
        """
        self.triplets = None
        self.weights = None

        return self



    #                               Apache License
    #                         Version 2.0, January 2004
    #                      http://www.apache.org/licenses/
    #
    # TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
    #
    # 1. Definitions.
    #
    #    "License" shall mean the terms and conditions for use, reproduction,
    #    and distribution as defined by Sections 1 through 9 of this document.
    #
    #    "Licensor" shall mean the copyright owner or entity authorized by
    #    the copyright owner that is granting the License.
    #
    #    "Legal Entity" shall mean the union of the acting entity and all
    #    other entities that control, are controlled by, or are under common
    #    control with that entity. For the purposes of this definition,
    #    "control" means (i) the power, direct or indirect, to cause the
    #    direction or management of such entity, whether by contract or
    #    otherwise, or (ii) ownership of fifty percent (50%) or more of the
    #    outstanding shares, or (iii) beneficial ownership of such entity.
    #
    #    "You" (or "Your") shall mean an individual or Legal Entity
    #    exercising permissions granted by this License.
    #
    #    "Source" form shall mean the preferred form for making modifications,
    #    including but not limited to software source code, documentation
    #    source, and configuration files.
    #
    #    "Object" form shall mean any form resulting from mechanical
    #    transformation or translation of a Source form, including but
    #    not limited to compiled object code, generated documentation,
    #    and conversions to other media types.
    #
    #    "Work" shall mean the work of authorship, whether in Source or
    #    Object form, made available under the License, as indicated by a
    #    copyright notice that is included in or attached to the work
    #    (an example is provided in the Appendix below).
    #
    #    "Derivative Works" shall mean any work, whether in Source or Object
    #    form, that is based on (or derived from) the Work and for which the
    #    editorial revisions, annotations, elaborations, or other modifications
    #    represent, as a whole, an original work of authorship. For the purposes
    #    of this License, Derivative Works shall not include works that remain
    #    separable from, or merely link (or bind by name) to the interfaces of,
    #    the Work and Derivative Works thereof.
    #
    #    "Contribution" shall mean any work of authorship, including
    #    the original version of the Work and any modifications or additions
    #    to that Work or Derivative Works thereof, that is intentionally
    #    submitted to Licensor for inclusion in the Work by the copyright owner
    #    or by an individual or Legal Entity authorized to submit on behalf of
    #    the copyright owner. For the purposes of this definition, "submitted"
    #    means any form of electronic, verbal, or written communication sent
    #    to the Licensor or its representatives, including but not limited to
    #    communication on electronic mailing lists, source code control systems,
    #    and issue tracking systems that are managed by, or on behalf of, the
    #    Licensor for the purpose of discussing and improving the Work, but
    #    excluding communication that is conspicuously marked or otherwise
    #    designated in writing by the copyright owner as "Not a Contribution."
    #
    #    "Contributor" shall mean Licensor and any individual or Legal Entity
    #    on behalf of whom a Contribution has been received by Licensor and
    #    subsequently incorporated within the Work.
    #
    # 2. Grant of Copyright License. Subject to the terms and conditions of
    #    this License, each Contributor hereby grants to You a perpetual,
    #    worldwide, non-exclusive, no-charge, royalty-free, irrevocable
    #    copyright license to reproduce, prepare Derivative Works of,
    #    publicly display, publicly perform, sublicense, and distribute the
    #    Work and such Derivative Works in Source or Object form.
    #
    # 3. Grant of Patent License. Subject to the terms and conditions of
    #    this License, each Contributor hereby grants to You a perpetual,
    #    worldwide, non-exclusive, no-charge, royalty-free, irrevocable
    #    (except as stated in this section) patent license to make, have made,
    #    use, offer to sell, sell, import, and otherwise transfer the Work,
    #    where such license applies only to those patent claims licensable
    #    by such Contributor that are necessarily infringed by their
    #    Contribution(s) alone or by combination of their Contribution(s)
    #    with the Work to which such Contribution(s) was submitted. If You
    #    institute patent litigation against any entity (including a
    #    cross-claim or counterclaim in a lawsuit) alleging that the Work
    #    or a Contribution incorporated within the Work constitutes direct
    #    or contributory patent infringement, then any patent licenses
    #    granted to You under this License for that Work shall terminate
    #    as of the date such litigation is filed.
    #
    # 4. Redistribution. You may reproduce and distribute copies of the
    #    Work or Derivative Works thereof in any medium, with or without
    #    modifications, and in Source or Object form, provided that You
    #    meet the following conditions:
    #
    #    (a) You must give any other recipients of the Work or
    #        Derivative Works a copy of this License; and
    #
    #    (b) You must cause any modified files to carry prominent notices
    #        stating that You changed the files; and
    #
    #    (c) You must retain, in the Source form of any Derivative Works
    #        that You distribute, all copyright, patent, trademark, and
    #        attribution notices from the Source form of the Work,
    #        excluding those notices that do not pertain to any part of
    #        the Derivative Works; and
    #
    #    (d) If the Work includes a "NOTICE" text file as part of its
    #        distribution, then any Derivative Works that You distribute must
    #        include a readable copy of the attribution notices contained
    #        within such NOTICE file, excluding those notices that do not
    #        pertain to any part of the Derivative Works, in at least one
    #        of the following places: within a NOTICE text file distributed
    #        as part of the Derivative Works; within the Source form or
    #        documentation, if provided along with the Derivative Works; or,
    #        within a display generated by the Derivative Works, if and
    #        wherever such third-party notices normally appear. The contents
    #        of the NOTICE file are for informational purposes only and
    #        do not modify the License. You may add Your own attribution
    #        notices within Derivative Works that You distribute, alongside
    #        or as an addendum to the NOTICE text from the Work, provided
    #        that such additional attribution notices cannot be construed
    #        as modifying the License.
    #
    #    You may add Your own copyright statement to Your modifications and
    #    may provide additional or different license terms and conditions
    #    for use, reproduction, or distribution of Your modifications, or
    #    for any such Derivative Works as a whole, provided Your use,
    #    reproduction, and distribution of the Work otherwise complies with
    #    the conditions stated in this License.
    #
    # 5. Submission of Contributions. Unless You explicitly state otherwise,
    #    any Contribution intentionally submitted for inclusion in the Work
    #    by You to the Licensor shall be under the terms and conditions of
    #    this License, without any additional terms or conditions.
    #    Notwithstanding the above, nothing herein shall supersede or modify
    #    the terms of any separate license agreement you may have executed
    #    with Licensor regarding such Contributions.
    #
    # 6. Trademarks. This License does not grant permission to use the trade
    #    names, trademarks, service marks, or product names of the Licensor,
    #    except as required for reasonable and customary use in describing the
    #    origin of the Work and reproducing the content of the NOTICE file.
    #
    # 7. Disclaimer of Warranty. Unless required by applicable law or
    #    agreed to in writing, Licensor provides the Work (and each
    #    Contributor provides its Contributions) on an "AS IS" BASIS,
    #    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
    #    implied, including, without limitation, any warranties or conditions
    #    of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
    #    PARTICULAR PURPOSE. You are solely responsible for determining the
    #    appropriateness of using or redistributing the Work and assume any
    #    risks associated with Your exercise of permissions under this License.
    #
    # 8. Limitation of Liability. In no event and under no legal theory,
    #    whether in tort (including negligence), contract, or otherwise,
    #    unless required by applicable law (such as deliberate and grossly
    #    negligent acts) or agreed to in writing, shall any Contributor be
    #    liable to You for damages, including any direct, indirect, special,
    #    incidental, or consequential damages of any character arising as a
    #    result of this License or out of the use or inability to use the
    #    Work (including but not limited to damages for loss of goodwill,
    #    work stoppage, computer failure or malfunction, or any and all
    #    other commercial damages or losses), even if such Contributor
    #    has been advised of the possibility of such damages.
    #
    # 9. Accepting Warranty or Additional Liability. While redistributing
    #    the Work or Derivative Works thereof, You may choose to offer,
    #    and charge a fee for, acceptance of support, warranty, indemnity,
    #    or other liability obligations and/or rights consistent with this
    #    License. However, in accepting such obligations, You may act only
    #    on Your own behalf and on Your sole responsibility, not on behalf
    #    of any other Contributor, and only if You agree to indemnify,
    #    defend, and hold each Contributor harmless for any liability
    #    incurred by, or claims asserted against, such Contributor by reason
    #    of your accepting any such warranty or additional liability.
    #
    # END OF TERMS AND CONDITIONS
    #
    # APPENDIX: How to apply the Apache License to your work.
    #
    #    To apply the Apache License to your work, attach the following
    #    boilerplate notice, with the fields enclosed by brackets "[]"
    #    replaced with your own identifying information. (Don't include
    #    the brackets!)  The text should be enclosed in the appropriate
    #    comment syntax for the file format. We also recommend that a
    #    file or class name and description of purpose be included on the
    #    same "printed page" as the copyright notice for easier
    #    identification within third-party archives.
    #
    # Copyright 2018 Ehsan Amid
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    #
    #
    #