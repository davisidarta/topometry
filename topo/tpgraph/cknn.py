# Code to perform Continuous k-Nearest Neighbors(CkNN), proposed in the paper
# 'Consistent Manifold Representation for Topological Data Analysis'
# (https://arxiv.org/pdf/1606.02353.pdf)
#
# Based on the implementation by Naoto MINAMI (https://github.com/chlorochrule/cknn),
# with some API and performance improvements (majorly by allowing precomputed kNN instead of pairwise metrics),
# under the following MIT license:
#
# MIT LICENSE
#
# Copyright (c) 2018 Naoto MINAMI, minami.polly@gmail.com
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances

from topo.base.dists import \
    (euclidean,
     standardised_euclidean,
     cosine,
     correlation,
     bray_curtis,
     canberra,
     chebyshev,
     manhattan,
     mahalanobis,
     minkowski,
     dice,
     hamming,
     jaccard,
     kulsinski,
     rogers_tanimoto,
     russellrao,
     sokal_michener,
     sokal_sneath,
     yule)


def cknn_graph(X, n_neighbors, delta=1.0, metric='euclidean', t='inf',
                       include_self=False, is_sparse=True,
                       return_instance=False):

    c_knn = CkNearestNeighbors(n_neighbors=n_neighbors, delta=delta,
                              metric=metric, t=t, include_self=include_self,
                              is_sparse=is_sparse)
    graph = c_knn.cknneighbors_graph(X)
    graph[(np.arange(graph.shape[0]), np.arange(graph.shape[0]))] = 0

    if return_instance:
        return c_knn
    else:
        return graph

class CkNearestNeighbors(object):
    """This object provides the all logic of CkNN.
    Args:
        n_neighbors: int, optional, default=5
            Number of neighbors to estimate the density around the point.
            It appeared as a parameter `k` in the paper.
        delta: float, optional, default=1.0
            A parameter to decide the radius for each points. The combination
            radius increases in proportion to this parameter.
        metric: str, optional, default='euclidean'
            The metric of each points. This parameter depends on the parameter
            `metric` of scipy.spatial.distance.pdist.
        t: 'inf' or float or int, optional, default='inf'
            The decay parameter of heat kernel. The weights are calculated as
            follow:
                W_{ij} = exp(-(||x_{i}-x_{j}||^2)/t)
            For more infomation, read the paper 'Laplacian Eigenmaps for
            Dimensionality Reduction and Data Representation', Belkin, et. al.
        include_self: bool, optional, default=True
            All diagonal elements are 1.0 if this parameter is True.
        is_sparse: bool, optional, default=True
            The method `cknneighbors_graph` returns csr_matrix object if this
            parameter is True else returns ndarray object.
        return_adjacency: bool, optional, default=False
            Whether to return the adjacency matrix instead of the estimated similarity.

        """

    def __repr__(self):
        if (self.n is not None):
            msg = "CkNearestNeighbors() object with %i fitted samples" % (self.n)
        else:
            msg = "CkNearestNeighbors() object object without any fitted data."
        return msg

    def __init__(self, n_neighbors=10, delta=1.0, metric='euclidean', t='inf',
                 include_self=False, is_sparse=True):
        self.n_neighbors = n_neighbors
        self.delta = delta
        self.metric = metric
        self.t = t
        self.include_self = include_self
        self.is_sparse = is_sparse
        self.n = None
        self.K = None
        self.n = None
        self.m = None
        if self.metric == 'euclidean':
            self.metric_fun = euclidean
        elif metric == 'standardised_euclidean':
            self.metric_fun = standardised_euclidean
        elif metric == 'cosine':
            self.metric_fun = cosine
        elif metric == 'correlation':
            self.metric_fun = correlation
        elif metric == 'bray_curtis':
            self.metric_fun = bray_curtis
        elif metric == 'canberra':
            self.metric_fun = canberra
        elif metric == 'chebyshev':
            self.metric_fun = chebyshev
        elif metric == 'manhattan':
            self.metric_fun = manhattan
        elif metric == 'mahalanobis':
            self.metric_fun = mahalanobis
        elif metric == 'minkowski':
            self.metric_fun = minkowski
        elif metric == 'dice':
            self.metric_fun = dice
        elif metric == 'hamming':
            self.metric_fun = hamming
        elif metric == 'jaccard':
            self.metric_fun = jaccard
        elif metric == 'kulsinski':
            self.metric_fun = kulsinski
        elif metric == 'rogers_tanimoto':
            self.metric_fun = rogers_tanimoto
        elif metric == 'russellrao':
            self.metric_fun = russellrao
        elif metric == 'sokal_michener':
            self.metric_fun = sokal_michener
        elif metric == 'sokal_sneath':
            self.metric_fun = sokal_sneath
        elif metric == 'yule':
            self.metric_fun = yule

    def cknneighbors_graph(self, X):
        """A method to calculate the CkNN graph
        Args:
            X: ndarray
                The data matrix.

        return: csr_matrix (if self.is_sparse is True)
                or ndarray(if self.is_sparse is False)
        """
        self.n = np.shape(X)[0]
        n_neighbors = self.n_neighbors
        delta = self.delta
        metric = self.metric
        t = self.t
        include_self = self.include_self
        is_sparse = self.is_sparse

        n_samples = X.shape[0]

        if n_neighbors < 1 or n_neighbors > n_samples-1:
            raise ValueError("`n_neighbors` must be "
                             "in the range 1 to number of samples")
        if len(X.shape) != 2:
            raise ValueError("`X` must be 2D matrix")
        if n_samples < 2:
            raise ValueError("At least 2 data points are required")

        if metric == 'precomputed':
            if X.shape[0] != X.shape[1]:
                raise ValueError("`X` must be square matrix")
            dmatrix = X
        else:
            dmatrix = pairwise_distances(X, metric=metric)

        darray_n_nbrs = np.partition(dmatrix, n_neighbors)[:, [n_neighbors]]
        # prevent approximately null results (div by 0)
        div_matrix = np.sqrt(darray_n_nbrs.dot(darray_n_nbrs.T)) + 1e-12
        ratio_matrix = dmatrix / div_matrix
        diag_ptr = np.arange(n_samples)

        if isinstance(delta, (int, float)):
            ValueError("Invalid argument type. "
                       "Type of `delta` must be float or int")
        A = csr_matrix(ratio_matrix < delta)

        if include_self:
            A[diag_ptr, diag_ptr] = True
        else:
            A[diag_ptr, diag_ptr] = False

        if t == 'inf':
            K = A.astype(np.float)
        else:
            mask = A.nonzero()
            weights = np.exp(-np.power(dmatrix[mask], 2)/t)
            dmatrix[:] = 0.
            dmatrix[mask] = weights
            K = csr_matrix(dmatrix)
        if not is_sparse:
            K = K.toarray()
        return K

