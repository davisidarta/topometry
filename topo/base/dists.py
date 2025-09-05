# These are some distance functions implemented in UMAP with numba, added here as module
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

"""Efficient distance computations used throughout the package.

Only a small subset of the original UMAP distance functions are kept here:
euclidean, poincare and cosine distances together with their gradients.
The implementations favour vectorised numpy operations and numba's JIT
compilation for speed.  A few convenience functions for pairwise distance
computation are also provided and make use of numba's parallelisation when
available.
"""

# Silly trick to workaround ReadTheDocs documentation build
# (it does not install numba and fails)
try:  # pragma: no cover - numba is optional
    import numba

    _have_numba = True
except ImportError:  # pragma: no cover - executed only without numba
    _have_numba = False

from sklearn.metrics import pairwise_distances as sklearn_pairwise_distances

if _have_numba:
    import numpy as np

    # ------------------------------------------------------------------
    # Single distance evaluations
    # ------------------------------------------------------------------

    @numba.njit(fastmath=True)
    def euclidean(x, y):
        """Standard Euclidean distance."""

        diff = x - y
        return np.sqrt(np.dot(diff, diff))

    @numba.njit(fastmath=True)
    def euclidean_grad(x, y):
        """Euclidean distance and gradient with respect to ``x``."""

        diff = x - y
        dist = np.sqrt(np.dot(diff, diff))
        grad = diff / (1e-8 + dist)
        return dist, grad

    @numba.njit(fastmath=True)
    def poincare(u, v):
        """Poincaré ball distance."""

        uu = np.dot(u, u)
        vv = np.dot(v, v)
        diff = u - v
        duv = np.dot(diff, diff)
        denom = (1.0 - uu) * (1.0 - vv)
        return np.arccosh(1.0 + 2.0 * duv / denom)

    @numba.njit(fastmath=True)
    def poincare_grad(u, v):
        """Poincaré distance and gradient with respect to ``u``."""

        uu = np.dot(u, u)
        vv = np.dot(v, v)
        diff = u - v
        duv = np.dot(diff, diff)
        alpha = 1.0 - uu
        beta = 1.0 - vv
        arg = 1.0 + 2.0 * duv / (alpha * beta)
        denom = alpha * alpha * beta * np.sqrt(arg - 1.0) * np.sqrt(arg + 1.0)
        grad = 4.0 * ((diff * alpha) + duv * u) / denom
        dist = np.arccosh(arg)
        return dist, grad

    @numba.njit(fastmath=True)
    def cosine(x, y):
        """Cosine distance."""

        num = np.dot(x, y)
        norm_x = np.sqrt(np.dot(x, x))
        norm_y = np.sqrt(np.dot(y, y))
        if norm_x == 0.0 and norm_y == 0.0:
            return 0.0
        if norm_x == 0.0 or norm_y == 0.0:
            return 1.0
        return 1.0 - num / (norm_x * norm_y)

    @numba.njit(fastmath=True)
    def cosine_grad(x, y):
        """Cosine distance and gradient with respect to ``x``."""

        num = np.dot(x, y)
        norm_x = np.sqrt(np.dot(x, x))
        norm_y = np.sqrt(np.dot(y, y))
        if norm_x == 0.0 and norm_y == 0.0:
            return 0.0, np.zeros_like(x)
        if norm_x == 0.0 or norm_y == 0.0:
            return 1.0, np.zeros_like(x)
        dist = 1.0 - num / (norm_x * norm_y)
        grad = -(x * num - y * norm_x * norm_x) / (norm_x ** 3 * norm_y)
        return dist, grad

    # ------------------------------------------------------------------
    # Pairwise distances
    # ------------------------------------------------------------------

    @numba.njit(parallel=True, fastmath=True)
    def _pairwise_euclidean(X, Y):
        result = np.empty((X.shape[0], Y.shape[0]), dtype=np.float32)
        for i in numba.prange(X.shape[0]):
            for j in range(Y.shape[0]):
                diff = X[i] - Y[j]
                result[i, j] = np.sqrt(np.dot(diff, diff))
        return result

    @numba.njit(parallel=True, fastmath=True)
    def _pairwise_poincare(X, Y):
        result = np.empty((X.shape[0], Y.shape[0]), dtype=np.float32)
        for i in numba.prange(X.shape[0]):
            for j in range(Y.shape[0]):
                result[i, j] = poincare(X[i], Y[j])
        return result

    @numba.njit(parallel=True, fastmath=True)
    def _pairwise_cosine(X, Y):
        result = np.empty((X.shape[0], Y.shape[0]), dtype=np.float32)
        for i in numba.prange(X.shape[0]):
            xi = X[i]
            norm_x = np.sqrt(np.dot(xi, xi))
            for j in range(Y.shape[0]):
                yj = Y[j]
                norm_y = np.sqrt(np.dot(yj, yj))
                if norm_x == 0.0 and norm_y == 0.0:
                    result[i, j] = 0.0
                elif norm_x == 0.0 or norm_y == 0.0:
                    result[i, j] = 1.0
                else:
                    result[i, j] = 1.0 - np.dot(xi, yj) / (norm_x * norm_y)
        return result

    def pairwise_euclidean(X, Y=None, n_jobs=-1):
        if Y is None:
            Y = X
        if n_jobs != -1:
            numba.set_num_threads(n_jobs)
        return _pairwise_euclidean(X, Y)

    def pairwise_poincare(X, Y=None, n_jobs=-1):
        if Y is None:
            Y = X
        if n_jobs != -1:
            numba.set_num_threads(n_jobs)
        return _pairwise_poincare(X, Y)

    def pairwise_cosine(X, Y=None, n_jobs=-1):
        if Y is None:
            Y = X
        if n_jobs != -1:
            numba.set_num_threads(n_jobs)
        return _pairwise_cosine(X, Y)

    @numba.njit(parallel=True, fastmath=True)
    def _matrix_pairwise_distance(a, metric):
        n = a.shape[0]
        out = np.empty((n, n), dtype=np.float32)
        for i in numba.prange(n):
            ai = a[i]
            for j in range(n):
                out[i, j] = metric(ai, a[j])
        return out

    @numba.njit(parallel=True, fastmath=True)
    def _matrix_to_matrix_distance(a, b, metric):
        n, m = a.shape[0], b.shape[0]
        out = np.empty((n, m), dtype=np.float32)
        for i in numba.prange(n):
            ai = a[i]
            for j in range(m):
                out[i, j] = metric(ai, b[j])
        return out

    @numba.njit(parallel=True, fastmath=True)
    def _cosine_vector_to_matrix(u, m):
        out = np.empty(m.shape[0], dtype=np.float32)
        norm_u = np.sqrt(np.dot(u, u))
        for i in numba.prange(m.shape[0]):
            mi = m[i]
            norm_m = np.sqrt(np.dot(mi, mi))
            if norm_u == 0.0 and norm_m == 0.0:
                out[i] = 0.0
            elif norm_u == 0.0 or norm_m == 0.0:
                out[i] = 1.0
            else:
                out[i] = 1.0 - np.dot(u, mi) / (norm_u * norm_m)
        return out

    def cosine_vector_to_matrix(u, m, n_jobs=-1):
        if n_jobs != -1:
            numba.set_num_threads(n_jobs)
        return _cosine_vector_to_matrix(u, m)

    def cosine_pairwise_distance(a, n_jobs=-1):
        if n_jobs != -1:
            numba.set_num_threads(n_jobs)
        return _pairwise_cosine(a, a)

    def matrix_pairwise_distance(a, metric, n_jobs=-1):
        metric_func = named_distances[metric] if isinstance(metric, str) else metric
        if n_jobs != -1:
            numba.set_num_threads(n_jobs)
        return _matrix_pairwise_distance(a, metric_func)

    def matrix_to_matrix_distance(a, b, metric, n_jobs=-1):
        metric_func = named_distances[metric] if isinstance(metric, str) else metric
        if n_jobs != -1:
            numba.set_num_threads(n_jobs)
        return _matrix_to_matrix_distance(a, b, metric_func)

    # Mapping dictionaries -------------------------------------------------
    named_distances = {
        "euclidean": euclidean,
        "l2": euclidean,
        "poincare": poincare,
        "cosine": cosine,
    }

    named_distances_with_gradients = {
        "euclidean": euclidean_grad,
        "l2": euclidean_grad,
        "poincare": poincare_grad,
        "cosine": cosine_grad,
    }

    def pairwise_distances(X, Y=None, metric="euclidean", n_jobs=-1):
        """Compute pairwise distances between rows of ``X`` and ``Y``."""

        if metric in ("euclidean", "l2"):
            return pairwise_euclidean(X, Y, n_jobs=n_jobs)
        if metric == "poincare":
            return pairwise_poincare(X, Y, n_jobs=n_jobs)
        if metric == "cosine":
            return pairwise_cosine(X, Y, n_jobs=n_jobs)
        raise ValueError("Unknown metric: %s" % metric)

else:  # ------------------------------------------------------------------
    import numpy as np

    def euclidean(x, y):
        diff = x - y
        return float(np.linalg.norm(diff))

    def euclidean_grad(x, y):
        diff = x - y
        dist = np.linalg.norm(diff)
        grad = diff / (1e-8 + dist)
        return dist, grad

    def poincare(u, v):
        uu = np.dot(u, u)
        vv = np.dot(v, v)
        diff = u - v
        duv = np.dot(diff, diff)
        denom = (1.0 - uu) * (1.0 - vv)
        return np.arccosh(1.0 + 2.0 * duv / denom)

    def poincare_grad(u, v):
        uu = np.dot(u, u)
        vv = np.dot(v, v)
        diff = u - v
        duv = np.dot(diff, diff)
        alpha = 1.0 - uu
        beta = 1.0 - vv
        arg = 1.0 + 2.0 * duv / (alpha * beta)
        denom = alpha * alpha * beta * np.sqrt(arg - 1.0) * np.sqrt(arg + 1.0)
        grad = 4.0 * ((diff * alpha) + duv * u) / denom
        dist = np.arccosh(arg)
        return dist, grad

    def cosine(x, y):
        num = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        if norm_x == 0.0 and norm_y == 0.0:
            return 0.0
        if norm_x == 0.0 or norm_y == 0.0:
            return 1.0
        return 1.0 - num / (norm_x * norm_y)

    def cosine_grad(x, y):
        num = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        if norm_x == 0.0 and norm_y == 0.0:
            return 0.0, np.zeros_like(x)
        if norm_x == 0.0 or norm_y == 0.0:
            return 1.0, np.zeros_like(x)
        dist = 1.0 - num / (norm_x * norm_y)
        grad = -(x * num - y * norm_x * norm_x) / (norm_x ** 3 * norm_y)
        return dist, grad

    def pairwise_euclidean(X, Y=None, n_jobs=-1):
        return sklearn_pairwise_distances(X, Y, metric="euclidean")

    def pairwise_poincare(X, Y=None, n_jobs=-1):
        if Y is None:
            Y = X
        result = np.empty((X.shape[0], Y.shape[0]), dtype=float)
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                result[i, j] = poincare(X[i], Y[j])
        return result

    def pairwise_cosine(X, Y=None, n_jobs=-1):
        return sklearn_pairwise_distances(X, Y, metric="cosine")

    def cosine_pairwise_distance(a, n_jobs=-1):
        return pairwise_cosine(a)

    def cosine_vector_to_matrix(u, m, n_jobs=-1):
        norm_u = np.linalg.norm(u)
        out = np.empty(m.shape[0], dtype=float)
        for i in range(m.shape[0]):
            norm_m = np.linalg.norm(m[i])
            if norm_u == 0.0 and norm_m == 0.0:
                out[i] = 0.0
            elif norm_u == 0.0 or norm_m == 0.0:
                out[i] = 1.0
            else:
                out[i] = 1.0 - np.dot(u, m[i]) / (norm_u * norm_m)
        return out

    def matrix_pairwise_distance(a, metric, n_jobs=-1):
        metric_func = named_distances[metric] if isinstance(metric, str) else metric
        n = a.shape[0]
        out = np.empty((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                out[i, j] = metric_func(a[i], a[j])
        return out

    def matrix_to_matrix_distance(a, b, metric, n_jobs=-1):
        metric_func = named_distances[metric] if isinstance(metric, str) else metric
        n, m = a.shape[0], b.shape[0]
        out = np.empty((n, m), dtype=float)
        for i in range(n):
            for j in range(m):
                out[i, j] = metric_func(a[i], b[j])
        return out

    def pairwise_distances(X, Y=None, metric="euclidean", n_jobs=-1):
        if metric in ("euclidean", "l2", "cosine"):
            return sklearn_pairwise_distances(X, Y, metric=metric if metric != "l2" else "euclidean")
        if metric == "poincare":
            return pairwise_poincare(X, Y, n_jobs=n_jobs)
        raise ValueError("Unknown metric: %s" % metric)

    named_distances = {
        "euclidean": euclidean,
        "l2": euclidean,
        "poincare": poincare,
        "cosine": cosine,
    }

    named_distances_with_gradients = {
        "euclidean": euclidean_grad,
        "l2": euclidean_grad,
        "poincare": poincare_grad,
        "cosine": cosine_grad,
    }

