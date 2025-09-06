"""
Sparse distance metrics optimised with numba.

This module provides Euclidean, Poincaré and cosine distances for
vectors stored in sparse CSR format (index and data arrays).  The
implementations are refactored for efficiency and include gradient
calculations with respect to the first argument.  Based on the original
sparse utilities by Leland McInnes.
"""

import numba
import numpy as np

from topo.utils.umap_utils import norm


@numba.njit(fastmath=True)
def sparse_euclidean(ind1, data1, ind2, data2):
    """Euclidean distance between two sparse vectors."""
    i1 = 0
    i2 = 0
    result = 0.0
    while i1 < ind1.shape[0] and i2 < ind2.shape[0]:
        j1 = ind1[i1]
        j2 = ind2[i2]
        if j1 == j2:
            diff = data1[i1] - data2[i2]
            result += diff * diff
            i1 += 1
            i2 += 1
        elif j1 < j2:
            val = data1[i1]
            result += val * val
            i1 += 1
        else:
            val = data2[i2]
            result += val * val
            i2 += 1
    while i1 < ind1.shape[0]:
        val = data1[i1]
        result += val * val
        i1 += 1
    while i2 < ind2.shape[0]:
        val = data2[i2]
        result += val * val
        i2 += 1
    return np.sqrt(result)


@numba.njit(parallel=True, fastmath=True)
def sparse_euclidean_grad(ind1, data1, ind2, data2):
    """Euclidean distance and gradient with respect to ``ind1``/``data1``."""
    i1 = 0
    i2 = 0
    max_len = ind1.shape[0] + ind2.shape[0]
    grad_ind = np.empty(max_len, dtype=ind1.dtype)
    grad_data = np.empty(max_len, dtype=np.float32)
    nnz = 0
    dist_sq = 0.0
    while i1 < ind1.shape[0] and i2 < ind2.shape[0]:
        j1 = ind1[i1]
        j2 = ind2[i2]
        if j1 == j2:
            diff = data1[i1] - data2[i2]
            if diff != 0.0:
                grad_ind[nnz] = j1
                grad_data[nnz] = diff
                nnz += 1
            dist_sq += diff * diff
            i1 += 1
            i2 += 1
        elif j1 < j2:
            diff = data1[i1]
            grad_ind[nnz] = j1
            grad_data[nnz] = diff
            nnz += 1
            dist_sq += diff * diff
            i1 += 1
        else:
            diff = -data2[i2]
            grad_ind[nnz] = j2
            grad_data[nnz] = diff
            nnz += 1
            dist_sq += diff * diff
            i2 += 1
    while i1 < ind1.shape[0]:
        diff = data1[i1]
        grad_ind[nnz] = ind1[i1]
        grad_data[nnz] = diff
        nnz += 1
        dist_sq += diff * diff
        i1 += 1
    while i2 < ind2.shape[0]:
        diff = -data2[i2]
        grad_ind[nnz] = ind2[i2]
        grad_data[nnz] = diff
        nnz += 1
        dist_sq += diff * diff
        i2 += 1
    dist = np.sqrt(dist_sq)
    if dist == 0.0:
        return 0.0, grad_ind[:0], grad_data[:0]
    denom = dist + 1e-8
    for k in numba.prange(nnz):
        grad_data[k] = grad_data[k] / denom
    return dist, grad_ind[:nnz], grad_data[:nnz]


@numba.njit(fastmath=True)
def sparse_poincare(ind1, data1, ind2, data2):
    """Poincaré distance between two sparse vectors."""
    i1 = 0
    i2 = 0
    uu = 0.0
    vv = 0.0
    duv = 0.0
    while i1 < ind1.shape[0] and i2 < ind2.shape[0]:
        j1 = ind1[i1]
        j2 = ind2[i2]
        if j1 == j2:
            x = data1[i1]
            y = data2[i2]
            diff = x - y
            uu += x * x
            vv += y * y
            duv += diff * diff
            i1 += 1
            i2 += 1
        elif j1 < j2:
            x = data1[i1]
            uu += x * x
            duv += x * x
            i1 += 1
        else:
            y = data2[i2]
            vv += y * y
            duv += y * y
            i2 += 1
    while i1 < ind1.shape[0]:
        x = data1[i1]
        uu += x * x
        duv += x * x
        i1 += 1
    while i2 < ind2.shape[0]:
        y = data2[i2]
        vv += y * y
        duv += y * y
        i2 += 1
    alpha = 1.0 - uu
    beta = 1.0 - vv
    denom = alpha * beta
    arg = 1.0 + 2.0 * duv / denom
    return np.arccosh(arg)


@numba.njit(parallel=True, fastmath=True)
def sparse_poincare_grad(ind1, data1, ind2, data2):
    """Poincaré distance and gradient with respect to ``ind1``/``data1``."""
    i1 = 0
    i2 = 0
    max_len = ind1.shape[0] + ind2.shape[0]
    grad_ind = np.empty(max_len, dtype=ind1.dtype)
    diff_vals = np.empty(max_len, dtype=np.float32)
    x_vals = np.empty(max_len, dtype=np.float32)
    nnz = 0
    uu = 0.0
    vv = 0.0
    duv = 0.0
    while i1 < ind1.shape[0] and i2 < ind2.shape[0]:
        j1 = ind1[i1]
        j2 = ind2[i2]
        if j1 == j2:
            x = data1[i1]
            y = data2[i2]
            diff = x - y
            uu += x * x
            vv += y * y
            duv += diff * diff
            grad_ind[nnz] = j1
            diff_vals[nnz] = diff
            x_vals[nnz] = x
            nnz += 1
            i1 += 1
            i2 += 1
        elif j1 < j2:
            x = data1[i1]
            uu += x * x
            duv += x * x
            grad_ind[nnz] = j1
            diff_vals[nnz] = x
            x_vals[nnz] = x
            nnz += 1
            i1 += 1
        else:
            y = data2[i2]
            vv += y * y
            duv += y * y
            grad_ind[nnz] = j2
            diff_vals[nnz] = -y
            x_vals[nnz] = 0.0
            nnz += 1
            i2 += 1
    while i1 < ind1.shape[0]:
        x = data1[i1]
        uu += x * x
        duv += x * x
        grad_ind[nnz] = ind1[i1]
        diff_vals[nnz] = x
        x_vals[nnz] = x
        nnz += 1
        i1 += 1
    while i2 < ind2.shape[0]:
        y = data2[i2]
        vv += y * y
        duv += y * y
        grad_ind[nnz] = ind2[i2]
        diff_vals[nnz] = -y
        x_vals[nnz] = 0.0
        nnz += 1
        i2 += 1
    alpha = 1.0 - uu
    beta = 1.0 - vv
    denom_base = alpha * beta
    arg = 1.0 + 2.0 * duv / denom_base
    if arg <= 1.0 or alpha <= 0.0 or beta <= 0.0:
        return 0.0, grad_ind[:0], diff_vals[:0]
    denom = alpha * alpha * beta * np.sqrt(arg - 1.0) * np.sqrt(arg + 1.0)
    grad_data = diff_vals[:nnz]
    for k in numba.prange(nnz):
        diff_k = diff_vals[k]
        grad_data[k] = 4.0 * ((diff_k * alpha) + duv * x_vals[k]) / denom
    dist = np.arccosh(arg)
    return dist, grad_ind[:nnz], grad_data


@numba.njit(fastmath=True)
def sparse_cosine(ind1, data1, ind2, data2):
    """Cosine distance between two sparse vectors."""
    i1 = 0
    i2 = 0
    dot = 0.0
    while i1 < ind1.shape[0] and i2 < ind2.shape[0]:
        j1 = ind1[i1]
        j2 = ind2[i2]
        if j1 == j2:
            dot += data1[i1] * data2[i2]
            i1 += 1
            i2 += 1
        elif j1 < j2:
            i1 += 1
        else:
            i2 += 1
    norm1 = norm(data1)
    norm2 = norm(data2)
    if norm1 == 0.0 and norm2 == 0.0:
        return 0.0
    if norm1 == 0.0 or norm2 == 0.0:
        return 1.0
    return 1.0 - dot / (norm1 * norm2)


@numba.njit(parallel=True, fastmath=True)
def sparse_cosine_grad(ind1, data1, ind2, data2):
    """Cosine distance and gradient with respect to ``ind1``/``data1``."""
    i1 = 0
    i2 = 0
    max_len = ind1.shape[0] + ind2.shape[0]
    grad_ind = np.empty(max_len, dtype=ind1.dtype)
    x_vals = np.empty(max_len, dtype=np.float32)
    y_vals = np.empty(max_len, dtype=np.float32)
    nnz = 0
    dot = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0
    while i1 < ind1.shape[0] and i2 < ind2.shape[0]:
        j1 = ind1[i1]
        j2 = ind2[i2]
        if j1 == j2:
            x = data1[i1]
            y = data2[i2]
            dot += x * y
            norm1_sq += x * x
            norm2_sq += y * y
            grad_ind[nnz] = j1
            x_vals[nnz] = x
            y_vals[nnz] = y
            nnz += 1
            i1 += 1
            i2 += 1
        elif j1 < j2:
            x = data1[i1]
            norm1_sq += x * x
            grad_ind[nnz] = j1
            x_vals[nnz] = x
            y_vals[nnz] = 0.0
            nnz += 1
            i1 += 1
        else:
            y = data2[i2]
            norm2_sq += y * y
            grad_ind[nnz] = j2
            x_vals[nnz] = 0.0
            y_vals[nnz] = y
            nnz += 1
            i2 += 1
    while i1 < ind1.shape[0]:
        x = data1[i1]
        norm1_sq += x * x
        grad_ind[nnz] = ind1[i1]
        x_vals[nnz] = x
        y_vals[nnz] = 0.0
        nnz += 1
        i1 += 1
    while i2 < ind2.shape[0]:
        y = data2[i2]
        norm2_sq += y * y
        grad_ind[nnz] = ind2[i2]
        x_vals[nnz] = 0.0
        y_vals[nnz] = y
        nnz += 1
        i2 += 1
    norm1 = np.sqrt(norm1_sq)
    norm2 = np.sqrt(norm2_sq)
    if norm1 == 0.0 and norm2 == 0.0:
        return 0.0, grad_ind[:0], x_vals[:0]
    if norm1 == 0.0 or norm2 == 0.0:
        return 1.0, grad_ind[:0], x_vals[:0]
    denom = norm1 * norm1 * norm1 * norm2
    grad_data = np.empty(nnz, dtype=np.float32)
    for k in numba.prange(nnz):
        grad_data[k] = -(x_vals[k] * dot - y_vals[k] * norm1_sq) / denom
    dist = 1.0 - dot / (norm1 * norm2)
    return dist, grad_ind[:nnz], grad_data


sparse_named_distances = {
    "euclidean": sparse_euclidean,
    "l2": sparse_euclidean,
    "poincare": sparse_poincare,
    "cosine": sparse_cosine,
}

sparse_named_distances_with_gradients = {
    "euclidean": sparse_euclidean_grad,
    "l2": sparse_euclidean_grad,
    "poincare": sparse_poincare_grad,
    "cosine": sparse_cosine_grad,
}

sparse_need_n_features = ()
SPARSE_SPECIAL_METRICS = {}
