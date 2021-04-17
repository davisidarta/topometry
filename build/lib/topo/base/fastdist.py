from numba import jit
import numpy as np
import math


@jit(nopython=False, fastmath=True)
def init_w(w, n):
    """
    :purpose:
    Initialize a weight array consistent of 1s if none is given
    This is called at the start of each function containing a w param
    :params:
    w      : a weight vector, if one was given to the initial function, else None
             NOTE: w MUST be an array of np.float64. so, even if you want a boolean w,
             convert it to np.float64 (using w.astype(np.float64)) before passing it to
             any function
    n      : the desired length of the vector of 1s (often set to len(u))
    :returns:
    w      : an array of 1s with shape (n,) if w is None, else return w un-changed
    """
    if w is None:
        return np.ones(n)
    else:
        return w


@jit(nopython=False, fastmath=True)
def braycurtis(u, v, w=None):
    """
    :purpose:
    Computes the Bray-Curtis distance between two 1D arrays
    :params:
    u, v   : input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output
    :returns:
    braycurtis : float, the Bray-Curtis distance between u and v
    :example:
    >>> u, v, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.braycurtis(u, v, w)
    0.3359619981199086
    """
    n = len(u)
    w = init_w(w, n)
    num, denom = 0, 0
    for i in range(n):
        num += abs(u[i] - v[i]) * w[i]
        denom += abs(u[i] + v[i]) * w[i]
    return num / denom


@jit(nopython=False, fastmath=True)
def canberra(u, v, w=None):
    """
    :purpose:
    Computes the Canberra distance between two 1D arrays
    :params:
    u, v   : input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output
    :returns:
    canberra : float, the Canberra distance between u and v
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> u, v, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.canberra(u, v, w)
    1951.0399135013315
    """
    n = len(u)
    w = init_w(w, n)
    dist = 0
    for i in range(n):
        num = abs(u[i] - v[i])
        denom = abs(u[i]) + abs(v[i])
        dist += num / denom * w[i]
    return dist


@jit(nopython=False, fastmath=True)
def chebyshev(u, v, w=None):
    """
    :purpose:
    Computes the Chebyshev distance between two 1D arrays
    :params:
    u, v   : input arrays, both of shape (n,)
    w      : here, w does nothing. it is only here for consistency
             with the other functions
    :returns:
    chebyshev : float, the Chebyshev distance between u and v
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> u, v, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.chebyshev(u, v, w)
    0.9934922585052587
    """
    return max(np.abs(u - v))


@jit(nopython=False, fastmath=True)
def cityblock(u, v, w=None):
    """
    :purpose:
    Computes the City Block distance between two 1D arrays
    :params:
    u, v   : input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output
    :returns:
    cityblock : float, the City Block distance between u and v
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> u, v, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.cityblock(u, v, w)
    1667.904767711218
    """
    n = len(u)
    w = init_w(w, n)
    dist = 0
    for i in range(n):
        dist += abs(u[i] - v[i]) * w[i]
    return dist


@jit(nopython=False, fastmath=True)
def correlation(u, v, w=None, centered=True):
    """
    :purpose:
    Computes the correlation between two 1D arrays
    :params:
    u, v   : input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output
    :returns:
    correlation : float, the correlation between u and v
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> u, v, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.correlation(u, v, w)
    0.9907907248975348
    """
    n = len(u)
    w = init_w(w, n)
    u_centered, v_centered = u - np.mean(u), v - np.mean(v)
    num = 0
    u_norm, v_norm = 0, 0
    for i in range(n):
        num += u_centered[i] * v_centered[i] * w[i]
        u_norm += abs(u_centered[i]) ** 2 * w[i]
        v_norm += abs(v_centered[i]) ** 2 * w[i]

    denom = (u_norm * v_norm) ** (1 / 2)
    return 1 - num / denom


@jit(nopython=False, fastmath=True)
def cosine(u, v, w=None):
    """
    :purpose:
    Computes the cosine similarity between two 1D arrays
    Unlike scipy's cosine distance, this returns similarity, which is 1 - distance
    :params:
    u, v   : input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output
    :returns:
    cosine  : float, the cosine similarity between u and v
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> u, v, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.cosine(u, v, w)
    0.7495065944399267
    """
    n = len(u)
    w = init_w(w, n)
    num = 0
    u_norm, v_norm = 0, 0
    for i in range(n):
        num += u[i] * v[i] * w[i]
        u_norm += abs(u[i]) ** 2 * w[i]
        v_norm += abs(v[i]) ** 2 * w[i]

    denom = (u_norm * v_norm) ** (1 / 2)
    return num / denom


@jit(nopython=False, fastmath=True)
def cosine_vector_to_matrix(u, m):
    """
    :purpose:
    Computes the cosine similarity between a 1D array and rows of a matrix
    :params:
    u      : input vector of shape (n,)
    m      : input matrix of shape (m, n)
    :returns:
    cosine vector  : np.array, of shape (m,) vector containing cosine similarity between u
                     and the rows of m
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> u = np.random.RandomState(seed=0).rand(10)
    >>> m = np.random.RandomState(seed=0).rand(100, 10)
    >>> fastdist.cosine_vector_to_matrix(u, m)
    (returns an array of shape (100,))
    """
    norm = 0
    for i in range(len(u)):
        norm += abs(u[i]) ** 2
    u = u / norm ** (1 / 2)
    for i in range(m.shape[0]):
        norm = 0
        for j in range(len(m[i])):
            norm += abs(m[i][j]) ** 2
        m[i] = m[i] / norm ** (1 / 2)
    return np.dot(u, m.T)


@jit(nopython=False, fastmath=True)
def cosine_matrix_to_matrix(a, b):
    """
    :purpose:
    Computes the cosine similarity between the rows of two matrices
    :params:
    a, b   : input matrices of shape (m, n) and (k, n)
             the matrices must share a common dimension at index 1
    :returns:
    cosine matrix  : np.array, an (m, k) array of the cosine similarity
                     between the rows of a and b
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> a = np.random.RandomState(seed=0).rand(10, 50)
    >>> b = np.random.RandomState(seed=0).rand(100, 50)
    >>> fastdist.cosine_matrix_to_matrix(a, b)
    (returns an array of shape (10, 100))
    """
    for i in range(a.shape[0]):
        norm = 0
        for j in range(len(a[i])):
            norm += abs(a[i][j]) ** 2
        a[i] = a[i] / norm ** (1 / 2)
    for i in range(b.shape[0]):
        norm = 0
        for j in range(len(b[i])):
            norm += abs(b[i][j]) ** 2
        b[i] = b[i] / norm ** (1 / 2)
    return np.dot(a, b.T)


@jit(nopython=False, fastmath=True)
def cosine_pairwise_distance(a, return_matrix=False):
    """
    :purpose:
    Computes the cosine similarity between the pairwise combinations of the rows of a matrix
    :params:
    a      : input matrix of shape (n, k)
    return_matrix : bool, whether to return the similarity as an (n, n) matrix
                    in which the (i, j) element is the cosine similarity
                    between rows i and j. if true, return the matrix.
                    if false, return a (n choose 2, 1) vector of the
                    similarities
    :returns:
    cosine matrix  : np.array, either an (n, n) matrix if return_matrix=True,
                     or an (n choose 2, 1) array if return_matrix=False
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> a = np.random.RandomState(seed=0).rand(10, 50)
    >>> fastdist.cosine_pairwise_distance(a, return_matrix=False)
    (returns an array of shape (45, 1))
    alternatively, with return_matrix=True:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> a = np.random.RandomState(seed=0).rand(10, 50)
    >>> fastdist.cosine_pairwise_distance(a, return_matrix=True)
    (returns an array of shape (10, 10))
    """
    n = a.shape[0]
    rows = np.arange(n)
    perm = [(rows[i], rows[j]) for i in range(n) for j in range(i + 1, n)]
    for i in range(n):
        norm = 0
        for j in range(len(a[i])):
            norm += abs(a[i][j]) ** 2
        a[i] = a[i] / norm ** (1 / 2)

    if return_matrix:
        out_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i):
                out_mat[i][j] = np.dot(a[i], a[j])
        out_mat += out_mat.T
        np.fill_diagonal(out_mat,1)
        return out_mat
    else:
        out = np.zeros((len(perm), 1))
        for i in range(len(perm)):
            out[i] = np.dot(a[perm[i][0]], a[perm[i][1]])
        return out


@jit(nopython=False, fastmath=True)
def euclidean(u, v, w=None):
    """
    :purpose:
    Computes the Euclidean distance between two 1D arrays
    :params:
    u, v   : input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output
    :returns:
    euclidean : float, the Euclidean distance between u and v
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> u, v, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.euclidean(u, v, w)
    28.822558591834163
    """
    n = len(u)
    w = init_w(w, n)
    dist = 0
    for i in range(n):
        dist += abs(u[i] - v[i]) ** 2 * w[i]
    return dist ** (1 / 2)


@jit(nopython=False, fastmath=True)
def rel_entr(x, y):
    """
    :purpose:
    Computes the relative entropy between two 1D arrays
    Used primarily for the jensenshannon function
    :params:
    x, y   : input arrays, both of shape (n,)
             to get a numerical value, x and y should be strictly non-negative;
             negative values result in infinite relative entropy
    :returns:
    rel_entr : float, the relative entropy distance of x and y
    """
    total_entr = 0
    for i in range(len(x)):
        if x[i] > 0 and y[i] > 0:
            total_entr += x[i] * math.log(x[i] / y[i])
        elif x[i] == 0 and y[i] >= 0:
            total_entr += 0
        else:
            total_entr += np.inf
    return total_entr


@jit(nopython=False, fastmath=True)
def jensenshannon(p, q, base=None):
    """
    :purpose:
    Computes the Jensen-Shannon divergence between two 1D probability arrays
    :params:
    u, v   : input probability arrays, both of shape (n,)
             note that because these are probability arrays, they are strictly non-negative
    base   : the base of the logarithm for the output
    :returns:
    jensenshannon : float, the Jensen-Shannon divergence between u and v
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> u, v = np.random.RandomState(seed=0).uniform(size=(10000, 2)).T
    >>> fastdist.jensenshannon(u, v, base=2)
    0.39076147897868996
    """
    p_sum, q_sum = 0, 0
    for i in range(len(p)):
        p_sum += p[i]
        q_sum += q[i]
    p, q = p / p_sum, q / q_sum
    m = (p + q) / 2
    num = rel_entr(p, m) + rel_entr(q, m)
    if base is not None:
        num /= math.log(base)
    return (num / 2) ** (1 / 2)


@jit(nopython=False, fastmath=True)
def mahalanobis(u, v, VI):
    """
    :purpose:
    Computes the Mahalanobis distance between two 1D arrays
    :params:
    u, v   : input arrays, both of shape (n,)
    VI     : the inverse of the covariance matrix of u and v
             note that some arrays will result in a VI containing
             very high values, leading to some imprecision
    :returns:
    mahalanobis : float, the Mahalanobis distance between u and v
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> u, v = np.array([2, 0, 0]).astype(np.float64), np.array([0, 1, 0]).astype(np.float64)
    >>> VI = np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
    >>> fastdist.mahalanobis(u, v, VI)
    1.7320508075688772
    """
    delta = (u - v)
    return np.dot(np.dot(delta, VI), delta) ** (1 / 2)


@jit(nopython=False, fastmath=True)
def minkowski(u, v, p, w=None):
    """
    :purpose:
    Computes the Minkowski distance between two 1D arrays
    :params:
    u, v   : input arrays, both of shape (n,)
    p      : the order of the norm (p=2 is the same as Euclidean)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output
    :returns:
    minkowski : float, the Minkowski distance between u and v
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> u, v, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> p = 3
    >>> fastdist.minkowski(u, v, p, w)
    7.904971256091215
    """
    n = len(u)
    w = init_w(w, n)
    dist = 0
    for i in range(n):
        dist += abs(u[i] - v[i]) ** p * w[i]
    return dist ** (1 / p)


@jit(nopython=False, fastmath=True)
def seuclidean(u, v, V):
    """
    :purpose:
    Computes the standardized Euclidean distance between two 1D arrays
    :params:
    u, v   : input arrays, both of shape (n,)
    V      : array of shape (n,) containing component variances
    :returns:
    seuclidean : float, the standardized Euclidean distance between u and v
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> u, v, V = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.seuclidean(u, v, V)
    116.80739235578636
    """
    return euclidean(u, v, w=1 / V)


@jit(nopython=False, fastmath=True)
def sqeuclidean(u, v, w=None):
    """
    :purpose:
    Computes the squared Euclidean distance between two 1D arrays
    :params:
    u, v   : input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output
    :returns:
    sqeuclidean : float, the squared Euclidean distance between u and v
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> u, v, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.sqeuclidean(u, v, w)
    830.7398837797134
    """
    n = len(u)
    w = init_w(w, n)
    dist = 0
    for i in range(n):
        dist += abs(u[i] - v[i]) ** 2 * w[i]
    return dist


@jit(nopython=False, fastmath=True)
def dice(u, v, w=None):
    """
    :purpose:
    Computes the Dice dissimilarity between two boolean 1D arrays
    :params:
    u, v   : boolean input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output
    :returns:
    dice : float, the Dice dissimilarity between u and v
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> u, v = np.random.RandomState(seed=0).randint(2, size=(10000, 2)).T
    >>> w = np.random.RandomState(seed=0).rand(10000)
    >>> fastdist.dice(u, v, w)
    0.5008483098538385
    """
    n = len(u)
    w = init_w(w, n)
    num, denom = 0, 0
    for i in range(n):
        num += u[i] * v[i] * w[i]
        denom += (u[i] + v[i]) * w[i]
    return 1 - 2 * num / denom


@jit(nopython=False, fastmath=True)
def hamming(u, v, w=None):
    """
    :purpose:
    Computes the Hamming distance between two boolean 1D arrays
    :params:
    u, v   : boolean input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output
    :returns:
    hamming : float, the Hamming distance between u and v
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> u, v = np.random.RandomState(seed=0).randint(2, size=(10000, 2)).T
    >>> w = np.random.RandomState(seed=0).rand(10000)
    >>> fastdist.hamming(u, v, w)
    0.5061006361240681
    """
    n = len(u)
    w = init_w(w, n)
    num, denom = 0, 0
    for i in range(n):
        if u[i] != v[i]:
            num += w[i]
        denom += w[i]
    return num / denom


@jit(nopython=False, fastmath=True)
def jaccard(u, v, w=None):
    """
    :purpose:
    Computes the Jaccard-Needham dissimilarity between two boolean 1D arrays
    :params:
    u, v   : boolean input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output
    :returns:
    jaccard : float, the Jaccard-Needham dissimilarity between u and v
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> u, v = np.random.RandomState(seed=0).randint(2, size=(10000, 2)).T
    >>> w = np.random.RandomState(seed=0).rand(10000)
    >>> fastdist.jaccard(u, v, w)
    0.6674202936639468
    """
    n = len(u)
    w = init_w(w, n)
    num, denom = 0, 0
    for i in range(n):
        if u[i] != v[i]:
            num += w[i]
            denom += w[i]
        denom += u[i] * v[i] * w[i]
    return num / denom


@jit(nopython=False, fastmath=True)
def kulsinski(u, v, w=None):
    """
    :purpose:
    Computes the Kulsinski dissimilarity between two boolean 1D arrays
    :params:
    u, v   : boolean input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output
    :returns:
    kulsinski : float, the Kulsinski dissimilarity between u and v
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> u, v = np.random.RandomState(seed=0).randint(2, size=(10000, 2)).T
    >>> w = np.random.RandomState(seed=0).rand(10000)
    >>> fastdist.kulsinski(u, v, w)
    0.8325522836573094
    """
    n = len(u)
    w = init_w(w, n)
    num, denom = 0, 0
    for i in range(n):
        num += (1 - u[i] * v[i]) * w[i]
        if u[i] != v[i]:
            num += w[i]
            denom += w[i]
        denom += w[i]
    return num / denom


@jit(nopython=False, fastmath=True)
def rogerstanimoto(u, v, w=None):
    """
    :purpose:
    Computes the Rogers-Tanimoto dissimilarity between two boolean 1D arrays
    :params:
    u, v   : boolean input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output
    :returns:
    rogerstanimoto : float, the Rogers-Tanimoto dissimilarity between u and v
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> u, v = np.random.RandomState(seed=0).randint(2, size=(10000, 2)).T
    >>> w = np.random.RandomState(seed=0).rand(10000)
    >>> fastdist.rogerstanimoto(u, v, w)
    0.672067488699178
    """
    n = len(u)
    w = init_w(w, n)
    r, denom = 0, 0
    for i in range(n):
        if u[i] != v[i]:
            r += 2 * w[i]
        else:
            denom += w[i]
    return r / (denom + r)


@jit(nopython=False, fastmath=True)
def russellrao(u, v, w=None):
    """
    :purpose:
    Computes the Ruseell-Rao dissimilarity between two boolean 1D arrays
    :params:
    u, v   : boolean input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output
    :returns:
    russelrao : float, the Russell-Rao dissimilarity between u and v
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> u, v = np.random.RandomState(seed=0).randint(2, size=(10000, 2)).T
    >>> w = np.random.RandomState(seed=0).rand(10000)
    >>> fastdist.russellrao(u, v, w)
    0.7478068878987577
    """
    n = len(u)
    w = init_w(w, n)
    num, n = 0, 0
    for i in range(n):
        num += u[i] * v[i] * w[i]
        n += w[i]
    return (n - num) / n


@jit(nopython=False, fastmath=True)
def sokalmichener(u, v, w=None):
    """
    :purpose:
    Computes the Sokal-Michener dissimilarity between two boolean 1D arrays
    :params:
    u, v   : boolean input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output
    :returns:
    sokalmichener : float, the Sokal-Michener dissimilarity between u and v
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> u, v = np.random.RandomState(seed=0).randint(2, size=(10000, 2)).T
    >>> w = np.random.RandomState(seed=0).rand(10000)
    >>> fastdist.sokalmichener(u, v, w)
    0.672067488699178
    :note:
    scipy's implementation returns a different value in the above example.
    when no w is given, our implementation and scipy's are the same.
    to replicate scipy's result of 0.8046210454292805, we can replace
    r += 2 * w[i] with r += 2, but then that does not apply the weights.
    so, we use (what we think) is the correct weight implementation
    """
    n = len(u)
    w = init_w(w, n)
    r, s = 0, 0
    for i in range(n):
        if u[i] != v[i]:
            r += 2 * w[i]
        else:
            s += w[i]
    return r / (s + r)


@jit(nopython=False, fastmath=True)
def sokalsneath(u, v, w=None):
    """
    :purpose:
    Computes the Sokal-Sneath dissimilarity between two boolean 1D arrays
    :params:
    u, v   : boolean input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output
    :returns:
    sokalsneath : float, the Sokal-Sneath dissimilarity between u and v
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> u, v = np.random.RandomState(seed=0).randint(2, size=(10000, 2)).T
    >>> w = np.random.RandomState(seed=0).rand(10000)
    >>> fastdist.sokalsneath(u, v, w)
    0.8005423661929552
    """
    n = len(u)
    w = init_w(w, n)
    r, denom = 0, 0
    for i in range(n):
        if u[i] != v[i]:
            r += 2 * w[i]
        denom += u[i] * v[i] * w[i]
    return r / (r + denom)


@jit(nopython=False, fastmath=True)
def yule(u, v, w=None):
    """
    :purpose:
    Computes the Yule dissimilarity between two boolean 1D arrays
    :params:
    u, v   : boolean input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output
    :returns:
    yule   : float, the Sokal-Sneath dissimilarity between u and v
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> u, v = np.random.RandomState(seed=0).randint(2, size=(10000, 2)).T
    >>> w = np.random.RandomState(seed=0).rand(10000)
    >>> fastdist.yule(u, v, w)
    1.0244476251862624
    """
    n = len(u)
    w = init_w(w, n)
    ctf, cft, ctt, cff = 0, 0, 0, 0
    for i in range(n):
        if u[i] != v[i] and u[i] == 1:
            ctf += w[i]
        elif u[i] != v[i] and u[i] == 0:
            cft += w[i]
        elif u[i] == v[i] == 1:
            ctt += w[i]
        elif u[i] == v[i] == 0:
            cff += w[i]
    return (2 * ctf * cft) / (ctt * cff + ctf * cft)


@jit(nopython=False, fastmath=True)
def vector_to_matrix_distance(u, m, metric, metric_name):
    """
    :purpose:
    Computes the distance between a vector and the rows of a matrix using any given metric
    :params:
    u      : input vector of shape (n,)
    m      : input matrix of shape (m, n)
    metric : the function used to calculate the distance
    metric_name : str of the function name. this is only used for
                  the if statement because cosine similarity has its
                  own function
    distance vector  : np.array, of shape (m,) vector containing the distance between u
                       and the rows of m
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> u = np.random.RandomState(seed=0).rand(10)
    >>> m = np.random.RandomState(seed=0).rand(100, 10)
    >>> fastdist.vector_to_matrix_distance(u, m, fastdist.cosine, "cosine")
    (returns an array of shape (100,))
    :note:
    the cosine similarity uses its own function, cosine_vector_to_matrix.
    this is because normalizing the rows and then taking the dot product
    of the vector and matrix heavily optimizes the computation. the other similarity
    metrics do not have such an optimization, so we loop through them
    """

    if metric_name == "cosine":
        return cosine_vector_to_matrix(u, m)

    n = m.shape[0]
    out = np.zeros((n))
    for i in range(n):
        out[i] = metric(u, m[i])
    return out


@jit(nopython=False, fastmath=True)
def matrix_to_matrix_distance(a, b, metric, metric_name):
    """
    :purpose:
    Computes the distance between the rows of two matrices using any given metric
    :params:
    a, b   : input matrices either of shape (m, n) and (k, n)
             the matrices must share a common dimension at index 1
    metric : the function used to calculate the distance
    metric_name : str of the function name. this is only used for
                  the if statement because cosine similarity has its
                  own function
    :returns:
    distance matrix  : np.array, an (m, k) array of the distance
                       between the rows of a and b
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> a = np.random.RandomState(seed=0).rand(10, 50)
    >>> b = np.random.RandomState(seed=0).rand(100, 50)
    >>> fastdist.matrix_to_matrix_distance(a, b, fastdist.cosine, "cosine")
    (returns an array of shape (10, 100))
    :note:
    the cosine similarity uses its own function, cosine_matrix_to_matrix.
    this is because normalizing the rows and then taking the dot product
    of the two matrices heavily optimizes the computation. the other similarity
    metrics do not have such an optimization, so we loop through them
    """
    if metric_name == "cosine":
        return cosine_matrix_to_matrix(a, b)
    n, m = a.shape[0], b.shape[0]
    out = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            out[i][j] = metric(a[i], b[j])
    return out


@jit(nopython=False, fastmath=True)
def matrix_pairwise_distance(a, metric, metric_name, return_matrix=False):
    """
    :purpose:
    Computes the distance between the pairwise combinations of the rows of a matrix
    :params:
    a      : input matrix of shape (n, k)
    metric : the function used to calculate the distance
    metric_name   : str of the function name. this is only used for
                    the if statement because cosine similarity has its
                    own function
    return_matrix : bool, whether to return the similarity as an (n, n) matrix
                    in which the (i, j) element is the cosine similarity
                    between rows i and j. if true, return the matrix.
                    if false, return a (n choose 2, 1) vector of the
                    similarities
    :returns:
    distance matrix  : np.array, either an (n, n) matrix if return_matrix=True,
                       or an (n choose 2, 1) array if return_matrix=False
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> a = np.random.RandomState(seed=0).rand(10, 50)
    >>> fastdist.matrix_pairwise_distance(a, fastdist.euclidean, "euclidean", return_matrix=False)
    (returns an array of shape (45, 1))
    alternatively, with return_matrix=True:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> a = np.random.RandomState(seed=0).rand(10, 50)
    >>> fastdist.matrix_pairwise_distance(a, fastdist.euclidean, "euclidean", return_matrix=True)
    (returns an array of shape (10, 10))
    """
    if metric_name == "cosine":
        return cosine_pairwise_distance(a, return_matrix)

    else:
        n = a.shape[0]
        rows = np.arange(n)
        perm = [(rows[i], rows[j]) for i in range(n) for j in range(i + 1, n)]
        if return_matrix:
            out_mat = np.zeros((n, n))
            for i in range(n):
                for j in range(i):
                    out_mat[i][j] = metric(a[i], a[j])
            return out_mat + out_mat.T
        else:
            out = np.zeros((len(perm), 1))
            for i in range(len(perm)):
                out[i] = metric(a[perm[i][0]], a[perm[i][1]])
            return out


## START OF SKLEARN METRICS IMPLEMENTATION

@jit(nopython=False, fastmath=True)
def variance(u, w=None):
    """
    :purpose:
    Computes the variance of a 1D array, used for r2 and explained variance score
    :params:
    u      : input array of shape (n,)
    w      : weights at each index of u. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output
    :returns:
    variance : float, the variance of u
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> u, w = np.random.RandomState(seed=0).rand(10000, 2).T
    >>> fastdist.variance(u, w)
    0.08447496068498446
    """
    n = len(u)
    w = init_w(w, n)
    num, denom = 0, 0
    for i in range(n):
        num += u[i] * w[i]
        denom += w[i]
    m = num / denom
    num, denom = 0, 0
    for i in range(n):
        num += abs(u[i] - m) ** 2 * w[i]
        denom += w[i]
    return num / denom


@jit(nopython=False, fastmath=True)
def r2_score(true, pred, w=None):
    """
    :purpose:
    Computes the r2 score between a predictions array and a target array
    :params:
    true, pred : input arrays, both of shape (n,)
    w          : weights at each index of true and pred. array of shape (n,)
                 if no w is set, it is initialized as an array of ones
                 such that it will have no impact on the output
    :returns:
    r2_score : float, the r2 score of the targets and predictions
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> true, pred, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.r2_score(true, pred, w)
    -0.9797313432213313
    """
    n = len(true)
    w = init_w(w, n)
    var_true = variance(true, w)
    num, denom = 0, 0
    for i in range(n):
        num += (pred[i] - true[i]) ** 2 * w[i]
        denom += w[i]
    return 1 - ((num / denom) / var_true)


@jit(nopython=False, fastmath=True)
def explained_variance_score(true, pred, w=None):
    """
    :purpose:
    Computes the explained variance score between a predictions array and a target array
    :params:
    true, pred : input arrays, both of shape (n,)
    w          : weights at each index of true and pred. array of shape (n,)
                 if no w is set, it is initialized as an array of ones
                 such that it will have no impact on the output
    :returns:
    explained_variance_score : float, the explained variance score of the targets and predictions
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> true, pred, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.explained_variance_score(true, pred, w)
    -0.979414934822632
    """
    var_true = variance(true, w=w)
    var_diff = variance(pred - true, w=w)
    return 1 - (var_diff / var_true)


@jit(nopython=False, fastmath=True)
def max_error(true, pred, w=None):
    """
    :purpose:
    Computes the max error between a predictions array and a target array
    :params:
    true, pred : input arrays, both of shape (n,)
    w          : here, w does nothing. it is only here for consistency
                 with the other functions
    :returns:
    max_error : float, the max error of the targets and predictions
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> true, pred, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.max_error(true, pred, w)
    0.9934922585052587
    """
    return max(np.abs(true - pred))


@jit(nopython=False, fastmath=True)
def mean_absolute_error(true, pred, w=None):
    """
    :purpose:
    Computes the mean absolute error between a predictions array and a target array
    :params:
    true, pred : input arrays, both of shape (n,)
    w          : weights at each index of true and pred. array of shape (n,)
                 if no w is set, it is initialized as an array of ones
                 such that it will have no impact on the output
    :returns:
    mean_absolute_error : float, the mean absolute error of the targets and predictions
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> true, pred, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.mean_absolute_error(true, pred, w)
    0.3353421174411754
    """
    n = len(true)
    w = init_w(w, n)
    num, denom = 0, 0
    for i in range(n):
        num += abs(true[i] - pred[i]) * w[i]
        denom += w[i]
    return num / denom


@jit(nopython=False, fastmath=True)
def mean_squared_error(true, pred, w=None, squared=True):
    """
    :purpose:
    Computes the mean squared error between a predictions array and a target array
    (can also be used for root mean squared error by setting squared=False)
    :params:
    true, pred : input arrays, both of shape (n,)
    w          : weights at each index of true and pred. array of shape (n,)
                 if no w is set, it is initialized as an array of ones
                 such that it will have no impact on the output
    squared    : whether to return MSE or RMSE, defaults to True, which
                 returns MSE (set to false for RMSE)
    :returns:
    mean_squared_error : float, the mean squared error of the targets and predictions
    :example for mean_squared_error:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> true, pred, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.mean_squared_error(true, pred, w, squared=True)
    0.16702516658178812
    :example for root_mean_squared error:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> true, pred, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.mean_squared_error(true, pred, w, squared=False)
    0.40868712553956005
    """
    to_square = 1 if squared else 2
    n = len(true)
    w = init_w(w, n)
    num, denom = 0, 0
    for i in range(n):
        num += abs(true[i] - pred[i]) ** 2 * w[i]
        denom += w[i]
    return (num / denom) ** (1 / to_square)


@jit(nopython=False, fastmath=True)
def mean_squared_log_error(true, pred, w=None):
    """
    :purpose:
    Computes the mean squared log error between a predictions array and a target array
    :params:
    true, pred : input arrays, both of shape (n,)
    w          : weights at each index of true and pred. array of shape (n,)
                 if no w is set, it is initialized as an array of ones
                 such that it will have no impact on the output
    :returns:
    mean_squared_log_error : float, the mean squared log error of the targets and predictions
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> true, pred, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.mean_squared_log_error(true, pred, w)
    0.07840806721686663
    """
    n = len(true)
    w = init_w(w, n)
    num, denom = 0, 0
    for i in range(n):
        num += abs(math.log(true[i] + 1) - math.log(pred[i] + 1)) ** 2 * w[i]
        denom += w[i]
    return num / denom


@jit(nopython=False, fastmath=True)
def median_absolute_error(true, pred, w=None):
    """
    :purpose:
    Computes the median absolute error between a predictions array and a target array
    :params:
    true, pred : input arrays, both of shape (n,)
    w          : here, w does nothing. it is only here for consistency
                 with the other functions
    :returns:
    median_absolute_error : float, the median absolute error of the targets and predictions
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> true, pred, w = np.random.RandomState(seed=0).rand(10000, 3).T
    >>> fastdist.median_absolute_error(true, pred, w)
    0.2976962111211224
    """
    return np.median(np.abs(true - pred))


@jit(nopython=False, fastmath=True)
def confusion_matrix(targets, preds, w=None, normalize=None):
    """
    :purpose:
    Creates a confusion matrix for an array of target and predicted classes
    (used in most of the other classification metrics, along with having its own use)
    :params:
    targets, preds : discrete input arrays, both of shape (n,)
    w              : weights at each index of true and pred. array of shape (n,)
                     if no w is set, it is initialized as an array of ones
                     such that it will have no impact on the output
    normalize      : how to normalize (if at all) the confusion matrix. options are
                     "true", which makes each row sum to 1, "pred", which makes columns
                     sum to 1, and "all", which makes the entire matrix sum to 1
    :returns:
    confusion_matrix : a confusion matrix (np.array) of shape (n_classes, n_classes)
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> true = np.random.RandomState(seed=0).randint(2, size=10000)
    >>> pred = np.random.RandomState(seed=1).randint(2, size=10000)
    >>> fastdist.confusion_matrix(true, pred)
    array([[2412., 2503.],
           [2594., 2491.]])
    """
    w = init_w(w, len(targets))
    n = max(len(np.unique(targets)), len(np.unique(preds)))
    cm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            correct = 0
            for val in range(len(targets)):
                if targets[val] == i and preds[val] == j:
                    correct += w[val]
            cm[i][j] = correct

    if normalize is None:
        return cm
    elif normalize == 'true':
        for i in range(n):
            row_sum = 0
            for j in range(n):
                row_sum += cm[i][j]
            cm[i] /= row_sum
        return cm
    elif normalize == 'pred':
        for i in range(n):
            col_sum = 0
            for j in range(n):
                col_sum += cm[j][i]
            cm[:, i] /= col_sum
        return cm
    elif normalize == 'all':
        total_sum = 0
        for i in range(n):
            for j in range(n):
                total_sum += cm[i][j]
        return cm / total_sum


@jit(nopython=False, fastmath=True)
def accuracy_score(targets, preds, w=None, normalize=True):
    """
    :purpose:
    Calculates the accuracy score between a discrete target and pred array
    :params:
    targets, preds : discrete input arrays, both of shape (n,)
    w              : weights at each index of true and pred. array of shape (n,)
                     if no w is set, it is initialized as an array of ones
                     such that it will have no impact on the output
    normalize      : bool. if true, the function returns (correct / total),
                     if false, the function returns (correct). defaults to true
    :returns:
    accuracy_score : float, the accuracy score of the targets and preds array
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> true = np.random.RandomState(seed=0).randint(2, size=10000)
    >>> pred = np.random.RandomState(seed=1).randint(2, size=10000)
    >>> fastdist.accuracy_score(true, pred)
    0.4903
    """
    w = init_w(w, len(targets))
    num, denom = 0, 0
    for i in range(len(targets)):
        if targets[i] == preds[i]:
            num += w[i]
        denom += w[i]
    return num / denom if normalize else num


@jit(nopython=False, fastmath=True)
def balanced_accuracy_score(targets, preds, cm=None, w=None, adjusted=False):
    """
    :purpose:
    Calculates the balanced accuracy score between a discrete target and pred array
    :params:
    targets, preds : discrete input arrays, both of shape (n,)
    cm             : if you have previously calculated a confusion matrix, pass it here to save the computation.
                     set as None, which makes the function calculate the confusion matrix
    w              : weights at each index of true and pred. array of shape (n,)
                     if no w is set, it is initialized as an array of ones
                     such that it will have no impact on the output
    adjusted       : bool. if true, adjust the output for chance (making 0 the worst
                     and 1 the best score). defaults to false
    :returns:
    balanced_accuracy_score : float, the balanced accuracy score of the targets and preds array
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> true = np.random.RandomState(seed=0).randint(2, size=10000)
    >>> pred = np.random.RandomState(seed=1).randint(2, size=10000)
    >>> fastdist.balanced_accuracy_score(true, pred)
    0.49030739883826424
    """
    w = init_w(w, len(targets))
    if cm is None:
        cm = confusion_matrix(targets, preds, w=w)
    n = cm.shape[0]
    diag, row_sums = np.zeros(n), np.zeros(n)
    for i in range(n):
        diag[i] = cm[i][i]
        for j in range(n):
            row_sums[i] += cm[i][j]

    class_div = diag / row_sums
    div_mean = 0
    for i in range(n):
        div_mean += class_div[i]
    div_mean /= n

    if adjusted:
        div_mean -= 1 / n
        div_mean /= 1 - 1 / n
    return div_mean


@jit(nopython=False, fastmath=True)
def brier_score_loss(targets, probs, w=None):
    """
    :purpose:
    Calculates the Brier score loss between an array of discrete targets and an array of probabilities
    :params:
    targets : discrete input array of shape (n,)
    probs   : input array of predicted probabilities for sample of shape (n,)
    w       : weights at each index of true and pred. array of shape (n,)
              if no w is set, it is initialized as an array of ones
              such that it will have no impact on the output
    :returns:
    brier_score_loss : float, the Brier score loss of the targets and probs array
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> true = np.random.RandomState(seed=0).randint(2, size=10000)
    >>> prob = np.random.RandomState(seed=0).uniform(size=10000)
    >>> fastdist.brier_score_loss(true, prob)
    0.5097
    """
    w = init_w(w, len(targets))
    num, denom = 0, 0
    for i in range(len(targets)):
        num += (probs[i] - targets[i]) ** 2 * w[i]
        denom += w[i]
    return num / denom


@jit(nopython=False, fastmath=True)
def precision_score(targets, preds, cm=None, w=None, average='binary'):
    """
    :purpose:
    Calculates the precision score between a discrete target and pred array
    :params:
    targets, preds : discrete input arrays, both of shape (n,)
    cm             : if you have previously calculated a confusion matrix, pass it here to save the computation.
                     set as None, which makes the function calculate the confusion matrix.
                     note that for your specific average (i.e., micro, macro, none, or binary), you must compute the confusion
                     matrix correctly corresponding to the one you would like to use. so, for "macro" or "none", the cm
                     must be computed with normalize="pred"
    w              : weights at each index of true and pred. array of shape (n,)
                     if no w is set, it is initialized as an array of ones
                     such that it will have no impact on the output
    average        : str, either "micro", "macro", "none", or "binary".
                     if "micro", computes precision globally
                     if "macro", take the mean of precision for each class (unweighted)
                     if "none", return a list of the precision for each class
                     if "binary", return precision in a binary classification problem
                     defaults to "binary", so for multi-class problems, you must change this
    :returns:
    precision_score : np.array, the precision score of the targets and preds array
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> true = np.random.RandomState(seed=0).randint(2, size=10000)
    >>> pred = np.random.RandomState(seed=1).randint(2, size=10000)
    >>> fastdist.precision_score(true, pred)
    array([0.49879856])
    """
    w = init_w(w, len(targets))
    if average == 'micro':
        if cm is None:
            cm = confusion_matrix(targets, preds, w=w)
        n = cm.shape[0]

        diag, row_sums = np.zeros(n), np.zeros(n)
        for i in range(n):
            diag[i] = cm[i][i]
            for j in range(n):
                row_sums += cm[i][j]
        class_div = diag / row_sums
        div_mean = 0.
        for i in range(n):
            div_mean += class_div[i]
        return np.array([div_mean])

    elif average == 'macro':
        if cm is None:
            cm = confusion_matrix(targets, preds, w=w, normalize='pred')
        n = cm.shape[0]

        diag, row_sums = np.zeros(n), np.zeros(n)
        for i in range(n):
            diag[i] = cm[i][i]
            for j in range(n):
                row_sums += cm[i][j]
        class_div = diag / row_sums * n
        class_mean = 0
        for i in range(n):
            class_mean += class_div[i]
        return np.array([class_mean / n])

    elif average == 'none':
        if cm is None:
            cm = confusion_matrix(targets, preds, w=w, normalize='pred')
        n = cm.shape[0]

        diag, row_sums = np.zeros(n), np.zeros(n)
        for i in range(n):
            diag[i] = cm[i][i]
            for j in range(n):
                row_sums += cm[i][j]
        class_div = diag / row_sums * n
        return class_div

    elif average == 'binary':
        if cm is None:
            cm = confusion_matrix(targets, preds, w=w)
        return np.array([cm[1][1] / (cm[1][1] + cm[0][1])])


@jit(nopython=False, fastmath=True)
def recall_score(targets, preds, cm=None, w=None, average='binary'):
    """
    :purpose:
    Calculates the recall score between a discrete target and pred array
    :params:
    targets, preds : discrete input arrays, both of shape (n,)
    cm             : if you have previously calculated a confusion matrix, pass it here to save the computation.
                     set as None, which makes the function calculate the confusion matrix.
                     note that for your specific average (i.e., micro, macro, none, or binary), you must compute the confusion
                     matrix correctly corresponding to the one you would like to use. so, for "macro" or "none", the cm
                     must be computed with normalize="true"
    w              : weights at each index of true and pred. array of shape (n,)
                     if no w is set, it is initialized as an array of ones
                     such that it will have no impact on the output
    average        : str, either "micro", "macro", "none", or "binary".
                     if "micro", computes recall globally
                     if "macro", take the mean of recall for each class (unweighted)
                     if "none", return a list of the recall for each class
                     if "binary", return recall in a binary classification problem
                     defaults to "binary", so for multi-class problems, you must change this
    :returns:
    recall_score : np.array, the recall score of the targets and preds array
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> true = np.random.RandomState(seed=0).randint(2, size=10000)
    >>> pred = np.random.RandomState(seed=1).randint(2, size=10000)
    >>> fastdist.recall_score(true, pred)
    array([0.48987217])
    """
    w = init_w(w, len(targets))
    if average == 'micro':
        if cm is None:
            cm = confusion_matrix(targets, preds, w=w)
        n = cm.shape[0]

        diag, row_sums = np.zeros(n), np.zeros(n)
        for i in range(n):
            diag[i] = cm[i][i]
            for j in range(n):
                row_sums += cm[i][j]
        class_div = diag / row_sums
        div_mean = 0.
        for i in range(n):
            div_mean += class_div[i]
        return np.array([div_mean])

    elif average == 'macro':
        if cm is None:
            cm = confusion_matrix(targets, preds, w=w, normalize='true')
        n = cm.shape[0]

        diag, row_sums = np.zeros(n), np.zeros(n)
        for i in range(n):
            diag[i] = cm[i][i]
            for j in range(n):
                row_sums += cm[i][j]
        class_div = diag / row_sums * n
        class_mean = 0
        for i in range(n):
            class_mean += class_div[i]
        return np.array([class_mean / n])

    elif average == 'none':
        if cm is None:
            cm = confusion_matrix(targets, preds, w=w, normalize='true')
        n = cm.shape[0]

        diag, row_sums = np.zeros(n), np.zeros(n)
        for i in range(n):
            diag[i] = cm[i][i]
            for j in range(n):
                row_sums += cm[i][j]
        class_div = diag / row_sums * n
        return class_div

    elif average == 'binary':
        if cm is None:
            cm = confusion_matrix(targets, preds, w=w)
        return np.array([cm[1][1] / (cm[1][1] + cm[1][0])])

@jit(nopython=False, fastmath=True)
def f1_score(targets, preds, w=None, average='binary'):
    """
    :purpose:
    Calculates the F1 score between a discrete target and pred array
    :params:
    targets, preds : discrete input arrays, both of shape (n,)
    w              : weights at each index of true and pred. array of shape (n,)
                     if no w is set, it is initialized as an array of ones
                     such that it will have no impact on the output
    average        : str, either "micro", "macro", "none", or "binary".
                     if "micro", computes F1 globally
                     if "macro", take the mean of F1 for each class (unweighted)
                     if "none", return a list of the F1 for each class
                     if "binary", return F1 in a binary classification problem
                     defaults to "binary", so for multi-class problems, you must change this
    :returns:
    f1_score : np.array, the F1 score of the targets and preds array
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> true = np.random.RandomState(seed=0).randint(2, size=10000)
    >>> pred = np.random.RandomState(seed=1).randint(2, size=10000)
    >>> fastdist.f1_score(true, pred)
    array([0.49429507])
    """
    w = init_w(w, len(targets))
    precision = precision_score(targets, preds, w, average)
    recall = recall_score(targets, preds, w, average)
    return np.array([2]) * precision * recall / (precision + recall)


@jit(nopython=False, fastmath=True)
def log_loss(targets, probs, w=None):
    """
    :purpose:
    Calculates the log loss between an array of discrete targets and an array of probabilities
    :params:
    targets : discrete input array of shape (n,)
    probs   : input array of predicted probabilities for sample of shape (n,)
    w       : weights at each index of true and pred. array of shape (n,)
              if no w is set, it is initialized as an array of ones
              such that it will have no impact on the output
    :returns:
    log_loss : float, the log loss score of the targets and probs array
    :example:
    >>> from topo.base import fastdist
    >>> import numpy as np
    >>> true = np.random.RandomState(seed=0).randint(2, size=10000)
    >>> prob = np.random.RandomState(seed=0).uniform(size=10000)
    >>> fastdist.log_loss(true, prob)
    1.0023371622966895
    """
    w = init_w(w, len(targets))
    num, denom = 0, 0
    for i in range(len(targets)):
        if targets[i] == 1:
            num += -math.log(probs[i]) * w[i]
        else:
            num += -math.log(1 - probs[i]) * w[i]
        denom += w[i]
    return num / denom