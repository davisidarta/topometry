# topo_metrics.py
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import spearmanr, wasserstein_distance
from scipy.linalg import orthogonal_procrustes, subspace_angles

# ----------------------------
# Utilities
# ----------------------------
def _ensure_csr(P):
    if not sp.isspmatrix_csr(P):
        P = P.tocsr()
    P.eliminate_zeros()
    return P

def _top_eigs_of_P(P, r=64, which='LM', tol=1e-4, maxiter=None, v0=None, symmetric_hint=False):
    """
    Compute top-r eigenpairs of P (row-stochastic Markov operator).
    If you used a symmetric diffusion operator earlier, set symmetric_hint=True for
    improved stability (we then eigendecompose the symmetrized operator).
    Returns evals (r,), evecs (n,r). Sorted by |lambda| desc.
    """
    P = _ensure_csr(P)
    n = P.shape[0]
    r = min(r, n-1)
    if r < 1:
        raise ValueError("r must be >= 1 and < n.")
    G = (P + P.T) * 0.5 if symmetric_hint else P
    # eigsh works for symmetric; for non-symmetric use eigs
    if symmetric_hint:
        w, V = spla.eigsh(G, k=r, which='LM', tol=tol, maxiter=maxiter, v0=v0)
    else:
        w, V = spla.eigs(G, k=r, which='LR', tol=tol, maxiter=maxiter, v0=v0)
        w = np.real(w); V = np.real(V)
    # sort by magnitude (descending), drop trivial ~1 at index 0 later where needed
    idx = np.argsort(-np.abs(w))
    w, V = w[idx], V[:, idx]
    return w, V

def diffusion_coordinates(evals, evecs, t, drop_first=True, r_use=None, normalize_cols=True):
    """
    Phi_t(P) = [lambda_1^t psi_1, ..., lambda_r^t psi_r], optionally skipping the trivial first.
    """
    lam = np.array(evals, dtype=float)
    Psi = np.array(evecs, dtype=float)
    start = 1 if drop_first else 0
    if r_use is None:
        r_use = Psi.shape[1] - start
    r_use = max(1, min(r_use, Psi.shape[1] - start))
    lam_t = lam[start:start+r_use] ** float(t)
    Phi = Psi[:, start:start+r_use] * lam_t[None, :]
    if normalize_cols:
        # shrink numerical drift
        s = np.linalg.norm(Phi, axis=0) + 1e-12
        Phi = Phi / s
    return Phi

def diffusion_distance_from_eigs(evals, evecs, t, r_use=None, drop_first=True, squared=False):
    """
    Compute diffusion distance matrix (via truncated eigendecomposition).
    D^2(i,j) = sum_l lambda_l^{2t} (psi_l(i)-psi_l(j))^2
    Returns a dense (n,n) matrix for convenience; for large n use sampling.
    """
    Phi = diffusion_coordinates(evals, evecs, t, drop_first=drop_first, r_use=r_use, normalize_cols=False)
    # pairwise squared Euclidean distances with weighted coordinates (which already include lambda^t)
    # efficient trick: ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    G = Phi @ Phi.T  # Gram
    diag = np.clip(np.diag(G), 0.0, np.inf)
    D2 = diag[:, None] + diag[None, :] - 2.0 * G
    np.maximum(D2, 0.0, out=D2)  # numerical floor
    return D2 if squared else np.sqrt(D2)

def _upper_triangle_vec(M):
    i, j = np.triu_indices(M.shape[0], k=1)
    return M[i, j]

def _knn_from_distance(D, k):
    """Return indices of k nearest neighbors for each row (excluding self)."""
    n = D.shape[0]
    k = min(k, n-1)
    # partial sort for scalability
    idx = np.argpartition(D, kth=k, axis=1)[:, :k+1]  # includes self likely
    # refine to get exact top-k by value, drop self
    rows = np.arange(n)[:, None]
    vals = D[rows, idx]
    # remove self if present: set its distance to +inf then take top-k
    mask_self = idx == rows
    vals[mask_self] = np.inf
    order = np.argsort(vals, axis=1)
    sel = order[:, :k]
    nbrs = idx[rows, sel]
    return nbrs  # (n,k)

def _row_js_divergence(p, q, eps=1e-12):
    """JS divergence between two row vectors p,q that sum to 1 (dense 1D arrays)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p + eps; q = q + eps
    p = p / p.sum(); q = q / q.sum()
    m = 0.5*(p+q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)))
    kl_qm = np.sum(q * (np.log(q) - np.log(m)))
    return 0.5*(kl_pm + kl_qm)

def _row_union_dense(P_row, P_ind, Q_row, Q_ind, n=None, eps=1e-12):
    """
    Build dense aligned vectors over the union of supports of two sparse rows.
    Returns p_vec, q_vec (same length), where each is probability-mass-aligned.
    """
    # supports
    Su = np.union1d(P_ind, Q_ind)
    # maps
    mp = {c: i for i, c in enumerate(P_ind)}
    mq = {c: i for i, c in enumerate(Q_ind)}
    p = np.zeros(Su.size, dtype=float)
    q = np.zeros(Su.size, dtype=float)
    for j, col in enumerate(Su):
        if col in mp:
            p[j] = P_row[mp[col]]
        if col in mq:
            q[j] = Q_row[mq[col]]
    # tiny prior + renormalize
    p = p + eps; q = q + eps
    p = p / p.sum(); q = q / q.sum()
    return p, q, Su

def _topk_support_from_row(data, ind, k):
    if k is None or k >= data.size:
        return set(ind.tolist())
    sel = np.argpartition(data, -k)[-k:]
    return set(ind[sel].tolist())

# ----------------------------
# 1) Global geometry
# ----------------------------
def rank_diffusion_correlation(Px, Py, times=(1, 2, 4, 8), r=64, symmetric_hint=False):
    """
    Global geometry agreement via Spearman correlation of diffusion distances.

    Parameters
    ----------
    Px, Py : (n, n) csr_matrix or ndarray
        Diffusion operators to compare.
    times : tuple of int, default=(1,2,4,8)
        Multiscale t values. The score is averaged over t.
    r : int, default=64
        Eigenpairs used for diffusion distances.
    symmetric_hint : bool, default=False
        See `diffusion_eigs`.

    Returns
    -------
    rho_avg : float in [-1, 1]
        Average Spearman correlation between the upper triangles of D_t(Px) and
        D_t(Py) across t (higher is better).

    Notes
    -----
    - Robust to monotone rescalings (rank-based).
    - Sensitive to global geometry preservation (coarse-to-fine as t grows).
    """
    wx, Vx = _top_eigs_of_P(Px, r=r, symmetric_hint=symmetric_hint)
    wy, Vy = _top_eigs_of_P(Py, r=r, symmetric_hint=symmetric_hint)
    rhos = []
    for t in times:
        Dx = diffusion_distance_from_eigs(wx, Vx, t)
        Dy = diffusion_distance_from_eigs(wy, Vy, t)
        vx = _upper_triangle_vec(Dx); vy = _upper_triangle_vec(Dy)
        rho, _ = spearmanr(vx, vy)
        rhos.append(float(rho))
    return np.nanmean(rhos)

def multiscale_diffusion_emd(Px, Py, times=(1,2,4,8), r=64, bins=64, symmetric_hint=False):
    """
    Global geometry preservation via multiscale Earth Mover’s Distance (EMD) on diffusion distances.

    Parameters
    ----------
    Px, Py : (n, n) csr_matrix or ndarray
        Diffusion operators to compare.
    times : tuple of int, default=(1,2,4,8)
        Diffusion timescales at which pairwise diffusion distances are computed.
        Scores are averaged across timescales.
    r : int, default=64
        Number of leading eigenpairs to use for diffusion distances.
    bins : int, default=64
        Number of histogram bins used to approximate the distribution of
        pairwise diffusion distances.
    symmetric_hint : bool, default=False
        Passed to `_top_eigs_of_P`; set to True if Px and Py are symmetric operators.

    Returns
    -------
    emd : float ≥ 0
        Mean Earth Mover’s Distance (1-Wasserstein distance) between the
        distributions of pairwise diffusion distances in Px and Py, averaged
        across timescales. Lower is better (0 indicates identical distributions).

    Notes
    -----
    - This is a *distributional* comparison: rather than matching each pairwise
      distance, it checks whether the **global distribution** of diffusion
      distances is preserved between Px and Py.
    - The measure is scale-sensitive: if distances in Py are systematically
      compressed or stretched relative to Px, the EMD will increase.
    - Bins are shared between Px and Py at each timescale to ensure fair
      histogram comparison.
    - Useful as a complement to correlation-based metrics (e.g. RDC), since it
      detects distributional distortions even if ranks are preserved.
    """
    wx, Vx = _top_eigs_of_P(Px, r=r, symmetric_hint=symmetric_hint)
    wy, Vy = _top_eigs_of_P(Py, r=r, symmetric_hint=symmetric_hint)
    emds = []
    for t in times:
        Dx = diffusion_distance_from_eigs(wx, Vx, t)
        Dy = diffusion_distance_from_eigs(wy, Vy, t)
        hx = _upper_triangle_vec(Dx)
        hy = _upper_triangle_vec(Dy)
        # histogram supports shared bins
        lo = min(hx.min(), hy.min()); hi = max(hx.max(), hy.max())
        edges = np.linspace(lo, hi + 1e-12, bins + 1)
        cx, _ = np.histogram(hx, bins=edges, density=True)
        cy, _ = np.histogram(hy, bins=edges, density=True)
        # approximate EMD with 1-Wasserstein on the midpoints
        mids = 0.5*(edges[:-1] + edges[1:])
        emd = wasserstein_distance(mids, mids, u_weights=cx, v_weights=cy)
        emds.append(float(emd))
    return np.nanmean(emds)

def spectral_procrustes(Px, Py, times=(1, 2, 4, 8), r=64, symmetric_hint=False, center=True):
    """
    Align diffusion coordinates via orthogonal Procrustes and report R^2.

    Parameters
    ----------
    Px, Py : (n, n) csr_matrix or ndarray
        Diffusion operators to compare.
    times : tuple of int, default=(1,2,4,8)
        Diffusion times; we build Φ_t for each and average the R^2.
    r : int, default=64
        Number of eigenpairs for coordinates.
    symmetric_hint : bool, default=False
        See `diffusion_eigs`.
    center : bool, default=True
        Mean-center coordinates before Procrustes.

    Returns
    -------
    R2_avg : float
        Average coefficient of determination across t (clipped to [0,1]).
        Higher is better (1.0 means perfect alignment up to a rotation).

    Notes
    -----
    - Procrustes finds the best orthogonal transform aligning Φ_t(Py) to Φ_t(Px),
      then reports R^2 of the fit.
    - Captures *coordinate-level* consistency, not just pairwise distances.
    """

    wx, Vx = _top_eigs_of_P(Px, r=r, symmetric_hint=symmetric_hint)
    wy, Vy = _top_eigs_of_P(Py, r=r, symmetric_hint=symmetric_hint)
    scores = []
    for t in times:
        Phix = diffusion_coordinates(wx, Vx, t, r_use=r_use)
        Phiy = diffusion_coordinates(wy, Vy, t, r_use=Phix.shape[1])
        # center
        Phix = Phix - Phix.mean(0, keepdims=True)
        Phiy = Phiy - Phiy.mean(0, keepdims=True)
        R, scale = orthogonal_procrustes(Phiy, Phix)  # maps Phiy @ R ~ Phix
        Yhat = Phiy @ R
        # R^2 with centered data (no intercept): 1 - SSE/SST
        sse = np.sum((Phix - Yhat)**2)
        sst = np.sum(Phix**2)
        r2 = 1.0 - (sse / (sst + 1e-12))
        scores.append(float(r2))
    return np.nanmean(scores)

# ----------------------------
# 2) Local geometry
# ----------------------------
def diffusion_knn_preservation(Px, Py, times=(1, 2, 4, 8), r=64, k=30, symmetric_hint=False):
    """
    Local geometry agreement via diffusion-space kNN overlap.

    Parameters
    ----------
    Px, Py : (n, n) csr_matrix or ndarray
        Diffusion operators to compare.
    times : tuple of int, default=(1,2,4,8)
        Diffusion times; we average over t.
    r : int, default=64
        Eigenpairs for diffusion distances.
    k : int, default=30
        Number of nearest neighbors per node under diffusion distance.
    symmetric_hint : bool, default=False
        See `diffusion_eigs`.

    Returns
    -------
    score : float in [0, 1]
        Average fraction of overlap of kNN sets per node across t. 1.0 means
        perfect local preservation in diffusion space.

    Notes
    -----
    - Rebuild kNN under D_t for each P and compare overlaps per node.
    - More faithful to the Markov/diffusive geometry than raw Euclidean kNN.
    """
    wx, Vx = _top_eigs_of_P(Px, r=r, symmetric_hint=symmetric_hint)
    wy, Vy = _top_eigs_of_P(Py, r=r, symmetric_hint=symmetric_hint)
    n = Px.shape[0]
    vals = []
    for t in times:
        Dx = diffusion_distance_from_eigs(wx, Vx, t)
        Dy = diffusion_distance_from_eigs(wy, Vy, t)
        Nx = _knn_from_distance(Dx, k)
        Ny = _knn_from_distance(Dy, k)
        overlap = [(len(set(Nx[i]).intersection(set(Ny[i]))) / float(k)) for i in range(n)]
        vals.append(np.mean(overlap))
    return float(np.mean(vals))

def diffusion_trustworthiness(Px, Py, times=(1,2,4,8), r=64, k=30, symmetric_hint=False):
    """
    Trustworthiness metric in diffusion space.

    Parameters
    ----------
    Px, Py : (n, n) csr_matrix or ndarray
        Diffusion operators. Px is the reference ("high-dimensional" diffusion),
        Py is the test representation ("low-dimensional" diffusion).
    times : tuple of int, default=(1,2,4,8)
        Diffusion timescales for evaluation.
    r : int, default=64
        Number of eigenpairs used for diffusion distances.
    k : int, default=30
        Number of neighbors per node to consider.
    symmetric_hint : bool, default=False
        See `diffusion_eigs`.

    Returns
    -------
    trust : float in [0,1]
        Average trustworthiness score across timescales. Higher is better.

    Notes
    -----
    - Trustworthiness penalizes points that are *false neighbors* in the
      low-dimensional space: nodes that appear in the kNN set of Py but are far
      in Px.
    - Computed in the diffusion space, not raw Euclidean.
    - Values close to 1 indicate that low-dimensional neighborhoods contain few
      false positives relative to diffusion neighborhoods in Px.
    """

    def _trust(D_high, D_low, k):
        n = D_high.shape[0]
        # ranks (argsort twice to get rank indices)
        R_high = np.argsort(np.argsort(D_high, axis=1), axis=1)
        R_low = np.argsort(np.argsort(D_low, axis=1), axis=1)
        t_sum = 0.0
        for i in range(n):
            # neighbors in low space (exclude self rank 0)
            nbrs_low = np.argsort(D_low[i])  # full order
            nbrs_low = nbrs_low[nbrs_low != i][:k]
            # penalty: those that are in low-k but rank_high > k
            for j in nbrs_low:
                rank_high = R_high[i, j]
                if rank_high > k:
                    t_sum += (rank_high - k)
        denom = n * k * (2*n - 3*k - 1)
        return 1.0 - (2.0 / denom) * t_sum
    wx, Vx = _top_eigs_of_P(Px, r=r, symmetric_hint=symmetric_hint)
    wy, Vy = _top_eigs_of_P(Py, r=r, symmetric_hint=symmetric_hint)
    vals = []
    for t in times:
        Dx = diffusion_distance_from_eigs(wx, Vx, t)
        Dy = diffusion_distance_from_eigs(wy, Vy, t)
        vals.append(_trust(Dx, Dy, k))
    return float(np.mean(vals))

def diffusion_continuity(Px, Py, times=(1,2,4,8), r=64, k=30, symmetric_hint=False):
    """
    Continuity metric in diffusion space (reverse trustworthiness).

    Parameters
    ----------
    Px, Py : (n, n) csr_matrix or ndarray
        Diffusion operators. Px is treated as the "reference" (high-dimensional),
        Py as the "embedding" (low-dimensional).
    times : tuple of int, default=(1,2,4,8)
        Diffusion timescales for evaluation.
    r : int, default=64
        Number of eigenpairs for diffusion distances.
    k : int, default=30
        Number of neighbors to consider.
    symmetric_hint : bool, default=False
        See `diffusion_eigs`.

    Returns
    -------
    cont : float in [0,1]
        Continuity score averaged across timescales. Higher is better.

    Notes
    -----
    - Continuity measures how well *high-dimensional neighbors* are recovered
      in the low-dimensional representation.
    - It is the counterpart of trustworthiness (which measures false neighbors).
    - Computed here by reusing `diffusion_trustworthiness` with Px,Py swapped.
    """

    return diffusion_trustworthiness(Py, Px, times=times, r=r, k=k, symmetric_hint=symmetric_hint)

def diffusion_rank_biased_overlap(Px, Py, times=(1,2,4,8), r=64, p=0.9, k_max=100, symmetric_hint=False):
    """
    Local geometry agreement via Rank-Biased Overlap (RBO) of diffusion neighbors.

    Parameters
    ----------
    Px, Py : (n, n) csr_matrix or ndarray
        Diffusion operators to compare.
    times : tuple of int, default=(1,2,4,8)
        Diffusion timescales; the score is averaged across them.
    r : int, default=64
        Number of leading eigenpairs to use for diffusion distances.
    p : float in (0,1), default=0.9
        Persistence parameter of RBO. Values closer to 1 give more weight to
        deeper ranks; smaller values emphasize the very top neighbors.
    k_max : int, default=100
        Maximum depth of neighbor lists to compare. Truncates infinite RBO.
    symmetric_hint : bool, default=False
        See `diffusion_eigs`.

    Returns
    -------
    score : float in [0,1]
        Average RBO similarity across timescales. 1.0 means identical ranked
        neighbor lists under diffusion distances.

    Notes
    -----
    - RBO compares *ordered* neighbor lists, unlike kNN overlap which ignores order.
    - Especially useful to assess whether diffusion rankings (top-100 neighbors)
      are preserved, not just set membership.
    - Parameter p controls top-heaviness: p=0.9 gives ≈86% of weight to top-10.
    """

    def _rbo(listA, listB, p, k):
        # listA/listB are ordered neighbor lists (no self), length >= k
        A = listA[:k]; B = listB[:k]
        score = 0.0
        Aseen, Bseen = set(), set()
        overlap = 0
        for d in range(1, k+1):
            Aseen.add(A[d-1]); Bseen.add(B[d-1])
            overlap = len(Aseen.intersection(Bseen))
            score += (overlap / d) * (p**(d-1))
        return (1 - p) * score
    wx, Vx = _top_eigs_of_P(Px, r=r, symmetric_hint=symmetric_hint)
    wy, Vy = _top_eigs_of_P(Py, r=r, symmetric_hint=symmetric_hint)
    n = Px.shape[0]
    vals = []
    for t in times:
        Dx = diffusion_distance_from_eigs(wx, Vx, t)
        Dy = diffusion_distance_from_eigs(wy, Vy, t)
        # full ranks; we’ll reuse top k_max
        Rx = np.argsort(Dx, axis=1)
        Ry = np.argsort(Dy, axis=1)
        rbo_i = []
        for i in range(n):
            ax = Rx[i][Rx[i] != i][:k_max]
            ay = Ry[i][Ry[i] != i][:k_max]
            if ax.size == 0 or ay.size == 0:
                continue
            rbo_i.append(_rbo(ax.tolist(), ay.tolist(), p=p, k=min(k_max, len(ax), len(ay))))
        vals.append(np.mean(rbo_i) if rbo_i else np.nan)
    return float(np.nanmean(vals))

# ----------------------------
# 3) Operator-level similarity
# ----------------------------
def rowwise_js_similarity(Px, Py, eps=1e-12):
    """
    Operator-level similarity via row-wise Jensen–Shannon (JS) similarity.

    Parameters
    ----------
    Px, Py : (n, n) csr_matrix or ndarray
        Row-stochastic operators to compare.
    eps : float, default=1e-12
        Small positive value added before renormalization for numerical stability.

    Returns
    -------
    sim : float in [0, 1]
        1 - mean(JS divergence) across rows. Higher is better.
        Uses per-row union of supports (sparse) to stay O(k) per node.

    Notes
    -----
    - JS is symmetric and bounded; we map to a similarity via 1 - JS.
    - Compares transition distributions directly (not distances).
    """

    Px = _ensure_csr(Px); Py = _ensure_csr(Py)
    n = Px.shape[0]
    js = []
    for i in range(n):
        p_row = Px.data[Px.indptr[i]:Px.indptr[i+1]]
        p_ind = Px.indices[Px.indptr[i]:Px.indptr[i+1]]
        q_row = Py.data[Py.indptr[i]:Py.indptr[i+1]]
        q_ind = Py.indices[Py.indptr[i]:Py.indptr[i+1]]
        if p_row.size == 0 and q_row.size == 0:
            continue
        p_vec, q_vec, _ = _row_union_dense(p_row, p_ind, q_row, q_ind)
        js.append(_row_js_divergence(p_vec, q_vec))
    return float(1.0 - np.mean(js)) if js else np.nan

def sparse_neighborhood_f1(Px, Py, k=None):
    """
    Operator-level set overlap: F1 of top-k transition neighborhoods per row.

    Parameters
    ----------
    Px, Py : (n, n) csr_matrix or ndarray
        Diffusion/transition operators.
    k : int or None, default=None
        Number of neighbors (by transition probability) to keep per row.
        If None, use the natural sparsity (nnz per row) or min(nnz_x, nnz_y).

    Returns
    -------
    f1_avg : float in [0, 1]
        Mean F1 score across rows. 1.0 → identical sparse neighborhoods.

    Notes
    -----
    - Build per-row sets of top-k columns by probability mass, then compute
      F1 = 2 |∩| / (|Sx| + |Sy|).
    - Complements JS: insensitive to weights but sensitive to support overlap.
    """

    Px = _ensure_csr(Px); Py = _ensure_csr(Py)
    n = Px.shape[0]
    f1s = []
    for i in range(n):
        pr = Px.data[Px.indptr[i]:Px.indptr[i+1]]
        pi = Px.indices[Px.indptr[i]:Px.indptr[i+1]]
        qr = Py.data[Py.indptr[i]:Py.indptr[i+1]]
        qi = Py.indices[Py.indptr[i]:Py.indptr[i+1]]
        A = _topk_support_from_row(pr, pi, k)
        B = _topk_support_from_row(qr, qi, k)
        if len(A) == 0 and len(B) == 0:
            continue
        tp = len(A & B); prec = tp / (len(B) + 1e-12); rec = tp / (len(A) + 1e-12)
        f1 = 0.0 if (prec + rec) == 0 else (2*prec*rec)/(prec+rec)
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else np.nan

def spectral_similarity(Px, Py, r=64, symmetric_hint=False, return_details=False):
    """
    Operator-level spectral agreement via eigenvalues and subspaces.

    Parameters
    ----------
    Px, Py : (n, n) csr_matrix or ndarray
        Diffusion operators.
    r : int, default=64
        Number of leading eigenpairs to consider.
    symmetric_hint : bool, default=False
        See `diffusion_eigs`.
    return_details : bool, default=False
        If True, return a dict with components.

    Returns
    -------
    sim : dict or float
        If return_details=False:
            cos_theta : float in [0,1]
                Cosine of the largest principal angle between top-r eigenspaces
                (closer to 1 is better).
        If return_details=True:
            {
              'cos_principal_angle': float in [0,1],
              'wasserstein_eigs': float >= 0,
              'evals_x': (r,) ndarray,
              'evals_y': (r,) ndarray
            }

    Notes
    -----
    - Subspace angle uses principal angles between spans of leading eigenvectors.
    - 'wasserstein_eigs' is a 1-Wasserstein distance between sorted spectra
      (smaller is better); not normalized to [0,1].
    """

    wx, Vx = _top_eigs_of_P(Px, r=r, symmetric_hint=symmetric_hint)
    wy, Vy = _top_eigs_of_P(Py, r=r, symmetric_hint=symmetric_hint)
    # drop trivial ~1
    wx = wx[1:]; wy = wy[1:]
    r_pair = min(len(wx), len(wy))
    # eigenvalue W1 (on real lines)
    w1 = wasserstein_distance(np.arange(r_pair), np.arange(r_pair), u_weights=np.abs(wx[:r_pair]), v_weights=np.abs(wy[:r_pair]))
    # principal angles (use same r_use)
    if r_use is None:
        r_use = min(Vx.shape[1]-1, Vy.shape[1]-1, 32)
    U = Vx[:, 1:1+r_use]
    V = Vy[:, 1:1+r_use]
    # orthonormalize columns (QR) to be safe
    U, _ = np.linalg.qr(U); V, _ = np.linalg.qr(V)
    ang = subspace_angles(U, V)  # ascending angles
    cos_largest = float(np.cos(ang[-1])) if ang.size > 0 else np.nan
    return {'eigenvalue_w1': float(w1), 'subspace_cos': cos_largest}

def commute_time_trace_gap(Px, Py, r=64, symmetric_hint=False, hutchinson_probes=None, random_state=None):
    """
    Global connectivity gap via (approximate) trace of Laplacian pseudoinverse.

    Parameters
    ----------
    Px, Py : (n, n) csr_matrix or ndarray
        Diffusion operators (will be symmetrized to build Laplacians).
    r : int, default=64
        If using low-rank approximation, number of leading modes for the trace.
    symmetric_hint : bool, default=False
        See `diffusion_eigs`.
    hutchinson_probes : int or None, default=None
        If provided, estimate trace via Hutchinson’s method with this many
        random probe vectors (for very large graphs).
    random_state : int or np.random.RandomState or None
        RNG for Hutchinson probes.

    Returns
    -------
    gap : float >= 0
        Absolute difference between (approximate) trace( L^+_x ) and trace( L^+_y ).
        Smaller is better (more similar commute-time geometry).

    Notes
    -----
    - Build A = (P + P^T)/2, then normalized Laplacian L. Commute-time distances
      relate to entries of L^+; its trace summarizes overall connectivity.
    - For speed, you can approximate using low-rank spectral sums or Hutchinson.
    """

    def _trace_pinv_laplacian(P):
        A = (P + P.T) * 0.5
        A = _ensure_csr(A)
        d = np.asarray(A.sum(axis=1)).ravel()
        d[d <= 0] = 1e-12
        Dm12 = sp.diags(1.0/np.sqrt(d))
        Lsym = sp.eye(A.shape[0], format='csr') - Dm12 @ A @ Dm12
        k = min(r+1, A.shape[0]-1)
        vals, _ = spla.eigsh(Lsym, k=k, which='SM', tol=1e-4, maxiter=A.shape[0]*5, v0=np.ones(A.shape[0]))
        vals = np.sort(vals)
        vals = vals[1:]  # drop the 0 eigenvalue
        vals = vals[vals > 1e-12]
        return float(np.sum(1.0/vals))
    tx = _trace_pinv_laplacian(Px)
    ty = _trace_pinv_laplacian(Py)
    return float(abs(tx - ty))

# ----------------------------
# 4) Composite score
# ----------------------------

def topo_preserve_score(
    Px, Py,
    times=(1, 2, 4, 8),
    k_local=30,
    r=64,
    symmetric_hint=False,
    weights=dict(RDC=0.35, DkNP=0.25, PF1=0.15, PJS=0.15, SP=0.10),
    k_for_pf1=None,
):
    """
    Compute the composite **TopoPreserve score** (≈[0,1], higher is better).

    This aggregates:
      - RDC  (global rank correlation of diffusion distances)      → higher better
      - DkNP (diffusion kNN preservation)                          → higher better
      - PF1  (sparse neighborhood F1 overlap)                      → higher better
      - PJS  (row-wise Jensen–Shannon similarity of transitions)   → higher better
      - SP   (spectral Procrustes R^2 alignment)                    → higher better

    Parameters
    ----------
    Px, Py : (n, n) csr_matrix or ndarray
        Diffusion operators to compare.
    times : tuple of int, default=(1,2,4,8)
        Diffusion timescales for multiscale averaging.
    k_local : int, default=30
        Neighborhood size for D-kNP.
    r : int, default=64
        Eigenpairs for spectral approximations.
    symmetric_hint : bool, default=False
        See `diffusion_eigs`.
    weights : dict, default={RDC=0.35, DkNP=0.25, PF1=0.15, PJS=0.15, SP=0.10}
        Mixing weights for the components; renormalized if any component is NaN.
    k_for_pf1 : int or None, default=None
        k used in PF1 (top-k transitions per row). If None, uses natural sparsity.

    Returns
    -------
    score : float
        Weighted average of component scores (≈[0,1]).
    parts : dict
        Component scores after mapping to [0,1] where needed:
          - 'RDC' : RDC mapped from [-1,1] → [0,1] via (ρ+1)/2
          - 'DkNP': [0,1]
          - 'PF1' : [0,1]
          - 'PJS' : [0,1]
          - 'SP'  : R^2 clipped to [0,1]

    Notes
    -----
    - Intended for topometry operators but agnostic to source.
    - Use the per-metric functions for detailed diagnostics; use this composite
      when you need a single headline score.
    """


    # Global
    rdc = rank_diffusion_correlation(Px, Py, times=times, r=r, symmetric_hint=symmetric_hint)  # [-1,1]
    rdc01 = 0.5*(rdc + 1.0)

    spR2 = spectral_procrustes(Px, Py, times=times, r=r, symmetric_hint=symmetric_hint)  # can be <0..1
    sp01 = max(0.0, min(1.0, spR2))

    # Local
    dknp = diffusion_knn_preservation(Px, Py, times=times, r=r, k=k_local, symmetric_hint=symmetric_hint)  # [0,1]

    # Operator-level
    pf1 = sparse_neighborhood_f1(Px, Py, k=k_for_pf1)  # [0,1]
    pjs = rowwise_js_similarity(Px, Py)                # ~[0,1]

    # Weighted sum (renormalize missing)
    parts = dict(RDC=rdc01, DkNP=dknp, PF1=pf1, PJS=pjs, SP=sp01)
    wsum = 0.0; acc = 0.0
    for k, w in weights.items():
        v = parts.get(k, np.nan)
        if np.isfinite(v):
            wsum += w; acc += w * v
    return float(acc / (wsum + 1e-12)), parts
