# topo_metrics.py
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import spearmanr, wasserstein_distance
from scipy.linalg import orthogonal_procrustes, subspace_angles
import scipy.sparse as sp
from topo.tpgraph.kernels import Kernel
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

def _topk_support_from_row(data, ind, k):
    if k is None or k >= data.size:
        return set(ind.tolist())
    sel = np.argpartition(data, -k)[-k:]
    return set(ind[sel].tolist())


def get_P(Y, **kwargs_for_kernel):
    """
    Convenience: build a diffusion operator P from data or a precomputed graph.

    Parameters
    ----------
    Y : array-like or sparse matrix or topo.tpgraph.kernels.Kernel
        - If a Kernel instance: returns its .P (computing it if needed).
        - If a rectangular (n x d) array/matrix: treated as data; a kernel and
          diffusion operator are built using the provided kwargs.
        - If a square (n x n) array/sparse matrix AND metric='precomputed' is
          provided (or auto-detected), Y is treated as a precomputed affinity/
          kernel matrix (NOT distances), and P is computed from it.

    **kwargs_for_kernel :
        Passed to Kernel(...). Useful options include:
          metric='cosine' | 'euclidean' | 'precomputed'
          n_neighbors=30
          adaptive_bw=True
          backend='nmslib' | 'hnswlib' 
          n_jobs=-1
          symmetrize=True
          anisotropy=1.0
          use_angular=True (for cosine)

    Returns
    -------
    P : scipy.sparse.csr_matrix
        The (symmetrized) diffusion operator.

    Notes
    -----
    - If Y is square and you did NOT set metric='precomputed', we auto-switch
      to 'precomputed' (assuming Y is an affinity/kernel). If Y is a *distance*
      matrix, convert it to an affinity first or pass the raw data.
    """
    # 1) If user already passed a fitted Kernel, just return its P
    if isinstance(Y, Kernel):
        Kobj = Y
        return Kobj.P.tocsr()

    # 2) Prepare defaults and merge user options
    params = dict(
        metric='cosine',
        n_neighbors=30,
        adaptive_bw=True,
        backend='nmslib',
        n_jobs=-1,
        symmetrize=True,
        use_angular=True,  # sensible for cosine on z-scored data
    )
    params.update(kwargs_for_kernel)

    # 3) Auto-detect precomputed case: square matrix input → kernel/affinity
    try:
        n0, n1 = Y.shape
        is_square = (n0 == n1)
    except Exception:
        is_square = False

    if is_square and params.get('metric', None) != 'precomputed':
        # Interpret as a precomputed affinity/kernel unless explicitly told otherwise
        params['metric'] = 'precomputed'

    # 4) Build the Kernel and compute P
    Kobj = Kernel(**params).fit(Y)
    P = Kobj.P  # already symmetrized internally
    return P.tocsr()

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
        See `_top_eigs_of_P`.
    center : bool, default=True
        Mean-center coordinates before Procrustes.

    Returns
    -------
    R2_avg : float
        Average coefficient of determination across t (clipped to [0,1]).
        Higher is better (1.0 means perfect alignment up to a rotation).
    """
    from scipy.linalg import orthogonal_procrustes

    wx, Vx = _top_eigs_of_P(Px, r=r, symmetric_hint=symmetric_hint)
    wy, Vy = _top_eigs_of_P(Py, r=r, symmetric_hint=symmetric_hint)

    # choose a consistent number of non-trivial modes (skip the first ~1.0 eigenvector)
    # limit by available columns and by r
    r_use = max(1, min(Vx.shape[1] - 1, Vy.shape[1] - 1, r))

    scores = []
    for t in times:
        Phix = diffusion_coordinates(wx, Vx, t, r_use=r_use, drop_first=True, normalize_cols=True)
        Phiy = diffusion_coordinates(wy, Vy, t, r_use=r_use, drop_first=True, normalize_cols=True)

        if center:
            Phix = Phix - Phix.mean(0, keepdims=True)
            Phiy = Phiy - Phiy.mean(0, keepdims=True)

        R, _ = orthogonal_procrustes(Phiy, Phix)  # map Phiy @ R ≈ Phix
        Yhat = Phiy @ R

        # R^2 with centered data (no intercept): 1 - SSE/SST
        sse = float(np.sum((Phix - Yhat) ** 2))
        sst = float(np.sum(Phix ** 2)) + 1e-12
        r2 = 1.0 - (sse / sst)
        # clip to [0,1] for stability
        scores.append(max(0.0, min(1.0, r2)))
    return float(np.nanmean(scores))


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

def rowwise_js_similarity(Px, Py, eps: float = 1e-12, topk: int = None, return_per_row: bool = False):
    """
    Row-wise Jensen–Shannon (JS) similarity between two diffusion operators.

    Given two (row-stochastic) operators Px and Py (csr_matrices or ndarrays),
    we compare each row i as a discrete probability distribution over columns
    (neighbors) and compute the Jensen–Shannon divergence JS(p_i, q_i).
    We then report a bounded similarity in [0, 1] via:

        JS-similarity = 1 - mean_i JS(p_i, q_i)

    where JS(·,·) = 0 for identical distributions and ≤ log(2) in nats;
    we use the standard normalized/base-e version implemented below.

    Parameters
    ----------
    Px, Py : (n, n) csr_matrix or ndarray
        Row-stochastic diffusion/transition operators to compare.
        (They need not be strictly stochastic; rows are renormalized internally.)
    eps : float, default=1e-12
        Small positive value added before per-row renormalization for numerical stability.
    topk : int or None, default=None
        If provided, restrict each row to its top-k entries by probability mass
        **before** comparing (helps robustness on very sparse / noisy rows).
        If None, we compare using the full sparse support (union of supports).
    return_per_row : bool, default=False
        If True, also return the per-row JS similarities (1 - JS_i) as a 1D array.

    Returns
    -------
    sim : float in [0, 1]
        1 - mean(JS divergence) across rows (higher is better).
    per_row : ndarray, optional
        Returned only if `return_per_row=True`. Per-row (1 - JS_i) scores.

    Notes
    -----
    • Operates sparsely: for each row we build vectors over the **union** of
      the supports of Px[i, :] and Py[i, :] (unless topk is set, in which case
      supports are first truncated to top-k).
    • Sensitive to **weights** (transition probabilities), unlike set-overlap metrics.
    • If a row is empty in both operators, it is skipped.
    """
    import numpy as np
    import scipy.sparse as sp

    def _ensure_csr(P):
        if not sp.isspmatrix_csr(P):
            P = sp.csr_matrix(P)
        P.eliminate_zeros()
        return P

    def _row_js_divergence(p, q):
        # p, q are dense 1D arrays over same support; not necessarily normalized
        p = np.asarray(p, dtype=float) + eps
        q = np.asarray(q, dtype=float) + eps
        p /= p.sum()
        q /= q.sum()
        m = 0.5 * (p + q)
        # KL in nats; JS = 0.5*(KL(p||m) + KL(q||m))
        js = 0.5 * (np.sum(p * (np.log(p) - np.log(m))) +
                    np.sum(q * (np.log(q) - np.log(m))))
        return float(js)

    def _topk_from_row(data, ind, k):
        if k is None or k >= data.size:
            return ind, data
        sel = np.argpartition(data, -k)[-k:]
        return ind[sel], data[sel]

    Px = _ensure_csr(Px); Py = _ensure_csr(Py)
    n = Px.shape[0]
    one_minus_js = []

    for i in range(n):
        # Sparse slices
        p_row = Px.data[Px.indptr[i]:Px.indptr[i+1]]
        p_ind = Px.indices[Px.indptr[i]:Px.indptr[i+1]]
        q_row = Py.data[Py.indptr[i]:Py.indptr[i+1]]
        q_ind = Py.indices[Py.indptr[i]:Py.indptr[i+1]]

        if p_row.size == 0 and q_row.size == 0:
            continue  # skip fully empty rows

        # Optional top-k truncation
        if topk is not None:
            p_ind, p_row = _topk_from_row(p_row, p_ind, topk)
            q_ind, q_row = _topk_from_row(q_row, q_ind, topk)

        # Align to union of supports
        Su = np.union1d(p_ind, q_ind)
        mp = {c: j for j, c in enumerate(p_ind)}
        mq = {c: j for j, c in enumerate(q_ind)}
        p_vec = np.zeros(Su.size, dtype=float)
        q_vec = np.zeros(Su.size, dtype=float)
        for t, col in enumerate(Su):
            if col in mp:
                p_vec[t] = p_row[mp[col]]
            if col in mq:
                q_vec[t] = q_row[mq[col]]

        js = _row_js_divergence(p_vec, q_vec)
        # Map divergence → similarity in [0,1]; JS ≤ ln(2); clip for safety
        js = min(js, np.log(2.0))
        one_minus_js.append(1.0 - (js / np.log(2.0)))

    sim = float(np.mean(one_minus_js)) if one_minus_js else np.nan
    if return_per_row:
        return sim, np.asarray(one_minus_js, dtype=float)
    return sim

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

    Returns
    -------
    If return_details=False:
        float in [0,1]: cosine of the largest principal angle between the
        top-r eigenspaces (higher is better).
    If return_details=True:
        dict with {'eigenvalue_w1', 'subspace_cos'}.
    """
    from scipy.stats import wasserstein_distance
    from scipy.linalg import subspace_angles

    wx, Vx = _top_eigs_of_P(Px, r=r, symmetric_hint=symmetric_hint)
    wy, Vy = _top_eigs_of_P(Py, r=r, symmetric_hint=symmetric_hint)

    # Drop the trivial first mode (~1.0)
    wx = wx[1:]; wy = wy[1:]

    r_pair = min(len(wx), len(wy))
    # Wasserstein-1 between (absolute) leading spectra (lower is better)
    w1 = wasserstein_distance(
        np.arange(r_pair), np.arange(r_pair),
        u_weights=np.abs(wx[:r_pair]), v_weights=np.abs(wy[:r_pair])
    )

    # principal angles between top subspaces (skip first vector)
    r_use = max(1, min(Vx.shape[1] - 1, Vy.shape[1] - 1, r))
    U = Vx[:, 1:1 + r_use]
    V = Vy[:, 1:1 + r_use]
    # Orthonormalize columns (QR) to be safe
    U, _ = np.linalg.qr(U); V, _ = np.linalg.qr(V)
    ang = subspace_angles(U, V)  # ascending
    cos_largest = float(np.cos(ang[-1])) if ang.size > 0 else np.nan

    if return_details:
        return {'eigenvalue_w1': float(w1), 'subspace_cos': cos_largest}
    else:
        return cos_largest
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
# Composite score
# ----------------------------

def topo_preserve_score(
    Px, Py,
    times=(1, 2, 4, 8),
    r: int = 64,
    symmetric_hint: bool = False,
    k_for_pf1: int = None,
    weights: dict = dict(PF1=0.30, PJS=0.30, SP=0.30),
):
    """
    Composite **TopoPreserve score** using four operator-aware metrics
    (higher is better; returns ≈[0,1] after internal normalizations).

    Components
    ----------
    • PF1  : F1@k on top-k transition neighborhoods per row (set overlap).
             Range [0,1], higher is better. (Weight-insensitive.)
    • PJS  : Row-wise Jensen–Shannon similarity of transitions (1 − JS, normalized).
             Range [0,1], higher is better. (Weight-sensitive.)
    • SP   : Spectral Procrustes R^2 alignment of diffusion coordinates (average over `times`).
             Range [0,1], higher is better. (Global/meso geometry.)

    Parameters
    ----------
    Px, Py : (n, n) csr_matrix or ndarray
        Diffusion (transition) operators to compare.
    times : tuple of int, default=(1, 2, 4, 8)
        Diffusion times for Spectral Procrustes. Ignored by the other components.
    r : int, default=64
        Leading eigenpairs used for spectral metrics (SP internals).
    symmetric_hint : bool, default=False
        If True, treat operators as symmetric for eigensolvers (stability hint).
    k_for_pf1 : int or None, default=None
        Top-k used in PF1. If None, each row uses its native sparsity.
    weights : dict, default={PF1=0.30, PJS=0.30, SP=0.30}
        Mixture weights for the four components. Any NaN component is skipped
        and remaining weights are renormalized.

    Returns
    -------
    score : float
        Weighted average of the component scores in ≈[0,1] (higher is better).
    parts : dict
        {
          'PF1'      : float in [0,1],
          'PJS'      : float in [0,1],
          'SP'       : float in [0,1],
        }

    Notes
    -----
    • PF1 (set) and PJS (weight) together capture **local** neighborhood fidelity.
    • SP captures **global/meso** geometry via diffusion eigencoordinates.
    • All components are operator-native (diffusion/graph-based), aligning with
      TopoMAP/DM objectives more directly than raw Euclidean metrics.
    """
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    # --- helpers (mirror commute_time_trace_gap internals so we can normalize CT) ---
    def _ensure_csr(P):
        if not sp.isspmatrix_csr(P):
            P = sp.csr_matrix(P)
        P.eliminate_zeros()
        return P

    def _trace_pinv_laplacian(P):
        # Symmetrize to an affinity, then build normalized Laplacian
        A = (P + P.T) * 0.5
        A = _ensure_csr(A)
        d = np.asarray(A.sum(axis=1)).ravel()
        d[d <= 0] = 1e-12
        Dm12 = sp.diags(1.0 / np.sqrt(d))
        Lsym = sp.eye(A.shape[0], format='csr') - Dm12 @ A @ Dm12
        # Smallest eigenvalues of Lsym; drop the trivial 0 mode
        k = min(int(r) + 1, A.shape[0] - 1) if A.shape[0] > 2 else 1
        vals, _ = spla.eigsh(Lsym, k=k, which='SM', tol=1e-4,
                             maxiter=A.shape[0]*5, v0=np.ones(A.shape[0]))
        vals = np.sort(vals)
        vals = vals[1:]  # drop the 0 eigenvalue
        vals = vals[vals > 1e-12]
        return float(np.sum(1.0 / vals)) if vals.size else np.inf

    # --- ensure CSR ---
    Px = _ensure_csr(Px); Py = _ensure_csr(Py)

    # --- components ---
    # PF1: set-overlap of top-k transition supports
    pf1 = sparse_neighborhood_f1(Px, Py, k=k_for_pf1)

    # PJS: weight-sensitive per-row transition similarity (bounded, normalized)
    pjs = rowwise_js_similarity(Px, Py)

    # SP: spectral Procrustes R^2 over diffusion coordinates
    spR2 = spectral_procrustes(Px, Py, times=times, r=r, symmetric_hint=symmetric_hint)

    parts = dict(PF1=pf1, PJS=pjs, SP=spR2)

    # --- weighted mixture with renormalization over finite components ---
    acc, wsum = 0.0, 0.0
    for key, w in weights.items():
        v = parts.get(key, np.nan)
        if np.isfinite(v):
            acc += float(w) * float(v)
            wsum += float(w)
    score = float(acc / (wsum + 1e-12)) if wsum > 0 else np.nan
    return score, parts
