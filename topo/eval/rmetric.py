######################################
# Riemannian metric estimation and visualization tools for TopOMetry
#
# Inspired by earlier work on Riemann metric estimation in the Megaman library
# by Marina Meila and collaborators (https://github.com/mmp2/megaman).
# Original Matlab function: megaman/geometry/rmetric.py by Dominique Perrault-Joncas.
#
# This code is a clean re-implementation with extensive new functionality,
# design, and visualization features. It is not a direct derivative of Megaman.
#
# Author: Davi Sidarta V. Rodrigues de Oliveira <david.oliveira@dpag.ox.ac.uk>
#         University of Oxford
# License: MIT
######################################

import numpy as np
import warnings

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib import cm, colors as mcolors
except Exception:
    plt = None
    Ellipse = None

try:
    from sklearn.neighbors import NearestNeighbors
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False


def _ensure_array(A):
    return A.toarray() if hasattr(A, "toarray") else np.asarray(A)


def _symmetrize(L):
    L = _ensure_array(L)
    return 0.5 * (L + L.T)


def _center(Y):
    Y = np.asarray(Y, dtype=float)
    return Y - Y.mean(0, keepdims=True)


def riemann_metric(Y, laplacian, n_dim=None):
    Y = _center(Y)
    L = _symmetrize(laplacian)

    n_samples = L.shape[0]
    n_dim_Y = Y.shape[1]
    if n_dim is None:
        n_dim = n_dim_Y

    H = np.zeros((n_samples, n_dim_Y, n_dim_Y), dtype=float)
    LY = L @ Y

    for i in range(n_dim_Y):
        yi = Y[:, i]
        Lyi = LY[:, i]
        for j in range(i, n_dim_Y):
            yj = Y[:, j]
            Lyj = LY[:, j]
            yij = yi * yj
            H[:, i, j] = 0.5 * ((L @ yij) - yj * Lyi - yi * Lyj)
            if j != i:
                H[:, j, i] = H[:, i, j]

    U, S, Vh = np.linalg.svd(H, full_matrices=False)
    eps = 1e-8
    Sinv = 1.0 / (S + eps)
    G = Vh.transpose(0, 2, 1) @ (Sinv[..., None] * U.transpose(0, 2, 1))

    return H, G, Vh, S, Sinv


def compute_G_from_H(H):
    U, S, Vh = np.linalg.svd(H, full_matrices=False)
    eps = 1e-8
    Sinv = 1.0 / (S + eps)
    G = Vh.transpose(0, 2, 1) @ (Sinv[..., None] * U.transpose(0, 2, 1))
    return G, H, Vh, S, Sinv


class RiemannMetric:
    def __init__(self, Y, L):
        self.Y = _center(Y)
        self.L = _symmetrize(L)
        self.n, self.mdimY = self.Y.shape
        self.mdimG = self.mdimY
        self.H = self.G = self.Hvv = self.Hsvals = self.Gsvals = self.detG = None

    def get_dual_rmetric(self, invert_h=False):
        if self.H is None:
            self.H, self.G, self.Hvv, self.Hsvals, self.Gsvals = riemann_metric(self.Y, self.L, self.mdimG)
        return (self.H, self.G) if invert_h else self.H

    def get_rmetric(self, return_svd=False):
        if self.G is None:
            self.H, self.G, self.Hvv, self.Hsvals, self.Gsvals = riemann_metric(self.Y, self.L, self.mdimG)
        return (self.G, self.Hvv, self.Hsvals, self.Gsvals) if return_svd else self.G

    def get_mdimG(self):
        return self.mdimG

    def get_detG(self, use_log=True):
        if self.G is None:
            self.H, self.G, self.Hvv, self.Hsvals, self.Gsvals = riemann_metric(self.Y, self.L, self.mdimG)
        if use_log:
            self.detG = -np.sum(np.log(self.Hsvals + 1e-8), axis=1)
            return self.detG
        else:
            self.detG = np.prod(1.0 / (self.Hsvals + 1e-8), axis=1)
            return self.detG

    def fit(self, Y, L=None):
        self.Y = _center(Y)
        if L is not None:
            self.L = _symmetrize(L)
        if self.L is None:
            raise ValueError("Laplacian matrix L is not set")
        self.n, self.mdimY = self.Y.shape
        self.mdimG = self.mdimY
        self.H = self.G = self.Hvv = self.Hsvals = self.Gsvals = self.detG = None
        return self.get_rmetric()

    def transform(self, Y, L=None):
        return self.fit(Y, L)


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    return vals[idx], vecs[:, idx]


def _ellipse_from_G(Gi, scale=1.0):
    vals, vecs = eigsorted(Gi)
    vals = np.clip(vals, 1e-12, None)
    a = np.sqrt(vals[0]) * scale
    b = np.sqrt(vals[-1]) * scale
    v = vecs[:, 0]
    theta = np.degrees(np.arctan2(v[1], v[0]))
    return a, b, theta


def _scaling_values(G, mode="logdet", eps=1e-8):
    if mode == "logdet":
        U, s, Vh = np.linalg.svd(G, full_matrices=False)
        # log det(G) = sum log s_i
        ld = np.sum(np.log(np.clip(s, eps, None)), axis=1)
        v = ld
    elif mode == "anisotropy":
        vals = np.linalg.eigvalsh(G)
        vals = np.clip(vals, eps, None)
        v = np.log(vals[:, -1] / vals[:, 0])
    else:
        v = np.ones(G.shape[0], dtype=float)
    v = v - np.nanmin(v)
    denom = np.nanmax(v) - np.nanmin(v) + eps
    return (v / denom) + eps


def _project_spd(Gi, eps=1e-8, norm="trace"):
    # Symmetrize and project to SPD with eigenvalue clipping; optional normalization.
    Gi = 0.5 * (Gi + Gi.T)
    vals, vecs = np.linalg.eigh(Gi)
    vals = np.clip(vals, eps, None)
    Gi = (vecs * vals) @ vecs.T
    if norm == "trace":
        tr = np.trace(Gi)
        if np.isfinite(tr) and tr > eps:
            Gi = Gi * (2.0 / tr)  # make trace ~ 2 for stable ellipse sizes
    elif norm == "det":
        det = np.linalg.det(Gi)
        if np.isfinite(det) and det > eps:
            Gi = Gi * (det ** (-1.0 / Gi.shape[0]))
    return Gi


def _ellipse_from_G(Gi, scale=1.0, eps=1e-8):
    # Use SPD-projected, trace-normalized Gi to get shape; return semi-axes & angle.
    Gi = _project_spd(Gi, eps=eps, norm="trace")
    vals, vecs = np.linalg.eigh(Gi)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    a = np.sqrt(max(vals[0], eps)) * scale
    b = np.sqrt(max(vals[-1], eps)) * scale
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return a, b, theta


def _scaling_values(G, mode="anisotropy", eps=1e-8):
    # Robust [0,1] scaling; fall back to ones if degenerate.
    if mode == "logdet":
        # log det(G) via eigenvalues (more stable than SVD on 2x2)
        vals = np.linalg.eigvalsh(G)
        vals = np.clip(vals, eps, None)
        s = np.sum(np.log(vals), axis=1)
    elif mode == "anisotropy":
        vals = np.linalg.eigvalsh(G)
        vals = np.clip(vals, eps, None)
        s = np.log(vals[:, -1] / vals[:, 0])
    else:
        s = np.ones(G.shape[0], dtype=float)
    s = s - np.nanmin(s)
    rng = np.nanmax(s)
    if not np.isfinite(rng) or rng < eps:
        return np.ones_like(s)
    return s / (rng + eps)

def _prepare_colors(c, n, cmap="viridis", vmin=None, vmax=None, default_alpha=None):
    if c is None:
        return None
    try:
        arr = np.asarray(c)
        if arr.ndim == 1 and arr.shape[0] == n and np.issubdtype(arr.dtype, np.number):
            norm = mcolors.Normalize(vmin=None if vmin is None else float(vmin),
                                     vmax=None if vmax is None else float(vmax))
            mapper = cm.get_cmap(cmap)
            rgba = mapper(norm(arr))
            if default_alpha is not None:
                rgba[:, 3] = default_alpha
            return rgba
        if arr.ndim == 2 and arr.shape[0] == n and arr.shape[1] in (3, 4):
            rgba = np.zeros((n, 4), float)
            rgba[:, :arr.shape[1]] = arr
            if arr.shape[1] == 3:
                rgba[:, 3] = 1.0 if default_alpha is None else default_alpha
            if default_alpha is not None:
                rgba[:, 3] = default_alpha
            return rgba
    except Exception:
        pass
    # Try list of color specs or categorical labels
    try:
        # If elements are valid color specs, convert each
        rgba = np.array([mcolors.to_rgba(ci) for ci in c], float)
        if rgba.shape[0] == n:
            if default_alpha is not None:
                rgba[:, 3] = default_alpha
            return rgba
    except Exception:
        pass
    # Treat as categorical labels
    labels, inv = np.unique(np.asarray(c), return_inverse=True)
    base = cm.get_cmap("tab20" if len(labels) > 10 else "tab10")
    palette = np.array([base(i / max(1, len(labels) - 1)) for i in range(len(labels))], float)
    rgba = palette[inv]
    if default_alpha is not None:
        rgba[:, 3] = default_alpha
    return rgba

def plot_riemann_metric_localized(
    Y,
    L,
    G_emb=None,
    n_plot=1500,
    scale_mode="anisotropy",
    scale_gain=1.0,
    scale_base="auto",
    alpha=0.25,
    edgecolor=None,
    facecolor=None,
    ax=None,
    seed=7,
    zorder=2,
    show_points=True,
    colors=None,              # per-sample colors (numeric, categorical, or color specs)
    cmap="viridis",
    vmin=None,
    vmax=None,
    point_alpha=0.6,
    ellipse_alpha=None,
    point_size=6,
    scatter_kw=None,
):
    if plt is None or Ellipse is None:
        raise RuntimeError("matplotlib is required for plotting.")
    Y = _center(Y)
    L = _symmetrize(L)
    if Y.shape[1] != 2:
        raise ValueError("plot_riemann_metric_localized expects 2D embeddings.")
    if G_emb is None:
        r = RiemannMetric(Y, L)
        G_emb = r.get_rmetric()

    # Pre-project G to SPD to avoid repeated eigs
    G_emb = np.asarray(G_emb)
    G_emb = np.stack([_project_spd(G_emb[i]) for i in range(G_emb.shape[0])], axis=0)

    rng = np.random.default_rng(seed)
    n = Y.shape[0]
    idx = rng.choice(n, size=min(n_plot, n), replace=False)

    if ax is None:
        ax = plt.gca()

    # Establish target axes limits EARLY (we will fit ellipses inside these)
    x0, y0 = Y.min(0); x1, y1 = Y.max(0)
    span = max(x1 - x0, y1 - y0)
    base = (0.05 * span) if scale_base == "auto" else float(scale_base)
    base = max(base, 1e-6)
    pad = 0.06 * span
    xL, xR = x0 - pad, x1 + pad
    yB, yT = y0 - pad, y1 + pad
    ax.set_xlim(xL, xR)
    ax.set_ylim(yB, yT)

    scales = _scaling_values(G_emb, mode=scale_mode)
    if facecolor is None:
        facecolor = "C0"
    if edgecolor is None:
        edgecolor = "k"

    rgba_all = _prepare_colors(colors, n, cmap=cmap, vmin=vmin, vmax=vmax,
                               default_alpha=point_alpha)

    if show_points:
        if rgba_all is not None:
            kw = dict(s=point_size, zorder=zorder - 2)
            if scatter_kw: kw.update(scatter_kw)
            ax.scatter(Y[:, 0], Y[:, 1], c=rgba_all, **kw)
        else:
            kw = dict(s=point_size, c="0.3", alpha=point_alpha, zorder=zorder - 2)
            if scatter_kw: kw.update(scatter_kw)
            ax.scatter(Y[:, 0], Y[:, 1], **kw)

    min_axis = 0.2 * base
    for i in idx:
        a, b, theta = _ellipse_from_G(G_emb[i], scale=scale_gain * base)
        a = max(a * (0.5 + 0.5 * scales[i]), min_axis)
        b = max(b * (0.5 + 0.5 * scales[i]), min_axis)
        if not (np.isfinite(a) and np.isfinite(b)):
            continue

        # Rotation-safe edge fitting: scale ellipse so its circumscribed circle fits in axes
        cx, cy = Y[i, 0], Y[i, 1]
        r = max(a, b)  # circumscribed circle radius (conservative bound)
        allowed = min(cx - xL, xR - cx, cy - yB, yT - cy)
        if allowed <= 0:
            # Center is outside or on edge; skip to avoid drawing outside the box
            continue
        s_edge = min(1.0, allowed / r)
        a *= s_edge
        b *= s_edge

        fc = facecolor
        ec = edgecolor
        if rgba_all is not None:
            rch, gch, bch, a_in = rgba_all[i]
            a_fill = alpha if ellipse_alpha is None else ellipse_alpha
            fc = (rch, gch, bch, a_fill)
            ec = (rch, gch, bch, 1.0 if ellipse_alpha is None else ellipse_alpha)

        e = Ellipse(
            (cx, cy),
            width=2 * a,
            height=2 * b,
            angle=theta,
            facecolor=fc if rgba_all is not None else facecolor,
            edgecolor=ec,
            alpha=None if rgba_all is not None else (alpha if ellipse_alpha is None else ellipse_alpha),
            linewidth=0.3,
            zorder=zorder,
        )
        e.set_clip_on(True)  # still clip (belt-and-braces), but scaling should keep it inside
        ax.add_patch(e)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    return ax



def plot_riemann_metric_global(
    Y,
    L,
    G_emb=None,
    grid_res=20,
    k_avg=16,
    scale_mode="anisotropy",
    scale_gain=1.0,
    scale_base="auto",
    alpha=0.35,
    edgecolor="k",
    cmap="coolwarm",
    vmin=None, vmax=None,
    ax=None,
    zorder=3,
    show_points=True,
    point_alpha=0.25,
    point_size=4,
    scatter_kw=None,
    min_sep_factor=0.9,
    choose_strong_first=True,
    deformation_vals=None,
    deformation_kwargs=None,
    respect_existing_limits=True,
):
    """
    Draw grid-averaged indicatrices with:
      • Poisson-like thinning to reduce ellipse overlap (min center separation).
      • Ellipse facecolor driven by local expansion/contraction (centered logdet).

    Coloring:
      deformation at a grid site = average of per-cell deformation of its k_avg neighbors.
      vmin/vmax can be supplied or inferred (symmetric clipping).

    Non-overlap:
      greedy selection of grid sites with pairwise spacing >= min_sep_factor * base.
    """
    if plt is None or Ellipse is None:
        raise RuntimeError("matplotlib is required for plotting.")
    from matplotlib import cm, colors as mcolors
    from topo.eval.rmetric import (
        RiemannMetric, _symmetrize as _sz, _project_spd, _scaling_values
    )

    Y_plot = np.asarray(Y, dtype=float)
    L = _sz(L)
    if Y_plot.shape[1] != 2:
        raise ValueError("plot_riemann_metric_global expects 2D embeddings.")

    if G_emb is None:
        G_emb = RiemannMetric(Y_plot, L).get_rmetric()
    G_emb = np.asarray(G_emb)
    G_emb = np.stack([_project_spd(G_emb[i]) for i in range(G_emb.shape[0])], axis=0)

    # Deformation (for colors)
    if deformation_vals is None:
        kwargs = dict(center="median", diffusion_t=0, normalize="symmetric",
                      clip_percentile=2.0, return_limits=True)
        if deformation_kwargs: kwargs.update(deformation_kwargs)
        deform_vals, (vmin_auto, vmax_auto) = calculate_deformation(Y_plot, L, G_emb=G_emb, **kwargs)
    else:
        deform_vals = np.asarray(deformation_vals)
        a = np.nanmax(np.abs(np.clip(
            deform_vals,
            np.nanpercentile(deform_vals, 2.0),
            np.nanpercentile(deform_vals, 98.0)
        )))
        vmin_auto, vmax_auto = -a, a

    if ax is None:
        ax = plt.gca()

    # Establish target axes limits EARLY; we will fit ellipses inside these
    x0, y0 = Y_plot.min(0); x1, y1 = Y_plot.max(0)
    span = max(x1 - x0, y1 - y0)
    pad = 0.06 * span
    if respect_existing_limits:
        xL, xR = ax.get_xlim()
        yB, yT = ax.get_ylim()
        # if axes are at defaults (0,1), fall back to embedding limits
        if (xL, xR) == (0.0, 1.0) and (yB, yT) == (0.0, 1.0):
            xL, xR = x0 - pad, x1 + pad
            yB, yT = y0 - pad, y1 + pad
            ax.set_xlim(xL, xR); ax.set_ylim(yB, yT)
    else:
        xL, xR = x0 - pad, x1 + pad
        yB, yT = y0 - pad, y1 + pad
        ax.set_xlim(xL, xR); ax.set_ylim(yB, yT)

    # Grid over the original extent
    gx = np.linspace(x0, x1, max(2, int(grid_res)))
    gy = np.linspace(y0, y1, max(2, int(grid_res)))
    XX, YY = np.meshgrid(gx, gy)
    grid_pts = np.c_[XX.ravel(), YY.ravel()]

    # Neighbors of each grid point
    k_eff = min(max(2, int(k_avg)), Y_plot.shape[0])
    if _HAVE_SK:
        nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto").fit(Y_plot)
        _, inds = nn.kneighbors(grid_pts, return_distance=True)
    else:
        diffs = grid_pts[:, None, :] - Y_plot[None, :, :]
        d2 = np.sum(diffs * diffs, axis=2)
        inds = np.argsort(d2, axis=1)[:, :k_eff]

    # Averages at grid points
    G_avg = np.mean(G_emb[inds], axis=1)
    deform_grid = np.mean(deform_vals[inds], axis=1)

    # Sizes and colors
    base = (0.05 * span) if scale_base == "auto" else float(scale_base)
    base = max(base, 1e-6)
    vmin_eff = vmin if vmin is not None else vmin_auto
    vmax_eff = vmax if vmax is not None else vmax_auto
    norm = mcolors.Normalize(vmin=vmin_eff, vmax=vmax_eff)
    mapper = cm.get_cmap(cmap)

    scales = _scaling_values(G_avg, mode=scale_mode)
    min_axis = 0.2 * base

    # Thinning to reduce overlaps
    keep = []
    taken = np.zeros(grid_pts.shape[0], dtype=bool)
    order = np.argsort(-np.abs(deform_grid)) if choose_strong_first else np.arange(grid_pts.shape[0])
    min_sep = min_sep_factor * base

    for j in order:
        if taken[j]:
            continue
        if not keep:
            keep.append(j); taken[j] = True; continue
        d2 = np.sum((grid_pts[keep] - grid_pts[j])**2, axis=1)
        if np.all(np.sqrt(d2) >= min_sep):
            keep.append(j); taken[j] = True
    keep = np.array(keep, dtype=int)

    # Optional background points
    if show_points:
        kw = dict(s=point_size, c="0.3", alpha=point_alpha, zorder=zorder - 3)
        if scatter_kw: kw.update(scatter_kw)
        ax.scatter(Y_plot[:, 0], Y_plot[:, 1], **kw)

    # Draw ellipses, scaling them to stay INSIDE the axes box.
    # Use rotation-safe constraint via circumscribed circle radius r = max(a, b).
    for i in keep:
        Gi = 0.5 * (G_avg[i] + G_avg[i].T)
        w, v = np.linalg.eigh(Gi)
        w = np.clip(w, 1e-12, None)
        idx = np.argsort(w)[::-1]

        a = np.sqrt(w[idx[0]]) * scale_gain * base
        b = np.sqrt(w[idx[-1]]) * scale_gain * base
        a = max(a * (0.5 + 0.5 * scales[i]), min_axis)
        b = max(b * (0.5 + 0.5 * scales[i]), min_axis)
        theta = np.degrees(np.arctan2(v[:, idx[0]][1], v[:, idx[0]][0]))

        cx, cy = grid_pts[i, 0], grid_pts[i, 1]
        r = max(a, b)  # circumscribed circle radius (conservative)

        # Allowed radius so ellipse (circle upper bound) stays within axes box
        allowed = min(cx - xL, xR - cx, cy - yB, yT - cy)
        if allowed <= 0:
            # outside bounds; skip drawing
            continue
        s_edge = min(1.0, allowed / r)
        a *= s_edge
        b *= s_edge

        fc = mapper(norm(deform_grid[i]))
        e = Ellipse(
            (cx, cy),
            width=2 * a,
            height=2 * b,
            angle=theta,
            facecolor=fc,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=0.4,
            zorder=zorder,
        )
        e.set_clip_on(True)
        ax.add_patch(e)

    # Aesthetics (limits already set)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    # Colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cb = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Contraction  ←  centered log det(G)  →  Expansion")

    return ax



def calculate_deformation(
    Y,
    L,
    G_emb=None,
    center="median",            # {'median','mean', float}
    use_dual=False,             # if True, compute with dual metric H so that val = -logdet(H)
    diffusion_t=0,              # integer steps of graph diffusion smoothing on the scalar field
    diffusion_op=None,          # optional Markov operator P; if None and diffusion_t>0, built from L
    re_center_after_diffusion=True,
    clip_percentile=2.0,        # robust clipping for color limits
    normalize="symmetric",      # {'symmetric','none'}
    return_limits=True,
):
    """
    Returns:
        vals : (n,) centered log-det(G) [<0 contraction, >0 expansion]
        (vmin, vmax) : effective color limits (if return_limits)
    """
    if plt is None:
        # still allow compute w/o plotting backend
        pass

    from topo.eval.rmetric import (
        RiemannMetric, _center as _ct, _symmetrize as _sz,
        _ensure_array as _ea
    )
    Y = _ct(Y); L = _sz(L)
    if G_emb is None:
        G_emb = RiemannMetric(Y, L).get_rmetric()

    if use_dual:
        # equivalently: val = -logdet(H)
        lam = np.linalg.eigvalsh(G_emb)  # treat input as H if provided
        lam = np.clip(lam, 1e-12, None)
        logdet = -np.sum(np.log(lam), axis=1)
    else:
        lam = np.linalg.eigvalsh(G_emb)
        lam = np.clip(lam, 1e-12, None)
        logdet = np.sum(np.log(lam), axis=1)

    # center
    if isinstance(center, str):
        ckey = center.lower()
        ref = np.nanmedian(logdet) if ckey == "median" else np.nanmean(logdet)
    else:
        ref = float(center)
    vals = logdet - ref

    # optional diffusion smoothing
    if int(diffusion_t) > 0:
        if diffusion_op is None:
            # build random-walk from L
            Ld = _ea(L)
            d = np.clip(np.diag(Ld).astype(float), 1e-12, None)
            W = np.diag(d) - Ld
            P = (W / d[:, None])
        else:
            P = diffusion_op
        v = vals.copy()
        for _ in range(int(diffusion_t)):
            v = P @ v
        vals = v
        if re_center_after_diffusion:
            if isinstance(center, str) and center.lower() == "median":
                vals -= np.nanmedian(vals)
            else:
                vals -= np.nanmean(vals)

    # robust color limits
    lo = np.nanpercentile(vals, clip_percentile)
    hi = np.nanpercentile(vals, 100.0 - clip_percentile)
    vals_clipped = np.clip(vals, lo, hi)
    if normalize == "symmetric":
        a = np.nanmax(np.abs(vals_clipped))
        vmin_eff, vmax_eff = -a, a
    else:
        vmin_eff, vmax_eff = lo, hi

    return (vals, (vmin_eff, vmax_eff)) if return_limits else vals


def plot_metric_contraction_expansion(
    Y,
    L,
    G_emb=None,
    center="median",
    normalize="symmetric",
    clip_percentile=2.0,
    s=6,
    alpha=0.9,
    cmap="coolwarm",
    vmin=None, vmax=None,
    show_colorbar=True,
    ax=None,
    zorder=1,
    use_dual=False,
    diffusion_t=0,
    diffusion_op=None,
    re_center_after_diffusion=True,
    plot_strong_last=True,
    legend_fontsize=9,         # NEW: control fontsize of colorbar label
    title="Local contraction / expansion",  # NEW: optional title
    title_fontsize=11,         # NEW: control fontsize of title
):
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting.")

    from topo.eval.rmetric import _center as _ct, _symmetrize as _sz
    Y = _ct(Y); L = _sz(L)
    if Y.shape[1] != 2:
        raise ValueError("plot_metric_contraction_expansion expects a 2D embedding (n,2).")

    vals, (vmin_eff_auto, vmax_eff_auto) = calculate_deformation(
        Y, L, G_emb=G_emb, center=center, use_dual=use_dual,
        diffusion_t=diffusion_t, diffusion_op=diffusion_op,
        re_center_after_diffusion=re_center_after_diffusion,
        clip_percentile=clip_percentile, normalize=normalize, return_limits=True
    )
    vmin_eff = vmin if vmin is not None else vmin_eff_auto
    vmax_eff = vmax if vmax is not None else vmax_eff_auto

    if ax is None:
        ax = plt.gca()

    order = np.arange(vals.shape[0])
    if plot_strong_last:
        order = np.argsort(np.abs(vals))

    sc = ax.scatter(
        Y[order, 0], Y[order, 1],
        c=vals[order],
        s=s,
        alpha=alpha,
        cmap=cmap,
        vmin=vmin_eff, vmax=vmax_eff,
        zorder=zorder,
    )

    x0, y0 = Y.min(0); x1, y1 = Y.max(0)
    span = max(x1 - x0, y1 - y0); pad = 0.06 * span
    ax.set_xlim(x0 - pad, x1 + pad); ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect("equal", adjustable="box"); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    if show_colorbar:
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Contraction  ←  log det(G)  →  Expansion", fontsize=legend_fontsize)

    if title:
        ax.set_title(title, fontsize=title_fontsize)

    return ax, vals


# Backward-compatibility shim (kept to avoid import errors in older code)
def get_eccentricity(emb, laplacian, G_emb=None):
    warnings.warn(
        "get_eccentricity is deprecated. Use plot_riemann_metric_localized/global with scale_mode='anisotropy' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    Y = _center(emb)
    L = _symmetrize(laplacian)
    if G_emb is None:
        r = RiemannMetric(Y, L)
        G_emb = r.get_rmetric()
    vals = np.linalg.eigvalsh(G_emb)
    vals = np.clip(vals, 1e-12, None)
    a = np.sqrt(vals[:, -1])
    b = np.sqrt(vals[:, 0])
    return (a - b) / (a + 1e-12)
