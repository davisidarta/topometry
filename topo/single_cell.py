# Wrapper functions for single-cell analysis with Scanpy and TopOMetry
# Author: David S Oliveira <david.oliveira(at)dpag(dot)ox(dot)ac(dot)uk>
# All of these functions call scanpy and thus require it for working
# However, I opted not to include it as a hard-dependency as not all users are interested in single-cell analysis
#
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.sparse import issparse, csr_matrix
from scipy.sparse.csgraph import connected_components
import colorsys
from matplotlib.backends.backend_pdf import PdfPages
import textwrap

try:
    import scanpy as sc
    _HAVE_SCANPY = True
except ImportError:
    _HAVE_SCANPY = False

# Functions will be defined only if user has scanpy installed.
if _HAVE_SCANPY:
    from anndata import AnnData

    from topo.topograph import TopOGraph, save_topograph, load_topograph
    from topo.eval.rmetric import (
        plot_riemann_metric_localized,
        plot_riemann_metric_global,
        plot_metric_contraction_expansion,
    )
    from topo.tpgraph.intrinsic_dim import IntrinsicDim

    # Geometry metrics
    from topo.eval.topo_metrics import (
            rank_diffusion_correlation,
            diffusion_knn_preservation,
            rowwise_js_similarity,
            sparse_neighborhood_f1,
            spectral_procrustes,
            topo_preserve_score,
        )


    # -----------------------
    # Helpers (internal API)
    # -----------------------
    def _first_non_none(*vals):
        for v in vals:
            if v is not None:
                return v
        return None
    
    def _safe_set_obsm(adata: "AnnData", key: str, val):
        if val is None:
            return
        arr = np.asarray(val)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[0] != adata.n_obs:
            return
        adata.obsm[key] = arr

    def _safe_set_obsp(adata: "AnnData", key: str, val):
        if val is None:
            return
        M = val
        if not sp.issparse(M):
            M = np.asarray(M)
        if M.shape != (adata.n_obs, adata.n_obs):
            return
        adata.obsp[key] = M

    def _ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)
        return path

    def _std_cols(Z, eps: float = 1e-12):
        Z = np.asarray(Z, float)
        Z = Z - np.nanmean(Z, axis=0, keepdims=True)
        sd = np.nanstd(Z, axis=0, keepdims=True)
        return Z / (sd + eps)

    def _spectral_alignment_by_label(
        adata: AnnData,
        labels_key: str,
        scaffold_key: str = "X_ms_spectral_scaffold",
        standardize: bool = True,
        top_k: int = 3,
        out_key: str = "spectral_alignment_summary",
    ):
        def _cohens_d(x, g):
            a, b = x[g], x[~g]
            if a.size < 2 or b.size < 2: return np.nan
            da, db = a.mean(), b.mean()
            va, vb = a.var(ddof=1), b.var(ddof=1)
            n1, n2 = a.size, b.size
            spooled = np.sqrt(((n1-1)*va + (n2-1)*vb) / max(1, n1+n2-2))
            if spooled <= 0: return np.nan
            return (da - db) / spooled

        Z = np.asarray(adata.obsm[scaffold_key], float)
        Zs = _std_cols(Z) if standardize else Z
        labels = adata.obs[labels_key].astype('category')

        rows = []
        for lab in labels.cat.categories:
            g = (labels.values == lab)
            ds = np.array([_cohens_d(Zs[:, k], g) for k in range(Zs.shape[1])])
            order = np.argsort(-np.abs(ds))
            for j in range(min(top_k, Zs.shape[1])):
                k = int(order[j])
                rows.append({"label": lab, "axis": k, "cohens_d": ds[k],
                            "axis_var": np.var(Zs[:, k], ddof=1)})
        df = pd.DataFrame(rows).sort_values(["label", "cohens_d"], ascending=[True, False]).reset_index(drop=True)
        adata.uns[out_key] = df
        return df

    def _eigvals_from_tg(tg: TopOGraph, variant: str = "msDM") -> np.ndarray:
        key = (("msDM with " + tg.base_kernel_version) if variant == "msDM"
               else ("DM with " + tg.base_kernel_version))
        try:
            _, evals = tg.EigenbasisDict[key].results(return_evals=True)
        except Exception:
            for _k, obj in getattr(tg, "EigenbasisDict", {}).items():
                try:
                    _, evals = obj.results(return_evals=True)
                    break
                except Exception:
                    continue
            else:
                return np.array([])
        ev = np.asarray(evals).ravel()
        ev = ev[np.isfinite(ev)]
        ev = ev[ev > 0]
        ev = np.sort(ev)[::-1]
        return ev

    def _palette_from_obs(adata: AnnData, key: str):
        if key not in adata.obs:
            return None, None, None
        labels = adata.obs[key].astype("category")
        cats = labels.cat.categories
        palette = adata.uns.get(f"{key}_colors", None)
        if palette is None or len(palette) < len(cats):
            palette = sc.pl.palettes.default_102
        lut = dict(zip(cats, palette[:len(cats)]))
        colors = labels.map(lut).astype(str).values
        return labels, lut, colors

    def _hex_to_hsv(hex_color: str):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)/255.0
        g = int(hex_color[2:4], 16)/255.0
        b = int(hex_color[4:6], 16)/255.0
        return colorsys.rgb_to_hsv(r, g, b)

    def _hsv_to_hex(h, s, v):
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return '#%02x%02x%02x' % (int(round(r*255)), int(round(g*255)), int(round(b*255)))

    def _componentwise_pseudotime_colors(adata: AnnData, tg: TopOGraph, cluster_key: str = "topo_clusters",
                                         base_brightness: float = 0.25, max_brightness: float = 1.0,
                                         pt_key_out: str = "topo_pseudotime_component",
                                         color_key_out: str = "pseudotime_color_hex"):
        P = getattr(tg, "P_of_msZ", None)
        if P is None:
            return None
        A = P
        if not issparse(A):
            A = csr_matrix(A)
        A = A.maximum(A.T)
        A.data[:] = 1.0
        n_comp, labels_cc = connected_components(A, directed=False, connection='weak')
        adata.obs['_topo_component'] = pd.Categorical(labels_cc.astype(int))

        lbls, lut, colors = _palette_from_obs(adata, cluster_key)
        if lbls is None:
            lut = {c: "#1f77b4" for c in np.unique(labels_cc)}
            colors = np.array([lut[np.unique(labels_cc)[0]]] * adata.n_obs)

        try:
            pt_full = adata.obs['topo_pseudotime'].values
        except Exception:
            return None

        pt_scaled = np.zeros_like(pt_full, dtype=float)
        for c in range(n_comp):
            idx = np.where(labels_cc == c)[0]
            if idx.size == 0:
                continue
            v = pt_full[idx]
            vmin = np.nanmin(v); vmax = np.nanmax(v)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                pt_scaled[idx] = 0.5
            else:
                pt_scaled[idx] = (v - vmin) / (vmax - vmin)

        alpha = 0.7
        V = np.clip(max_brightness - alpha * pt_scaled, 0.0, 1.0)

        hex_base = colors
        out_hex = []
        for i in range(adata.n_obs):
            hb = hex_base[i] if isinstance(hex_base[i], str) else "#1f77b4"
            try:
                h, s, _ = _hex_to_hsv(hb)
            except Exception:
                h, s = 0.58, 0.65
            out_hex.append(_hsv_to_hex(h, s, max(base_brightness, V[i])))

        adata.obs[pt_key_out] = pt_scaled
        adata.obs[color_key_out] = out_hex
        return pt_key_out, color_key_out, n_comp

    def _composition_barplot(ax, adata: AnnData, column_key: str, color_key: str, invert: bool = False):
        df = adata.obs.groupby([column_key, color_key]).size().unstack(fill_value=0)
        df = df.div(df.sum(axis=1), axis=0)
        if invert:
            df.plot(kind='barh', stacked=True, colormap='tab20', ax=ax, legend=False)
            ax.set_xlabel(column_key, fontsize=11); ax.set_ylabel("")
        else:
            df.plot(kind='bar', stacked=True, colormap='tab20', ax=ax, legend=False)
            ax.set_xlabel(column_key, fontsize=11); ax.set_ylabel("")
        ax.grid(False)
        ax.tick_params(axis='x', labelrotation=90, labelsize=8)
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            handles = [plt.Rectangle((0,0),1,1,fc='gray', ec='none')]
            labels = [color_key]
        leg = ax.legend(handles, labels, title=color_key, bbox_to_anchor=(1.02,1), loc='upper left', frameon=False, fontsize=8)
        for t in leg.get_texts():
            t.set_fontsize(8)

    def _decay_plot_axes_original(ax_curve, ax_diff, evals: np.ndarray, title: str | None = None):
        if evals is None or len(evals) == 0:
            ax_curve.axis('off'); ax_diff.axis('off')
            ax_curve.text(0.5, 0.5, "No eigenvalues", ha='center', va='center')
            return
        max_eigs = int(np.sum(evals > 0, axis=0))
        first_diff = np.diff(evals)
        eigengap = np.argmax(first_diff) + 1

        if title is not None:
            ax_curve.set_title(title, fontsize=14)
        ax_curve.plot(range(0, len(evals)), evals, 'b')
        ax_curve.set_ylabel('Eigenvalues', fontsize=12)
        ax_curve.set_xlabel('Eigenvectors', fontsize=12)
        if max_eigs == len(evals):
            ax_curve.vlines(eigengap, ax_curve.get_ylim()[0], ax_curve.get_ylim()[1], linestyles="--", label='Eigengap')
            ax_curve.legend(prop={'size': 12}, fontsize=12, loc='best')
        else:
            ax_curve.vlines(max_eigs, ax_curve.get_ylim()[0], ax_curve.get_ylim()[1], linestyles="--", label='Eigengap')
            ax_curve.legend(prop={'size': 12}, fontsize=12, loc='best')

        ax_diff.set_yscale('log')
        ax_diff.scatter(range(0, len(first_diff)), np.abs(first_diff), s=8)
        ax_diff.set_ylabel('Eigenvalues first derivatives (abs)', fontsize=12)
        ax_diff.set_xlabel('Eigenvalues', fontsize=12)
        ax_diff.tick_params(axis='y', labelleft=False)
        if max_eigs == len(evals):
            ax_diff.vlines(eigengap, ax_diff.get_ylim()[0], ax_diff.get_ylim()[1], linestyles="--", label='Eigengap')
        else:
            ax_diff.vlines(max_eigs, ax_diff.get_ylim()[0], ax_diff.get_ylim()[1], linestyles="--", label='Eigengap')

    def _plot_id_histograms_original(ax_fsa, ax_mle, id_est: IntrinsicDim | None):
        def _one(ax, method_name: str):
            if (id_est is None) or (method_name not in id_est.local_id) or (len(id_est.local_id[method_name]) == 0):
                ax.axis('off'); ax.text(0.5,0.5,"N/A", ha='center', va='center'); return
            for key in id_est.local_id[method_name].keys():
                x = id_est.local_id[method_name][key]
                label = 'k = ' + key + '    ( estim.i.d. = ' + str(int(id_est.global_id[method_name][key])) + ' )'
                ax.hist(x, bins=30, histtype='step', stacked=True, density=True, log=False, label=label)
            ax.set_title(method_name.upper(), fontsize=14, pad=8)
            ax.legend(prop={'size': 9}, fontsize=9)
            ax.set_xlabel('Estimated intrinsic dimension', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.tick_params(axis='y', labelleft=False)
        _one(ax_fsa, 'fsa')
        _one(ax_mle, 'mle')

    # ---------- PaCMAP ensuring ----------
    def _ensure_pacmap(tg: TopOGraph):
        dm_ok = False
        ms_ok = False
        try:
            _ = tg.PaCMAP
            dm_ok = getattr(tg, "PaCMAP", None) is not None
        except Exception:
            pass
        try:
            _ = tg.msPaCMAP
            ms_ok = getattr(tg, "msPaCMAP", None) is not None
        except Exception:
            pass

        if not (dm_ok and ms_ok):
            proj = getattr(tg, "project", None)
            if callable(proj):
                try:
                    proj(projection_method="PaCMAP", which="Z")
                    dm_ok = getattr(tg, "PaCMAP", None) is not None
                except Exception:
                    pass
                try:
                    proj(projection_method="PaCMAP", which="msZ")
                    ms_ok = getattr(tg, "msPaCMAP", None) is not None
                except Exception:
                    pass

        def _try(method_name, *args, **kwargs):
            f = getattr(tg, method_name, None)
            if callable(f):
                try:
                    return f(*args, **kwargs)
                except Exception:
                    return None
            return None

        if not dm_ok:
            for name in ("compute_pacmap", "compute_PaCMAP", "pacmap", "PaCMAP_fit"):
                _try(name)
                if getattr(tg, "PaCMAP", None) is not None:
                    dm_ok = True
                    break

        if not ms_ok:
            for name in ("compute_pacmap", "compute_PaCMAP", "pacmap", "PaCMAP_fit"):
                _try(name, multiscale=True)
                if getattr(tg, "msPaCMAP", None) is not None:
                    ms_ok = True
                    break

        return dm_ok, ms_ok

    # ---------- Simple kNN → row-stochastic P from coordinates ----------
    def _rowstochastic_from_coords(Y: np.ndarray, k: int = 30) -> sp.csr_matrix:
        Y = np.asarray(Y, float)
        n = Y.shape[0]
        k = int(max(1, min(k, n - 1)))

        try:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
            nn.fit(Y)
            dists, idx = nn.kneighbors(Y, return_distance=True)
        except Exception:
            D2 = np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2)
            idx = np.argpartition(D2, kth=(k + 1), axis=1)[:, :k + 1]
            dists = np.take_along_axis(D2, idx, axis=1)

        rows = np.repeat(np.arange(n), k)
        cols = idx[:, 1:].ravel()  # drop self
        data = np.ones(rows.size, dtype=float)
        P = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        row_sums = np.asarray(P.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        inv = sp.diags(1.0 / row_sums)
        P = inv @ P
        return P

    # ---------- aspect ratio enforcement ----------
    def _force_equal_axes(fig: plt.Figure):
        """Set 1:1 aspect on ALL axes in a figure."""
        for ax in fig.get_axes():
            try:
                ax.set_aspect('equal', adjustable='box')
            except Exception:
                pass

    # =========================
    # STEPWISE API (new)
    # =========================

    def fit_adata(
        adata: AnnData,
        tg: TopOGraph | None = None,
        *,
        projections: tuple[str, ...] = ("MAP", "PaCMAP"),
        do_leiden: bool = True,
        leiden_key_base: str = "topo_clusters",
        leiden_resolutions: list[float] | tuple[float, ...] = (0.2, 0.8, 1.2),
        leiden_primary_index: int = 1,
        **topograph_kwargs,
    ):
        """
        Fit or reuse a TopOGraph, populate adata with scaffolds, projections, and
        cluster the refined DM graph (P_of_Z == tg.graph_kernel.P).

        Returns
        -------
        tg : TopOGraph
        """
        if tg is None:
            tg = TopOGraph(**topograph_kwargs).fit(adata.X)
        else:
            # (re)fit if user passed new kwargs
            if topograph_kwargs:
                tg.set_params(**topograph_kwargs)
                tg.fit(adata.X)

        # --- spectral scaffolds
        try:
            _safe_set_obsm(adata, 'X_ms_spectral_scaffold', tg.spectral_scaffold(multiscale=True))
        except Exception:
            pass
        try:
            _safe_set_obsm(adata, 'X_spectral_scaffold', tg.spectral_scaffold(multiscale=False))
        except Exception:
            pass

        # --- ensure requested projections over {DM, msDM}
        proj_fn = getattr(tg, "project", None)
        for multiscale in (False, True):
            which = "msZ" if multiscale else "Z"
            ms_prefix = "X_msTopo" if multiscale else "X_Topo"
            for proj in projections:
                key = f"{ms_prefix}{proj}"
                try:
                    Y = proj_fn(projection_method=proj, which=multiscale) if callable(proj_fn) else None
                    if Y is None:
                        if proj.upper() == "MAP":
                            Y = getattr(tg, "msTopoMAP" if multiscale else "TopoMAP", None)
                        elif proj.upper() == "PACMAP":
                            _ensure_pacmap(tg)
                            Y = getattr(tg, "msTopoPaCMAP" if multiscale else "TopoPaCMAP", None)
                    _safe_set_obsm(adata, key, Y)
                except Exception:
                    # last-ditch fallback
                    try:
                        if proj.upper() == "MAP":
                            _safe_set_obsm(adata, key, getattr(tg, "msTopoMAP" if multiscale else "TopoMAP", None))
                        elif proj.upper() == "PACMAP":
                            _ensure_pacmap(tg)
                            _safe_set_obsm(adata, key, getattr(tg, "msTopoPaCMAP" if multiscale else "TopoPaCMAP", None))
                    except Exception:
                        pass

        # Default alias
        if adata.obsm.get('X_msTopoMAP', None) is not None:
            _safe_set_obsm(adata, 'X_TopoMAP_default', adata.obsm['X_msTopoMAP'])
        if adata.obsm.get('X_TopoMAP', None) is not None:
            _safe_set_obsm(adata, 'X_TopoMAP_default', adata.obsm['X_TopoMAP'])
        if adata.obsm.get('X_msTopoPaCMAP', None) is not None:
            _safe_set_obsm(adata, 'X_TopoPaCMAP_default', adata.obsm['X_msTopoPaCMAP'])
        if adata.obsm.get('X_TopoPaCMAP', None) is not None:
            _safe_set_obsm(adata, 'X_TopoPaCMAP_default', adata.obsm['X_TopoPaCMAP'])

        # --- refined Markov operators (DM + msDM)
        _safe_set_obsp(adata, 'topometry_connectivities_dm', getattr(tg, 'P_of_Z', None))
        _safe_set_obsp(adata, 'topometry_connectivities_ms', getattr(tg, 'P_of_msZ', None))
        if adata.obsp.get('topometry_connectivities_ms', None) is not None:
            _safe_set_obsp(adata, 'topometry_connectivities', adata.obsp['topometry_connectivities_ms'])

        # --- clustering on refined DM graph (P_of_Z)
        P_dm = adata.obsp.get('topometry_connectivities_dm', None)
        if do_leiden and P_dm is not None:
            res_list = list(leiden_resolutions) if isinstance(leiden_resolutions, (list, tuple)) else [float(leiden_resolutions)]
            for res in res_list:
                key = f"{leiden_key_base}_res{res}"
                try:
                    sc.tl.leiden(adata, adjacency=P_dm, key_added=key, resolution=float(res))
                except Exception as e:
                    print(f"[topometry] Leiden clustering failed at res={res}: {e}")
            try:
                prim_key = f"{leiden_key_base}_res{res_list[int(leiden_primary_index)]}"
                if prim_key in adata.obs:
                    adata.obs[leiden_key_base] = adata.obs[prim_key].astype("category")
                    if f"{prim_key}_colors" in adata.uns:
                        adata.uns[f"{leiden_key_base}_colors"] = adata.uns[f"{prim_key}_colors"]
            except Exception:
                pass

        return tg


    def intrinsic_dim(
        adata: AnnData,
        tg: TopOGraph | None = None,
        *,
        id_methods: list[str] = ("fsa", "mle"),
        id_k_values: list[int] | None = None,
        n_jobs: int = -1,
    ):
        """
        Populate adata with global+local intrinsic dimension summaries.
        Uses tg summaries when available and also runs IntrinsicDim on adata.X.
        """
        if tg is not None:
            adata.uns['topometry_id_details'] = getattr(tg, "_id_details", None)
            adata.uns['topometry_id_global_mle'] = tg.global_id_mle()
            adata.uns['topometry_id_global_fsa'] = tg.global_id_fsa()
            _loc = tg.local_ids()
            if _loc is not None:
                if isinstance(_loc, dict):
                    for name, vec in _loc.items():
                        if vec is not None:
                            adata.obs[f'local_id_{name}'] = np.asarray(vec)
                else:
                    adata.obs['local_id'] = np.asarray(_loc)

        # Explicit estimator over adata.X
        if id_k_values is None:
            id_k_values = list(range(10, 110, 20))
        try:
            id_est = IntrinsicDim(
                methods=list(id_methods),
                k=list(id_k_values),
                backend='hnswlib',
                metric='euclidean',
                n_jobs=n_jobs,
                plot=False
            )
            id_est.fit(adata.X)
            adata.uns['intrinsic_dim_estimator'] = id_est
            k_low  = str(min(id_k_values))
            k_high = str(max(id_k_values))
            if 'fsa' in id_est.local_id and k_low in id_est.local_id['fsa']:
                adata.obs['id_fsa_k'+k_low] = id_est.local_id['fsa'][k_low]
            if 'fsa' in id_est.local_id and k_high in id_est.local_id['fsa']:
                adata.obs['id_fsa_k'+k_high] = id_est.local_id['fsa'][k_high]
            if 'mle' in id_est.local_id and k_low in id_est.local_id['mle']:
                adata.obs['id_mle_k'+k_low] = id_est.local_id['mle'][k_low]
            if 'mle' in id_est.local_id and k_high in id_est.local_id['mle']:
                adata.obs['id_mle_k'+k_high] = id_est.local_id['mle'][k_high]
        except Exception as e:
            print(f"[TopOMetry] IntrinsicDim estimation skipped: {e}")


    def spectral_selectivity(
        adata: AnnData,
        tg: TopOGraph,
        *,
        weight_mode: str = "lambda_over_one_minus_lambda",
        k_neighbors: int = 30,
        smooth_P: str | None = None,  # {'X','Z','msZ'} or None
        smooth_t: int = 0,
        groupby_candidates: list[str] | None = None,
    ):
        """
        Compute spectral selectivity on the multiscale scaffold and store to adata.
        Also computes alignment-by-label against the first available grouping key.
        """
        spec = tg.spectral_selectivity(
            multiscale=True,
            weight_mode=weight_mode,
            k_neighbors=k_neighbors,
            smooth_P=smooth_P,
            smooth_t=smooth_t,
        )
        adata.obs['spectral_EAS']       = spec['EAS']
        adata.obs['spectral_RayScore']  = spec['RayScore']
        adata.obs['spectral_LAC']       = spec['LAC']
        adata.obs['spectral_axis']      = pd.Categorical(spec['axis'].astype(int))
        adata.obs['spectral_axis_sign'] = pd.Categorical(spec['axis_sign'].astype(int))
        adata.obs['spectral_radius']    = spec['radius']

        # alignment-by-label on ms scaffold, if present
        candidates = groupby_candidates or ['topo_clusters', 'leiden', 'cell_type']
        align_key = next((k for k in candidates if k in adata.obs), None)
        if align_key and ('X_ms_spectral_scaffold' in adata.obsm):
            _spectral_alignment_by_label(
                adata, labels_key=align_key,
                scaffold_key='X_ms_spectral_scaffold',
                top_k=3,
                out_key='spectral_alignment_summary',
            )


    def riemann_diagnostics(
        adata: AnnData,
        tg: TopOGraph,
        *,
        center: str = "median",
        diffusion_t: int = 8,
        diffusion_op: str | None = "X",   # {'X','Z','msZ'} or None
        normalize: str = "symmetric",
        clip_percentile: float = 2.0,
    ):
        """
        Run Riemann diagnostics on EVERY 2-D embedding in adata.obsm.
        Stores per-embedding deformation vectors in adata.obs as:
            metric_deformation__<obsm_key>
        and limits in:
            adata.uns['metric_limits'][<obsm_key>]
        Also caches the Laplacian.
        """
        # Save Laplacian
        try:
            adata.obsp['topometry_laplacian'] = tg.base_kernel.L
        except Exception:
            pass

        adata.uns.setdefault('metric_limits', {})

        for key, Y in adata.obsm.items():
            try:
                arr = np.asarray(Y)
                if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] != adata.n_obs:
                    continue
            except Exception:
                continue

            try:
                riem = tg.riemann_diagnostics(
                    Y=arr,
                    L=tg.base_kernel.L,
                    center=center,
                    diffusion_t=diffusion_t,
                    diffusion_op=diffusion_op,
                    normalize=normalize,
                    clip_percentile=clip_percentile,
                    return_limits=True,
                    compute_metric=False,
                    compute_scalars=True,
                )
                col = f"metric_deformation__{key}"
                adata.obs[col] = riem['deformation']
                adata.uns['metric_limits'][key] = riem.get('limits', None)
            except Exception as e:
                print(f"[TopOMetry] Riemann diagnostics skipped for {key}: {e}")

    def pseudotime_analysis(
        adata,
        tg,
        *,
        starting_cluster,
        groupby: str = "topo_clusters",
        lineage_mask=None,
        random_state: int = 42,
        verbose: bool = True,
    ):
        """
        Compute diffusion-based pseudotime for a *single lineage* (assumes `adata`
        contains only the cells of interest OR a mask is provided).

        This function:
        1) Selects a root cell by sampling one index from the provided cluster
            label in `adata.obs[groupby]` (within the working subset).
        2) Builds multiscale diffusion coordinates from the fitted `TopOGraph`
            (msDM scaffold with lambda/(1-lambda) weighting).
        3) Computes pseudotime as squared Euclidean distance in those coordinates
            to the chosen root, normalized to [0, 1] within the working subset.
        4) Stores results in `adata.obs['topo_pseudotime']`.

        Parameters
        ----------
        adata : AnnData
            If it has the *same* number of cells as the fitted `tg` (no subsetting),
            you may optionally pass `lineage_mask` to restrict the evaluation to a
            subset; values outside the mask are set to NaN.
            If it is a *subset* of the original data, you **must** provide a mask
            aligned to the original `tg` universe (see `lineage_mask` below).
        tg : TopOGraph
            Fitted TopOGraph instance for the original dataset (the one used to
            compute the spectral scaffold).
        starting_cluster : str or int (required)
            Cluster label within `adata.obs[groupby]` from which the root cell
            will be sampled (e.g., a progenitor/early cluster).
        groupby : str, default 'topo_clusters'
            Column in `adata.obs` with cluster labels used to select the root.
        lineage_mask : str or array-like of bool, optional
            - If a **string**, it refers to a boolean column in `adata.obs` that
            marks the working subset (True = included).
            *Use this when `adata.n_obs == tg.n` (no subsetting).*
            - If a **boolean vector**:
                * If `adata.n_obs == tg.n`, it must have length `tg.n` and marks
                the working subset directly.
                * If `adata.n_obs != tg.n` (you passed a subset AnnData), it must
                have length `tg.n` and select the rows in the original `tg`
                universe that correspond to the rows of `adata` (i.e., the exact
                mask you used for subsetting). In this case, pseudotime is computed
                only on those rows and written to `adata` (length matches).
        random_state : int, default 42
            RNG seed for selecting the root inside the starting cluster.
        verbose : bool, default True
            Print progress.

        Returns
        -------
        dict
            {'root': int, 'k_use': int}
            Root is the chosen row index *within the working subset*, and k_use is
            the number of spectral coordinates used.

        Notes
        -----
        - We compute pseudotime directly from msDM spectral coordinates to avoid
        shape mismatches with `tg.pseudotime` when `adata` is a subset.
        - If `adata` is a subset, provide a boolean `lineage_mask` aligned to the
        original `tg` universe so we can subset the spectral coordinates correctly.
        """

        # --- helper: sample root from a cluster within a given AnnData view ---
        def _sample_from_cluster(_adata, *, groupby, group, strategy='random', seed=42):
            if groupby not in _adata.obs.columns:
                raise KeyError(f"`{groupby}` not found in adata.obs")
            labels = _adata.obs[groupby].astype(str).to_numpy()
            target = str(group)
            idxs = np.where(labels == target)[0]
            if idxs.size == 0:
                raise ValueError(f"No cells found for {groupby} == {group!r}")
            if strategy == 'first':
                return int(idxs[0])
            rng = np.random.RandomState(seed)
            return int(rng.choice(idxs))

        # --- 0) Full spectral scaffold + weights from TopOGraph ---
        Z_full = tg.spectral_scaffold(multiscale=True)              # shape (N_full, m)
        key = 'msDM with ' + str(tg.base_kernel_version)
        evals = tg.EigenbasisDict[key].eigenvalues                  # includes λ0
        # choose k (drop the trivial first eigenpair)
        k_use = int(min(64, Z_full.shape[1] - 1)) if Z_full.shape[1] > 1 else 1
        # weights: lambda/(1-lambda), like in tg.pseudotime
        lam = np.asarray(evals[1:k_use+1], float)
        w = (lam / (1.0 - lam + 1e-12))[None, :]
        Psi_full = Z_full[:, :k_use] * w                            # (N_full, k_use)

        N_full = Psi_full.shape[0]
        N_here = adata.n_obs

        # --- 1) Resolve working subset (mask_full) and the AnnData view used for root selection ---
        direct_mode = (N_here == N_full)  # adata covers the same rows as tg in the same order

        if isinstance(lineage_mask, str):
            # string column from *adata.obs*
            if lineage_mask not in adata.obs.columns:
                raise KeyError(f"`{lineage_mask}` not found in adata.obs")
            mask_from_obs = adata.obs[lineage_mask].astype(bool).to_numpy()
            if direct_mode:
                mask_full = mask_from_obs
                adata_view = adata[mask_full]  # view for root selection
            else:
                # We cannot map a short mask (length N_here) to the full `tg` universe.
                raise ValueError(
                    "You passed a string `lineage_mask` but `adata` is a subset "
                    "(adata.n_obs != tg.n). Please provide a boolean `lineage_mask` "
                    "aligned to the *original* tg universe (length == tg.n)."
                )
        elif lineage_mask is None:
            if direct_mode:
                # use all cells
                mask_full = np.ones(N_full, dtype=bool)
                adata_view = adata
            else:
                raise ValueError(
                    "adata.n_obs != tg.n (you passed a subset). Please provide "
                    "`lineage_mask` as a boolean array aligned to the original tg universe."
                )
        else:
            # boolean vector provided
            mask_arr = np.asarray(lineage_mask)
            if mask_arr.dtype != bool:
                mask_arr = mask_arr.astype(bool)
            if mask_arr.shape[0] != N_full:
                raise ValueError(
                    f"`lineage_mask` must have length {N_full} (tg.n). Got {mask_arr.shape[0]}."
                )
            mask_full = mask_arr
            # For root sampling, use the current `adata` as-is; it should be the subset defined by mask_full.
            # (Length of adata is sum(mask_full) if it was created as adata_full[mask_full].)
            adata_view = adata

        # --- 2) Build the working Psi (subset) and pick the root within that subset ---
        Psi_work = Psi_full[mask_full]               # (N_work, k_use)
        if Psi_work.shape[0] != adata_view.n_obs:
            raise ValueError(
                "The working subset derived from `lineage_mask` does not match the number "
                "of rows in `adata` used for root selection. Ensure your `adata` is either "
                "(a) the full dataset (no subsetting), or (b) the exact subset produced by "
                "the same boolean mask you provided as `lineage_mask`."
            )

        # root within the *subset view*
        root_sub = _sample_from_cluster(
            adata_view, groupby=groupby, group=str(starting_cluster),
            strategy='random', seed=random_state
        )
        if verbose:
            print(f"[pseudotime_analysis] Root selected from {groupby} == {starting_cluster!r}: "
                f"subset-index {root_sub}")

        # --- 3) Pseudotime in the working subset (msDM weighted coordinates) ---
        # squared distances to root in Psi space
        diff = Psi_work - Psi_work[root_sub, :]
        d2 = np.sum(diff * diff, axis=1)
        # normalize to [0,1] within the working subset
        pt_work = (d2 - d2.min()) / (d2.max() - d2.min() + 1e-12)

        # --- 4) Write back to adata.obs['topo_pseudotime'] ---
        if direct_mode:
            # fill a full-length vector with NaN outside the mask (if mask_full not all True)
            pt_full = np.full(N_full, np.nan, dtype=float)
            pt_full[mask_full] = pt_work
            adata.obs['topo_pseudotime'] = pt_full
        else:
            # adata is the subset; lengths match
            adata.obs['topo_pseudotime'] = pt_work

        if verbose:
            print(f"[pseudotime_analysis] Stored pseudotime in adata.obs['topo_pseudotime'] "
                f"(n={adata.n_obs}).")

        return {'root': int(root_sub), 'k_use': int(k_use)}

    def impute_adata(
        adata: AnnData,
        tg: TopOGraph,
        *,
        layer: str = "X",
        raw: bool = False,
        which: str = "msZ",
        impute_t_grid: list[int] | tuple[int, ...] = (1, 2, 4, 8, 16),
        null_K: int = 1000,
        heatmap_top_genes: int = 100,
        seed: int = 13,
    ):
        """
        Automatically detect best diffusion step t via null-based QC and store ONLY
        the best imputation in adata.layers['topo_imputation'].
        Also stores QC info in adata.uns['imputation_qc'].
        """
        # choose source matrix
        if raw:
            Xsrc = adata.raw.to_adata().X
        else:
            Xsrc = adata.X if layer == "X" else adata.layers[layer]

        # helper to get variable genes
        try:
            X_for_var = Xsrc
            if sp.issparse(X_for_var):
                mean = np.asarray(X_for_var.mean(axis=0)).ravel()
                mean_sq = np.asarray(X_for_var.multiply(X_for_var).mean(axis=0)).ravel()
                var = np.maximum(0.0, mean_sq - mean**2)
            else:
                var = np.var(np.asarray(X_for_var), axis=0)
            top_n = int(max(5, min(heatmap_top_genes, adata.n_vars)))
            top_idx = np.argsort(var)[::-1][:top_n]
            top_genes = adata.var_names[top_idx].tolist()
        except Exception:
            # fallback
            top_idx = np.arange(min(heatmap_top_genes, adata.n_vars))
            top_genes = adata.var_names[top_idx].tolist()

        def _dense_subset(X, idx):
            if sp.issparse(X):
                return X[:, idx].toarray()
            return np.asarray(X)[:, idx]

        P = tg.P_of_msZ
        def _diffuse_block(M, t):
            F = M
            for _ in range(int(t)):
                F = P @ F
            return F

        def _corr_genes(M):
            if M.ndim != 2 or M.shape[1] < 2:
                return None
            return np.corrcoef(M, rowvar=False)

        X_top = _dense_subset(Xsrc, top_idx)
        corr_raw = _corr_genes(X_top)

        t_grid = sorted(set(int(t) for t in impute_t_grid))
        stats = []
        corr_by_t = {}
        rng = np.random.default_rng(seed)

        for tval in t_grid:
            X_imp_top = _diffuse_block(X_top, tval)
            C_imp = _corr_genes(X_imp_top)
            corr_by_t[int(tval)] = C_imp
            if C_imp is not None:
                g = C_imp.shape[0]
                mask = ~np.eye(g, dtype=bool)
                S_obs = float(np.mean(np.abs(C_imp[mask])))
            else:
                S_obs = np.nan

            null_vals = []
            for k in range(int(null_K)):
                X_null = X_top.copy()
                for j in range(X_null.shape[1]):
                    rng.shuffle(X_null[:, j])
                X_null_f = _diffuse_block(X_null, tval)
                C_null = _corr_genes(X_null_f)
                if C_null is None:
                    continue
                g = C_null.shape[0]
                mask = ~np.eye(g, dtype=bool)
                null_vals.append(float(np.mean(np.abs(C_null[mask]))))
            null_vals = np.asarray(null_vals, float)
            null_mean = float(np.nanmean(null_vals)) if null_vals.size else np.nan
            null_std  = float(np.nanstd(null_vals)) if null_vals.size else np.nan

            if null_vals.size:
                p_emp = (np.sum(null_vals >= S_obs) + 1.0) / (null_vals.size + 1.0)
            else:
                p_emp = np.nan

            z = (S_obs - null_mean) / (null_std + 1e-9) if np.isfinite(null_mean) and np.isfinite(null_std) and null_std > 0 else np.nan

            stats.append({
                "t": int(tval),
                "score_mean_abs_corr": S_obs,
                "null_mean": null_mean,
                "null_std": null_std,
                "zscore": z,
                "p_empirical": p_emp,
            })

        df_stats = pd.DataFrame(stats).sort_values("t").reset_index(drop=True)
        best_idx = int(np.nanargmin(df_stats["p_empirical"].values)) if df_stats["p_empirical"].notna().any() else 0
        if df_stats["p_empirical"].notna().any():
            min_p = df_stats.loc[best_idx, "p_empirical"]
            cand = df_stats.index[df_stats["p_empirical"] == min_p].tolist()
            if len(cand) > 1:
                best_idx = int(df_stats.loc[cand, "zscore"].idxmax())
        best_t = int(df_stats.loc[best_idx, "t"]) if len(df_stats) else int(t_grid[0])

        # final imputation on the chosen source
        try:
            X_imp_best = tg.impute(Xsrc, t=best_t, which=which)
            adata.layers["topo_imputation"] = X_imp_best  # <- ONLY best result persisted
        except Exception as e:
            print(f"[TopOMetry] Imputation failed at best_t={best_t}: {e}")

        adata.uns["imputation_qc"] = {
            "t_grid": [int(t) for t in t_grid],
            "stats": df_stats,
            "best_t": best_t,
            "heatmap_genes": top_genes,
            "corr_raw": corr_raw,
            "corr_imp_best": corr_by_t.get(best_t, None),
        }


    def geometry_preservation(
        adata: AnnData,
        tg: TopOGraph | None = None,
        *,
        verbose: bool = False,
    ):
        """
        Compute geometry-preservation metrics for ALL 2D/ND representations in adata.obsm.
        Stores a formatted table in adata.uns['geometry_metrics_table'].
        """
        Px = _first_non_none(
            adata.obsp.get('topometry_connectivities_ms', None),
            adata.obsp.get('topometry_connectivities_dm', None)
        )
        if Px is None:
            adata.uns["geometry_metrics_table"] = None
            if verbose:
                print("[TopOMetry] Geometry metrics skipped: no connectivities found.")
            return

        # collect all candidate reps
        reps = {}
        for k, Y in adata.obsm.items():
            if Y is None:
                continue
            try:
                arr = np.asarray(Y)
                if arr.ndim == 2 and arr.shape[0] == adata.n_obs and arr.shape[1] >= 2:
                    reps[k] = arr
            except Exception:
                continue
        # ensure scaffolds included if present
        for k in ('X_spectral_scaffold', 'X_ms_spectral_scaffold'):
            if k in adata.obsm:
                reps[k] = adata.obsm[k]

        # Build operators
        Py_ops = {}
        for name, Y in reps.items():
            try:
                Py_ops[name] = _rowstochastic_from_coords(Y, k=int(max(10, min(30, Px.shape[0]-1))))
            except Exception:
                continue

        rows = []
        for name, Py in Py_ops.items():
            try:
                rdc = rank_diffusion_correlation(Px, Py, times=(1,2,4,8), r=64, symmetric_hint=False)
                dknp15 = diffusion_knn_preservation(Px, Py, times=(1,2,4,8), r=64, k=15)
                dknp30 = diffusion_knn_preservation(Px, Py, times=(1,2,4,8), r=64, k=30)
                dknp60 = diffusion_knn_preservation(Px, Py, times=(1,2,4,8), r=64, k=60 if Px.shape[0] > 60 else max(5, Px.shape[0]//20))
                dknp = float(np.nanmean([dknp15, dknp30, dknp60]))
                pf1 = sparse_neighborhood_f1(Px, Py, k=None)
                pjs = rowwise_js_similarity(Px, Py)
                spR2 = spectral_procrustes(Px, Py, times=(1,2,4,8), r=64, symmetric_hint=False)
                comp, parts = topo_preserve_score(Px, Py, times=(1,2,4,8), k_local=30, r=64, symmetric_hint=False)
                rows.append(dict(
                    representation=name,
                    RDC=rdc, DkNP=dknp, PF1=pf1, PJS=pjs, SP=spR2, Composite=comp
                ))
            except Exception:
                continue

        if rows:
            df = pd.DataFrame(rows).set_index("representation").sort_index()
            df_display = df.copy()
            df_display["RDC"] = 0.5 * (df_display["RDC"] + 1.0)
            adata.uns["geometry_metrics_table"] = df_display
            if verbose:
                print(df_display.round(3))
        else:
            adata.uns["geometry_metrics_table"] = None
            if verbose:
                print("[TopOMetry] Geometry metrics produced no rows.")


    # --------------------------------------
    # Core: run full analysis on an AnnData
    # --------------------------------------

    def run_topometry_analysis(
        adata: AnnData,
        *,
        # TopOGraph hyperparameters (passed to fit_adata via **kwargs)
        base_knn: int = 30,
        graph_knn: int = 30,
        n_eigs: int = 100,
        base_metric: str = "cosine",
        graph_metric: str = "euclidean",
        graph_kernel_version: str = "bw_adaptive",
        diff_t: int = 1,
        n_jobs: int = -1,
        verbosity: int = 1,
        random_state: int = 42,
        id_method: str = "mle",
        id_ks: int | list[int] = 50,
        id_min_components: int = 16,
        id_max_components: int = 512,
        id_headroom: float = 0.5,
        # projections
        projections: tuple[str, ...] = ("MAP", "PaCMAP"),
        # clustering
        do_leiden: bool = True,
        leiden_key_base: str = "topo_clusters",
        leiden_resolutions: list[float] | tuple[float, ...] = (0.2, 0.8, 1.2),
        leiden_primary_index: int = 1,
        # spectral selectivity
        spec_weight_mode: str = "lambda_over_one_minus_lambda",
        spec_k_neighbors: int = 30,
        spec_smooth_P: str | None = None,
        spec_smooth_t: int = 0,
        groupby_candidates: list[str] | None = None,
        # riemann diagnostics
        riem_center: str = "median",
        riem_diffusion_t: int = 8,
        riem_diffusion_op: str | None = "X",
        riem_normalize: str = "symmetric",
        riem_clip_percentile: float = 2.0,
        # imputation
        impute_layer: str = "X",
        impute_raw: bool = False,
        impute_which: str = "msZ",
        impute_t_grid: list[int] | tuple[int, ...] = (1, 2, 4, 8, 16),
        impute_null_K: int = 1000,
        impute_heatmap_top_genes: int = 100,
    ):
        """
        Run the full pipeline by composing the new stepwise functions.
        Returns
        -------
        (adata, tg)
        """
        tg = fit_adata(
            adata, tg=None,
            projections=projections,
            do_leiden=do_leiden,
            leiden_key_base=leiden_key_base,
            leiden_resolutions=leiden_resolutions,
            leiden_primary_index=leiden_primary_index,
            base_knn=base_knn,
            graph_knn=graph_knn,
            n_eigs=n_eigs,
            base_metric=base_metric,
            graph_metric=graph_metric,
            graph_kernel_version=graph_kernel_version,
            diff_t=diff_t,
            n_jobs=n_jobs,
            verbosity=verbosity,
            random_state=random_state,
            id_method=id_method,
            id_ks=id_ks,
            id_min_components=id_min_components,
            id_max_components=id_max_components,
            id_headroom=id_headroom,
        )

        intrinsic_dim(
            adata, tg,
            id_methods=("fsa", "mle"),
            id_k_values=None,
            n_jobs=n_jobs,
        )

        spectral_selectivity(
            adata, tg,
            weight_mode=spec_weight_mode,
            k_neighbors=spec_k_neighbors,
            smooth_P=spec_smooth_P,
            smooth_t=spec_smooth_t,
            groupby_candidates=(groupby_candidates or [leiden_key_base, "leiden", "cell_type"]),
        )

        riemann_diagnostics(
            adata, tg,
            center=riem_center,
            diffusion_t=riem_diffusion_t,
            diffusion_op=riem_diffusion_op,
            normalize=riem_normalize,
            clip_percentile=riem_clip_percentile,
        )

        impute_adata(
            adata, tg,
            layer=impute_layer,
            raw=impute_raw,
            which=impute_which,
            impute_t_grid=impute_t_grid,
            null_K=impute_null_K,
            heatmap_top_genes=impute_heatmap_top_genes,
        )

        geometry_preservation(adata, tg, verbose=False)

        return adata, tg



    def run_topometry_analysis(
        adata: AnnData,
        # --- TopOGraph hyperparameters (exposed; sane defaults) ---
        base_knn: int = 30,
        graph_knn: int = 30,
        n_eigs: int = 100,
        base_metric: str = "cosine",
        graph_metric: str = "euclidean",
        graph_kernel_version: str = "bw_adaptive",
        diff_t: int = 1,
        n_jobs: int = -1,
        verbosity: int = 1,
        random_state: int = 42,
        # Automated scaffold sizing
        id_method: str = "mle",
        id_ks: int | list[int] = 50,
        id_min_components: int = 16,
        id_max_components: int = 512,
        id_headroom: float = 0.5,
        # --- Clustering (multi-granularity) ---
        do_leiden: bool = True,
        leiden_key_base: str = "topo_clusters",
        leiden_resolutions: list[float] = (0.2, 0.8, 1.2),
        leiden_primary_index: int = 1,
        # --- Extra categorical variables for plots / filtering ---
        categorical_plot_keys: list[str] | None = None,
        filtering_label_key: str | None = None,
        # --- Spectral selectivity (stored to .obs) ---
        spec_weight_mode: str = "lambda_over_one_minus_lambda",
        spec_k_neighbors: int = 30,
        spec_smooth_P: str | None = None,
        spec_smooth_t: int = 0,
        # --- Riemann diagnostics ---
        riem_center: str = "median",
        riem_diffusion_t: int = 8,
        riem_diffusion_op: str | None = "X",
        riem_normalize: str = "symmetric",
        riem_clip_percentile: float = 2.0,
        # --- Graph filtering demo ---
        filtering_noise_level: float = 0.15,
        filtering_diffusion_t: int = 3,
        filtering_null_t: int = 1,
        filtering_null_K: int = 500,
        # --- Imputation ---
        impute_t: int = 3,
        impute_which: str = "msZ",
        # --- Imputation QC (report page) ---
        impute_t_grid: list[int] | tuple[int, ...] = (1, 2, 4, 8, 16),
        impute_null_K: int = 1000,
        impute_heatmap_top_genes: int = 100,
        # --- Intrinsic dimension page ---
        id_methods: list[str] = ("fsa", "mle"),
        id_k_values: list[int] | None = None,
        # --- NEW: projections to compute/store ---
        projections: tuple[str, ...] = ("MAP", "PaCMAP"),
    ):
        """
        Run the full TopOMetry pipeline on `adata` and cache results into the object.
        Returns
        -------
        tg : TopOGraph
        """
        # 1) Fit TopOGraph
        tg = TopOGraph(
            base_knn=base_knn,
            graph_knn=graph_knn,
            n_eigs=n_eigs,
            base_metric=base_metric,
            graph_metric=graph_metric,
            graph_kernel_version=graph_kernel_version,
            diff_t=diff_t,
            n_jobs=n_jobs,
            verbosity=verbosity,
            random_state=random_state,
            id_method=id_method,
            id_ks=id_ks,
            id_min_components=id_min_components,
            id_max_components=id_max_components,
            id_headroom=id_headroom,
        ).fit(adata.X)

        # 1) Persist core outputs to adata
        # --- spectral scaffolds
        try:
            _safe_set_obsm(adata, 'X_ms_spectral_scaffold', tg.spectral_scaffold(multiscale=True))
        except Exception:
            pass
        try:
            _safe_set_obsm(adata, 'X_spectral_scaffold', tg.spectral_scaffold(multiscale=False))
        except Exception:
            pass

        # --- Projections: iterate requested methods × {DM, msDM}
        proj_fn = getattr(tg, "project", None)
        for multiscale in (False, True):
            which = "msZ" if multiscale else "Z"
            ms_prefix = "X_ms_topo" if multiscale else "X_topo"
            for proj in projections:
                key = f"{ms_prefix}{proj}"
                try:
                    Y = None
                    if callable(proj_fn):
                        Y = proj_fn(projection_method=proj, which=which)
                    else:
                        # Fall back to existing attributes for MAP/PaCMAP if available
                        if proj.upper() == "MAP":
                            Y = getattr(tg, "msMAP" if multiscale else "MAP", None)
                        elif proj.upper() == "PACMAP":
                            _ensure_pacmap(tg)
                            Y = getattr(tg, "msPaCMAP" if multiscale else "PaCMAP", None)
                    _safe_set_obsm(adata, key, Y)
                except Exception:
                    # try best-effort specific fallbacks
                    try:
                        if proj.upper() == "MAP":
                            _safe_set_obsm(adata, key, getattr(tg, "msMAP" if multiscale else "MAP", None))
                        elif proj.upper() == "PACMAP":
                            _ensure_pacmap(tg)
                            _safe_set_obsm(adata, key, getattr(tg, "msPaCMAP" if multiscale else "PaCMAP", None))
                    except Exception:
                        pass

        # Legacy default alias for embeddings
        if adata.obsm.get('X_ms_TopoMAP', None) is not None:
            _safe_set_obsm(adata, 'X_TopoMAP_default', adata.obsm['X_ms_TopoMAP'])
        elif adata.obsm.get('X_TopoMAP', None) is not None:
            _safe_set_obsm(adata, 'X_TopoMAP_default', adata.obsm['X_TopoMAP'])

        # --- Refined Markov operators
        _safe_set_obsp(adata, 'topometry_connectivities_ms', getattr(tg, 'P_of_msZ', None))
        if adata.obsp.get('topometry_connectivities_ms', None) is not None:
            _safe_set_obsp(adata, 'topometry_connectivities', adata.obsp['topometry_connectivities_ms'])
        _safe_set_obsp(adata, 'topometry_connectivities_dm', getattr(tg, 'P_of_Z', None))

        # Clustering over ms connectivities
        if do_leiden and adata.obsp.get('topometry_connectivities_ms', None) is not None:
            res_list = list(leiden_resolutions) if isinstance(leiden_resolutions, (list, tuple)) else [float(leiden_resolutions)]
            for res in res_list:
                key = f"{leiden_key_base}_res{res}"
                try:
                    sc.tl.leiden(
                        adata,
                        adjacency=adata.obsp['topometry_connectivities_ms'],
                        key_added=key,
                        resolution=float(res),
                    )
                except Exception as e:
                    print(f"[TopOMetry] Leiden clustering failed at res={res}: {e}")
            try:
                prim_key = f"{leiden_key_base}_res{res_list[int(leiden_primary_index)]}"
                if prim_key in adata.obs:
                    adata.obs[leiden_key_base] = adata.obs[prim_key].astype("category")
                    if f"{prim_key}_colors" in adata.uns:
                        adata.uns[f"{leiden_key_base}_colors"] = adata.uns[f"{prim_key}_colors"]
            except Exception:
                pass

        # 2) Store ID summaries
        adata.uns['topometry_id_details'] = getattr(tg, "_id_details", None)
        adata.uns['topometry_id_global_mle'] = tg.global_id_mle()
        adata.uns['topometry_id_global_fsa'] = tg.global_id_fsa()
        _loc = tg.local_ids()
        if _loc is not None:
            if isinstance(_loc, dict):
                for name, vec in _loc.items():
                    if vec is not None:
                        adata.obs[f'local_id_{name}'] = np.asarray(vec)
            else:
                adata.obs['local_id'] = np.asarray(_loc)

        # 3) Spectral selectivity (multiscale)
        spec = tg.spectral_selectivity(
            multiscale=True,
            weight_mode=spec_weight_mode,
            k_neighbors=spec_k_neighbors,
            smooth_P=spec_smooth_P,
            smooth_t=spec_smooth_t,
        )
        adata.obs['spectral_EAS']       = spec['EAS']
        adata.obs['spectral_RayScore']  = spec['RayScore']
        adata.obs['spectral_LAC']       = spec['LAC']
        adata.obs['spectral_axis']      = pd.Categorical(spec['axis'].astype(int))
        adata.obs['spectral_axis_sign'] = pd.Categorical(spec['axis_sign'].astype(int))
        adata.obs['spectral_radius']    = spec['radius']

        # Alignment-by-label table
        categorical_plot_keys = list(categorical_plot_keys) if categorical_plot_keys else []
        align_key_candidates = []
        if do_leiden:
            if leiden_key_base in adata.obs:
                align_key_candidates.append(leiden_key_base)
            for res in (leiden_resolutions if isinstance(leiden_resolutions, (list, tuple)) else [leiden_resolutions]):
                rk = f"{leiden_key_base}_res{res}"
                if rk in adata.obs: align_key_candidates.append(rk)
        align_key_candidates += [k for k in [filtering_label_key, "cell_type", "leiden"] if (k and k in adata.obs)]
        align_key = next((k for k in align_key_candidates if k in adata.obs), None)
        if align_key and ('X_ms_spectral_scaffold' in adata.obsm):
            _spectral_alignment_by_label(
                adata, labels_key=align_key,
                scaffold_key='X_ms_spectral_scaffold',
                top_k=3,
                out_key='spectral_alignment_summary',
            )

        # 4) Riemann diagnostics (scalar field only computed once on ms TopoMAP)
        riem = tg.riemann_diagnostics(
            Y=adata.obsm['X_ms_TopoMAP'] if 'X_ms_TopoMAP' in adata.obsm else adata.obsm.get('X_TopoMAP', None),
            L=tg.base_kernel.L,
            center=riem_center,
            diffusion_t=riem_diffusion_t,
            diffusion_op=riem_diffusion_op,
            normalize=riem_normalize,
            clip_percentile=riem_clip_percentile,
            return_limits=True,
            compute_metric=False,
            compute_scalars=True,
        )
        adata.obs['metric_deformation'] = riem['deformation']
        adata.uns['metric_limits']      = riem.get('limits', None)
        # add L to adata.obsp

        # 5) Imputation
        X_imp = tg.impute(adata.X, t=impute_t, which=impute_which)
        adata.layers['topo_imputation'] = X_imp

        # Imputation QC (omitted here for brevity — identical to previous version)
        try:
            X_for_var = adata.X
            if sp.issparse(X_for_var):
                mean = np.asarray(X_for_var.mean(axis=0)).ravel()
                mean_sq = np.asarray(X_for_var.multiply(X_for_var).mean(axis=0)).ravel()
                var = np.maximum(0.0, mean_sq - mean**2)
            else:
                var = np.var(np.asarray(X_for_var), axis=0)
            top_n = int(max(5, min(impute_heatmap_top_genes, adata.n_vars)))
            top_idx = np.argsort(var)[::-1][:top_n]
            top_genes = adata.var_names[top_idx].tolist()

            def _dense_subset(X, idx):
                if sp.issparse(X):
                    return X[:, idx].toarray()
                Xnp = np.asarray(X)
                return Xnp[:, idx]

            P = tg.P_of_msZ
            def _diffuse_block(M, t):
                F = M
                for _ in range(int(t)):
                    F = P @ F
                return F

            def _corr_genes(M):
                if M.ndim != 2 or M.shape[1] < 2:
                    return None
                return np.corrcoef(M, rowvar=False)

            X_top = _dense_subset(adata.X, top_idx)
            corr_raw = _corr_genes(X_top)

            t_grid = sorted(set(list(impute_t_grid) + [int(impute_t)]))
            stats = []
            corr_by_t = {}
            rng = np.random.default_rng(13)

            for tval in t_grid:
                X_imp_top = _diffuse_block(X_top, tval)
                C_imp = _corr_genes(X_imp_top)
                corr_by_t[int(tval)] = C_imp
                if C_imp is not None:
                    g = C_imp.shape[0]
                    mask = ~np.eye(g, dtype=bool)
                    S_obs = float(np.mean(np.abs(C_imp[mask])))
                else:
                    S_obs = np.nan

                null_vals = []
                for k in range(int(impute_null_K)):
                    X_null = X_top.copy()
                    for j in range(X_null.shape[1]):
                        rng.shuffle(X_null[:, j])
                    X_null_f = _diffuse_block(X_null, tval)
                    C_null = _corr_genes(X_null_f)
                    if C_null is None:
                        continue
                    g = C_null.shape[0]
                    mask = ~np.eye(g, dtype=bool)
                    null_vals.append(float(np.mean(np.abs(C_null[mask]))))
                null_vals = np.asarray(null_vals, float)
                null_mean = float(np.nanmean(null_vals)) if null_vals.size else np.nan
                null_std  = float(np.nanstd(null_vals)) if null_vals.size else np.nan

                if null_vals.size:
                    p_emp = (np.sum(null_vals >= S_obs) + 1.0) / (null_vals.size + 1.0)
                else:
                    p_emp = np.nan

                z = (S_obs - null_mean) / (null_std + 1e-9) if np.isfinite(null_mean) and np.isfinite(null_std) and null_std > 0 else np.nan

                stats.append({
                    "t": int(tval),
                    "score_mean_abs_corr": S_obs,
                    "null_mean": null_mean,
                    "null_std": null_std,
                    "zscore": z,
                    "p_empirical": p_emp,
                })

            df_stats = pd.DataFrame(stats).sort_values("t").reset_index(drop=True)
            best_idx = int(np.nanargmin(df_stats["p_empirical"].values)) if df_stats["p_empirical"].notna().any() else 0
            if df_stats["p_empirical"].notna().any():
                min_p = df_stats.loc[best_idx, "p_empirical"]
                cand = df_stats.index[df_stats["p_empirical"] == min_p].tolist()
                if len(cand) > 1:
                    best_idx = int(df_stats.loc[cand, "zscore"].idxmax())

            best_t = int(df_stats.loc[best_idx, "t"]) if len(df_stats) else int(impute_t)

            adata.uns["imputation_qc"] = {
                "t_grid": [int(t) for t in t_grid],
                "stats": df_stats,
                "best_t": best_t,
                "heatmap_genes": top_genes,
                "corr_raw": corr_raw,
                "corr_imp_best": corr_by_t.get(best_t, None),
            }

            try:
                X_imp_best = tg.impute(adata.X, t=best_t, which=impute_which)
                adata.layers["topo_imputation_best"] = X_imp_best
            except Exception:
                pass

        except Exception as e:
            print(f"[TopOMetry] Imputation QC skipped: {e}")

        # Intrinsic dimension (same as before)
        if id_k_values is None:
            id_k_values = list(range(10, 110, 20))
        try:
            id_est = IntrinsicDim(
                methods=list(id_methods),
                k=list(id_k_values),
                backend='hnswlib',
                metric='euclidean',
                n_jobs=n_jobs,
                plot=False
            )
            id_est.fit(adata.X)
            adata.uns['intrinsic_dim_estimator'] = id_est
            k_low  = str(min(id_k_values))
            k_high = str(max(id_k_values))
            if 'fsa' in id_est.local_id and k_low in id_est.local_id['fsa']:
                adata.obs['id_fsa_k'+k_low] = id_est.local_id['fsa'][k_low]
            if 'fsa' in id_est.local_id and k_high in id_est.local_id['fsa']:
                adata.obs['id_fsa_k'+k_high] = id_est.local_id['fsa'][k_high]
            if 'mle' in id_est.local_id and k_low in id_est.local_id['mle']:
                adata.obs['id_mle_k'+k_low] = id_est.local_id['mle'][k_low]
            if 'mle' in id_est.local_id and k_high in id_est.local_id['mle']:
                adata.obs['id_mle_k'+k_high] = id_est.local_id['mle'][k_high]
        except Exception as e:
            print(f"[TopOMetry] IntrinsicDim estimation skipped: {e}")

        # Geometry preservation metrics — dynamic over ALL obsm representations
        Px = _first_non_none(adata.obsp.get('topometry_connectivities_ms', None),
                                adata.obsp.get('topometry_connectivities_dm', None))
        if Px is not None:
            reps = {}
            for k, Y in adata.obsm.items():
                if Y is None:
                    continue
                try:
                    arr = np.asarray(Y)
                    if arr.ndim == 2 and arr.shape[0] == adata.n_obs and arr.shape[1] >= 2:
                        reps[k] = arr
                except Exception:
                    continue
            # Ensure spectral scaffolds are present if available
            for k in ('X_spectral_scaffold', 'X_ms_spectral_scaffold'):
                if k in adata.obsm:
                    reps[k] = adata.obsm[k]

            # Build operators
            Py_ops = {}
            for name, Y in reps.items():
                try:
                    Py_ops[name] = _rowstochastic_from_coords(Y, k=int(max(10, min(30, Px.shape[0]-1))))
                except Exception:
                    continue

            rows = []
            for name, Py in Py_ops.items():
                try:
                    rdc = rank_diffusion_correlation(Px, Py, times=(1,2,4,8), r=64, symmetric_hint=False)
                    dknp15 = diffusion_knn_preservation(Px, Py, times=(1,2,4,8), r=64, k=15)
                    dknp30 = diffusion_knn_preservation(Px, Py, times=(1,2,4,8), r=64, k=30)
                    dknp60 = diffusion_knn_preservation(Px, Py, times=(1,2,4,8), r=64, k=60 if Px.shape[0] > 60 else max(5, Px.shape[0]//20))
                    dknp = float(np.nanmean([dknp15, dknp30, dknp60]))
                    pf1 = sparse_neighborhood_f1(Px, Py, k=None)
                    pjs = rowwise_js_similarity(Px, Py)
                    spR2 = spectral_procrustes(Px, Py, times=(1,2,4,8), r=64, symmetric_hint=False)
                    comp, parts = topo_preserve_score(Px, Py, times=(1,2,4,8), k_local=30, r=64, symmetric_hint=False)
                    rows.append(dict(
                        representation=name,
                        RDC=rdc, DkNP=dknp, PF1=pf1, PJS=pjs, SP=spR2, Composite=comp
                    ))
                except Exception:
                    continue
            if rows:
                df = pd.DataFrame(rows).set_index("representation").sort_index()
                df_display = df.copy()
                df_display["RDC"] = 0.5 * (df_display["RDC"] + 1.0)
                adata.uns["geometry_metrics_table"] = df_display
        else:
            adata.uns["geometry_metrics_table"] = None

        return adata, tg

    # --------------------------------------
    # Report: single multi-page A4 landscape PDF
    # --------------------------------------

    def plot_topometry_report(
        adata: AnnData,
        tg: TopOGraph,
        output_dir: str = "./topometry_report",
        filename: str = "topometry_report.pdf",
        dpi: int = 300,
        a4_landscape_inches: tuple[float, float] = (11.69, 8.27),
        gene_for_imputation: str | None = None,
        labels_key_for_page_titles: str | None = None,
        categorical_plot_keys: list[str] | None = None,
        signal_plot_keys: list[str] | None = None,
        filtering_noise_level: float = 0.15,
        filtering_diffusion_t: int = 3,
        filtering_null_t: int = 1,
        filtering_null_K: int = 500,
    ):
        """
        Build a consolidated multi-page A4-landscape, 300 dpi PDF summarizing the analysis.

        For each "conceptual page", we render two pages:
          * Page X:  DM scaffold (row1=TopoMAP, row2=TopoPaCMAP)
          * Page X+1: msDM scaffold (row1=TopoMAP, row2=TopoPaCMAP)

        Page order:
          1&2) Clustering resolutions (2×3: row1 TopoMAP, row2 TopoPaCMAP)
          3&4) Riemann diagnostics (2×3 + bottom text)
          5&6) Spectral selectivity (2×4, cmap='Reds' for all)
          7)   Eigenspectrum & IDs (2×4; first row uses original decay+hist style; second row = ID maps on TopoMAP)
          8)   Simulated disease-state filtering page (kept as is)
          9)   Pure noise filtering control (kept as is, with text)
         10)   Pseudotime (fixed gradient shading)
         11)   Imputation (overlap fix in QC panel)
         12)   TopOMetry summary
        """
        _ensure_dir(output_dir)
        pdf_path = os.path.join(output_dir, filename)

        # Embedding helper (variant-aware)
        def _embedding(ax, color, basis_name: str, variant: str, title=None, cmap=None, legend_loc=None):
            """
            Draw embedding with 1:1 aspect and hidden ticks.
            basis_name in {'TopoMAP','TopoPaCMAP'} ; variant in {'DM','msDM'}
            """
            if basis_name == 'TopoMAP':
                key = 'X_TopoMAP' if variant == 'DM' else 'X_ms_TopoMAP'
                basis_arg = 'TopoMAP' if variant == 'DM' else 'ms_TopoMAP'
            else:
                key = 'X_TopoPaCMAP' if variant == 'DM' else 'X_ms_TopoPaCMAP'
                basis_arg = 'TopoPaCMAP' if variant == 'DM' else 'ms_TopoPaCMAP'
            if (key not in adata.obsm) or (adata.obsm.get(key) is None):
                ax.axis('off'); ax.text(0.5, 0.5, f"{basis_name} ({variant}) unavailable", ha='center', va='center')
                return
            sc.pl.embedding(
                adata, basis=basis_arg, color=color, cmap=cmap,
                legend_loc=legend_loc, show=False, ax=ax, title=title
            )
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])

        def _coords(basis_name: str, variant: str):
            if basis_name == 'TopoMAP':
                key = 'X_TopoMAP' if variant == 'DM' else 'X_ms_TopoMAP'
            else:
                key = 'X_TopoPaCMAP' if variant == 'DM' else 'X_ms_TopoPaCMAP'
            return adata.obsm.get(key, None)

        # Choose a label key for titles/legends
        lab_key = None
        for k in [labels_key_for_page_titles, 'topo_clusters', 'cell_type', 'leiden']:
            if k and (k in adata.obs):
                lab_key = k
                break

        # Collect resolution keys and sort ascending; keep also topo_clusters if no res-keys
        res_keys = [k for k in adata.obs.columns if k.startswith('topo_clusters_res')]
        pairs = []
        for k in res_keys:
            try:
                r = float(k.split('res',1)[1])
            except Exception:
                r = np.nan
            pairs.append((r, k))
        pairs.sort(key=lambda x: (np.isnan(x[0]), x[0]))
        res_keys = [k for _, k in pairs]
        if len(res_keys) == 0 and 'topo_clusters' in adata.obs:
            res_keys = ['topo_clusters']

        # Which categorical keys qualify
        cat_keys = []
        for k in (categorical_plot_keys or []):
            if k in adata.obs:
                s = adata.obs[k]
                if pd.api.types.is_categorical_dtype(s) or pd.api.types.is_bool_dtype(s) or pd.api.types.is_integer_dtype(s):
                    cat_keys.append(k)

        # Select three representative resolutions (min/median/max)
        def _pick_three(keys):
            if len(keys) <= 3:
                return keys
            # parse numeric
            vals = []
            for k in keys:
                try: vals.append(float(k.split('res',1)[1]))
                except Exception: vals.append(np.nan)
            idx = np.argsort(vals)
            ordered = [keys[i] for i in idx]
            lo = ordered[0]
            hi = ordered[-1]
            mid = ordered[len(ordered)//2]
            out = []
            for k in [lo, mid, hi]:
                if k not in out: out.append(k)
            return out

        # Convenience: variant order (DM first, then msDM)
        variant_order = ['DM', 'msDM']

        with PdfPages(pdf_path) as pdf:

            # ===== PAGES 1 & 2: CLUSTERING (2×3 grid) =====
            show_keys = _pick_three(res_keys)
            ncols = max(1, len(show_keys))
            for variant in variant_order:
                fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
                gs = fig.add_gridspec(2, ncols, left=0.04, right=0.98, top=0.92, bottom=0.12, wspace=0.15, hspace=0.20)
                # Row 1: TopoMAP
                for j, k in enumerate(show_keys):
                    ax = fig.add_subplot(gs[0, j])
                    _embedding(ax, k, basis_name='TopoMAP', variant=variant, title=f"Clustering ({k})", legend_loc='on data')
                # Row 2: TopoPaCMAP
                for j, k in enumerate(show_keys):
                    ax = fig.add_subplot(gs[1, j])
                    _embedding(ax, k, basis_name='TopoPaCMAP', variant=variant, title=f"Clustering ({k})", legend_loc='on data')
                fig.suptitle(f"Clustering across resolutions — {variant}", y=0.98, fontsize=12)
                pdf.savefig(fig, dpi=dpi); plt.close(fig)

            # ===== PAGES 3 & 4: RIEMANN DIAGNOSTICS (2×3 + bottom text) =====
            for variant in variant_order:
                fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
                outer = fig.add_gridspec(2, 1, height_ratios=[4.0, 1.1], left=0.04, right=0.98, top=0.94, bottom=0.10, hspace=0.23)
                top = outer[0].subgridspec(2, 3, wspace=0.18, hspace=0.20)

                L = tg.base_kernel.L
                Y_map = _coords('TopoMAP', variant)
                Y_pac = _coords('TopoPaCMAP', variant)

                # Colors from lab_key (if available)
                colors_series = None
                if lab_key and (lab_key in adata.obs):
                    labels, lut, col = _palette_from_obs(adata, lab_key)
                    if labels is not None:
                        colors_series = pd.Series(col)

                # Row 1: TopoMAP (Localized / Global / Contraction)
                ax = fig.add_subplot(top[0, 0])
                try:
                    plot_riemann_metric_localized(
                        Y_map, L,
                        n_plot=max(100, adata.shape[0]//8),
                        scale_mode="logdet",
                        scale_gain=1.0,
                        alpha=0.01,
                        ax=ax,
                        seed=7,
                        show_points=True,
                        colors=colors_series,
                        point_alpha=0.6,
                        ellipse_alpha=0.35,
                        point_size=6,
                    )
                    ax.set_title('Localized indicatrices (TopoMAP)', fontsize=10); ax.set_aspect('equal')
                except Exception:
                    ax.axis('off'); ax.text(0.5,0.5,"N/A", ha='center', va='center')

                ax = fig.add_subplot(top[0, 1])
                try:
                    _embedding(ax, lab_key if lab_key else 'metric_deformation', 'TopoMAP', variant, title='Global indicatrices (overlay)')
                    deform_vals = adata.obs['metric_deformation'].values
                    (dmin, dmax) = adata.uns.get('metric_limits', (np.nanmin(deform_vals), np.nanmax(deform_vals)))
                    plot_riemann_metric_global(
                        Y_map, L,
                        grid_res=8, k_avg=30,
                        scale_mode="logdet",
                        scale_gain=1.0,
                        alpha=0.4,
                        ax=ax,
                        show_points=False,
                        zorder=3,
                        cmap="coolwarm",
                        vmin=dmin, vmax=dmax,
                        min_sep_factor=1.1,
                        choose_strong_first=True,
                        deformation_vals=deform_vals,
                    )
                except Exception:
                    ax.axis('off'); ax.text(0.5,0.5,"N/A", ha='center', va='center')

                ax = fig.add_subplot(top[0, 2])
                try:
                    deform_vals = adata.obs['metric_deformation'].values
                    (dmin, dmax) = adata.uns.get('metric_limits', (np.nanmin(deform_vals), np.nanmax(deform_vals)))
                    plot_metric_contraction_expansion(
                        Y_map, L,
                        center="median",
                        normalize="symmetric",
                        cmap="coolwarm",
                        s=6,
                        alpha=1,
                        diffusion_t=3,
                        diffusion_op=getattr(tg.base_kernel, "P", None),
                        vmin=dmin, vmax=dmax,
                        show_colorbar=True,
                        ax=ax,
                    )
                    ax.set_title('Local contraction/expansion (TopoMAP)', fontsize=10); ax.set_aspect('equal')
                except Exception:
                    ax.axis('off'); ax.text(0.5,0.5,"N/A", ha='center', va='center')

                # Row 2: TopoPaCMAP (same trio)
                ax = fig.add_subplot(top[1, 0])
                try:
                    plot_riemann_metric_localized(
                        Y_pac, L,
                        n_plot=max(100, adata.shape[0]//8),
                        scale_mode="logdet",
                        scale_gain=1.0,
                        alpha=0.01,
                        ax=ax,
                        seed=7,
                        show_points=True,
                        colors=colors_series,
                        point_alpha=0.6,
                        ellipse_alpha=0.35,
                        point_size=6,
                    )
                    ax.set_title('Localized indicatrices (TopoPaCMAP)', fontsize=10); ax.set_aspect('equal')
                except Exception:
                    ax.axis('off'); ax.text(0.5,0.5,"N/A", ha='center', va='center')

                ax = fig.add_subplot(top[1, 1])
                try:
                    _embedding(ax, lab_key if lab_key else 'metric_deformation', 'TopoPaCMAP', variant, title='Global indicatrices (overlay)')
                    deform_vals = adata.obs['metric_deformation'].values
                    (dmin, dmax) = adata.uns.get('metric_limits', (np.nanmin(deform_vals), np.nanmax(deform_vals)))
                    plot_riemann_metric_global(
                        Y_pac, L,
                        grid_res=8, k_avg=30,
                        scale_mode="logdet",
                        scale_gain=1.0,
                        alpha=0.4,
                        ax=ax,
                        show_points=False,
                        zorder=3,
                        cmap="coolwarm",
                        vmin=dmin, vmax=dmax,
                        min_sep_factor=1.1,
                        choose_strong_first=True,
                        deformation_vals=deform_vals,
                    )
                except Exception:
                    ax.axis('off'); ax.text(0.5,0.5,"N/A", ha='center', va='center')

                ax = fig.add_subplot(top[1, 2])
                try:
                    deform_vals = adata.obs['metric_deformation'].values
                    (dmin, dmax) = adata.uns.get('metric_limits', (np.nanmin(deform_vals), np.nanmax(deform_vals)))
                    plot_metric_contraction_expansion(
                        Y_pac, L,
                        center="median",
                        normalize="symmetric",
                        cmap="coolwarm",
                        s=6,
                        alpha=1,
                        diffusion_t=3,
                        diffusion_op=getattr(tg.base_kernel, "P", None),
                        vmin=dmin, vmax=dmax,
                        show_colorbar=True,
                        ax=ax,
                    )
                    ax.set_title('Local contraction/expansion (TopoPaCMAP)', fontsize=10); ax.set_aspect('equal')
                except Exception:
                    ax.axis('off'); ax.text(0.5,0.5,"N/A", ha='center', va='center')

                # Guidance text
                ax = fig.add_subplot(outer[1, 0]); ax.axis('off')
                guide = (
                    "Reading the Riemann metric panels:\n"
                    "• Localized indicatrices: each ellipse summarizes local anisotropy of the metric; large elongation indicates preferred directions of variation.\n"
                    "• Global indicatrices: grid-averaged ellipses overlaid on the embedding reveal consistent deformation; colors reflect contraction (blue) vs expansion (red).\n"
                    "• Contraction/expansion: points colored by centered log det(G) after mild diffusion smoothing."
                )
                ax.text(0, 1, guide, va='top', fontsize=9)
                fig.suptitle(f"Riemann diagnostics — {variant}", y=0.985, fontsize=12)
                pdf.savefig(fig, dpi=dpi); plt.close(fig)

            # ===== PAGES 5 & 6: SPECTRAL SELECTIVITY (2×4, Reds) =====
            for variant in variant_order:
                fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
                gs = fig.add_gridspec(2, 4, left=0.04, right=0.98, top=0.92, bottom=0.12, wspace=0.25, hspace=0.30)
                # Row 1 TopoMAP
                ax = fig.add_subplot(gs[0, 0]); _embedding(ax, 'spectral_EAS',      'TopoMAP',    variant, title='EAS',      cmap='Reds')
                ax = fig.add_subplot(gs[0, 1]); _embedding(ax, 'spectral_RayScore', 'TopoMAP',    variant, title='RayScore', cmap='Reds')
                ax = fig.add_subplot(gs[0, 2]); _embedding(ax, 'spectral_LAC',      'TopoMAP',    variant, title='LAC',      cmap='Reds')
                ax = fig.add_subplot(gs[0, 3]); _embedding(ax, 'spectral_radius',   'TopoMAP',    variant, title='Radius',   cmap='Reds')
                # Row 2 TopoPaCMAP
                ax = fig.add_subplot(gs[1, 0]); _embedding(ax, 'spectral_EAS',      'TopoPaCMAP', variant, title='EAS',      cmap='Reds')
                ax = fig.add_subplot(gs[1, 1]); _embedding(ax, 'spectral_RayScore', 'TopoPaCMAP', variant, title='RayScore', cmap='Reds')
                ax = fig.add_subplot(gs[1, 2]); _embedding(ax, 'spectral_LAC',      'TopoPaCMAP', variant, title='LAC',      cmap='Reds')
                ax = fig.add_subplot(gs[1, 3]); _embedding(ax, 'spectral_radius',   'TopoPaCMAP', variant, title='Radius',   cmap='Reds')
                fig.suptitle(f"Spectral selectivity — {variant}", y=0.98, fontsize=12)
                pdf.savefig(fig, dpi=dpi); plt.close(fig)

            # ===== PAGE 7: EIGENSPECTRUM & IDs (2×4) =====
            # First row: original decay+first-derivative + FSA/MLE histograms
            # Second row: TopoMAP colored by IDs at low/high k
            fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
            gs = fig.add_gridspec(2, 4, left=0.04, right=0.98, top=0.92, bottom=0.12, wspace=0.28, hspace=0.35)
            # Row 1: spectrum & diff
            ax_curve = fig.add_subplot(gs[0, 0]); ax_diff = fig.add_subplot(gs[0, 1])
            evals_ms = _eigvals_from_tg(tg, variant='msDM')  # show msDM spectrum by default
            _decay_plot_axes_original(ax_curve, ax_diff, evals_ms, title="Eigenspectrum & Eigengap")

            # ID histograms (original style) into two axes
            ax_fsa = fig.add_subplot(gs[0, 2]); ax_mle = fig.add_subplot(gs[0, 3])
            id_est = adata.uns.get('intrinsic_dim_estimator', None)
            _plot_id_histograms_original(ax_fsa, ax_mle, id_est)

            # Row 2: TopoMAPs colored by ID low/high for FSA and MLE (if present)
            def _find_id_keys(prefix: str):
                ks = [c for c in adata.obs.columns if c.startswith(prefix)]
                if not ks: return (None, None)
                nums = []
                for c in ks:
                    try:
                        nums.append(int(c.split('k',1)[1]))
                    except Exception:
                        nums.append(9999)
                order = np.argsort(nums)
                low = ks[order[0]]
                high = ks[order[-1]]
                return (low, high) if low != high else (low, None)

            fsa_low_key, fsa_high_key = _find_id_keys('id_fsa_k')
            mle_low_key, mle_high_key = _find_id_keys('id_mle_k')

            ax = fig.add_subplot(gs[1, 0]); _embedding(ax, fsa_low_key  if fsa_low_key  else 'spectral_radius', 'TopoMAP', 'msDM', title=f'FSA id ({fsa_low_key})')
            ax = fig.add_subplot(gs[1, 1]); _embedding(ax, fsa_high_key if fsa_high_key else 'spectral_radius', 'TopoMAP', 'msDM', title=f'FSA id ({fsa_high_key})')
            ax = fig.add_subplot(gs[1, 2]); _embedding(ax, mle_low_key  if mle_low_key  else 'spectral_radius', 'TopoMAP', 'msDM', title=f'MLE id ({mle_low_key})')
            ax = fig.add_subplot(gs[1, 3]); _embedding(ax, mle_high_key if mle_high_key else 'spectral_radius', 'TopoMAP', 'msDM', title=f'MLE id ({mle_high_key})')

            fig.suptitle("Eigenspectrum and intrinsic dimensionality (msDM)", y=0.98, fontsize=12)
            pdf.savefig(fig, dpi=dpi); plt.close(fig)

            # ===== PAGE 8: SIMULATED DISEASE-STATE FILTERING (kept, but with new title) =====
            # Simulate ~70% positives in three random clusters if no signal provided
            keys_for_signals = list(signal_plot_keys) if signal_plot_keys else []
            if not keys_for_signals:
                rng = np.random.default_rng(7)
                cluster_key = None
                for k in ['topo_clusters'] + [c for c in adata.obs.columns if c.startswith('topo_clusters_res')] + ['leiden', 'cell type']:
                    if k in adata.obs:
                        cluster_key = k
                        break
                if cluster_key is None:
                    adata.obs['_all'] = pd.Categorical(['all'] * adata.n_obs)
                    cluster_key = '_all'
                labels = adata.obs[cluster_key].astype('category')
                cats = list(labels.cat.categories)
                pick = rng.choice(cats, size=min(3, len(cats)), replace=False) if len(cats) else []
                mask = np.zeros(adata.n_obs, dtype=bool)
                for c in pick:
                    idx = np.where(labels.values == c)[0]
                    if idx.size == 0: continue
                    ksel = int(round(0.7 * idx.size))
                    chosen = rng.choice(idx, size=max(1, min(ksel, idx.size)), replace=False)
                    mask[chosen] = True
                sim_key = "_simulated_state_for_example"
                adata.obs[sim_key] = pd.Categorical(np.where(mask, 'diseased', 'healthy'))
                keys_for_signals = [sim_key]
                _cleanup_sim_key_after = sim_key
            else:
                _cleanup_sim_key_after = None

            # Use msDM operator for filtering demo
            P = adata.obsp.get('topometry_connectivities_ms', None)

            for sig_key in keys_for_signals:
                rng = np.random.default_rng(7)
                disease_state = (adata.obs[sig_key].astype(str) == 'diseased').astype(float).values \
                                if pd.api.types.is_categorical_dtype(adata.obs[sig_key]) else np.asarray(adata.obs[sig_key]).astype(float)
                noisy = np.clip(disease_state + filtering_noise_level * rng.standard_normal(adata.n_obs), 0, 1)

                filt_cat = noisy.copy()
                if P is not None:
                    for _ in range(int(max(1, filtering_diffusion_t))):
                        filt_cat = P @ filt_cat

                rand_raw = (rng.random(adata.n_obs) < 0.5).astype(float)
                rand_flt = rand_raw.copy()
                if P is not None:
                    for _ in range(int(max(1, filtering_diffusion_t))):
                        rand_flt = P @ rand_flt

                adata.obs['_gf_cat_raw'] = noisy
                adata.obs['_gf_cat_flt'] = filt_cat
                adata.obs['_gf_rand_raw'] = rand_raw
                adata.obs['_gf_rand_flt'] = rand_flt

                fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
                gs = fig.add_gridspec(2, 3, left=0.04, right=0.98, top=0.95, bottom=0.10, wspace=0.25, hspace=0.35)
                ax = fig.add_subplot(gs[0, 0]); _embedding(ax, sig_key,       'TopoMAP', 'msDM', title='Simulated disease state', legend_loc=None)
                ax = fig.add_subplot(gs[0, 1]); _embedding(ax, '_gf_cat_raw', 'TopoMAP', 'msDM', title='Noisy categorical readout', cmap='coolwarm')
                ax = fig.add_subplot(gs[0, 2]); _embedding(ax, '_gf_cat_flt', 'TopoMAP', 'msDM', title='Graph-filtered readout',   cmap='coolwarm')
                ax = fig.add_subplot(gs[1, 0]); ax.axis('off')
                ax = fig.add_subplot(gs[1, 1]); _embedding(ax, '_gf_rand_raw','TopoMAP', 'msDM', title='Random discrete (Bernoulli)', cmap='coolwarm')
                ax = fig.add_subplot(gs[1, 2]); _embedding(ax, '_gf_rand_flt','TopoMAP', 'msDM', title='Filtered random discrete',   cmap='coolwarm')
                pdf.savefig(fig, dpi=dpi); plt.close(fig)

                # cleanup temp obs fields for this signal (page 8 only)
                for tmp in ['_gf_cat_raw','_gf_cat_flt','_gf_rand_raw','_gf_rand_flt']:
                    if tmp in adata.obs: del adata.obs[tmp]

            if _cleanup_sim_key_after and (_cleanup_sim_key_after in adata.obs):
                del adata.obs[_cleanup_sim_key_after]

            # ===== PAGE 9: PURE NOISE FILTERING CONTROL =====
            # Build null mean/std with msDM operator and power attenuation curve
            P = adata.obsp.get('topometry_connectivities_ms', None)
            evecs, evals = None, None
            try:
                key_e = 'msDM with ' + tg.base_kernel_version
                evecs, evals = tg.EigenbasisDict[key_e].results(return_evals=True)
            except Exception:
                pass

            # Nulls
            if P is not None:
                K = int(max(10, filtering_null_K))
                t_null = int(max(0, filtering_null_t))
                vals = np.empty((K, adata.n_obs), float)
                for s in range(K):
                    r = np.random.default_rng(s)
                    y = r.standard_normal(adata.n_obs)
                    f = y.copy()
                    for _ in range(t_null):
                        f = P @ f
                    vals[s] = f
                adata.obs['_gf_null_mean'] = vals.mean(axis=0)
                adata.obs['_gf_null_std']  = vals.std(axis=0)

            # If we still have the previous simulated filtered signal in obs, recompute quick spec summaries
            spec_raw = np.array([]); spec_flt = np.array([])
            gtv_raw = np.nan; gtv_flt = np.nan; spec_energy_raw = np.nan; spec_energy_flt = np.nan
            try:
                # fabricate a quick raw/filtered pair from a new Bernoulli control to show attenuation
                rng = np.random.default_rng(11)
                raw = (rng.random(adata.n_obs) < 0.5).astype(float)
                flt = raw.copy()
                if P is not None:
                    for _ in range(int(max(1, filtering_diffusion_t))):
                        flt = P @ flt
                if evecs is not None:
                    U = evecs[:, 1:129]
                    cr = U.T @ raw
                    cf = U.T @ flt
                    spec_raw = np.sort(cr**2)[::-1]
                    spec_flt = np.sort(cf**2)[::-1]
                    spec_energy_raw = float(np.sum(cr**2))
                    spec_energy_flt = float(np.sum(cf**2))
                L = tg.base_kernel.L
                def _gtv(L_, f): return float(f.T @ (L_ @ f))
                gtv_raw = _gtv(L, raw)
                gtv_flt = _gtv(L, flt)
            except Exception:
                pass

            fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
            gs = fig.add_gridspec(2, 3, height_ratios=[4.0, 1.4],
                                  left=0.04, right=0.98, top=0.94, bottom=0.10, wspace=0.25, hspace=0.25)
            ax0 = fig.add_subplot(gs[0, 0]); _embedding(ax0, '_gf_null_mean', 'TopoMAP', 'msDM', title='Filtered pure-noise: mean', cmap='coolwarm')
            ax1 = fig.add_subplot(gs[0, 1]); _embedding(ax1, '_gf_null_std',  'TopoMAP', 'msDM', title='Filtered pure-noise: std',  cmap='viridis')
            ax2 = fig.add_subplot(gs[0, 2])
            ax2.plot(np.arange(spec_raw.size), spec_raw, lw=1, label='raw')
            ax2.plot(np.arange(spec_flt.size), spec_flt, lw=1, label='filtered')
            ax2.set_title('Power attenuation after diffusion', fontsize=10)
            ax2.set_xlabel('Mode rank (desc)'); ax2.set_yticklabels([]); ax2.legend(frameon=False, fontsize=8)
            axt = fig.add_subplot(gs[1, :]); axt.axis('off')
            def _fmt(x):
                try: return f"{float(x):.4g}"
                except Exception: return "n/a"
            txt = (
                "Interpretation:\n"
                "• The filtered pure-noise mean/std visualize how diffusion smoothing behaves under a null model; patterns should not align to biological labels.\n"
                "• Graph Total Variation (GTV) quantifies smoothness: lower values after filtering indicate attenuation of high-frequency noise.\n"
                "• The power attenuation curve shows spectral energy shifting towards low-frequency modes after diffusion.\n"
                f"GTV raw: {_fmt(gtv_raw)} | GTV filtered: {_fmt(gtv_flt)} | Δ: {_fmt(gtv_raw - gtv_flt)} | "
                f"Spectral energy (low+mid): raw={_fmt(spec_energy_raw)}, filtered={_fmt(spec_energy_flt)}"
            )
            axt.text(0.0, 1.0, txt, fontsize=9, va='top')
            pdf.savefig(fig, dpi=dpi); plt.close(fig)

            # Cleanup null fields
            for tmp in ['_gf_null_mean','_gf_null_std']:
                if tmp in adata.obs: del adata.obs[tmp]

            # ===== PAGE 10: PSEUDOTIME (gradient shading over cluster hue) =====
            res = _componentwise_pseudotime_colors(
                adata, tg, cluster_key=lab_key if lab_key else 'topo_clusters'
            )
            fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
            ax = fig.add_axes([0.06, 0.10, 0.88, 0.82])
            Yb = adata.obsm.get('X_ms_TopoMAP', None) or adata.obsm.get('X_TopoMAP', None)
            if res is not None and Yb is not None:
                pt_key, pt_color_key, n_comp = res
                cols = adata.obs[pt_color_key].astype(str).values
                ax.scatter(Yb[:, 0], Yb[:, 1], s=6, c=cols, linewidths=0, alpha=0.95)
                ax.set_title(f"Pseudotime within components (n={n_comp})", fontsize=12)
                ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
            else:
                ax.axis('off'); ax.text(0.5, 0.5, "Pseudotime colors unavailable", ha='center', va='center')
            pdf.savefig(fig, dpi=dpi); plt.close(fig)

            # ===== PAGE 11: IMPUTATION (overlap fix on QC axis) =====
            fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
            gs = fig.add_gridspec(2, 3, left=0.04, right=0.98, top=0.88, bottom=0.18, wspace=0.28, hspace=0.38)
            # choose gene
            g = gene_for_imputation or (adata.var_names[0] if len(adata.var_names) else None)

            if g is not None and g in adata.var_names and ('topo_imputation' in adata.layers):
                gi = adata.var_names.get_loc(g)
                X_csr = adata.X.tocsr(copy=False) if sp.issparse(adata.X) else sp.csr_matrix(np.asarray(adata.X))
                X_imp = adata.layers['topo_imputation']
                raw = (X_csr[:, gi].toarray().ravel() if sp.issparse(X_csr) else np.asarray(X_csr[:, gi]).ravel())
                imp = (X_imp[:, gi].toarray().ravel() if sp.issparse(X_imp) else np.asarray(X_imp[:, gi]).ravel())
                adata.obs['_gene_raw'] = raw
                adata.obs['_gene_imputed'] = imp
                ax = fig.add_subplot(gs[0, 0]); _embedding(ax, '_gene_raw',     'TopoMAP', 'msDM', title=f'Raw: {g}',     cmap='Reds')
                ax = fig.add_subplot(gs[0, 1]); _embedding(ax, '_gene_imputed', 'TopoMAP', 'msDM', title=f'Imputed: {g}', cmap='Reds')
            else:
                for slot in [gs[0, 0], gs[0, 1]]:
                    ax = fig.add_subplot(slot); ax.axis('off'); ax.text(0.5, 0.5, "Imputation layer or gene missing", ha='center', va='center')

            # heatmaps
            iqc = adata.uns.get("imputation_qc", {})
            corr_raw = iqc.get("corr_raw", None)
            corr_imp = iqc.get("corr_imp_best", None)
            genes = iqc.get("heatmap_genes", None)

            def _plot_heatmap(ax, C, title):
                ax.set_title(title, fontsize=10)
                if C is None:
                    ax.axis('off'); ax.text(0.5, 0.5, "N/A", ha='center', va='center'); return
                im = ax.imshow(C, vmin=-1, vmax=1, cmap='coolwarm', interpolation='nearest', aspect='auto')
                ax.tick_params(axis='x', labelsize=6)
                ax.tick_params(axis='y', labelsize=6)
                if genes is not None and len(genes) == C.shape[0] and C.shape[0] <= 50:
                    ax.set_xticks(np.arange(len(genes))); ax.set_yticks(np.arange(len(genes)))
                    ax.set_xticklabels(genes, rotation=90); ax.set_yticklabels(genes)
                else:
                    ax.set_xticks([]); ax.set_yticks([])
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                div = make_axes_locatable(ax); cax = div.append_axes("right", size="3%", pad=0.15)
                cb = plt.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=6)

            ax_h1 = fig.add_subplot(gs[1, 0]); _plot_heatmap(ax_h1, corr_raw, "Gene–gene corr (raw)")
            ax_h2 = fig.add_subplot(gs[1, 1]); _plot_heatmap(ax_h2, corr_imp, "Gene–gene corr (imputed @ best t)")

            # QC score vs t – remove y-ticks and shrink label fontsize to avoid overlap
            sub = gs[:, 2].subgridspec(2, 1, hspace=0.10)
            ax = fig.add_subplot(sub[1, 0])
            ax.set_title("Imputation QC score across t", fontsize=10)
            df_stats = iqc.get("stats", None)
            best_t = iqc.get("best_t", None)
            if isinstance(df_stats, pd.DataFrame) and not df_stats.empty:
                xs = df_stats["t"].to_numpy()
                ys = df_stats["score_mean_abs_corr"].to_numpy()
                nul = df_stats["null_mean"].to_numpy()
                ax.plot(xs, ys, marker='o', linewidth=1.2, label='obs |mean|corr|')
                ax.plot(xs, nul, marker='o', linewidth=1.0, linestyle='--', label='null mean')
                if best_t is not None:
                    ax.axvline(float(best_t), color='k', linewidth=0.8, linestyle=':')
                ax.set_xlabel("t (diffusion steps)")
                ax.set_ylabel("mean |corr|", fontsize=9)
                ax.tick_params(axis='y', labelsize=0)  # effectively removes y-ticks labels
                ax.legend(frameon=False, fontsize=8)
            else:
                ax.axis('off'); ax.text(0.5, 0.5, "No QC stats", ha='center', va='center')

            # bottom explanation band
            ax_exp = fig.add_axes([0.06, 0.06, 0.88, 0.06]); ax_exp.axis('off')
            ax_exp.text(0.0, 0.5,
                        "Imputation uses diffusion (P^t) on the TopOMetry graph to denoise expression. "
                        "QC compares mean absolute gene–gene correlations against null (per-gene permutations) across t. "
                        "Best t minimizes the empirical null p-value (ties broken by max z-score).",
                        fontsize=9, va='center')
            pdf.savefig(fig, dpi=dpi); plt.close(fig)

            # Cleanup temp gene maps
            for tmp in ['_gene_raw','_gene_imputed']:
                if tmp in adata.obs: del adata.obs[tmp]

            # ===== PAGE 12: TOPOmetry SUMMARY (text page) =====
            fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
            ax = fig.add_axes([0.06, 0.12, 0.88, 0.80]); ax.axis('off')
            gid_mle = adata.uns.get('topometry_id_global_mle', None)
            gid_fsa = adata.uns.get('topometry_id_global_fsa', None)
            def _fmt_g(v):
                try: return f"{float(v):.3g}"
                except Exception: return "n/a"
            txt = (f"TopOMetry summary\n"
                   f"• Global ID (MLE): {_fmt_g(gid_mle)}\n"
                   f"• Global ID (FSA proxy): {_fmt_g(gid_fsa)}\n"
                   f"• Root index (pseudotime): {adata.uns.get('topo_pseudotime_root', 'n/a')}\n"
                   f"• Base kernel: {tg.base_kernel_version} | Graph kernel: {tg.graph_kernel_version}\n"
                   f"• n_eigs: {tg.n_eigs} | base_knn: {tg.base_knn} | graph_knn: {tg.graph_knn}")
            ax.text(0.0, 1.0, txt, fontsize=11, va='top')
            pdf.savefig(fig, dpi=dpi); plt.close(fig)

        return pdf_path


    # ----------------------------------------------------
    # Convenience wrapper: run + plot in one call
    # ----------------------------------------------------

    def run_and_report(
        adata: AnnData,
        # --- Pass-through analysis knobs (same defaults as run_topometry_analysis) ---
        base_knn: int = 30,
        graph_knn: int = 30,
        n_eigs: int = 100,
        base_metric: str = "cosine",
        graph_metric: str = "euclidean",
        graph_kernel_version: str = "bw_adaptive",
        diff_t: int = 1,
        n_jobs: int = -1,
        verbosity: int = 1,
        random_state: int = 42,
        id_method: str = "mle",
        id_ks: int | list[int] = 50,
        id_min_components: int = 16,
        id_max_components: int = 512,
        id_headroom: float = 0.5,
        # multi-granularity clustering
        do_leiden: bool = True,
        leiden_key_base: str = "topo_clusters",
        leiden_resolutions: list[float] = (0.2, 0.8, 1.2),
        leiden_primary_index: int = 1,
        # categorical keys
        categorical_plot_keys: list[str] | None = None,
        filtering_label_key: str | None = None,
        # spectral selectivity
        spec_weight_mode: str = "lambda_over_one_minus_lambda",
        spec_k_neighbors: int = 30,
        spec_smooth_P: str | None = None,
        spec_smooth_t: int = 0,
        # riemann
        riem_center: str = "median",
        riem_diffusion_t: int = 8,
        riem_diffusion_op: str | None = "X",
        riem_normalize: str = "symmetric",
        riem_clip_percentile: float = 2.0,
        # filtering knobs (used only in report pages)
        filtering_noise_level: float = 0.15,
        filtering_diffusion_t: int = 3,
        filtering_null_t: int = 1,
        filtering_null_K: int = 500,
        # pseudotime
        pseudotime_null_seeds: int = 200,
        pseudotime_multiscale: bool = True,
        pseudotime_k: int = 64,
        # imputation
        impute_t: int = 8,
        impute_which: str = "msZ",
        # ID page
        id_methods: list[str] = ("fsa", "mle"),
        id_k_values: list[int] | None = None,
        # --- Report / I/O ---
        output_dir: str = "./topometry_report",
        filename: str = "topometry_report.pdf",
        dpi: int = 300,
        a4_landscape_inches: tuple[float, float] = (11.69, 8.27),
        gene_for_imputation: str | None = None,
        labels_key_for_page_titles: str | None = None,
        # >>> NEW: graph filtering report controls <<<
        signal_plot_keys: list[str] | None = None,
    ):
        """
        Run the full analysis then emit a multi-page PDF report.

        Returns
        -------
        tg : TopOGraph
        pdf_path : str
        """
        tg = run_topometry_analysis(
            adata,
            base_knn=base_knn,
            graph_knn=graph_knn,
            n_eigs=n_eigs,
            base_metric=base_metric,
            graph_metric=graph_metric,
            graph_kernel_version=graph_kernel_version,
            diff_t=diff_t,
            n_jobs=n_jobs,
            verbosity=verbosity,
            random_state=random_state,
            id_method=id_method,
            id_ks=id_ks,
            id_min_components=id_min_components,
            id_max_components=id_max_components,
            id_headroom=id_headroom,
            do_leiden=do_leiden,
            leiden_key_base=leiden_key_base,
            leiden_resolutions=leiden_resolutions,
            leiden_primary_index=leiden_primary_index,
            categorical_plot_keys=categorical_plot_keys,
            filtering_label_key=filtering_label_key,
            spec_weight_mode=spec_weight_mode,
            spec_k_neighbors=spec_k_neighbors,
            spec_smooth_P=spec_smooth_P,
            spec_smooth_t=spec_smooth_t,
            riem_center=riem_center,
            riem_diffusion_t=riem_diffusion_t,
            riem_diffusion_op=riem_diffusion_op,
            riem_normalize=riem_normalize,
            riem_clip_percentile=riem_clip_percentile,
            filtering_noise_level=filtering_noise_level,
            filtering_diffusion_t=filtering_diffusion_t,
            filtering_null_t=filtering_null_t,
            filtering_null_K=filtering_null_K,
            pseudotime_null_seeds=pseudotime_null_seeds,
            pseudotime_multiscale=pseudotime_multiscale,
            pseudotime_k=pseudotime_k,
            impute_t=impute_t,
            impute_which=impute_which,
            id_methods=id_methods,
            id_k_values=id_k_values,
        )
        # --- right before calling plot_topometry_report: synthesize a temporary signal if needed ---
        # Decide which signals to plot. If none provided, synthesize a demo key:
        _sim_key = None
        _signal_keys_for_report = list(signal_plot_keys) if signal_plot_keys else None

        if not _signal_keys_for_report:
            import numpy as np, pandas as pd
            rng = np.random.default_rng(7)

            # choose a clustering key available in obs
            cluster_key = None
            for k in ['topo_clusters'] + [c for c in adata.obs.columns if c.startswith('topo_clusters_res')] + ['leiden', 'cell type']:
                if k in adata.obs:
                    cluster_key = k
                    break
            if cluster_key is None:
                # fallback: single-trivial cluster
                adata.obs['_all'] = pd.Categorical(['all'] * adata.n_obs)
                cluster_key = '_all'

            labels = adata.obs[cluster_key].astype('category')
            cats = list(labels.cat.categories) if labels.dtype.name == 'category' else np.unique(labels)
            # pick a random cluster
            if len(cats) == 0:
                pick_mask = np.ones(adata.n_obs, dtype=bool)
            else:
                c = rng.choice(cats)
                pick_mask = (labels.astype(str).values == str(c))

            # ~80% positives within the chosen cluster, ~0% elsewhere
            pos = np.zeros(adata.n_obs, dtype=bool)
            idx = np.where(pick_mask)[0]
            if idx.size > 0:
                k = max(1, int(round(0.8 * idx.size)))
                chosen = rng.choice(idx, size=min(k, idx.size), replace=False)
                pos[chosen] = True

            _sim_key = 'simulated_state_for_example'
            # store as categorical for plotting; values are {'enriched','depleted'}-like but not using that string
            adata.obs[_sim_key] = pd.Categorical(np.where(pos, 'positive', 'negative'))
            _signal_keys_for_report = [_sim_key]

        try:
            pdf_path = plot_topometry_report(
                adata, tg,
                output_dir=output_dir,
                filename=filename,
                dpi=dpi,
                a4_landscape_inches=a4_landscape_inches,
                gene_for_imputation=gene_for_imputation,
                labels_key_for_page_titles=labels_key_for_page_titles or (
                    leiden_key_base if (do_leiden and (leiden_key_base in adata.obs)) else "cell_type"
                ),
                categorical_plot_keys=categorical_plot_keys,
                # >>> pass graph-filtering signals & knobs <<<
                signal_plot_keys=_signal_keys_for_report,
                filtering_noise_level=filtering_noise_level,
                filtering_diffusion_t=filtering_diffusion_t,
                filtering_null_t=filtering_null_t,
                filtering_null_K=filtering_null_K,
            )
        finally:
            # Clean up the temporary simulation key so it never persists beyond the report
            if _sim_key is not None and _sim_key in adata.obs:
                del adata.obs[_sim_key]

        pdf_path = plot_topometry_report(
            adata, tg,
            output_dir=output_dir,
            filename=filename,
            dpi=dpi,
            a4_landscape_inches=a4_landscape_inches,
            gene_for_imputation=gene_for_imputation,
            labels_key_for_page_titles=labels_key_for_page_titles or (
                leiden_key_base if (do_leiden and (leiden_key_base in adata.obs)) else "cell_type"
            ),
            categorical_plot_keys=categorical_plot_keys,
        )

        return tg, pdf_path


    # ----------------------------------------------------
    # Convenience wrapper: run + plot in one call
    # ----------------------------------------------------

    def run_and_report(
        adata: AnnData,
        # --- Pass-through analysis knobs (same defaults as run_topometry_analysis) ---
        base_knn: int = 30,
        graph_knn: int = 30,
        n_eigs: int = 100,
        base_metric: str = "cosine",
        graph_metric: str = "euclidean",
        graph_kernel_version: str = "bw_adaptive",
        diff_t: int = 1,
        n_jobs: int = -1,
        verbosity: int = 1,
        random_state: int = 42,
        id_method: str = "fsa",
        id_ks: int | list[int] = 100,
        id_min_components: int = 16,
        id_max_components: int = 512,
        id_headroom: float = 0.1,
        # multi-granularity clustering
        do_leiden: bool = True,
        leiden_key_base: str = "topo_clusters",
        leiden_resolutions: list[float] = (0.2, 0.8, 1.2),
        leiden_primary_index: int = 1,
        # categorical keys
        categorical_plot_keys: list[str] | None = None,
        filtering_label_key: str | None = None,
        # spectral selectivity
        spec_weight_mode: str = "lambda_over_one_minus_lambda",
        spec_k_neighbors: int = 30,
        spec_smooth_P: str | None = None,
        spec_smooth_t: int = 0,
        # riemann
        riem_center: str = "median",
        riem_diffusion_t: int = 8,
        riem_diffusion_op: str | None = "X",
        riem_normalize: str = "symmetric",
        riem_clip_percentile: float = 2.0,
        # filtering knobs (used only in report pages)
        filtering_noise_level: float = 0.15,
        filtering_diffusion_t: int = 3,
        filtering_null_t: int = 1,
        filtering_null_K: int = 500,
        # imputation
        impute_t: int = 8,
        impute_which: str = "msZ",
        # ID page
        id_methods: list[str] = ("fsa", "mle"),
        id_k_values: list[int] | None = None,
        # --- Report / I/O ---
        output_dir: str = "./topometry_report",
        filename: str = "topometry_report.pdf",
        dpi: int = 300,
        a4_landscape_inches: tuple[float, float] = (11.69, 8.27),
        gene_for_imputation: str | None = None,
        labels_key_for_page_titles: str | None = None,
        # Graph filtering report controls
        signal_plot_keys: list[str] | None = None,
        # --- NEW ---
        projections: tuple[str, ...] = ("MAP", "PaCMAP"),
    ):
        """
        Run the full analysis then emit a single consolidated PDF report.
        Returns
        -------
        tg : TopOGraph
        pdf_path : str
        """
        adata, tg = run_topometry_analysis(
            adata,
            base_knn=base_knn,
            graph_knn=graph_knn,
            n_eigs=n_eigs,
            base_metric=base_metric,
            graph_metric=graph_metric,
            graph_kernel_version=graph_kernel_version,
            diff_t=diff_t,
            n_jobs=n_jobs,
            verbosity=verbosity,
            random_state=random_state,
            id_method=id_method,
            id_ks=id_ks,
            id_min_components=id_min_components,
            id_max_components=id_max_components,
            id_headroom=id_headroom,
            do_leiden=do_leiden,
            leiden_key_base=leiden_key_base,
            leiden_resolutions=leiden_resolutions,
            leiden_primary_index=leiden_primary_index,
            categorical_plot_keys=categorical_plot_keys,
            filtering_label_key=filtering_label_key,
            spec_weight_mode=spec_weight_mode,
            spec_k_neighbors=spec_k_neighbors,
            spec_smooth_P=spec_smooth_P,
            spec_smooth_t=spec_smooth_t,
            riem_center=riem_center,
            riem_diffusion_t=riem_diffusion_t,
            riem_diffusion_op=riem_diffusion_op,
            riem_normalize=riem_normalize,
            riem_clip_percentile=riem_clip_percentile,
            filtering_noise_level=filtering_noise_level,
            filtering_diffusion_t=filtering_diffusion_t,
            filtering_null_t=filtering_null_t,
            filtering_null_K=filtering_null_K,
            impute_t=impute_t,
            impute_which=impute_which,
            id_methods=id_methods,
            id_k_values=id_k_values,
            projections=projections,   # <— pass through
        )

        # Ensure PaCMAP computed (redundant if projections contained 'PaCMAP', but safe)
        _ensure_pacmap(tg)

        pdf_path = plot_topometry_report(
            adata, tg,
            output_dir=output_dir,
            filename=filename,
            dpi=dpi,
            a4_landscape_inches=a4_landscape_inches,
            gene_for_imputation=gene_for_imputation,
            labels_key_for_page_titles=labels_key_for_page_titles or (
                leiden_key_base if (do_leiden and (leiden_key_base in adata.obs)) else "cell_type"
            ),
            categorical_plot_keys=categorical_plot_keys,
            signal_plot_keys=signal_plot_keys,
            filtering_noise_level=filtering_noise_level,
            filtering_diffusion_t=filtering_diffusion_t,
            filtering_null_t=filtering_null_t,
            filtering_null_K=filtering_null_K,
        )

        adata.uns["topometry_report_path"] = pdf_path
        return tg, pdf_path







# ----------------------------------------------------
# Old functions kept for backwards compatibility
# ----------------------------------------------------

    def preprocess(AnnData, normalize=True, log=True, target_sum=1e4, min_mean=0.0125, max_mean=8, min_disp=0.3, max_value=10, save_to_raw=True, plot_hvg=False, scale=True, **kwargs):
        """
        A wrapper around Scanpy's preprocessing functions. Normalizes RNA library by size, logarithmizes it and
        selects highly variable genes for subsetting the AnnData object. Automatically subsets the Anndata
        object and saves the full expression matrix to AnnData.raw.


        Parameters
        ----------
        AnnData : the target AnnData object.

        normalize : bool (optional, default True).
            Whether to size-normalize each cell.
        
        log : bool (optional, default True).
            Whether to log-transform for variance stabilization.

        target_sum : int (optional, default 1e4).
            constant for library size normalization.

        min_mean : float (optional, default 0.0125).
            Minimum gene expression level for inclusion as highly-variable gene.

        max_mean : float (optional, default 8.0).
            Maximum gene expression level for inclusion as highly-variable gene.

        min_disp : float (optional, default 0.3).
            Minimum expression dispersion for inclusion as highly-variable gene.

        save_to_raw : bool (optional, default True).
            Whether to save the full expression matrix to AnnData.raw.

        plot_hvg : bool (optional, default False).
            Whether to plot the high-variable genes plot.

        scale : bool (optional, default True).
            Whether to zero-center and scale the data to unit variance.

        max_value : float (optional, default 10.0).
            Maximum value for clipping the data after scaling.

        **kwargs : dict (optional, default {})
            Additional keyword arguments for `sc.pp.highly_variable_genes()`.

        Returns
        -------

        Updated AnnData object.

        """
        if normalize:
            sc.pp.normalize_total(AnnData, target_sum=target_sum)
        if log:
            sc.pp.log1p(AnnData)
        sc.pp.highly_variable_genes(
            AnnData, min_mean=min_mean, max_mean=max_mean, min_disp=min_disp, **kwargs)
        if plot_hvg:
            sc.pl.highly_variable_genes(AnnData)
        if save_to_raw:
            AnnData.raw = AnnData
        AnnData = AnnData[:, AnnData.var.highly_variable]
        if scale:
            sc.pp.scale(AnnData, max_value=max_value)
        return AnnData.copy()

    def default_workflow(AnnData, n_neighbors=15, n_pcs=50, metric='euclidean',
                         resolution=0.8, min_dist=0.5, spread=1, maxiter=600):
        """

        A wrapper around scanpy's default workflow: mean-center and scale the data, perform PCA and use its
        output to construct k-Nearest-Neighbors graphs. These graphs are used for louvain clustering and
        laid out with UMAP. For simplicity, only the main arguments are included; we recommend you tailor
        a simple personalized workflow if you want to optimize hyperparameters.

        Parameters
        ----------

        AnnData: the target AnnData object.

        n_neighbors: int (optional, default 15).
            Number of neighbors for kNN graph construction.

        n_pcs: int (optional, default 50).
            Number of principal components to retain for downstream analysis.

        metric: str (optional, default 'euclidean').
            Metric used for neighborhood graph construction. Common values are 'euclidean' and 'cosine'.

        resolution: float (optional, default 0.8).
            Resolution parameter for the leiden graph community clustering algorithm.

        min_dist: float (optional, default 0.5).
            Key hyperparameter for UMAP embedding. Smaller values lead to more 'concentrated' graphs, however can
            lead to loss of global structure. Recommended values are between 0.3 and 0.8.

        spread: float (optional, default 1.0).
            Key hyperparameter for UMAP embedding. Controls the global spreading of data in the embedidng during
            optimization. Larger values lead to more spread out layouts, but can lead to loss of local structure.
            Ideally, this parameter should vary with `min_dist`.

        maxiter: int (optional, 600).
            Number of maximum iterations for the UMAP embedding optimization.


        Returns
        -------

        Updated AnnData object.

        """
        sc.tl.pca(AnnData, n_comps=n_pcs)
        sc.pp.neighbors(AnnData, n_neighbors=n_neighbors, metric=metric, n_pcs=n_pcs)
        sc.tl.leiden(AnnData, resolution=resolution)
        sc.tl.umap(AnnData, min_dist=min_dist, spread=spread, maxiter=maxiter)
        AnnData.obsm['X_pca_umap'] = AnnData.obsm['X_umap']
        AnnData.obs['pca_leiden'] = AnnData.obs['leiden']
        return AnnData.copy()

    def default_integration_workflow(AnnData,
                                     integration_method=[
                                         'harmony', 'scanorama', 'bbknn'],
                                     batch_key='batch',
                                     n_neighbors=15,
                                     n_pcs=50,
                                     metric='euclidean',
                                     resolution=0.8,
                                     min_dist=0.5,
                                     spread=1,
                                     maxiter=600,
                                     **kwargs):
        """

        A wrapper around scanpy's default integration workflows: harmony, scanorama and bbknn.

        Parameters
        ----------

        AnnData: the target AnnData object.

        integration_method: str( optional, default ['harmony', 'scanorama', 'bbknn', 'scvi']).
            Which integration methods to run. Defaults to all.

        n_neighbors: int (optional, default 15).
            Number of neighbors for kNN graph construction.

        n_pcs: int (optional, default 50).
            Number of principal components to retain for downstream analysis in scanorama.

        metric: str (optional, default 'euclidean').
            Metric used for neighborhood graph construction. Common values are 'euclidean' and 'cosine'.

        resolution: float (optional, default 0.8).
            Resolution parameter for the leiden graph community clustering algorithm.

        min_dist: float (optional, default 0.5).
            Key hyperparameter for UMAP embedding. Smaller values lead to more 'concentrated' graphs, however can
            lead to loss of global structure. Recommended values are between 0.3 and 0.8.

        spread: float (optional, default 1.0).
            Key hyperparameter for UMAP embedding. Controls the global spreading of data in the embedidng during
            optimization. Larger values lead to more spread out layouts, but can lead to loss of local structure.
            Ideally, this parameter should vary with `min_dist`.

        maxiter: int (optional, 600).
            Number of maximum iterations for the UMAP embedding optimization.

        kwargs: additional parameters to be passed for the integration method. To use this option,
            select only one integration method at a time - otherwise, it'll raise several errors.


        Returns
        -------

        Batch-corrected and updated AnnData object.

        """
        # Batch-correct latent representations
        # With harmony

        if 'harmony' in integration_method:
            sce.pp.harmony_integrate(AnnData, key=batch_key, basis='X_pca',
                                     adjusted_basis='X_pca_harmony', **kwargs)
            sc.pp.neighbors(AnnData, use_rep='X_pca_harmony',
                            n_neighbors=n_neighbors, metric=metric)
            sc.tl.leiden(AnnData, key_added='pca_harmony_leiden',
                         resolution=resolution)
            sc.tl.umap(AnnData, min_dist=min_dist,
                       maxiter=maxiter, spread=spread)
            AnnData.obsm['X_pca_harmony_umap'] = AnnData.obsm['X_umap']

        if 'scanorama' in integration_method:
            try:
                import scanorama
            except ImportError:
                return ((print("scanorama is required for using scanorama as batch-correction method."
                               " Please install it with `pip install scanorama`. ")))

            # subset the individual dataset to the same variable genes as in MNN-correct.
            # split per batch into new objects.
            batches = AnnData.obs[batch_key].cat.categories.tolist()
            alldata = {}
            for batch in batches:
                alldata[batch] = AnnData[AnnData.obs[batch_key] == batch, ]

            # convert to list of AnnData objects
            adatas = list(alldata.values())
            # run scanorama.integrate
            scanorama.integrate_scanpy(adatas, dimred=n_pcs, **kwargs)
            # Get all the integrated matrices.
            scanorama_int = [ad.obsm['X_scanorama'] for ad in adatas]
            # make into one matrix.
            all_s = np.concatenate(scanorama_int)
            print(all_s.shape)
            # add to the AnnData object
            AnnData.obsm["X_pca_scanorama"] = all_s
            sc.pp.neighbors(AnnData, use_rep='X_pca_scanorama',
                            n_neighbors=n_neighbors, metric=metric)
            sc.tl.leiden(AnnData, key_added='pca_scanorama_leiden',
                         resolution=resolution)
            sc.tl.umap(AnnData, min_dist=min_dist,
                       maxiter=maxiter, spread=spread)
            AnnData.obsm['X_pca_scanorama_umap'] = AnnData.obsm['X_umap']

        if 'bbknn' in integration_method:
            try:
                import bbknn
            except ImportError:
                return ((print("bbknn is required for using BBKNN as batch-correction method."
                               " Please install it with `pip install bbknn`. ")))
            if 'pca_leiden' not in AnnData.obs.keys():
                sc.pp.neighbors(
                    AnnData, n_neighbors=n_neighbors, metric=metric)
                sc.tl.leiden(AnnData, resolution=resolution,
                             key_added='pca_leiden')
            bbknn.ridge_regression(AnnData, batch_key=batch_key,
                                   confounder_key=['pca_leiden'])
            bbknn.bbknn(AnnData, batch_key=batch_key, use_rep='X_pca',
                        n_pcs=None, **kwargs)
            bbknn_graph = csr_matrix(AnnData.obsp['connectivities'])
            sc.tl.leiden(AnnData, key_added='pca_BBKNN_leiden',
                         adjacency=bbknn_graph, resolution=resolution)
            sc.tl.umap(AnnData, min_dist=min_dist,
                       maxiter=maxiter, spread=spread)
            AnnData.obsm['X_pca_bbknn_umap'] = AnnData.obsm['X_umap']

        # if 'scvi' in integration_method:
        #     try:
        #         import scvi
        #     except ImportError:
        #         return((print("scvi is required for using scvi as batch-correction method."
        #                       " Please install it with `pip install scvi-tools`. ")))
        #     scvi.data.setup_anndata(AnnData, batch_key=batch_key)
        #     vae = scvi.model.SCVI(AnnData, n_layers=5, n_latent=n_pcs, gene_likelihood="nb")
        #     vae.train()
        #     AnnData.obsm["X_scVI"] = vae.get_latent_representation()
        #     sc.pp.neighbors(AnnData, use_rep='X_scvi', n_neighbors=n_neighbors, metric=metric)
        #     sc.tl.leiden(AnnData, key_added='scvi_leiden', resolution=resolution)
        #     sc.tl.umap(AnnData, min_dist=min_dist, maxiter=maxiter, spread=spread)
        #     AnnData.obsm['X_scvi_umap'] = AnnData.obsm['X_umap']

        return AnnData

    def topological_workflow(AnnData, topograph=None,
                             kernels=['fuzzy', 'cknn',
                                      'bw_adaptive'],
                             eigenmap_methods=['DM', 'LE'],
                             projections=['Isomap', 'MAP'],
                             resolution=0.8,
                             X_to_csr=False, **kwargs):
        """

        A wrapper around TopOMetry's topological workflow. Clustering is performed
        with the leiden algorithm on TopOMetry's topological graphs. This wrapper takes an AnnData object containing a
        cell per feature matrix (np.ndarray or scipy.sparse.csr_matrix) and a TopOGraph object. If no TopOGraph object is
        provided, it will generate a new one, *which will not be stored*. The TopOGraph object keeps all analysis results,
        but similarity matrices, dimensional reductions and clustering results are also added to the AnnData object.

        All parameters for the topological analysis must have been added to the TopOGraph object beforehand; otherwise,
        default parameters will be used. Within this wrapper, users only select which kernel and eigendecomposition models to use
        and what graph-layout layout options to use them with. For hyperparameter tuning, the embeddings must be obtained separetely.


        Parameters
        ----------

        AnnData: the target AnnData object.

        topograph: celltometry.TopOGraph (optional).
            The TopOGraph object containing parameters for the topological analysis.

        kernels : list of str (optional, default ['fuzzy', 'cknn', 'bw_adaptive']).
            List of kernel versions to run. These will be used to learn an eigenbasis and to learn a new graph kernel from it.
            Options are:
            * 'fuzzy'
            * 'cknn'
            * 'bw_adaptive'
            * 'bw_adaptive_alpha_decaying'
            * 'bw_adaptive_nbr_expansion'
            * 'bw_adaptive_alpha_decaying_nbr_expansion'
            * 'gaussian'
            Will not run all by default to avoid long waiting times in reckless calls.

        eigenmap_methods : list of str (optional, default ['DM', 'LE', 'top']).
            List of eigenmap methods to run. Options are:
            * 'DM'
            * 'LE'
            * 'top'
            * 'bottom'
        
        projections : list of str (optional, default ['Isomap', 'MAP']).
            List of projection methods to run. Options are the same of the `topo.layouts.Projector()` object:
            * '(L)Isomap'
            * 't-SNE'
            * 'MAP'
            * 'UMAP'
            * 'PaCMAP'
            * 'TriMAP'
            * 'IsomorphicMDE' - MDE with preservation of nearest neighbors
            * 'IsometricMDE' - MDE with preservation of pairwise distances
            * 'NCVis'

        resolution : float (optional, default 0.8).
            Resolution parameter for the leiden graph community clustering algorithm.

        X_to_csr : bool (optional, default False).
            Whether to convert the data matrix in AnnData.X to a csr_matrix format prior to analysis. This is quite
            useful if the data matrix is rather sparse and may significantly speed up computations.

        kwargs : dict (optional)
            Additional parameters to be passed to the sc.tl.leiden() function for clustering.

        Returns
        -------

        Updated AnnData object.

        """

        if topograph is None:
            topograph = TopOGraph()
        if X_to_csr:
            from scipy.sparse import csr_matrix
            data = csr_matrix(AnnData.X)
        else:
            data = AnnData.X
        topograph.run_models(data,
                   kernels=kernels,
                   eigenmap_methods=eigenmap_methods,
                   projections=projections)
        
        # Get results to AnnData
        for base_kernel in kernels:  
            for eigenmap_method in eigenmap_methods:
                if eigenmap_method in ['msDM','DM', 'LE']:
                    basis_key = eigenmap_method + ' with ' + str(base_kernel)
                elif eigenmap_method == 'top':
                    basis_key = 'Top eigenpairs with ' + str(base_kernel)
                elif eigenmap_method == 'bottom':
                    basis_key = 'Bottom eigenpairs with ' + str(base_kernel)
                else:
                    raise ValueError('Unknown eigenmap method.')
                AnnData.obsm['X_' + basis_key] = topograph.EigenbasisDict[basis_key].transform(data) # returns the scaled eigenvectors

                for graph_kernel in kernels:
                    graph_key = graph_kernel + ' from ' + basis_key
                    AnnData.obsp[basis_key + '_distances'] = topograph.eigenbasis_knn_graph
                    AnnData.obsp[graph_key + '_connectivities'] = topograph.GraphKernelDict[graph_key].P
                    sc.tl.leiden(AnnData, adjacency = topograph.GraphKernelDict[graph_key].P, resolution=resolution, key_added = graph_key + '_leiden', **kwargs)
                    for projection in projections:
                        if projection in ['MAP', 'UMAP', 'MDE', 'Isomap']: 
                            suffix_key = graph_key
                        else:
                            suffix_key = basis_key
                        projection_key = projection + ' of ' + suffix_key
                        AnnData.obsm['X_' + projection_key] = topograph.ProjectionDict[projection_key]
        return AnnData



    def explained_variance_by_hvg(AnnData, title='scRNAseq data', n_pcs=200, gene_number_range = [250, 1000, 3000], figsize=(12,6), sup_title_fontsize=20, title_fontsize=16, return_dicts=False):
        """
        Plots the explained variance by PCA with varying number of highly variable genes.

        Parameters
        ----------
        AnnData: the target AnnData object.

        title: str (optional, default 'scRNAseq data').

        n_pcs: int (optional, default 200).
            Number of principal components to use.

        gene_number_range: list of int (optional, default [250, 1000, 3000]).
            List of numbers of highly variable genes to test.

        figsize: tuple of int (optional, default (12,6)).

        sup_title_fontsize: int (optional, default 20).

        title_fontsize: int (optional, default 16).

        return_dicts: bool (optional, default False).
            Whether to return explained covariance ratio and singular values dictionaries.

        Returns
        -------
        A plot. If `return_dicts=True`, also returns a tuple of dictionaries (explained_cov_ratio, singular_values) with the keys
        being strings with the number of genes and the values being the explained covariance ratio
        and the singular values for PCA.
        
        """
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        explained_cov_ratio = {}
        singular_values = {}
        for n_genes in gene_number_range:
            sc.pp.highly_variable_genes(AnnData, n_top_genes=n_genes)
            AnnData.raw = AnnData.copy()
            AnnData = AnnData[:, AnnData.var.highly_variable]
            sc.pp.scale(AnnData, max_value=10)
            pca = PCA(n_components=n_pcs)
            pca.fit(AnnData.X)
            AnnData.obsm['X_pca'] = pca.transform(AnnData.X)
            explained_cov_ratio[n_genes] = pca.explained_variance_ratio_
            singular_values[n_genes] = pca.singular_values_
        plt.figure(figsize=figsize)
        plt.subplots_adjust(left=0.2, right=0.98, bottom=0.001,
                            top=0.9, wspace=0.15, hspace=0.01)
        plt.suptitle(title, fontsize=sup_title_fontsize)
        for j, gene_number in enumerate(gene_number_range):
            plt.subplot(1, 2, 1)
            plt.plot(singular_values[gene_number], label='{} genes'.format(gene_number), color=colors[j])
            plt.title('Eigenspectrum', fontsize=title_fontsize)
            plt.xlabel('Principal component', fontsize=title_fontsize-6)
            plt.ylabel('Singular values', fontsize=title_fontsize-6)
            plt.legend(fontsize=11)
            plt.subplot(1, 2, 2)
            plt.plot(explained_cov_ratio[gene_number].cumsum(), label='{} genes'.format(gene_number), color=colors[j])
            plt.title('Total explained variance', fontsize=title_fontsize)
            plt.xlabel('Principal component', fontsize=13)
            plt.ylabel('Cumulative explained variance', fontsize=title_fontsize-6)
            plt.legend(fontsize=11)
        plt.tight_layout()
        plt.show()
        if return_dicts:
            return explained_cov_ratio, singular_values