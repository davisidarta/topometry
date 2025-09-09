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

    # -----------------------
    # Helpers (internal API)
    # -----------------------

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
        scaffold_key: str = "X_spectral_scaffold_ms",
        standardize: bool = True,
        top_k: int = 3,
        out_key: str = "spectral_alignment_summary",
    ):
        """Per-label: best 1D separating axis (Cohen's d) table (stored in adata.uns[out_key])."""
        def _cohens_d(x, g):
            a, b = x[g], x[~g]
            if a.size < 2 or b.size < 2: return np.nan
            da, db = a.mean(), b.mean()
            va, vb = a.var(ddof=1), b.var(ddof=1)
            n1, n2 = a.size, b.size
            sp = np.sqrt(((n1-1)*va + (n2-1)*vb) / max(1, n1+n2-2))
            if sp <= 0: return np.nan
            return (da - db) / sp

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
        """Return (sorted descending) positive eigenvalues for the chosen diffusion variant."""
        key = (("msDM with " + tg.base_kernel_version) if variant == "msDM"
               else ("DM with " + tg.base_kernel_version))
        try:
            _, evals = tg.EigenbasisDict[key].results(return_evals=True)
        except Exception:
            # Fallback to any available
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
        """Return labels (Categorical), lut and color list aligned to obs order for a categorical obs key."""
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
        """
        Detect connected components on TopOMetry graph and create a brightness-modulated color per cell.
        Pseudotime values are min-max scaled within each component, then mapped to brightness along
        the base hue for the cell's cluster color (from `cluster_key`).
        """
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
            pt_out = tg.pseudotime(multiscale=True, k=64, null_n_seeds=0)
            pt_full = pt_out['pseudotime']
            adata.obs['topo_pseudotime'] = pt_full

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
        """Stacked normalized composition barplot on given axis, no seaborn dependency."""
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
        # legend outside
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            handles = [plt.Rectangle((0,0),1,1,fc='gray', ec='none')]
            labels = [color_key]
        leg = ax.legend(handles, labels, title=color_key, bbox_to_anchor=(1.02,1), loc='upper left', frameon=False, fontsize=8)
        for t in leg.get_texts():
            t.set_fontsize(8)

    def _decay_plot_axes(ax_curve, ax_diff, evals: np.ndarray, title: str | None = None):
        """Draw spectrum decay and abs(first derivative) on provided axes (no plt.show)."""
        if title is not None:
            ax_curve.set_title(title, fontsize=12)
        if evals.size == 0:
            ax_curve.text(0.5,0.5,"No eigenvalues", ha='center', va='center')
            ax_diff.text(0.5,0.5,"No eigenvalues", ha='center', va='center')
            return
        xs = np.arange(0, len(evals))
        ax_curve.plot(xs, evals, 'b', lw=1)
        ax_curve.set_ylabel('Eigenvalues', fontsize=10)
        ax_curve.set_xlabel('Eigenvectors', fontsize=10)
        first_diff = np.diff(evals)
        sec_diff = np.diff(first_diff)
        max_eigs = int(np.sum(evals > 0))
        eigengap = np.argmax(first_diff) + 1
        if max_eigs == len(evals):
            ax_curve.vlines(eigengap, ymin=np.min(evals), ymax=np.max(evals), linestyles='--', lw=0.8)
        else:
            ax_curve.vlines(max_eigs, ymin=np.min(evals), ymax=np.max(evals), linestyles='--', lw=0.8)
        ax_curve.grid(False)

        ax_diff.set_yscale('log')
        ax_diff.scatter(np.arange(0, len(first_diff)), np.abs(first_diff), s=6)
        ax_diff.set_ylabel('abs(first diff)', fontsize=10)
        ax_diff.set_xlabel('Eigenvalues', fontsize=10)
        if max_eigs == len(evals):
            ax_diff.vlines(eigengap, ymin=np.min(np.abs(first_diff[np.isfinite(first_diff)] + 1e-12)), ymax=np.max(np.abs(first_diff)), linestyles='--', lw=0.8)
        else:
            ax_diff.vlines(max_eigs, ymin=np.min(np.abs(first_diff[np.isfinite(first_diff)] + 1e-12)), ymax=np.max(np.abs(first_diff)), linestyles='--', lw=0.8)
        ax_diff.grid(False)

    # --------------------------------------
    # Core: run TopOMetry on an AnnData
    # --------------------------------------

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
        # which of the resolutions to designate as "primary" topo_clusters
        leiden_primary_index: int = 1,
        # --- Extra categorical variables for plots / filtering ---
        categorical_plot_keys: list[str] | None = None,
        filtering_label_key: str | None = None,
        # --- Spectral selectivity (stored to .obs) ---
        spec_weight_mode: str = "lambda_over_one_minus_lambda",
        spec_k_neighbors: int = 30,
        spec_smooth_P: str | None = None,  # {'X','Z','msZ'} or None
        spec_smooth_t: int = 0,
        # --- Riemann diagnostics ---
        riem_center: str = "median",
        riem_diffusion_t: int = 8,
        riem_diffusion_op: str | None = "X",  # {'X','Z','msZ'} or None
        riem_normalize: str = "symmetric",
        riem_clip_percentile: float = 2.0,
        # --- Graph filtering demo (defaults; pages may override) ---
        filtering_noise_level: float = 0.15,
        filtering_diffusion_t: int = 3,
        filtering_null_t: int = 1,
        filtering_null_K: int = 500,
        # --- Pseudotime ---
        pseudotime_null_seeds: int = 200,
        pseudotime_multiscale: bool = True,
        pseudotime_k: int = 64,
        # --- Imputation ---
        impute_t: int = 3,
        impute_which: str = "msZ",
        # --- Imputation QC (report page) ---
        impute_t_grid: list[int] | tuple[int, ...] = (1, 2, 4, 8, 16),
        impute_null_K: int = 1000,
        impute_heatmap_top_genes: int = 100,
        # --- Intrinsic dimension page ---
        id_methods: list[str] = ("fsa", "mle"),
        id_k_values: list[int] | None = None,  # default range(10,110,20)
    ):
        """
        Run the full TopOMetry pipeline on `adata` and cache results into the object.

        Returns
        -------
        tg : TopOGraph
            The fitted TopOGraph instance (also persists outputs into `adata`).
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

        # 2) Persist core outputs to adata
        # Spectral scaffolds
        adata.obsm['X_spectral_scaffold_ms'] = tg.spectral_scaffold(multiscale=True)
        adata.obsm['X_spectral_scaffold_dm'] = tg.spectral_scaffold(multiscale=False)

        # Embeddings (topoMAP)
        adata.obsm['X_topoMAP_ms'] = getattr(tg, 'msMAP', None)
        if adata.obsm['X_topoMAP_ms'] is None and hasattr(tg, 'msMAP'):
            adata.obsm['X_topoMAP_ms'] = tg.msMAP
        # Fixed-t DM MAP (if available)
        try:
            adata.obsm['X_topoMAP_dm'] = tg.MAP
        except Exception:
            adata.obsm['X_topoMAP_dm'] = adata.obsm['X_topoMAP_ms']

        # PaCMAP (multiscale and optional DM)
        try:
            adata.obsm['X_topoPaCMAP_ms'] = tg.msPaCMAP
        except Exception:
            pass
        try:
            adata.obsm['X_topoPaCMAP_dm'] = tg.PaCMAP
        except Exception:
            adata.obsm['X_topoPaCMAP_dm'] = adata.obsm.get('X_topoPaCMAP_ms', None)

        # Refined operators (store for both variants if accessible)
        adata.obsp['topometry_connectivities_ms'] = getattr(tg, 'P_of_msZ', None)
        adata.obsp['topometry_connectivities_dm'] = getattr(tg, 'P_of_Z', getattr(tg, 'P_of_msZ', None))

        # Clustering (optional): run for multiple resolutions on ms connectivities
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

        # Store ID details computed internally by TopOGraph (when available)
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
        if align_key:
            _spectral_alignment_by_label(
                adata, labels_key=align_key,
                scaffold_key='X_spectral_scaffold_ms',
                top_k=3,
                out_key='spectral_alignment_summary',
            )

        # 4) Riemann diagnostics (metric deformation scalar for plots)
        riem = tg.riemann_diagnostics(
            Y=adata.obsm['X_topoMAP_ms'],
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

        # 5) Pseudotime once (global), component-wise scaling & colors later
        pt = tg.pseudotime(
            multiscale=pseudotime_multiscale,
            k=pseudotime_k,
            null_n_seeds=pseudotime_null_seeds,
        )
        adata.obs['topo_pseudotime'] = pt['pseudotime']
        if 'null_mean' in pt: adata.obs['pt_null_mean'] = pt['null_mean']
        if 'null_std' in pt:  adata.obs['pt_null_std']  = pt['null_std']
        adata.uns['topo_pseudotime_root'] = pt['root']

        # 6) Imputation (P^t @ X)
        X_imp = tg.impute(adata.X, t=impute_t, which=impute_which)
        adata.layers['topo_imputation'] = X_imp

        # 7) Imputation QC & summaries
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

        # 8) Intrinsic dimension (explicit estimator for plots)
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
            # store representative vectors (if present) for quick coloring
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

        return tg

    # --------------------------------------
    # Report: multi-page A4 landscape PDF
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
        variant: str = "msDM",  # {"msDM","DM"}
        basis_name: str = "topoMAP",  # {"topoMAP","topoPaCMAP"}
    ):
        """
        Build a multi-page A4-landscape, 300 dpi PDF summarizing the analysis for a given variant + basis.

        Page order (per variant & basis):
          1) Clustering resolutions (condensed to 1–N pages, 1×3 grids in ascending res)
          2) For each categorical key: topoMAP and composition barplot
          3) Riemann metrics page: Localized, Global, Contraction/Expansion (top row), guide text bottom
          4) Spectral selectivity (EAS, RayScore, LAC, Radius) – 1:1 aspect
          5) Eigenspectrum + ID page (2×4): spectrum & first diff; ID hists; topoMAP colored by ID (low/high k)
          6) Graph filtering main page(s): for each signal key (or simulated), ground truth + noisy/filtered + Bernoulli
          7) Graph filtering controls: null mean/std + power attenuation + explanatory text
          8) Pseudotime within components (cluster hue shaded by brightness)
          9) Imputation QC page (maps, heatmaps, score-vs-t + explanatory text)
         10) TopOMetry summary page (text only)
        """
        _ensure_dir(output_dir)
        pdf_path = os.path.join(output_dir, filename)

        # Basis mapping
        if variant not in {"msDM", "DM"}:
            variant = "msDM"
        if basis_name not in {"topoMAP", "topoPaCMAP"}:
            basis_name = "topoMAP"

        if variant == "msDM":
            basis_key = "X_topoMAP_ms" if basis_name == "topoMAP" else "X_topoPaCMAP_ms"
            P_ref = adata.obsp.get('topometry_connectivities_ms', None)
            scaffold_key = 'X_spectral_scaffold_ms'
        else:
            basis_key = "X_topoMAP_dm" if basis_name == "topoMAP" else "X_topoPaCMAP_dm"
            P_ref = adata.obsp.get('topometry_connectivities_dm', None)
            scaffold_key = 'X_spectral_scaffold_dm'

        # Simple wrapper for embeddings enforcing 1:1 aspect
        def _emb(ax, color, title=None, cmap=None, legend_loc=None):
            bk = 'topoMAP' if basis_name == 'topoMAP' else ('msPaCMAP' if basis_name == 'topoPaCMAP' and 'X_topoPaCMAP_ms' in adata.obsm else 'topoMAP')
            try:
                sc.pl.embedding(
                    adata, basis=bk, color=color, show=False, ax=ax,
                    title=title, cmap=cmap, legend_loc=legend_loc
                )
            except Exception:
                ax.axis('off'); ax.text(0.5,0.5,f"{color} N/A", ha='center', va='center')
                return
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])

        # Choose a label key for titles/legends
        lab_key = None
        for k in [labels_key_for_page_titles, 'topo_clusters', 'cell_type', 'leiden']:
            if k and (k in adata.obs):
                lab_key = k
                break
        cat_keys = []
        for k in (categorical_plot_keys or []):
            if k in adata.obs:
                s = adata.obs[k]
                if pd.api.types.is_categorical_dtype(s) or pd.api.types.is_bool_dtype(s) or pd.api.types.is_integer_dtype(s):
                    cat_keys.append(k)

        # Collect resolution keys and sort ascending
        res_keys = [k for k in adata.obs.columns if k.startswith('topo_clusters_res')]
        res_pairs = []
        for k in res_keys:
            try:
                r = float(k.split('res',1)[1])
            except Exception:
                r = np.nan
            res_pairs.append((r, k))
        res_pairs.sort(key=lambda x: (np.isnan(x[0]), x[0]))
        res_keys = [k for _, k in res_pairs]
        if len(res_keys) == 0 and 'topo_clusters' in adata.obs:
            res_keys = ['topo_clusters']

        with PdfPages(pdf_path) as pdf:
            # ===== 1) CLUSTERING RESOLUTION PAGES (1×3 condensed) =====
            if len(res_keys) <= 3:
                show_keys = res_keys
                fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
                gs = fig.add_gridspec(1, len(show_keys), left=0.04, right=0.98, top=0.92, bottom=0.12, wspace=0.15)
                for i, k in enumerate(show_keys):
                    ax = fig.add_subplot(gs[0, i])
                    _emb(ax, k, title=f"Clustering ({k})", legend_loc='on data')
                pdf.savefig(fig, dpi=dpi); plt.close(fig)
            else:
                # slide windows of size 3
                for i in range(0, len(res_keys), 3):
                    show_keys = res_keys[i:i+3]
                    fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
                    gs = fig.add_gridspec(1, len(show_keys), left=0.04, right=0.98, top=0.92, bottom=0.12, wspace=0.15)
                    for j, k in enumerate(show_keys):
                        ax = fig.add_subplot(gs[0, j])
                        _emb(ax, k, title=f"Clustering ({k})", legend_loc='on data')
                    pdf.savefig(fig, dpi=dpi); plt.close(fig)

            # ===== 2) CATEGORICAL KEYS (map + composition barplot) =====
            for k in cat_keys:
                # Page A: embedding colored by key
                fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
                ax = fig.add_axes([0.06, 0.12, 0.88, 0.80])
                _emb(ax, k, title=f"Category: {k}", legend_loc='on data')
                pdf.savefig(fig, dpi=dpi); plt.close(fig)

                # Page B: composition vs primary clustering
                comp_against = 'topo_clusters' if 'topo_clusters' in adata.obs else (res_keys[0] if len(res_keys)>0 else k)
                fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
                ax = fig.add_axes([0.06, 0.12, 0.70, 0.80])
                try:
                    _composition_barplot(ax, adata, column_key=k, color_key=comp_against, invert=False)
                except Exception:
                    ax.text(0.5,0.5,"Composition N/A", ha='center', va='center'); ax.axis('off')
                pdf.savefig(fig, dpi=dpi); plt.close(fig)

            # ===== 3) RIEMANN METRICS (1×3 + text) =====
            fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
            outer = fig.add_gridspec(2, 1, height_ratios=[4.0, 1.3], left=0.04, right=0.98, top=0.94, bottom=0.10, hspace=0.25)
            top = outer[0].subgridspec(1, 3, wspace=0.18)
            L = tg.base_kernel.L
            Y = adata.obsm.get('X_topoMAP_ms', None) if basis_name == 'topoMAP' else adata.obsm.get('X_topoPaCMAP_ms', None)
            if Y is None:
                Y = adata.obsm.get('X_topoMAP_dm', None)

            # colors from lab_key (if available)
            colors_series = None
            if lab_key and (lab_key in adata.obs):
                labels, lut, col = _palette_from_obs(adata, lab_key)
                if labels is not None:
                    colors_series = pd.Series(col)

            ax = fig.add_subplot(top[0, 0])
            try:
                plot_riemann_metric_localized(
                    Y, L,
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
                ax.set_title('Localized indicatrices', fontsize=11); ax.set_aspect('equal')
            except Exception:
                ax.axis('off'); ax.text(0.5,0.5,"Localized metric N/A", ha='center', va='center')

            ax = fig.add_subplot(top[0, 1])
            try:
                _emb(ax, lab_key if lab_key else 'metric_deformation', title='Global indicatrices (overlay)')
                deform_vals = adata.obs['metric_deformation'].values
                (dmin, dmax) = adata.uns.get('metric_limits', (np.nanmin(deform_vals), np.nanmax(deform_vals)))
                plot_riemann_metric_global(
                    Y, L,
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
                ax.axis('off'); ax.text(0.5,0.5,"Global indicatrices N/A", ha='center', va='center')

            ax = fig.add_subplot(top[0, 2])
            try:
                deform_vals = adata.obs['metric_deformation'].values
                (dmin, dmax) = adata.uns.get('metric_limits', (np.nanmin(deform_vals), np.nanmax(deform_vals)))
                plot_metric_contraction_expansion(
                    Y, L,
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
                ax.set_title('Local contraction / expansion', fontsize=11); ax.set_aspect('equal')
            except Exception:
                ax.axis('off'); ax.text(0.5,0.5,"Contraction/Expansion N/A", ha='center', va='center')

            # guidance text bottom
            ax = fig.add_subplot(outer[1, 0]); ax.axis('off')
            guide = (
                "Reading the Riemann metric panels:\n"
                "• Localized indicatrices: each ellipse summarizes local anisotropy of the metric; large elongation indicates preferred directions of variation.\n"
                "• Global indicatrices: grid-averaged ellipses overlaid on the embedding reveal regions of consistent deformation; colors reflect local contraction (blue) vs expansion (red).\n"
                "• Contraction/expansion: points colored by centered log det(G) after mild diffusion smoothing."
            )
            ax.text(0, 1, guide, va='top', fontsize=9)
            pdf.savefig(fig, dpi=dpi); plt.close(fig)

            # ===== 4) SPECTRAL SELECTIVITY =====
            fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
            gs = fig.add_gridspec(2, 2, left=0.04, right=0.98, top=0.92, bottom=0.12, wspace=0.25, hspace=0.30)
            ax = fig.add_subplot(gs[0, 0]); _emb(ax, 'spectral_EAS',       title='Axis selectivity (EAS)', cmap='Reds')
            ax = fig.add_subplot(gs[0, 1]); _emb(ax, 'spectral_RayScore',  title='Ray score',              cmap='Reds')
            ax = fig.add_subplot(gs[1, 0]); _emb(ax, 'spectral_LAC',       title='Local axial coherence',  cmap='Reds')
            ax = fig.add_subplot(gs[1, 1]); _emb(ax, 'spectral_radius',    title='Spectral radius',        cmap='viridis')
            pdf.savefig(fig, dpi=dpi); plt.close(fig)

            # ===== 5) EIGENSPECTRUM & ID (2×4) =====
            fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
            gs = fig.add_gridspec(2, 4, left=0.04, right=0.98, top=0.92, bottom=0.12, wspace=0.28, hspace=0.35)
            # Row 1: spectrum & diff
            ax_curve = fig.add_subplot(gs[0, 0]); ax_diff = fig.add_subplot(gs[0, 1])
            evals = _eigvals_from_tg(tg, variant=variant)
            _decay_plot_axes(ax_curve, ax_diff, evals, title="Eigenspectrum & Eigengap")

            # Row 1: ID histograms
            ax_fsa = fig.add_subplot(gs[0, 2]); ax_mle = fig.add_subplot(gs[0, 3])
            id_est = adata.uns.get('intrinsic_dim_estimator', None)
            def _hist(ax, data, title):
                if data is None:
                    ax.axis('off'); ax.text(0.5,0.5,"N/A", ha='center', va='center'); return
                ax.hist(np.asarray(data).ravel(), bins=30, edgecolor='none')
                ax.set_title(title, fontsize=10)

            try:
                # collect any stored local IDs in obs for low/high k
                fsa_low  = next((adata.obs[c] for c in adata.obs.columns if c.startswith('id_fsa_k')), None)
                mle_low  = next((adata.obs[c] for c in adata.obs.columns if c.startswith('id_mle_k')), None)
                _hist(ax_fsa, fsa_low, "Local intrinsic dim (FSA)")
                _hist(ax_mle, mle_low, "Local intrinsic dim (MLE)")
            except Exception:
                ax_fsa.axis('off'); ax_fsa.text(0.5,0.5,"N/A", ha='center', va='center')
                ax_mle.axis('off'); ax_mle.text(0.5,0.5,"N/A", ha='center', va='center')

            # Row 2: topoMAPs colored by ID low/high for FSA and MLE (if present)
            # Choose lowest and highest k available by key names
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

            ax = fig.add_subplot(gs[1, 0]); _emb(ax, fsa_low_key  if fsa_low_key  else 'spectral_radius', title=f'FSA id ({fsa_low_key})')
            ax = fig.add_subplot(gs[1, 1]); _emb(ax, fsa_high_key if fsa_high_key else 'spectral_radius', title=f'FSA id ({fsa_high_key})')
            ax = fig.add_subplot(gs[1, 2]); _emb(ax, mle_low_key  if mle_low_key  else 'spectral_radius', title=f'MLE id ({mle_low_key})')
            ax = fig.add_subplot(gs[1, 3]); _emb(ax, mle_high_key if mle_high_key else 'spectral_radius', title=f'MLE id ({mle_high_key})')

            pdf.savefig(fig, dpi=dpi); plt.close(fig)

            # ===== 6) GRAPH FILTERING — one page per signal (or simulated) =====
            keys_for_signals = list(signal_plot_keys) if signal_plot_keys else []
            if not keys_for_signals:
                # Simulate ~70% positives in three random clusters
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

            # Precompute graph operator & spectrum bits for diagnostics
            P = P_ref if P_ref is not None else getattr(tg, 'P_of_msZ', None)
            evecs, evals = None, None
            try:
                key_e = 'msDM with ' + tg.base_kernel_version if variant == 'msDM' else 'DM with ' + tg.base_kernel_version
                evecs, evals = tg.EigenbasisDict[key_e].results(return_evals=True)
            except Exception:
                pass

            for sig_key in keys_for_signals:
                # Build raw/noisy/filtered + random Bernoulli
                rng = np.random.default_rng(7)
                disease_state = (adata.obs[sig_key].astype(str) == str(adata.obs[sig_key].cat.categories[0])).astype(float).values \
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

                # Page: 2×3 as specified
                fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
                gs = fig.add_gridspec(2, 3, left=0.04, right=0.98, top=0.95, bottom=0.10, wspace=0.25, hspace=0.35)
                ax = fig.add_subplot(gs[0, 0]); _emb(ax, sig_key, title='Simulated disease state', legend_loc=None)
                ax = fig.add_subplot(gs[0, 1]); _emb(ax, '_gf_cat_raw', title='Noisy categorical readout', cmap='coolwarm')
                ax = fig.add_subplot(gs[0, 2]); _emb(ax, '_gf_cat_flt', title='Graph-filtered readout',   cmap='coolwarm')
                ax = fig.add_subplot(gs[1, 0]); ax.axis('off')
                ax = fig.add_subplot(gs[1, 1]); _emb(ax, '_gf_rand_raw', title='Random discrete (Bernoulli)', cmap='coolwarm')
                ax = fig.add_subplot(gs[1, 2]); _emb(ax, '_gf_rand_flt', title='Filtered random discrete',   cmap='coolwarm')
                pdf.savefig(fig, dpi=dpi); plt.close(fig)

                # Controls: null mean/std + power attenuation
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

                # power attenuation + GTV
                spec_raw = np.array([])
                spec_flt = np.array([])
                gtv_raw = np.nan; gtv_flt = np.nan; spec_energy_raw = np.nan; spec_energy_flt = np.nan
                try:
                    if evecs is not None:
                        U = evecs[:, 1:129]
                        cr = U.T @ np.asarray(adata.obs['_gf_cat_raw'].values, float)
                        cf = U.T @ np.asarray(adata.obs['_gf_cat_flt'].values, float)
                        spec_raw = np.sort(cr**2)[::-1]
                        spec_flt = np.sort(cf**2)[::-1]
                        spec_energy_raw = float(np.sum(cr**2))
                        spec_energy_flt = float(np.sum(cf**2))
                    L = tg.base_kernel.L
                    def _gtv(L_, f): return float(f.T @ (L_ @ f))
                    gtv_raw = _gtv(L, np.asarray(adata.obs['_gf_cat_raw'].values, float))
                    gtv_flt = _gtv(L, np.asarray(adata.obs['_gf_cat_flt'].values, float))
                except Exception:
                    pass

                fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
                gs = fig.add_gridspec(2, 3, height_ratios=[4.0, 1.4],
                                      left=0.04, right=0.98, top=0.94, bottom=0.10, wspace=0.25, hspace=0.25)
                ax0 = fig.add_subplot(gs[0, 0]); _emb(ax0, '_gf_null_mean', title='Filtered pure-noise: mean', cmap='coolwarm')
                ax1 = fig.add_subplot(gs[0, 1]); _emb(ax1, '_gf_null_std',  title='Filtered pure-noise: std',  cmap='viridis')
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

                # cleanup temp obs fields for this signal
                for tmp in ['_gf_cat_raw','_gf_cat_flt','_gf_rand_raw','_gf_rand_flt','_gf_null_mean','_gf_null_std']:
                    if tmp in adata.obs: del adata.obs[tmp]

            if '_simulated_state_for_example' in locals() and _cleanup_sim_key_after:
                if _cleanup_sim_key_after in adata.obs:
                    del adata.obs[_cleanup_sim_key_after]

            # ===== 7) PSEUDOTIME PAGE =====
            res = _componentwise_pseudotime_colors(
                adata, tg, cluster_key=lab_key if lab_key else 'topo_clusters'
            )
            fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
            ax = fig.add_axes([0.06, 0.10, 0.88, 0.82])
            Yb = adata.obsm.get('X_topoMAP_ms', None) if basis_name == 'topoMAP' else adata.obsm.get('X_topoPaCMAP_ms', None)
            if Yb is None:
                Yb = adata.obsm.get('X_topoMAP_dm', None)
            if res is not None:
                pt_key, pt_color_key, n_comp = res
                cols = adata.obs[pt_color_key].astype(str).values
                ax.scatter(Yb[:, 0], Yb[:, 1], s=6, c=cols, linewidths=0, alpha=0.95)
                ax.set_title(f"Pseudotime within components (n={n_comp})", fontsize=12)
                ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
            else:
                ax.axis('off'); ax.text(0.5, 0.5, "Pseudotime colors unavailable", ha='center', va='center')
            pdf.savefig(fig, dpi=dpi); plt.close(fig)

            # ===== 8) IMPUTATION QC PAGE =====
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
                ax = fig.add_subplot(gs[0, 0]); _emb(ax, '_gene_raw',     title=f'Raw: {g}',     cmap='Reds')
                ax = fig.add_subplot(gs[0, 1]); _emb(ax, '_gene_imputed', title=f'Imputed: {g}', cmap='Reds')
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

            # QC score vs t
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
                ax.set_xlabel("t (diffusion steps)"); ax.set_ylabel("mean |corr|")
                ax.tick_params(axis='y', labelsize=8)
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

            # ===== 9) TOPOmetry SUMMARY (text page) =====
            fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
            ax = fig.add_axes([0.06, 0.12, 0.88, 0.80]); ax.axis('off')
            gid_mle = adata.uns.get('topometry_id_global_mle', None)
            gid_fsa = adata.uns.get('topometry_id_global_fsa', None)
            def _fmt_g(v):
                try: return f"{float(v):.3g}"
                except Exception: return "n/a"
            txt = (f"TopOMetry summary ({variant}, {basis_name})\n"
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
        # Graph filtering report controls
        signal_plot_keys: list[str] | None = None,
        # NEW: emit both variants
        report_variants: tuple[str, ...] = ("msDM", "DM"),
    ):
        """
        Run the full analysis then emit one or more PDF reports (msDM and DM variants).

        Returns
        -------
        tg : TopOGraph
        pdf_path : str         # path to the msDM topoMAP report (for backward compatibility)
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

        # Decide which signals to plot; if none, a per-variant simulation will be created inside plot function as needed
        _signal_keys_for_report = list(signal_plot_keys) if signal_plot_keys else None

        # Build base filename without extension
        base, ext = os.path.splitext(filename)
        if ext.lower() != ".pdf":
            ext = ".pdf"

        outputs = {}
        # For each variant, emit two reports (topoMAP & topoPaCMAP)
        for variant in report_variants:
            # topoMAP
            fname_map = f"{base}_{variant}_topoMAP{ext}"
            p1 = plot_topometry_report(
                adata, tg,
                output_dir=output_dir,
                filename=fname_map,
                dpi=dpi,
                a4_landscape_inches=a4_landscape_inches,
                gene_for_imputation=gene_for_imputation,
                labels_key_for_page_titles=labels_key_for_page_titles or (
                    leiden_key_base if (do_leiden and (leiden_key_base in adata.obs)) else "cell_type"
                ),
                categorical_plot_keys=categorical_plot_keys,
                signal_plot_keys=_signal_keys_for_report,
                filtering_noise_level=filtering_noise_level,
                filtering_diffusion_t=filtering_diffusion_t,
                filtering_null_t=filtering_null_t,
                filtering_null_K=filtering_null_K,
                variant=variant,
                basis_name="topoMAP",
            )
            # topoPaCMAP
            fname_pac = f"{base}_{variant}_topoPaCMAP{ext}"
            p2 = plot_topometry_report(
                adata, tg,
                output_dir=output_dir,
                filename=fname_pac,
                dpi=dpi,
                a4_landscape_inches=a4_landscape_inches,
                gene_for_imputation=gene_for_imputation,
                labels_key_for_page_titles=labels_key_for_page_titles or (
                    leiden_key_base if (do_leiden and (leiden_key_base in adata.obs)) else "cell_type"
                ),
                categorical_plot_keys=categorical_plot_keys,
                signal_plot_keys=_signal_keys_for_report,
                filtering_noise_level=filtering_noise_level,
                filtering_diffusion_t=filtering_diffusion_t,
                filtering_null_t=filtering_null_t,
                filtering_null_K=filtering_null_K,
                variant=variant,
                basis_name="topoPaCMAP",
            )
            outputs[variant] = {"topoMAP": p1, "topoPaCMAP": p2}

        # Store paths in adata.uns for convenience
        adata.uns["topometry_report_paths"] = outputs

        # For backward compatibility: return msDM topoMAP path as the second tuple element
        fallback_path = outputs.get("msDM", {}).get("topoMAP", None)
        return tg, (fallback_path if fallback_path is not None else list(outputs.values())[0]["topoMAP"])


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