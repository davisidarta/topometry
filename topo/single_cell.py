# Wrapper functions for single-cell analysis with Scanpy and TopOMetry
#
# All of these functions call scanpy and thus require it for working
# However, I opted not to include it as a hard-dependency as not all users are interested in single-cell analysis
#
from __future__ import annotations
try:
    import scanpy as sc
    _HAVE_SCANPY = True
except ImportError:
    _HAVE_SCANPY = False

# Functions will be defined only if user has scanpy installed.
if _HAVE_SCANPY:
    import os
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import matplotlib.pyplot as plt

    from matplotlib.backends.backend_pdf import PdfPages
    from anndata import AnnData

    from topo.topograph import TopOGraph, save_topograph, load_topograph
    from topo.eval.rmetric import (
        RiemannMetric,
        plot_riemann_metric_localized,
        plot_riemann_metric_global,
        plot_metric_contraction_expansion,
        calculate_deformation,
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
        # --- Graph filtering demo (cluster-boost + random discrete) ---
        filtering_noise_level: float = 0.15,
        filtering_diffusion_t: int = 3,
        filtering_null_t: int = 1,
        filtering_null_K: int = 500,
        boost_cluster_label: str = "0",
        bernoulli_p: float = 0.5,
        # --- Pseudotime ---
        pseudotime_null_seeds: int = 200,
        pseudotime_multiscale: bool = True,
        pseudotime_k: int = 64,
        # --- Imputation ---
        impute_t: int = 8,
        impute_which: str = "msZ",
        # --- Intrinsic dimension page ---
        id_methods: list[str] = ("fsa", "mle"),
        id_k_values: list[int] | None = None,  # default range(10,110,10)
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
        adata.obsm['X_spectral_scaffold_ms'] = tg.spectral_scaffold(multiscale=True)
        adata.obsm['X_spectral_scaffold']    = tg.spectral_scaffold(multiscale=False)
        adata.obsm['X_topoMAP']              = tg.msMAP
        # refined operator on msDM scaffold (default)
        adata.obsp['topometry_connectivities'] = tg.P_of_msZ

        # Clustering (optional): run for multiple resolutions
        if do_leiden:
            res_list = list(leiden_resolutions) if isinstance(leiden_resolutions, (list, tuple)) else [float(leiden_resolutions)]
            for i, res in enumerate(res_list):
                key = f"{leiden_key_base}_res{res}"
                try:
                    sc.tl.leiden(
                        adata,
                        adjacency=adata.obsp['topometry_connectivities'],
                        key_added=key,
                        resolution=float(res),
                    )
                except Exception as e:
                    print(f"[TopOMetry] Leiden clustering failed at res={res}: {e}")
            # choose "primary" topo_clusters alias
            try:
                prim_key = f"{leiden_key_base}_res{res_list[int(leiden_primary_index)]}"
                if prim_key in adata.obs:
                    adata.obs[leiden_key_base] = adata.obs[prim_key].astype("category")
                    # propagate palette if available
                    if f"{prim_key}_colors" in adata.uns:
                        adata.uns[f"{leiden_key_base}_colors"] = adata.uns[f"{prim_key}_colors"]
            except Exception:
                pass

        # Intrinsic dimension details (from TopOGraph automated sizing)
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

        # 3) Spectral selectivity
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

        # Alignment-by-label table (choose topo_clusters if present else provided key)
        # Also prepare list of categorical keys user wants in plots
        categorical_plot_keys = list(categorical_plot_keys) if categorical_plot_keys else []
        align_key_candidates = []
        if do_leiden:
            # prefer the alias "topo_clusters" if we created it
            if leiden_key_base in adata.obs:
                align_key_candidates.append(leiden_key_base)
            # any of the resolution-specific keys
            for res in leiden_resolutions if isinstance(leiden_resolutions, (list, tuple)) else [leiden_resolutions]:
                rk = f"{leiden_key_base}_res{res}"
                if rk in adata.obs:
                    align_key_candidates.append(rk)
        # user-provided globals
        align_key_candidates += [k for k in [filtering_label_key, "cell_type", "leiden"] if (k and k in adata.obs)]
        # final align key (first found)
        align_key = next((k for k in align_key_candidates if k in adata.obs), None)
        if align_key:
            _spectral_alignment_by_label(
                adata, labels_key=align_key,
                scaffold_key='X_spectral_scaffold_ms',
                top_k=3,
                out_key='spectral_alignment_summary',
            )
        # ensure plot keys exist
        categorical_plot_keys = [k for k in categorical_plot_keys if k in adata.obs]

        # 4) Riemann diagnostics (metric + scalars + deformation)
        riem = tg.riemann_diagnostics(
            Y=adata.obsm['X_topoMAP'],
            L=tg.base_kernel.L,
            center=riem_center,
            diffusion_t=riem_diffusion_t,
            diffusion_op=riem_diffusion_op,
            normalize=riem_normalize,
            clip_percentile=riem_clip_percentile,
            return_limits=True,
            compute_metric=True,
            compute_scalars=True,
        )
        adata.obs['metric_anisotropy']  = riem['anisotropy']
        adata.obs['metric_logdetG']     = riem['logdetG']
        adata.obs['metric_deformation'] = riem['deformation']
        adata.uns['metric_limits']      = riem.get('limits', None)

        # 5) Graph filtering demo (cluster-boost + random discrete + nulls)
        # Choose a labels key (for per-cluster enrichment)
        if align_key is None:
            # fallback: make a single category
            adata.obs['_all'] = pd.Categorical(['all']*adata.n_obs)
            align_key = '_all'
        labels_series = adata.obs[align_key].astype('category')
        cats = labels_series.cat.categories
        rng = np.random.default_rng(7)

        # cluster-enriched "disease_state" via per-cluster sampling
        CLUSTER_KEY = str(boost_cluster_label)
        p_base = 0.05
        p_boost = 10.0
        is_enriched = np.zeros(adata.n_obs, dtype=bool)
        for c in cats:
            idx = np.where(labels_series.values == c)[0]
            if idx.size == 0:
                continue
            p = min(1.0, p_base * (p_boost if str(c) == CLUSTER_KEY else 1.0))
            k = max(1, int(round(p * idx.size)))
            pick = rng.choice(idx, size=min(k, idx.size), replace=False)
            is_enriched[pick] = True
        disease_state = is_enriched.astype(float)
        adata.obs['disease_state'] = pd.Categorical(np.where(is_enriched, 'enriched', 'depleted'))

        # noisy categorical readout
        noisy_categorical_signal = disease_state + filtering_noise_level * rng.standard_normal(adata.n_obs)
        noisy_categorical_signal = np.clip(noisy_categorical_signal, 0, 1)

        # discrete random (Bernoulli)
        random_discrete = (rng.random(adata.n_obs) < float(bernoulli_p)).astype(float)
        adata.obs['random_discrete_raw'] = random_discrete

        # Diffusion filtering over refined graph
        P = tg.P_of_msZ
        t_filter = int(max(1, filtering_diffusion_t))

        filt_cat = noisy_categorical_signal.copy()
        for _ in range(t_filter):
            filt_cat = P @ filt_cat
        adata.obs['cat_signal_raw'] = noisy_categorical_signal
        adata.obs['cat_signal_filtered'] = filt_cat

        filt_rand = random_discrete.copy()
        for _ in range(t_filter):
            filt_rand = P @ filt_rand
        adata.obs['random_discrete_filtered'] = filt_rand

        # Null control: filtered pure-noise across seeds (mean/std)
        K = int(max(1, filtering_null_K))
        t_null = int(max(0, filtering_null_t))
        vals = np.empty((K, adata.n_obs), dtype=float)
        for s in range(K):
            r = np.random.default_rng(s)
            y = r.standard_normal(adata.n_obs)
            f = y.copy()
            for _ in range(t_null):
                f = P @ f
            vals[s] = f
        adata.obs['null_mean'] = vals.mean(axis=0)
        adata.obs['null_std']  = vals.std(axis=0)

        # Diagnostics: GTV and spectral attenuation curves
        def _graph_total_variation(L, f): return float(f.T @ (L @ f))
        L_base = tg.base_kernel.L
        adata.uns['gtv_raw'] = _graph_total_variation(L_base, adata.obs['cat_signal_raw'].values)
        adata.uns['gtv_flt'] = _graph_total_variation(L_base, adata.obs['cat_signal_filtered'].values)

        evecs, evals = tg.EigenbasisDict['msDM with ' + tg.base_kernel_version].results(return_evals=True)
        U = evecs[:, 1:129]
        coef_raw = U.T @ adata.obs['cat_signal_raw'].values
        coef_flt = U.T @ adata.obs['cat_signal_filtered'].values
        adata.uns['spec_energy_raw'] = float(np.sum(coef_raw**2))
        adata.uns['spec_energy_flt'] = float(np.sum(coef_flt**2))
        adata.uns['spec_power_raw'] = np.sort(coef_raw**2)[::-1]
        adata.uns['spec_power_flt'] = np.sort(coef_flt**2)[::-1]

        # 6) Pseudotime (+ null summaries)
        pt = tg.pseudotime(
            multiscale=pseudotime_multiscale,
            k=pseudotime_k,
            null_n_seeds=pseudotime_null_seeds,
        )
        adata.obs['topo_pseudotime'] = pt['pseudotime']
        if 'null_mean' in pt:
            adata.obs['pt_null_mean'] = pt['null_mean']
        if 'null_std' in pt:
            adata.obs['pt_null_std']  = pt['null_std']
        adata.uns['topo_pseudotime_root'] = pt['root']

        # 7) Imputation (P^t @ X)
        X_imp = tg.impute(adata.X, t=impute_t, which=impute_which)
        adata.layers['topo_imputation'] = X_imp

        # 8) Intrinsic dimension (explicit estimator for plots)
        if id_k_values is None:
            id_k_values = list(range(10, 110, 10))
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
            adata.uns['intrinsic_dim_estimator'] = id_est  # store for report page

            # store representative k=50 if available
            kshow = '50' if '50' in id_est.local_id.get('fsa', {}) else next(iter(id_est.local_id.get('fsa', {})), None)
            if kshow:
                adata.obs['id_fsa_k'+kshow] = id_est.local_id['fsa'][kshow]
            kshow_mle = '50' if '50' in id_est.local_id.get('mle', {}) else next(iter(id_est.local_id.get('mle', {})), None)
            if kshow_mle:
                adata.obs['id_mle_k'+kshow_mle] = id_est.local_id['mle'][kshow_mle]
        except Exception as e:
            print(f"[TopOMetry] IntrinsicDim estimation skipped: {e}")

        # 9) Try compute PaCMAP (ms) if not already present (optional)
        try:
            _ = tg.msPaCMAP
            adata.obsm['X_msPaCMAP'] = tg.msPaCMAP
        except Exception:
            pass

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
        labels_key_for_page_titles: str | None = None,   # e.g., 'topo_clusters' or 'cell_type'
        # extra plot keys for the "categorical overview" page
        categorical_plot_keys: list[str] | None = None,
    ):
        """
        Build a multi-page A4-landscape, 300 dpi PDF summarizing the analysis.
        """
        _ensure_dir(output_dir)
        pdf_path = os.path.join(output_dir, filename)

        # Choose a label key for titles/legends
        lab_key = None
        for k in [labels_key_for_page_titles, 'topo_clusters', 'cell_type']:
            if k and (k in adata.obs):
                lab_key = k
                break

        cat_keys = [k for k in (categorical_plot_keys or []) if k in adata.obs]

        with PdfPages(pdf_path) as pdf:

            # -------- Page 1: Eigengap + Spectral selectivity panels --------
            fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
            gs = fig.add_gridspec(2, 3, left=0.04, right=0.98, top=0.95, bottom=0.08, wspace=0.3, hspace=0.35)

            # (1a) Eigengap
            ax = fig.add_subplot(gs[0, 0])
            tg.eigenspectrum()  # uses internal plt

            # (1b–1d) Spectral maps
            ax1 = fig.add_subplot(gs[0, 1])
            sc.pl.embedding(adata, basis='topoMAP', color='spectral_EAS', cmap='Reds',
                            title='Axis selectivity (EAS)', show=False, ax=ax1)
            ax2 = fig.add_subplot(gs[0, 2])
            sc.pl.embedding(adata, basis='topoMAP', color='spectral_RayScore', cmap='Reds',
                            title='Ray score', show=False, ax=ax2)
            ax3 = fig.add_subplot(gs[1, 0])
            sc.pl.embedding(adata, basis='topoMAP', color='spectral_LAC', cmap='Reds',
                            title='Local axial coherence', show=False, ax=ax3)

            # (1e) Optional: show labels
            ax4 = fig.add_subplot(gs[1, 1])
            if lab_key:
                sc.pl.embedding(adata, basis='topoMAP', color=lab_key, legend_loc='right margin',
                                title=f'Labels: {lab_key}', show=False, ax=ax4)
            else:
                sc.pl.embedding(adata, basis='topoMAP', color='spectral_radius', cmap='viridis',
                                title='Spectral radius', show=False, ax=ax4)

            # (1f) Alignment table preview (top few rows)
            ax5 = fig.add_subplot(gs[1, 2])
            df = adata.uns.get('spectral_alignment_summary', None)
            ax5.axis('off')
            ax5.set_title('Spectral alignment by label (top axes)', fontsize=10)
            if df is not None and not df.empty:
                txt = df.head(10).to_string(index=False)
            else:
                txt = "No alignment summary available."
            ax5.text(0.0, 1.0, txt, fontsize=8, family='monospace', va='top')

            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)

            # -------- Page 2: Riemannian panels (localized, global, scalars, expansion) --------
            fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
            gs = fig.add_gridspec(2, 2, left=0.04, right=0.98, top=0.95, bottom=0.08, wspace=0.25, hspace=0.35)

            L = tg.base_kernel.L
            Y = adata.obsm['X_topoMAP']
            # colors from labels if available
            colors = None
            if lab_key and (lab_key in adata.obs):
                if f'{lab_key}_colors' in adata.uns:
                    labels = adata.obs[lab_key]
                    palette = adata.uns[f'{lab_key}_colors']
                    cats = labels.cat.categories if labels.dtype.name == 'category' else np.unique(labels)
                    lut = dict(zip(cats, palette))
                    colors = labels.map(lut)

            ax = fig.add_subplot(gs[0, 0])
            plot_riemann_metric_localized(
                Y, L,
                n_plot=adata.shape[0]//10,
                scale_mode="logdet",
                scale_gain=1.0,
                alpha=0.01,
                ax=ax,
                seed=7,
                show_points=True,
                colors=colors,
                point_alpha=0.6,
                ellipse_alpha=0.35,
                point_size=6,
            )
            ax.set_title('Localized indicatrices', fontsize=10)
            ax.set_aspect('equal')

            # Global with shared color scale from earlier deformation
            deform_vals = adata.obs['metric_deformation'].values
            (dmin, dmax) = adata.uns.get('metric_limits', (np.nanmin(deform_vals), np.nanmax(deform_vals)))

            ax = fig.add_subplot(gs[0, 1])
            sc.pl.embedding(
                adata, basis='topoMAP', color=lab_key if lab_key else 'metric_deformation',
                title='Global indicatrices (overlay)', legend_loc=None, show=False, return_fig=False, ax=ax
            )
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
            ax.set_aspect('equal')

            sub = gs[1, 0].subgridspec(1, 2, wspace=0.15)
            ax_a = fig.add_subplot(sub[0, 0])
            sc.pl.embedding(
                adata, basis='topoMAP', color='metric_anisotropy',
                cmap='inferno', show=False, ax=ax_a
            )
            ax_a.set_title('Anisotropy', fontsize=10)

            ax_b = fig.add_subplot(sub[0, 1])
            sc.pl.embedding(
                adata, basis='topoMAP', color='metric_logdetG',
                cmap='inferno', show=False, ax=ax_b
            )
            ax_b.set_title('log det(G)', fontsize=10)

            ax = fig.add_subplot(gs[1, 1])
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
            ax.set_title('Local contraction/expansion', fontsize=10)
            ax.set_aspect('equal')

            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)

            # -------- Page 3: Graph filtering demo (ground truth/noisy/filtered + random + null + diagnostics) --------
            fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
            gs = fig.add_gridspec(2, 3, left=0.04, right=0.98, top=0.95, bottom=0.08, wspace=0.25, hspace=0.35)

            ax = fig.add_subplot(gs[0, 0])
            sc.pl.embedding(adata, basis='topoMAP', color='disease_state', show=False, legend_loc=None,
                            title='Ground truth (categorical, cluster-boosted)', ax=ax)

            ax = fig.add_subplot(gs[0, 1])
            sc.pl.embedding(adata, basis='topoMAP', color='cat_signal_raw', show=False, cmap='coolwarm',
                            title='Noisy categorical readout', ax=ax)

            ax = fig.add_subplot(gs[0, 2])
            sc.pl.embedding(adata, basis='topoMAP', color='cat_signal_filtered', show=False, cmap='coolwarm',
                            title='Graph-filtered readout', ax=ax)

            ax = fig.add_subplot(gs[1, 0])
            sc.pl.embedding(adata, basis='topoMAP', color='random_discrete_raw', show=False, cmap='coolwarm',
                            title='Random discrete (Bernoulli)', ax=ax)

            ax = fig.add_subplot(gs[1, 1])
            sc.pl.embedding(adata, basis='topoMAP', color='random_discrete_filtered', show=False, cmap='coolwarm',
                            title='Filtered random discrete', ax=ax)

            # Diagnostics: GTV and spectral attenuation curves
            ax = fig.add_subplot(gs[1, 2])
            spec_raw = np.asarray(adata.uns.get('spec_power_raw', []))
            spec_flt = np.asarray(adata.uns.get('spec_power_flt', []))
            ax.plot(np.arange(spec_raw.size), spec_raw, lw=1, label='raw')
            ax.plot(np.arange(spec_flt.size), spec_flt, lw=1, label='filtered')
            gtv_raw = adata.uns.get('gtv_raw', np.nan)
            gtv_flt = adata.uns.get('gtv_flt', np.nan)
            ax.set_title(f"Power attenuation | GTV Δ={gtv_raw - gtv_flt:.3g}", fontsize=10)
            ax.set_xlabel("Mode rank (desc)")
            ax.set_ylabel("Sorted spectral power")
            ax.legend(frameon=False)

            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)

            # -------- Page 4: Pseudotime & Imputation --------
            fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
            gs = fig.add_gridspec(2, 3, left=0.04, right=0.98, top=0.95, bottom=0.08, wspace=0.25, hspace=0.35)

            ax = fig.add_subplot(gs[0, 0])
            sc.pl.embedding(adata, basis='topoMAP', color='topo_pseudotime', cmap='viridis',
                            title='Pseudotime', show=False, ax=ax)

            ax = fig.add_subplot(gs[0, 1])
            if 'pt_null_mean' in adata.obs:
                sc.pl.embedding(adata, basis='topoMAP', color='pt_null_mean', cmap='viridis',
                                title='Pseudotime (null mean)', show=False, ax=ax)
            else:
                ax.axis('off'); ax.text(0.5, 0.5, "No null summary", ha='center', va='center')

            ax = fig.add_subplot(gs[0, 2])
            if 'pt_null_std' in adata.obs:
                sc.pl.embedding(adata, basis='topoMAP', color='pt_null_std', cmap='magma',
                                title='Pseudotime (null std)', show=False, ax=ax)
            else:
                ax.axis('off'); ax.text(0.5, 0.5, "No null summary", ha='center', va='center')

            # Imputation: pick a gene (or first available)
            g = gene_for_imputation
            if g is None:
                g = adata.var_names[0] if len(adata.var_names) else None

            if g is not None and g in adata.var_names and ('topo_imputation' in adata.layers):
                gi = adata.var_names.get_loc(g)
                X_csr = adata.X.tocsr(copy=False) if sp.issparse(adata.X) else sp.csr_matrix(np.asarray(adata.X))
                X_imp = adata.layers['topo_imputation']

                raw = (X_csr[:, gi].toarray().ravel() if sp.issparse(X_csr) else np.asarray(X_csr[:, gi]).ravel())
                imp = (X_imp[:, gi].toarray().ravel() if sp.issparse(X_imp) else np.asarray(X_imp[:, gi]).ravel())
                adata.obs['gene_raw'] = raw
                adata.obs['gene_imputed'] = imp

                ax = fig.add_subplot(gs[1, 0])
                sc.pl.embedding(adata, basis='topoMAP', color='gene_raw', cmap='Reds', title=f'Raw: {g}',
                                show=False, ax=ax)

                ax = fig.add_subplot(gs[1, 1])
                sc.pl.embedding(adata, basis='topoMAP', color='gene_imputed', cmap='Reds', title=f'Imputed: {g}',
                                show=False, ax=ax)
            else:
                # placeholders
                for slot in [gs[1, 0], gs[1, 1]]:
                    ax = fig.add_subplot(slot)
                    ax.axis('off'); ax.text(0.5, 0.5, "Imputation layer or gene missing", ha='center', va='center')

            # Summary text
            ax = fig.add_subplot(gs[1, 2])
            ax.axis('off')
            gid_mle_raw = adata.uns.get('topometry_id_global_mle', None)
            gid_fsa_raw = adata.uns.get('topometry_id_global_fsa', None)

            def _fmt_g(v):
                try:
                    return f"{float(v):.3g}"
                except Exception:
                    return "n/a"

            gid_mle = _fmt_g(gid_mle_raw)
            gid_fsa = _fmt_g(gid_fsa_raw)

            txt = (f"TopOMetry summary\n"
                f"• Global ID (MLE): {gid_mle}\n"
                f"• Global ID (FSA proxy): {gid_fsa}\n"
                f"• Root index (pseudotime): {adata.uns.get('topo_pseudotime_root', 'n/a')}\n"
                f"• Base kernel: {tg.base_kernel_version} | Graph kernel: {tg.graph_kernel_version}\n"
                f"• n_eigs: {tg.n_eigs} | base_knn: {tg.base_knn} | graph_knn: {tg.graph_knn}")
            ax.text(0.0, 1.0, txt, fontsize=9, va='top')

            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)

            # -------- Page 5: Intrinsic dimensionality page (embeddings + histograms) --------
            id_est = adata.uns.get('intrinsic_dim_estimator', None)
            if id_est is not None:
                fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
                gs = fig.add_gridspec(2, 2, left=0.04, right=0.98, top=0.95, bottom=0.08, wspace=0.25, hspace=0.35)

                # TopoMAP colored by ID (prefer FSA k=50 if present)
                ax = fig.add_subplot(gs[0, 0])
                id_key = None
                if 'id_fsa_k50' in adata.obs:
                    id_key = 'id_fsa_k50'
                else:
                    # find any fsa key
                    id_key = next((k for k in adata.obs.columns if k.startswith('id_fsa_k')), None)
                if id_key is not None:
                    sc.pl.embedding(adata, basis='topoMAP', color=id_key, cmap='inferno',
                                    title=f'Intrinsic dim (FSA: {id_key})', show=False, ax=ax)
                else:
                    ax.axis('off'); ax.text(0.5, 0.5, "No FSA i.d. vector", ha='center', va='center')

                # msPaCMAP (if available) colored by MLE (k=50 if present)
                ax = fig.add_subplot(gs[0, 1])
                basis = 'msPaCMAP' if 'X_msPaCMAP' in adata.obsm else 'topoMAP'
                id_key_mle = next((k for k in adata.obs.columns if k.startswith('id_mle_k')), None)
                if id_key_mle is not None:
                    sc.pl.embedding(adata, basis=basis, color=id_key_mle, cmap='inferno',
                                    title=f'Intrinsic dim (MLE: {id_key_mle}) on {basis}', show=False, ax=ax)
                else:
                    ax.axis('off'); ax.text(0.5, 0.5, "No MLE i.d. vector", ha='center', va='center')

                # Histogram(s) from IntrinsicDim.plot_id
                # The method makes/uses its own figure; capture and save.
                id_est.plot_id(bins=30, figsize=(6, 8), titlesize=18, labelsize=12, legendsize=9)
                fig_hist = plt.gcf()
                pdf.savefig(fig, dpi=dpi)
                plt.close(fig)
                pdf.savefig(fig_hist, dpi=dpi)
                plt.close(fig_hist)

            # -------- Optional Page 6: Categorical overview (topoMAP) --------
            color_list = []
            if 'topo_clusters' in adata.obs:
                color_list.append('topo_clusters')
            color_list.extend([k for k in cat_keys if k not in color_list])
            if len(color_list) > 0:
                # Use Scanpy's built-in grid when multiple colors; do not pass ax
                try:
                    fig = sc.pl.embedding(
                        adata,
                        basis='topoMAP',
                        color=color_list,
                        legend_loc='on data',
                        legend_fontsize=8,
                        legend_fontoutline=2,
                        show=False,
                        return_fig=True,
                    )
                    pdf.savefig(fig, dpi=dpi)
                    plt.close(fig)
                except Exception as e:
                    print(f"[TopOMetry] Categorical overview page skipped: {e}")

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
        # filtering demo
        filtering_noise_level: float = 0.15,
        filtering_diffusion_t: int = 3,
        filtering_null_t: int = 1,
        filtering_null_K: int = 500,
        boost_cluster_label: str = "0",
        bernoulli_p: float = 0.5,
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
            boost_cluster_label=boost_cluster_label,
            bernoulli_p=bernoulli_p,
            pseudotime_null_seeds=pseudotime_null_seeds,
            pseudotime_multiscale=pseudotime_multiscale,
            pseudotime_k=pseudotime_k,
            impute_t=impute_t,
            impute_which=impute_which,
            id_methods=id_methods,
            id_k_values=id_k_values,
        )

        pdf_path = plot_topometry_report(
            adata, tg,
            output_dir=output_dir,
            filename=filename,
            dpi=dpi,
            a4_landscape_inches=a4_landscape_inches,
            gene_for_imputation=gene_for_imputation,
            labels_key_for_page_titles=labels_key_for_page_titles or (leiden_key_base if (do_leiden and (leiden_key_base in adata.obs)) else "cell_type"),
            categorical_plot_keys=categorical_plot_keys,
        )

        return tg, pdf_path


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