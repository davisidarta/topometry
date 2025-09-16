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

    def _decay_plot_axes_original(ax_curve, ax_diff, evals: np.ndarray, title: str | None = None, title_fontsize: int = 12):
        if evals is None or len(evals) == 0:
            ax_curve.axis('off'); ax_diff.axis('off')
            ax_curve.text(0.5, 0.5, "No eigenvalues", ha='center', va='center')
            return
        max_eigs = int(np.sum(evals > 0, axis=0))
        first_diff = np.diff(evals)
        eigengap = np.argmax(first_diff) + 1

        if title is not None:
            ax_curve.set_title(title, fontsize=title_fontsize)
        ax_curve.plot(range(0, len(evals)), evals, 'b')
        ax_curve.set_ylabel('Eigenvalues', fontsize=title_fontsize-2)
        ax_curve.set_xlabel('Eigenvectors', fontsize=title_fontsize-2)
        if max_eigs == len(evals):
            ax_curve.vlines(eigengap, ax_curve.get_ylim()[0], ax_curve.get_ylim()[1], linestyles="--", label='Eigengap')
            ax_curve.legend(prop={'size': 12}, fontsize=title_fontsize-2, loc='best')
        else:
            ax_curve.vlines(max_eigs, ax_curve.get_ylim()[0], ax_curve.get_ylim()[1], linestyles="--", label='Eigengap')
            ax_curve.legend(prop={'size': 12}, fontsize=title_fontsize-2, loc='best')

        ax_diff.set_yscale('log')
        ax_diff.scatter(range(0, len(first_diff)), np.abs(first_diff), s=8)
        ax_diff.set_ylabel('Eigenvalues first derivatives (abs)', fontsize=title_fontsize-2)
        ax_diff.set_xlabel('Eigenvalues', fontsize=title_fontsize-2)
        ax_diff.tick_params(axis='y', labelleft=False)
        if max_eigs == len(evals):
            ax_diff.vlines(eigengap, ax_diff.get_ylim()[0], ax_diff.get_ylim()[1], linestyles="--", label='Eigengap')
        else:
            ax_diff.vlines(max_eigs, ax_diff.get_ylim()[0], ax_diff.get_ylim()[1], linestyles="--", label='Eigengap')

    def plot_id_histograms(ax_fsa, ax_mle, id_est: IntrinsicDim | None):
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

    def _heatmap(data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
        if ax is None: ax = plt.gca()
        if cbar_kw is None: cbar_kw = {}
        im = ax.imshow(data, **kwargs)
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
        ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
        ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        plt.setp(ax.get_xticklabels(), rotation=-40, ha="right", rotation_mode="anchor")
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
        return im, cbar


    def plot_evaluation_heatmap(df_eval, out_png=None, title="Geometry Preservation Scores"):
        """
        Render a compact heatmap of geometry-preservation scores per representation.

        Expects a DataFrame with a 'representation' column and any subset of
        the score columns {"PF1", "PJS", "SP"} in [0, 1]. Values are shown
        as a colored matrix (rows = representations, columns = metrics) with
        on-cell numeric annotations. Optionally saves the figure to disk.

        Parameters
        ----------
        df_eval : pandas.DataFrame
            Table with at least a 'representation' column and one or more of
            {'PF1','PJS','SP'}.
        out_png : str or None, default None
            If provided, path where the PNG is saved.
        title : str, default "Geometry Preservation Scores"
            Figure title.

        Returns
        -------
        None
            Displays the figure inline (if in a notebook) and optionally saves it.

        Notes
        -----
        - Rows are displayed in the DataFrame order.
        - Text annotation color in each cell flips based on the cell’s value
        relative to half the column’s max to keep labels readable.
        """

        if df_eval is None or len(df_eval) == 0:
            return

        cols = [c for c in ["PF1", "PJS", "SP"] if c in df_eval.columns]
        if not cols:
            return

        df_show = df_eval.copy()
        M = df_show[cols].astype(float).values
        reps = df_show["representation"].astype(str).tolist()

        fig, ax = plt.subplots(
            figsize=(max(6, 0.75*len(cols)+2), max(4, 0.4*len(reps)+2)),
            dpi=150,
            facecolor="white"
        )
        ax.set_facecolor("white")

        # Heatmap
        im, cbar = _heatmap(
            M, reps, cols, ax=ax, cmap="viridis", cbarlabel="Score (0–1)"
        )

        # Remove grid (minor ticks)
        ax.grid(False)
        ax.set_xticks(np.arange(M.shape[1]))
        ax.set_yticks(np.arange(M.shape[0]))

        # Annotate values with reversed text color logic
        def _annotate_heatmap_reversed(im, data=None, valfmt="{x:.3f}", textcolors=("white","black"), threshold=None, **textkw):
            if data is None:
                data = im.get_array()
            if threshold is None:
                threshold = np.nanmax(data) / 2.0
            kw = dict(horizontalalignment="center", verticalalignment="center")
            kw.update(textkw)
            texts = []
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    color_idx = int(data[i, j] > threshold)  # flipped back
                    kw.update(color=textcolors[color_idx])
                    text = im.axes.text(j, i, valfmt.format(x=data[i, j]), **kw)
                    texts.append(text)
            return texts

        _annotate_heatmap_reversed(im, data=M, valfmt="{x:.3f}", fontsize=8)

        ax.set_title(title, fontsize=12, pad=12)
        fig.tight_layout()
        plt.show()
        if out_png is not None:
            fig.savefig(out_png, dpi=150, facecolor="white")
            print(f"[plot] Saved heatmap → {out_png}")
        plt.close(fig)


    # =========================
    # STEPWISE API (new)
    # =========================


    def preprocess(
        AnnData,
        normalize=True,
        log=True,
        target_sum=1e4,
        min_mean=0.0125,
        max_mean=3.0,
        min_disp=0.5,
        max_value=10,
        save_to_raw=True,
        plot_hvg=False,
        scale=True,
        n_top_genes=3000,
        flavor="seurat_v3",
        **kwargs,
    ):
        """
        Scanpy-style legacy preprocessing:
            1) Save raw counts to .layers['counts']
            2) normalize_total (on X)
            3) log1p (on X)
            4) highly_variable_genes
            5) .raw snapshot (optional)
            6) subset to HVGs
            7) create .layers['scaled'] and scale there; set X <- scaled

        Returns
        -------
        AnnData (copy)
        """
        import numpy as np
        import scipy.sparse as sp
        import scanpy as sc

        adata = AnnData

        # 1) Keep a copy of raw counts before any normalization/log
        adata.layers["counts"] = adata.X.copy()

        # 2) Library-size normalization (on X)
        if normalize:
            sc.pp.normalize_total(adata, target_sum=target_sum)

        # 3) Log1p transform (on X)
        if log:
            sc.pp.log1p(adata)

        # 4) HVGs on raw counts layer (Seurat v3-style)
        sc.pp.highly_variable_genes(
            adata,
            layer="counts",
            n_top_genes=n_top_genes,
            min_mean=min_mean,
            max_mean=max_mean,
            min_disp=min_disp,
            flavor=flavor,
            **kwargs,
        )
        if plot_hvg:
            try:
                sc.pl.highly_variable_genes(adata, layer="counts")
            except TypeError:
                # older Scanpy versions may not support layer= in the plot call
                sc.pl.highly_variable_genes(adata)

        # 5) Save full matrix snapshot to .raw (optional)
        if save_to_raw:
            adata.raw = adata.copy()

        # 6) Subset to HVGs
        adata = adata[:, adata.var.highly_variable].copy()

        # 7) Scale into .layers['scaled'] then make X == scaled
        if scale:
            # stash a dense copy into 'scaled'
            Xdense = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
            adata.layers["scaled"] = Xdense.copy()

            # Some Scanpy versions don't support `layer=` in scale; handle both.
            try:
                sc.pp.scale(adata, layer="scaled", max_value=max_value)
                # ensure X follows the scaled layer
                adata.X = adata.layers["scaled"]
            except TypeError:
                # fallback: temporarily operate on X
                adata.X = adata.layers["scaled"].copy()
                sc.pp.scale(adata, max_value=max_value)
                adata.layers["scaled"] = adata.X.copy()

        else:
            # even if not scaling, keep X consistent with the (possibly log-normalized) values
            pass

        return adata.copy()


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
        Fit (or reuse) a TopOGraph and populate `adata` with scaffolds, projections,
        and optional Leiden clusters computed on the refined TopoMetry graph.

        Behavior
        --------
        - If `tg` is None, instantiates `TopOGraph(**topograph_kwargs)` and fits it on `adata.X`.
        - If `tg` is provided, reuses it unless new kwargs are given or its fitted
        shape does not match the current `adata`; in either case it is re-fit.
        - Stores dual spectral scaffolds and requested 2-D projections into `adata.obsm`.
        - If `do_leiden=True`, runs Leiden on the refined DM operator (`tg.P_of_Z`)
        and stores multi-resolution labels.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix whose `.X` is used to fit TopOGraph.
        tg : TopOGraph or None, default None
            Existing model to reuse; may be re-fit depending on kwargs / shape.
        projections : tuple of {"MAP","PaCMAP"} or str, default ("MAP","PaCMAP")
            Projections to ensure and store as 2-D embeddings.
            Stored keys: "X_msTopoMAP"/"X_TopoMAP", "X_msTopoPaCMAP"/"X_TopoPaCMAP".
        do_leiden : bool, default True
            Whether to run Leiden clustering on the refined DM graph.
        leiden_key_base : str, default "topo_clusters"
            Base name for cluster columns in `adata.obs`.
        leiden_resolutions : sequence of float, default (0.2, 0.8, 1.2)
            Leiden resolutions to compute.
        leiden_primary_index : int, default 1
            Index into `leiden_resolutions` to use as the primary cluster column.
        **topograph_kwargs
            Extra keyword arguments passed to `TopOGraph(...)` (on creation or refit).

        Returns
        -------
        tg : TopOGraph
            The fitted (or reused) TopOGraph instance.

        Side Effects
        ------------
        - adata.obsm["X_ms_spectral_scaffold"], adata.obsm["X_spectral_scaffold"]
        - adata.obsm["X_msTopoMAP"], adata.obsm["X_TopoMAP"] (if requested/available)
        - adata.obsm["X_msTopoPaCMAP"], adata.obsm["X_TopoPaCMAP"] (if requested/available)
        - adata.obsp["topometry_connectivities"] (refined DM operator), and the same
        matrix mirrored in `topometry_distances` for Scanpy compatibility.
        - If available, msDM operator stored as
        `topometry_connectivities_ms` / `topometry_distances_ms`.
        - If `do_leiden`, adds multi-resolution labels in `adata.obs` and a primary
        categorical column `adata.obs[leiden_key_base]`.
        """

        need_refit = False

        if tg is None:
            tg = TopOGraph(**topograph_kwargs)
            need_refit = True
        else:
            # Apply new params if provided
            if topograph_kwargs:
                tg.set_params(**topograph_kwargs)
                need_refit = True

            # Check if tg looks fitted & compatible with this adata
            if getattr(tg, "base_kernel", None) is None:
                need_refit = True
            elif getattr(tg.base_kernel, "P", None) is None:
                need_refit = True
            # size mismatch → refit
            elif (getattr(tg, "n", None) != adata.n_obs) or (getattr(tg, "m", None) != adata.n_vars):
                need_refit = True

        if need_refit:
            tg.fit(adata.X)

        # From here on, reuse the fitted tg (new or reused). Populate adata:
        # (1) store scaffolds
        adata.obsm["X_ms_spectral_scaffold"] = tg.spectral_scaffold(multiscale=True)[:, :getattr(tg, "_scaffold_components_ms", tg.n_eigs)]
        adata.obsm["X_spectral_scaffold"]    = tg.spectral_scaffold(multiscale=False)[:, :getattr(tg, "_scaffold_components_dm", tg.n_eigs)]

        # (2) store projections if requested/available
        # Normalize a single string to a tuple
        if isinstance(projections, str):
            projections = (projections,)

        # Pretty names used in adata.obsm keys
        pretty_map = {"MAP": "TopoMAP", "PaCMAP": "TopoPaCMAP"}

        # Map requested method & scale -> TopOGraph property name
        prop_map = {
            "MAP":    {False: "TopoMAP",     True: "msTopoMAP"},
            "PaCMAP": {False: "TopoPaCMAP",  True: "msTopoPaCMAP"},
        }

        def _get_projection_if_available(tg, method: str, multiscale: bool):
            """Return embedding array if the corresponding property exists, else None."""
            prop = prop_map[method][multiscale]
            try:
                return getattr(tg, prop)  # property raises AttributeError if missing
            except AttributeError:
                return None

        def _ensure_projection(tg, method: str, multiscale: bool):
            """Compute projection only if not already available; return the array or None."""
            Y = _get_projection_if_available(tg, method, multiscale)
            if Y is None:
                tg.project(projection_method=method, multiscale=multiscale)
                Y = _get_projection_if_available(tg, method, multiscale)
            return Y

        for proj in projections:
            if proj not in prop_map:
                # Unknown projection name; skip safely
                continue
            pretty = pretty_map.get(proj, proj)  # e.g., "MAP" -> "TopoMAP"

            # Multiscale (msDM)
            Y_ms = _ensure_projection(tg, proj, multiscale=True)
            if Y_ms is not None:
                adata.obsm[f"X_ms{pretty}"] = Y_ms      # X_msTopoMAP / X_msTopoPaCMAP

            # Single-scale DM
            Y_dm = _ensure_projection(tg, proj, multiscale=False)
            if Y_dm is not None:
                adata.obsm[f"X_{pretty}"] = Y_dm        # X_TopoMAP / X_TopoPaCMAP

        # (3) clustering on refined DM graph (tg.P_of_Z)
        if do_leiden:
            import scanpy as sc
            from scipy.sparse import issparse, csr_matrix

            def _csr(A):
                return A if issparse(A) else csr_matrix(A)

            # --- DM refined operator ---
            P = _csr(tg.P_of_Z)
            # Make it available to the report:
            adata.obsp["topometry_connectivities"] = P
            # (We use the same matrix as a reasonable proxy for distances)
            adata.obsp["topometry_distances"] = P

            # --- msDM refined operator (if present) ---
            P_ms = getattr(tg, "P_of_msZ", None)
            if P_ms is not None:
                P_ms = _csr(P_ms)
                adata.obsp["topometry_connectivities_ms"] = P_ms
                adata.obsp["topometry_distances_ms"] = P_ms

            # Build a temporary neighbors_key for Leiden using our P
            sc.pp.neighbors(
                adata,
                use_rep=None,
                n_neighbors=tg.graph_knn,
                method="umap",
                key_added="_topo_tmp",
            )
            adata.uns["_topo_tmp"]["connectivities"] = P
            adata.uns["_topo_tmp"]["distances"] = P  # acceptable proxy

            for i, res in enumerate(leiden_resolutions):
                key = f"{leiden_key_base}_res{res:g}"
                sc.tl.leiden(adata, resolution=res, neighbors_key="_topo_tmp", key_added=key)

            primary = f"{leiden_key_base}_res{leiden_resolutions[leiden_primary_index]:g}"
            adata.obs[leiden_key_base] = adata.obs[primary].astype("category")

        return tg



    def evaluate_representations(
        adata,
        tg,
        return_df: bool = False,
        print_results: bool = False,
        plot_results: bool = True,
        plot_path: str | None = None,
        *,
        # operator construction
        metric: str = "euclidean",
        n_neighbors: int = 30,
        backend: str = "hnswlib",
        n_jobs: int = -1,
        # evaluation hyperparams
        times = (1, 2, 4),
        r: int = 32,
        k_for_pf1: int = None,
        symmetric_hint: bool = True,
        # pretty printing
        strip_obsm_prefix: str = "X_",
    ):
        """
        Evaluate all representations in `adata.obsm` by comparing their induced
        diffusion operator to the reference operator `tg.base_kernel.P`.

        Uses `topo_preserve_score` (PF1, PJS, SP) and also reports spectral similarity diagnostics.

        Parameters
        ----------
        adata : AnnData
            Must contain one or more embeddings/representations in .obsm.
        tg : TopOGraph (fitted)
            Provides the reference diffusion operator via `tg.base_kernel.P`.
        return_df : bool, default=False
            If True, returns a DataFrame with the results.
        print_results : bool, default=False
            If True, prints a table with the results.
        plot_results : bool, default=True
            If True, plots a heatmap with the results.
        metric : str, default="euclidean"
            Distance used to build the operator on each representation.
        n_neighbors : int, default=30
            Graph degree used to build each representation's operator.
        backend : str, default="nmslib"
            ANN backend for kNN graph construction in `get_P`.
        n_jobs : int, default=-1
            Parallelism for neighbor search where supported.
        times : tuple(int), default=(1,2,4)
            Diffusion times for Spectral Procrustes (SP).
        r : int, default=32
            Leading eigenpairs for spectral metrics.
        k_for_pf1 : int or None, default=None
            Top-k used by PF1; if None, each row uses its native sparsity.
        symmetric_hint : bool, default=True
            Hint for eigensolvers (operators are symmetrized inside).
        strip_obsm_prefix : str, default="X_"
            Purely cosmetic: removes this prefix when printing column headers.

        Prints
        ------
        A table with columns = embeddings in `adata.obsm` (header stripped),
        and rows for:
        • PF1 (↑)
        • PJS (↑)
        • SP  (↑)
        """
        import numpy as np
        from scipy.sparse import csr_matrix, issparse

        from topo.eval.topo_metrics import (
            topo_preserve_score,
            spectral_similarity,
            commute_time_trace_gap,
            get_P,
        )

        # -----------------------------
        # 0) Reference operator (CSR)
        # -----------------------------
        PX_ref = tg.base_kernel.P
        if not issparse(PX_ref):
            PX_ref = csr_matrix(PX_ref)

        # -----------------------------
        # 1) Iterate over all obsm reps
        # -----------------------------
        obsm_keys = list(adata.obsm_keys())

        tp_score   = {}  # composite TopoPreserve score
        parts_all  = {}  # PF1, PJS, SP
        spec_evW1  = {}  # spectral eigenvalue W1 (lower better)
        spec_cos   = {}  # spectral subspace cosine   (higher better)

        for key in obsm_keys:
            Y = adata.obsm[key]  # (n, d)

            # Build diffusion operator for this representation
            PY = get_P(
                Y,
                metric=metric,
                n_neighbors=n_neighbors,
                backend=backend,
                n_jobs=n_jobs,
            )
            if not issparse(PY):
                PY = csr_matrix(PY)

            # Composite score + parts (PF1, PJS, SP)
            score, parts = topo_preserve_score(
                PX_ref, PY,
                times=times,
                r=r,
                symmetric_hint=symmetric_hint,
                k_for_pf1=k_for_pf1,
            )
            tp_score[key]  = score
            parts_all[key] = parts  # contains PF1, PJS, SP

            # Spectral similarity diagnostics
            spec = spectral_similarity(PX_ref, PY, r=r, symmetric_hint=symmetric_hint, return_details=True)
            spec_evW1[key] = spec['eigenvalue_w1']
            spec_cos[key]  = spec['subspace_cos']

        # -----------------------------
        # 2) Pretty print as a table
        # -----------------------------
        def nice(name):
            return name[len(strip_obsm_prefix):] if strip_obsm_prefix and name.startswith(strip_obsm_prefix) \
                else name

        cols = [nice(k) for k in obsm_keys]

        def row_from_dict(title, dct, fmt="{:.3f}"):
            vals = []
            for k in obsm_keys:
                v = dct.get(k, np.nan)
                vals.append(fmt.format(v) if (v is not None and np.isfinite(v)) else "nan")
            return title, vals

        rows = []
        rows.append(row_from_dict("PF1@k   (set overlap)      (↑)         ", {k: parts_all[k]['PF1']       for k in obsm_keys}))
        rows.append(row_from_dict("PJS     (1 - JS rows)      (↑)         ", {k: parts_all[k]['PJS']       for k in obsm_keys}))
        rows.append(row_from_dict("SP      (Procrustes R²)    (↑)         ", {k: parts_all[k]['SP']        for k in obsm_keys}))

        col_widths = [max(len(c), 9) for c in cols]
        row_name_w = max(38, max(len(r[0]) for r in rows))  # keep row titles aligned

        header = " " * row_name_w + " | " + " | ".join(c.ljust(w) for c, w in zip(cols, col_widths))
        sep    = "-" * len(header)

        if print_results:
            print(sep)
            print(header)
            print(sep)
            for name, vals in rows:
                line = name.ljust(row_name_w) + " | " + " | ".join(v.ljust(w) for v, w in zip(vals, col_widths))
                print(line)
            print(sep)

        # -----------------------------
        # 3) Persist a proper per-representation metrics table
        # -----------------------------

        if not obsm_keys:
            df = pd.DataFrame(columns=[
                "representation", "PF1", "PJS", "SP"
            ])
        else:
            def nice_key(k):
                return k[len(strip_obsm_prefix):] if strip_obsm_prefix and k.startswith(strip_obsm_prefix) else k

            records = []
            for k in obsm_keys:
                parts = parts_all.get(k, {})
                records.append({
                    "representation": nice_key(k),
                    "PF1":         float(parts.get("PF1", np.nan)),
                    "PJS":         float(parts.get("PJS", np.nan)),
                    "SP":          float(parts.get("SP", np.nan)),
                })

            df = pd.DataFrame.from_records(records)
            if "PF1" in df.columns:
                df = df.sort_values("PF1", ascending=False).reset_index(drop=True)

        adata.uns["topometry_representation_eval"] = df

        if plot_results:
            plot_evaluation_heatmap(df, out_png=plot_path)
        if return_df:
            return df

    

    def plot_riemann_diagnostics(
        adata,
        tg,
        proj_key='X_TopoMAP',
        groupby='topo_clusters',
        scale_gain=1.0,
        ellipse_alpha=0.15,
        point_size=6,
        do_all=True,
        verbose=True,
        title_fontsize=18,
        figsize=(18, 6),
        dpi=150,
        show=True,
    ):
        """
        Plot Riemannian distortion diagnostics for representations found in adata.obsm.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix with projections in adata.obsm.
        tg : TopoGraph
            Fitted TopoGraph object with base_kernel and Laplacian.
        proj_key : str
            Key in adata.obsm to use as the 2D embedding for plotting.
        groupby : str
            Key in adata.obs for categorical labels (used for coloring).
        scale_gain : float, default 1.0
            Global multiplicative scale for ellipse axes. Increase to make ellipses larger.
        ellipse_alpha : float, default 0.15
            Alpha transparency for ellipses.
        point_size : float, default 6
            Size of points in scatter plots.
        do_all : bool
            Whether to compute diagnostics for all projections in adata.obsm.
        verbose : bool
            Whether to print progress messages.
        title_fontsize : int
            Font size for titles.
        figsize : tuple
            Figure size for the 3-panel plot.
        dpi : int
            Figure DPI for the 3-panel plot.
        show : bool
            Whether to display the plots immediately.

        Returns
        -------
        None (plots are shown and saved in adata)

        """
        from topo.eval.rmetric import (
        RiemannMetric,
        plot_riemann_metric_localized,
        plot_riemann_metric_global,
        calculate_deformation,
        )
        if proj_key not in adata.obsm:
            raise KeyError(f"{proj_key} not found in adata.obsm")
        if not hasattr(tg, 'base_kernel') or not hasattr(tg.base_kernel, 'L'):
            raise ValueError("tg must have a fitted base_kernel with attribute L (Laplacian)")
        proj_nick = proj_key.replace("X_", "")
        # --- choose highest-resolution clustering if 'topo_clusters' has no palette ---
        if groupby == 'topo_clusters':
            if f'{groupby}_colors' in adata.uns:
                pass
            else:
                try:
                    # find all topo_clusters_res{number} columns and pick the highest resolution
                    res_keys = [k for k in adata.obs.columns if k.startswith('topo_clusters_res')]
                    if res_keys:
                        def _parse_res(k):
                            try:
                                return float(k.split('res', 1)[1])
                            except Exception:
                                return -np.inf
                        best_key = max(res_keys, key=_parse_res)
                        # create a stable alias plus palette entry
                        adata.obs['topo_clusters_highestres'] = adata.obs[best_key].astype('category')
                        best_col_key = f'{best_key}_colors'
                        if best_col_key in adata.uns:
                            adata.uns['topo_clusters_highestres_colors'] = adata.uns[best_col_key]
                        groupby = 'topo_clusters_highestres'
                except Exception:
                    Warning("Could not find a suitable topo_clusters_res* column for coloring.")
                    groupby = None
                    pass

        # Select colors as in scanpy (categorical)

        labels = adata.obs[groupby]
        palette = adata.uns[f'{groupby}_colors']
        cats = labels.cat.categories if labels.dtype.name == 'category' else np.unique(labels)
        lut = dict(zip(cats, palette))
        _colors = labels.map(lut)

        L = tg.base_kernel.L

        def _make_plot(Y, L, proj_key, proj_nick, show=show):            
            fig, axs = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
            # fig.suptitle(f"Riemannian diagnostics - {proj_nick}", fontsize=title_fontsize)
            # plot using the mapped colors directly (do not pass cmap)
            plot_riemann_metric_localized(
                Y, L,
                n_plot=adata.shape[0]//10,
                scale_mode="logdet",
                scale_gain=scale_gain,
                alpha=None,
                ax=axs[0],
                seed=7,
                show_points=True,
                colors=_colors,   # now numeric RGBA/hex codes from Scanpy palette
                point_alpha=0.1,
                ellipse_alpha=ellipse_alpha,
                point_size=point_size,
            )
            axs[0].set_title('Localized indicatrices', fontsize=title_fontsize)
            plt.gca().set_aspect('equal'); plt.tight_layout()

            # Compute deformation once (centered log det(G)), optionally smoothed
            deform_vals, (dmin, dmax) = calculate_deformation(
                Y, L,
                center="median",
                diffusion_t=8,
                diffusion_op=getattr(tg.base_kernel, "P", None),
                normalize="symmetric",
                clip_percentile=2.0,
                return_limits=True,
            )
            adata.obs[f'metric_contract_expand_{proj_nick}'] = deform_vals

            # (b) Global indicatrices (thinned grid-averaged ellipses) overlaid on the embedding,
            #     colored by local contraction/expansion using the shared color scale
            #fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
            sc.pl.embedding(
                adata,
                basis=proj_nick,
                color='topo_clusters',
                legend_loc=None,
                legend_fontsize=6,
                show=False,
                return_fig=False,
                ax=axs[1],
            )
            # Extra safety: if a legend exists, shrink its fontsize (works across Scanpy/mpl versions)
            leg = axs[1].get_legend()
            if leg is not None:
                try:
                    leg.set_title(leg.get_title().get_text(), prop={'size': 4})
                except Exception:
                    pass
                for txt in leg.get_texts():
                    try:
                        txt.set_fontsize(6)
                    except Exception:
                        pass
            plot_riemann_metric_global(
                Y, L,
                grid_res=8,
                k_avg=30,
                scale_mode="logdet",
                scale_gain=0.8,
                alpha=ellipse_alpha,
                ax=axs[1],
                show_points=False,
                zorder=3,
                cmap="coolwarm",
                vmin=dmin, vmax=dmax,               # keep color scale consistent with panel (d)
                min_sep_factor=1.1,                 # reduce ellipse overlap
                choose_strong_first=True,
                deformation_vals=deform_vals,       # reuse computed deformation
            )
            plt.gca().set_aspect('equal'); plt.tight_layout()
            axs[1].set_title("Global indicatrices (C/E overlay)", fontsize=title_fontsize)
            
            # (c) Metric-derived scalar maps (anisotropy and log-det(G))
            G = RiemannMetric(Y, L).get_rmetric()
            lam = np.linalg.eigvalsh(G); lam = np.clip(lam, 1e-12, None)
            adata.obs[f'metric_anisotropy_{proj_nick}'] = np.log(lam[:, -1] / lam[:, 0])
            adata.obs[f'metric_logdetG_{proj_nick}'] = np.sum(np.log(lam), axis=1)

            # (d) CONTRACTION vs EXPANSION PANEL (points), using the same deformation and limits
            # Try to suppress colorbar via Scanpy arg (newer versions)
            _axes_before = list(plt.gcf().axes)
            try:
                sc.pl.embedding(
                    adata,
                    basis=proj_key,
                    color=[f'metric_contract_expand_{proj_nick}'],
                    cmap='bwr',
                    wspace=0.25,
                    show=False,
                    ax=axs[2],
                    vmin=dmin,
                    vmax=dmax,
                    colorbar_loc=None,   # <-- preferred (if supported)
                )
            except TypeError:
                # Older Scanpy: draw normally, then remove any colorbar axes added
                sc.pl.embedding(
                    adata,
                    basis=proj_key,
                    color=[f'metric_contract_expand_{proj_nick}'],
                    cmap='bwr',
                    wspace=0.25,
                    show=False,
                    ax=axs[2],
                    vmin=dmin,
                    vmax=dmax,
                )
                _axes_after = list(plt.gcf().axes)
                # Any *new* axes (after the call) that are not one of our panel axes are likely colorbars; remove them
                for extra_ax in _axes_after:
                    if extra_ax not in _axes_before and extra_ax not in [axs[0], axs[1], axs[2]]:
                        try:
                            extra_ax.remove()
                        except Exception:
                            pass
            axs[2].set_title("Local contraction / expansion", fontsize=title_fontsize)
            plt.gca().set_aspect('equal'); plt.subplots_adjust(wspace=0.05); plt.tight_layout()
            if show:
                plt.show()      # optional for interactive use
                plt.close(fig)  # close only this fig after showing
                return None
            else:
                return fig      # caller will save & close

        if do_all:
            for proj_name in adata.obsm_keys():
                Y = np.asarray(adata.obsm[proj_name])
                proj_nick = proj_name.replace("X_", "")
                if verbose: print(f"Riemannian diagnostics for projection '{proj_nick}'")
                # For do_all, still avoid plt.show()/close-all; just show if explicitly requested
                _fig = _make_plot(Y, L, proj_name, proj_nick, show=show)
                if (not show) and (_fig is not None):
                    plt.close(_fig)  # safe cleanup if used interactively outside the PDF
            return None
        else:
            Y = np.asarray(adata.obsm[proj_key])
            if verbose: print(f"Riemannian diagnostics for projection '{proj_key}'")
            return _make_plot(Y, L, proj_key, proj_nick, show=show)


    def visualize_optimization(
        adata,
        tg,
        groupby: str = "topo_clusters",
        num_iters: int = 600,
        save_every: int = 10,
        dpi: int = 120,
        *,
        multiscale: bool = True,
        fps: int = 20,
        point_size: float = 3.0,
        filename: str = None,
        cmap: str = "inferno",
    ):
        """
        Create an animated GIF showing the evolution of MAP optimization.

        At regular checkpoints collected during `TopOGraph.project(..., save_every=...)`,
        draw the current 2-D embedding colored by a categorical/numeric label, or by
        a gene if `groupby` matches a variable name. Combines the snapshots into an
        animation.

        Parameters
        ----------
        adata : AnnData
            Source of colors/labels; also used for gene coloring when `groupby` is a gene.
        tg : TopOGraph
            Fitted model containing MAP snapshots (ms or DM).
        groupby : str, default "topo_clusters"
            Column in `adata.obs` (categorical or numeric) or a gene in `adata.var_names`.
        num_iters : int, default 600
            Total iterations to visualize (clips to available snapshots).
        save_every : int, default 10
            Snapshot frequency used during optimization; used to index frames.
        dpi : int, default 120
            DPI for frames in the resulting GIF.
        multiscale : bool, default True
            If True, visualize msMAP snapshots; otherwise DM MAP snapshots.
        fps : int, default 20
            Frames per second for the GIF.
        point_size : float, default 3.0
            Scatter marker size.
        filename : str or None, default None
            Output path (e.g., "map_optimization.gif"). If None, a name is auto-chosen.
        cmap : str, default "inferno"
            Colormap for numeric coloring.

        Returns
        -------
        filename : str
            Path to the saved GIF.

        Notes
        -----
        - Requires that snapshots were collected during optimization (see `TopOGraph.project`).
        - Preserves Scanpy categorical palettes when present in `adata.uns[f"{groupby}_colors"]`.
        """

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        n = adata.n_obs

        # --- 1) Build per-cell colors based on `groupby`
        if groupby in adata.var_names:
            # Gene expression coloring
            gidx = list(adata.var_names).index(groupby)
            x = adata.X[:, gidx]
            if hasattr(x, "toarray"):
                x = x.toarray().ravel()
            else:
                x = np.asarray(x).ravel()

            vmin, vmax = np.nanpercentile(x, [1, 99])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = np.nanmin(x), np.nanmax(x)
            t = (x - vmin) / (vmax - vmin + 1e-12)
            colors = plt.get_cmap(cmap)(np.clip(t, 0, 1))  # (n,4)

        elif groupby in adata.obs.columns:
            series = adata.obs[groupby]
            if series.dtype.name == "category" or str(series.dtype).startswith("category"):
                # Categorical — use scanpy palette if available, else make a stable one
                cats = series.cat.categories
                codes = series.cat.codes.to_numpy()  # -1 for missing
                palette_key = f"{groupby}_colors"
                if palette_key in adata.uns and len(adata.uns[palette_key]) >= len(cats):
                    palette = list(adata.uns[palette_key])
                else:
                    _cmap = plt.get_cmap("tab20")
                    palette = [mcolors.to_hex(_cmap(i % 20)) for i in range(len(cats))]
                    adata.uns[palette_key] = palette

                # Directly map codes -> palette (handle -1 as a neutral gray)
                neutral = (0.7, 0.7, 0.7, 0.6)
                colors = np.empty((n, 4), dtype=float)
                for i, c in enumerate(codes):
                    if c >= 0 and c < len(palette):
                        colors[i] = mcolors.to_rgba(palette[c])
                    else:
                        colors[i] = neutral  # missing/unassigned category
            else:
                # Numeric obs: viridis
                vals = np.asarray(series.values, float).ravel()
                vmin, vmax = np.nanpercentile(vals, [1, 99])
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
                t = (vals - vmin) / (vmax - vmin + 1e-12)
                colors = plt.get_cmap("viridis")(np.clip(t, 0, 1))  # (n,4)
        else:
            # Fallback uniform semi-dark gray
            colors = np.tile(np.array([0.15, 0.15, 0.15, 0.85], dtype=float)[None, :], (n, 1))

        # --- 2) Ensure snapshots exist for the requested scaffold (crucial) ---
        tg.project(
            projection_method="MAP",
            multiscale=bool(multiscale),
            num_iters=int(num_iters),
            save_every=int(save_every),
            include_init_snapshot=True,
        )

        # --- 3) Render the GIF using TopOGraph’s helper (uses precomputed snaps) ---
        out_path = tg.visualize_optimization(
            num_iters=num_iters,    # harmless if snapshots already exist
            save_every=save_every,
            dpi=dpi,
            color=colors,           # (n,4) RGBA float array
            multiscale=multiscale,
            fps=fps,
            point_size=point_size,
            filename=filename,
        )
        return out_path


    def intrinsic_dim(
        adata: AnnData,
        tg: TopOGraph | None = None,
        *,
        id_methods: list[str] = ("fsa", "mle"),
        id_k_values: list[int] | None = None,
        n_jobs: int = -1,
    ):
        """
        Compute and store intrinsic dimensionality summaries.

        Uses `TopOGraph`'s internal sizing results when available (global + local
        for FSA / MLE), and optionally runs additional estimators on `adata.X`
        for the provided k values.

        Parameters
        ----------
        adata : AnnData
            Target container to receive summaries in `.uns` and `.obs`.
        tg : TopOGraph or None, default None
            Fitted model to source existing ID estimates; may be None (then only
            direct computations on `adata.X` are performed).
        id_methods : list of {"fsa","mle"}, default ("fsa","mle")
            Which estimators to summarize.
        id_k_values : list[int] or None, default None
            Optional neighborhood sizes for additional direct computations.
        n_jobs : int, default -1
            Parallelism for direct computations.

        Returns
        -------
        None

        Side Effects
        ------------
        - adata.uns['topometry_id_global_fsa'], adata.uns['topometry_id_global_mle']
        - adata.obs[...] per-cell vectors when available
        - adata.uns['topometry_id_details'] with estimator-specific metadata
        """

        if tg is not None:
            adata.uns['topometry_id_details'] = getattr(tg, "_id_details", None)
            adata.uns[f'topometry_id_global_{tg.id_method}'] = tg.global_id
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
        groupby_candidates: list[str] | None = None,
        **spec_kwargs,
    ):
        """
        Quantify spectral selectivity on the multiscale scaffold and align with labels.

        Computes per-eigenvector selectivity scores under a chosen eigenvalue
        weighting scheme and (optionally) a smoothed operator. Also evaluates
        alignment with available grouping keys (first match from `groupby_candidates`).

        Parameters
        ----------
        adata : AnnData
            Target to store summaries (uns).
        tg : TopOGraph
            Fitted model providing eigenpairs and refined graphs.
        groupby_candidates : list[str] or None, default None
            Candidate obs keys for alignment-by-label; first available is used.

        Returns
        -------
        None

        Side Effects
        ------------
        - adata.uns['spectral_alignment_summary'] with selectivity/alignment tables.
        """

        spec = tg.spectral_selectivity(
            multiscale=True, **spec_kwargs
        )
        adata.obs['spectral_EAS']       = spec['EAS']
        adata.obs['spectral_RayScore']  = spec['RayScore']
        adata.obs['spectral_LAC']       = spec['LAC']
        adata.obs['spectral_axis']      = pd.Categorical(spec['axis'].astype(int))
        adata.obs['spectral_axis_sign'] = pd.Categorical(spec['axis_sign'].astype(int))
        adata.obs['spectral_radius']    = spec['radius']

        # alignment-by-label on ms scaffold, if present
        candidates = groupby_candidates or ['topo_clusters']
        align_key = next((k for k in candidates if k in adata.obs), None)
        if align_key and ('X_ms_spectral_scaffold' in adata.obsm):
            _spectral_alignment_by_label(
                adata, labels_key=align_key,
                scaffold_key='X_ms_spectral_scaffold',
                top_k=3,
                out_key='spectral_alignment_summary',
            )


    def filter_signal(
        adata,
        tg,
        signal_key: str = "disease_state",
        signal: str = "diseased",
        *,
        which: str = "msZ",              # {'msZ','Z','X'}: which operator to filter with
        diffusion_t: int = 8,            # number of graph filtering (diffusion) steps
        noise_level: float = 0.0,        # optional Gaussian noise added BEFORE filtering (for stress tests)
        normalize: str = "auto",         # {'auto','none','unit'}: rescale after filtering
        out_base: str | None = None,     # name stem for outputs in adata.obs (auto if None)
        return_array: bool = False,      # also return the filtered vector
        random_state: int = 7,
    ):
        """
        Graph-filter a user-provided signal using TopOMetry’s Markov operators.

        Behavior
        --------
        - If `adata.obs[signal_key]` is categorical, binarizes to 1 for entries equal to `signal`,
        else 0. If it is numeric, uses it directly.
        - Optionally adds Gaussian noise (`noise_level`) before filtering.
        - Filters the signal by repeatedly multiplying by the chosen Markov operator (`which`)
        for `diffusion_t` steps.
        - Optionally normalizes the filtered values.

        Parameters
        ----------
        adata : AnnData
            Container holding the signal in `adata.obs[signal_key]`.
        tg : TopOGraph
            Fitted TopOGraph providing Markov operators (P_of_msZ, P_of_Z, P_of_X).
        signal_key : str, default "disease_state"
            Column in `adata.obs` containing the signal (categorical or numeric).
        signal : str, default "diseased"
            Category treated as 1 when `signal_key` is categorical.
        which : {"msZ","Z","X"}, default "msZ"
            Which operator to use for filtering.
        diffusion_t : int, default 8
            Number of filtering (diffusion) steps.
        noise_level : float, default 0.0
            Std. dev. of Gaussian noise added to the raw signal prior to filtering.
        normalize : {"auto","none","unit"}, default "auto"
            - "auto": if input looks like a probability (0–1), keep range; otherwise unit-scale to 0–1.
            - "unit": force min–max to 0–1 after filtering.
            - "none": do not rescale.
        out_base : str or None, default None
            Base name for outputs in `adata.obs`. If None, uses f"{signal_key}__gf".
            Two columns are written:
            - f"{out_base}__raw"
            - f"{out_base}__filtered_t{diffusion_t}_{which}"
        return_array : bool, default False
            If True, also returns the filtered vector as a NumPy array.
        random_state : int, default 7
            RNG seed for optional noise.

        Returns
        -------
        np.ndarray or None
            The filtered vector if `return_array=True`, else None.

        Notes
        -----
        - Uses operators directly from `tg` when possible. If that fails, tries to fall
        back to `adata.obsp['topometry_connectivities_ms']` (for msZ) / `'topometry_connectivities'` (for Z).
        - Works with dense or sparse operators; computation is vector–matrix.

        Examples
        --------
        # categorical example
        # adata.obs['disease_state'] contains {'healthy','diseased'}
        filter_signal(adata, tg, signal_key='disease_state', signal='diseased',
                    which='msZ', diffusion_t=8, noise_level=0.0)

        # numeric example (e.g., a score in adata.obs['risk_score'])
        filter_signal(adata, tg, signal_key='risk_score', which='Z', diffusion_t=4, normalize='unit')

        """
        if signal_key not in adata.obs:
            raise KeyError(f"`{signal_key}` not found in adata.obs")

        # --- 1) Build raw numeric vector s_raw in [0,1] (when categorical) or as-is (numeric) ---
        ser = adata.obs[signal_key]
        if pd.api.types.is_categorical_dtype(ser) or ser.dtype == "object":
            s_raw = (ser.astype(str).values == str(signal)).astype(float)
        else:
            s_raw = pd.to_numeric(ser, errors="coerce").astype(float).values
            # gentle coercion if it already looks like a probability vector (within [0,1] allowing small eps)
            finite = np.isfinite(s_raw)
            looks_prob = np.nanmin(s_raw[finite]) >= -1e-6 and np.nanmax(s_raw[finite]) <= 1.0 + 1e-6
            if normalize == "auto" and not looks_prob:
                # defer normalization until after filtering
                pass

        s_raw = np.nan_to_num(s_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # optional pre-filter noise (stress test)
        if noise_level > 0.0:
            rng = np.random.default_rng(random_state)
            s_raw = np.clip(s_raw + noise_level * rng.standard_normal(adata.n_obs), 0.0, None)

        # --- 2) Choose Markov operator P ---
        def _csr(A):
            return A if issparse(A) else csr_matrix(A)

        P = None
        which_key = str(which).lower().strip()
        try:
            if which_key in {"msz", "ms"}:
                P = tg.P_of_msZ
            elif which_key in {"z", "dm"}:
                P = tg.P_of_Z
            elif which_key == "x":
                P = tg.P_of_X
        except Exception:
            P = None

        # fallback to adata.obsp if needed
        if P is None:
            if which_key in {"msz", "ms"}:
                P = adata.obsp.get("topometry_connectivities_ms", None)
            elif which_key in {"z", "dm"}:
                P = adata.obsp.get("topometry_connectivities", None)
            elif which_key == "x":
                # not typically stored; leave None
                P = None

        if P is None:
            raise RuntimeError(
                f"Could not obtain a Markov operator for '{which}'. "
                "Ensure `tg.fit(...)` was called or store the operator in `adata.obsp`."
            )

        P = _csr(P)

        # --- 3) Diffusion filtering: s_f = P^t * s_raw ---
        s_f = s_raw.copy()
        t_steps = int(max(0, diffusion_t))
        for _ in range(t_steps):
            s_f = P @ s_f

        s_f = np.asarray(s_f).ravel()

        # --- 4) Optional normalization after filtering ---
        if normalize == "unit" or (normalize == "auto" and (s_f.min() < -1e-6 or s_f.max() > 1.0 + 1e-6)):
            mn, mx = float(np.min(s_f)), float(np.max(s_f))
            if mx > mn:
                s_f = (s_f - mn) / (mx - mn)
            else:
                s_f = np.zeros_like(s_f)

        # --- 5) Persist to adata.obs ---
        stem = out_base or f"{signal_key}__gf"
        raw_key = f"{stem}__raw"
        flt_key = f"{stem}__filtered_t{t_steps}_{which_key}"

        adata.obs[raw_key] = pd.Series(s_raw, index=adata.obs_names, dtype=float)
        adata.obs[flt_key] = pd.Series(s_f,   index=adata.obs_names, dtype=float)

        # return if requested
        return s_f if return_array else None


    def riemann_diagnostics(
        adata: AnnData,
        tg: TopOGraph,
        *,
        center: str = "median",
        diffusion_t: int = 8,
        diffusion_op: str | None = "X",
        normalize: str = "symmetric",
        clip_percentile: float = 2.0,
    ):
        """
        Compute Riemannian deformation diagnostics for every 2-D embedding in `adata.obsm`.

        For each (n,2) embedding in `.obsm`, calculates the centered log-det deformation
        scalar field (optionally diffusion-smoothed) using the base Laplacian and stores:
        - a per-cell vector in `adata.obs[f"metric_deformation__{obsm_key}"]`
        - its display limits in `adata.uns['metric_limits'][obsm_key]`

        Parameters
        ----------
        adata : AnnData
            Container with 2-D embeddings in `.obsm`.
        tg : TopOGraph
            Fitted model providing the Laplacian / metric machinery.
        center : {"median","mean"} or float, default "median"
            How to center the log-det values.
        diffusion_t : int, default 8
            Steps of diffusion smoothing on the scalar field.
        diffusion_op : {"X","Z","msZ"} or None, default "X"
            Which operator to use for smoothing; None disables smoothing.
        normalize : {"symmetric","none"}, default "symmetric"
            Color-limit normalization mode used when computing limits.
        clip_percentile : float, default 2.0
            Percentile for robust clipping when deriving color limits.

        Returns
        -------
        None

        Side Effects
        ------------
        - adata.obsp['topometry_laplacian'] is set when available.
        - Deformation vectors and limits stored as described above.
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
                print(f"[TopoMetry] Riemann diagnostics skipped for {key}: {e}")

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
        - We compute pseudotime directly from multiscale spectral coordinates to avoid
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
        Diffusion-based imputation with automatic t selection and QC.

        Searches a grid of diffusion steps `impute_t_grid`, scores each against a
        null distribution (`null_K` permutations), selects the best t, and stores
        ONLY the chosen imputed matrix.

        Parameters
        ----------
        adata : AnnData
            Dataset to impute; results stored in `.layers`.
        tg : TopOGraph
            Fitted model providing the diffusion operators.
        layer : {"X", <layer_name>}, default "X"
            Which layer to impute.
        raw : bool, default False
            If True, use `adata.raw.X` as the source when available.
        which : {"msZ","Z","X"}, default "msZ"
            Space whose Markov operator drives imputation.
        impute_t_grid : sequence of int, default (1,2,4,8,16)
            Candidate diffusion steps.
        null_K : int, default 1000
            Null samples for QC.
        heatmap_top_genes : int, default 100
            Number of top genes to summarize in QC plots/tables.
        seed : int, default 13
            Random seed for null sampling.

        Returns
        -------
        None

        Side Effects
        ------------
        - adata.layers['topo_imputation'] : selected imputed matrix
        - adata.uns['imputation_qc'] : QC statistics and chosen t
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




    # --------------------------------------
    # Core: run full analysis on an AnnData
    # --------------------------------------

    def run_topometry_analysis(
        adata: AnnData,
        tg: TopOGraph | None = None,
        *,
        # TopOGraph hyperparameters (passed to fit_adata via **kwargs)
        base_knn: int = 30,
        graph_knn: int = 30,
        min_eigs: int = 100,
        base_metric: str = "cosine",
        graph_metric: str = "euclidean",
        graph_kernel_version: str = "bw_adaptive",
        diff_t: int = 1,
        n_jobs: int = -1,
        verbosity: int = 0,
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
        End-to-end TopOMetry pipeline on an AnnData object.

        Steps (in order):
        1) Fit (or reuse) TopOGraph and write scaffolds / projections to `.obsm`.
        2) (Optional) Leiden clustering on the refined DM graph.
        3) Spectral selectivity (and optional label alignment).
        4) Riemann diagnostics on all 2-D embeddings in `.obsm`.
        5) Diffusion imputation with null-based QC (stores only best t).

        Parameters
        ----------
        adata : AnnData
            Input dataset; will be populated with results (obsm/obsp/uns/layers).
        tg : TopOGraph or None, default None
            Existing model to reuse or None to fit anew.
        ... (see arguments; forwarded to constituent steps)

        Returns
        -------
        adata : AnnData
            The same object, enriched with TopOMetry outputs.
        tg : TopOGraph
            The fitted/reused model.
        """

        tg = fit_adata(
            adata, tg=tg,
            projections=projections,
            do_leiden=do_leiden,
            leiden_key_base=leiden_key_base,
            leiden_resolutions=leiden_resolutions,
            leiden_primary_index=leiden_primary_index,
            base_knn=base_knn,
            graph_knn=graph_knn,
            min_eigs=min_eigs,
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

        evaluate_representations(adata, tg, return_df=False, print_results=False, plot_results=False)

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
        Generate a multi-page A4-landscape PDF summarizing a TopOMetry run.

        Pages typically include:
        • Dataset overview and QC.
        • Dual scaffolds and requested 2-D projections (colored by categories/signals).
        • Clustering summaries (if computed).
        • Spectral selectivity / alignment.
        • Riemann diagnostics overlays and deformation maps.
        • Imputation QC (and optional gene example).
        • Optional graph-filtering visuals controlled by *filtering_* knobs.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix with TopOMetry results.
        tg : TopOGraph
            Fitted model used to produce the results.
        output_dir : str, default "./topometry_report"
            Directory where the PDF is written.
        filename : str, default "topometry_report.pdf"
            PDF file name.
        dpi : int, default 300
            Rendering resolution.
        a4_landscape_inches : (float, float), default (11.69, 8.27)
            Page size in inches (A4 landscape).
        gene_for_imputation : str or None, default None
            If provided, include a page showing imputed expression for this gene.
        labels_key_for_page_titles : str or None, default None
            If given, titles use this categorical obs key for context.
        categorical_plot_keys : list[str] or None, default None
            Extra obs keys to color projections.
        signal_plot_keys : list[str] or None, default None
            Extra numeric obs keys to display as signals.
        filtering_noise_level : float, default 0.15
            Noise parameter for graph-filtering demo page(s).
        filtering_diffusion_t : int, default 3
            Diffusion steps for filtering visuals.
        filtering_null_t : int, default 1
            Null t for filtering QC.
        filtering_null_K : int, default 500
            Null samples for filtering QC.

        Returns
        -------
        pdf_path : str
            Full path to the generated PDF.
        """

        # Defensive unwrap (in case someone passes (adata, tg))
        if isinstance(tg, tuple):
            tg = tg[1] if len(tg) >= 2 and hasattr(tg[1], "base_kernel") else tg[0]

        _ensure_dir(output_dir)
        pdf_path = os.path.join(output_dir, filename)

        # ---- helpers ----------------------------------------------------------
        def _embedding(ax, color, basis_name: str, variant: str, title=None, cmap=None, legend_loc=None, **kwargs):
            """
            Draw embedding with 1:1 aspect, show a frame, and label axes with projection names.
            basis_name in {'TopoMAP','TopoPaCMAP'} ; variant in {'DM','msDM'}
            """
            # Resolve keys, scanpy basis, and axis labels correctly per variant
            if basis_name == 'TopoMAP':
                if variant == 'DM':
                    key = 'X_TopoMAP'
                    basis_arg = 'TopoMAP'
                    xlab, ylab = 'TopoMAP_1', 'TopoMAP_2'
                else:  # msDM
                    key = 'X_msTopoMAP'
                    basis_arg = 'msTopoMAP'
                    xlab, ylab = 'msTopoMAP_1', 'msTopoMAP_2'
            elif basis_name == 'TopoPaCMAP':  # TopoPaCMAP
                if variant == 'DM':
                    key = 'X_TopoPaCMAP'
                    basis_arg = 'TopoPaCMAP'
                    xlab, ylab = 'TopoPaCMAP_1', 'TopoPaCMAP_2'
                else:  # msDM
                    key = 'X_msTopoPaCMAP'
                    basis_arg = 'msTopoPaCMAP'
                    xlab, ylab = 'msTopoPaCMAP_1', 'msTopoPaCMAP_2'
            else:
                if variant == 'DM':
                    key = 'X_' + basis_name
                    basis_arg = basis_name
                    xlab, ylab = basis_name + '_1', basis_name + '_2'
                else:  # msDM
                    key = 'X_ms' + basis_name
                    basis_arg = 'ms' + basis_name
                    xlab, ylab = 'ms' + basis_name + '_1', 'ms' + basis_name + '_2'
                

            Y = adata.obsm.get(key, None)
            if (Y is None) or (not isinstance(Y, np.ndarray)) or (Y.ndim != 2) or (Y.shape[1] < 2):
                ax.axis('off')
                ax.text(0.5, 0.5, f"{basis_name} ({variant}) unavailable", ha='center', va='center')
                return

            # Draw with scanpy (may toggle axis off)
            sc.pl.embedding(
                adata,
                basis=basis_arg,
                color=color,
                cmap=cmap,
                legend_loc=legend_loc,
                show=False,
                ax=ax,
                title=title,
                **kwargs
            )

            # Ensure axes are visible after scanpy call
            ax.set_axis_on()

            # Formatting: equal aspect, thin frame, hidden ticks, correct axis labels
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel(xlab, fontsize=9, labelpad=2)
            ax.set_ylabel(ylab, fontsize=9, labelpad=2)
            ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

            ax.set_frame_on(True)
            for side in ('left', 'right', 'top', 'bottom'):
                ax.spines[side].set_visible(True)
                ax.spines[side].set_linewidth(0.6)




        # Collect resolution keys and sort ascending; keep topo_clusters if no res-keys
        res_keys = [k for k in adata.obs.columns if k.startswith('topo_clusters_res')]
        pairs = []
        for k in res_keys:
            try:
                r = float(k.split('res', 1)[1])
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
            vals = []
            for k in keys:
                try:
                    vals.append(float(k.split('res', 1)[1]))
                except Exception:
                    vals.append(np.nan)
            idx = np.argsort(vals)
            ordered = [keys[i] for i in idx]
            lo = ordered[0]
            hi = ordered[-1]
            mid = ordered[len(ordered)//2]
            out = []
            for k in [lo, mid, hi]:
                if k not in out:
                    out.append(k)
            return out

        # Variant order (DM first, then msDM)
        variant_order = ['DM', 'msDM']

        # Pick a default deformation column produced by riemann_diagnostics
        def _pick_deformation_column():
            preferred = ['X_msTopoMAP', 'X_TopoMAP', 'X_msTopoPaCMAP', 'X_TopoPaCMAP']
            for k in preferred:
                col = f"metric_deformation__{k}"
                if col in adata.obs:
                    return col
            # fallback: any deformation column
            for c in adata.obs.columns:
                if c.startswith("metric_deformation__"):
                    return c
            return None

        deform_col = _pick_deformation_column()

        # Limits dictionary layout (per-embedding) from riemann_diagnostics
        def _get_limits_for(key_like):
            lims = adata.uns.get('metric_limits', {})
            if isinstance(lims, dict) and key_like in lims:
                return lims[key_like]
            # fallback: compute from deform_col if available
            if deform_col is not None:
                vals = adata.obs[deform_col].values
                return (np.nanmin(vals), np.nanmax(vals))
            return (None, None)

        with PdfPages(pdf_path) as pdf:
            # ===== topometry SUMMARY (stacked, no overlap) =====
            fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)

            # A single wide axis with page margins; we'll stack text blocks manually top→bottom.
            ax = fig.add_axes([0.06, 0.08, 0.88, 0.84])  # left, bottom, width, height
            ax.axis('off')

            # ----- Title -----
            fig.text(0.05, 0.95, "TopoMetry — summary", fontsize=16, weight='bold', va='center', ha='left')

            # Helpers
            def _fmt_num(x, nd=3):
                try:
                    return f"{float(x):.{nd}g}"
                except Exception:
                    return "n/a"

            def _safe(val, default="n/a"):
                return default if val is None else str(val)

            line_h_small  = 0.028   # normalized y step for small text
            line_h_medium = 0.032   # for section headings / slightly larger lines

            # Pull fitted stats
            ev_ms = _eigvals_from_tg(tg, variant='msDM')
            ev_dm = _eigvals_from_tg(tg, variant='DM') if (ev_ms is None or (hasattr(ev_ms, "size") and ev_ms.size == 0)) else None
            _evals = ev_ms if (ev_ms is not None and getattr(ev_ms, "size", 0) > 0) else (ev_dm if ev_dm is not None else np.array([]))
            sel_k  = tg.global_id if (hasattr(tg, 'global_id') and tg.global_id is not None) else tg.n_eigs if (hasattr(tg, 'n_eigs') and tg.n_eigs is not None) else None

            n_cells = adata.n_obs
            n_genes = adata.n_vars
            base_knn  = getattr(tg, 'base_knn', 'n/a')
            graph_knn = getattr(tg, 'graph_knn', 'n/a')
            n_eigs    = getattr(tg, 'n_eigs', 'n/a')
            base_metric  = getattr(tg, 'base_metric', 'n/a')
            graph_metric = getattr(tg, 'graph_metric', 'n/a')
            bk_ver = getattr(tg, 'base_kernel_version', 'n/a')
            gk_ver = getattr(tg, 'graph_kernel_version', 'n/a')

            # Graphs available
            graphs_available = []
            if 'topometry_connectivities' in adata.obsp:
                graphs_available.append("topometry_connectivities")
            if 'topometry_connectivities_ms' in adata.obsp:
                graphs_available.append("topometry_connectivities_ms (multiscale)")

            # Embeddings available
            embeddings_available = []
            for k in ('X_TopoMAP', 'X_msTopoMAP', 'X_TopoPaCMAP', 'X_msTopoPaCMAP'):
                if k in adata.obsm and adata.obsm.get(k) is not None:
                    embeddings_available.append(k.replace('X_', ''))

            # Cluster keys
            cluster_keys = []
            if 'topo_clusters' in adata.obs:
                cluster_keys.append('topo_clusters')
            cluster_keys += [c for c in adata.obs.columns if c.startswith('topo_clusters_res')]

            # Geometry preservation “best”
            best_geo_line = None
            try:
                df_eval = adata.uns.get('topometry_representation_eval', None)
                if df_eval is not None and not df_eval.empty:
                    best_idx = int(np.nanargmax(df_eval['TopoPreserve'].values))
                    best_rep = df_eval.iloc[best_idx]['representation']
                    best_geo_line = f"• Best geometry preservation (TopoPreserve): {best_rep}"
            except Exception:
                pass
            if best_geo_line is None:
                try:
                    gtbl = adata.uns.get('geometry_metrics_table', None)
                    if gtbl is not None and 'Composite' in gtbl:
                        best_rep = gtbl['Composite'].idxmax()
                        best_geo_line = f"• Best geometry preservation (composite): {best_rep}"
                except Exception:
                    pass

            # ---------- Stack content ----------
            y = 0.98  # start below title band (normalized figure coords inside our axis)

            # SECTION 1: Key ideas

            ax.text(0.0, y, "What TopoMetry does:", ha='left', va='top', fontsize=12, weight='bold')
            y -= line_h_small
            s1 = ("\n"
                "For a given dataset, the TopoMetry analysis:\n"
                "   1) builds neighborhood graphs,\n"
                "   2) refines their geometry with diffusion,\n"
                "   3) constructs a spectral scaffold capturing the data geometry (akin to diffusion maps),\n"
                "   4) learns a second, refined graph and its Laplacian operators,\n"
                "   5) uses the scaffold and refined graph to produce faithful 2-D views (TopoMAP / TopoPaCMAP) that"
                "      preserve global and local structure better than direct PCA/UMAP in many datasets.\n \n"
                "   • Quantifies distortions in low-dimensional representations and provides intuitive diagnostic plots.\n \n"
                "   • Extra tools: intrinsic dimensionality estimation, spectral selectivity (axes linked to biology),"
                " and graph-diffusion for pseudotime analysis, imputation, denoising and filtering."
            )
            ax.text(0.0, y, s1, ha='left', va='top', fontsize=11, linespacing=1.25, wrap=True)
            # advance y by a rough number of lines (count \n plus an estimate for wrapped lines)
            s1_lines = s1.count("\n") + 3  # small buffer for wrapped lines
            y -= s1_lines * line_h_small

            # add blank space between sections
            y -= 0.5 * line_h_small

            # SECTION 2: Fitted statistics
            ax.text(0.0, y, "Fitted statistics", ha='left', va='top', fontsize=12, weight='bold')
            y -= line_h_medium
            s2 = (
                f"• Cells × genes: {n_cells} × {n_genes}\n"
                f"• Global ID ({tg.id_method}): {int(tg.global_id)}\n"
                f"• spectral scaffold size: {int(tg.n_eigs)}\n"
                "\n"
                "Hyperparameters\n"
                f"• k-nearest neighbors for base graph: {base_knn} / metric: {base_metric} / kernel version: {_safe(bk_ver)}\n"
                f"• k-nearest neighbors for refined graph: {graph_knn} / metric: {graph_metric} / kernel version: {_safe(gk_ver)}\n"
            )
            ax.text(0.0, y, s2, ha='left', va='top', fontsize=10, linespacing=1.30, wrap=True)
            s2_lines = s2.count("\n") + 2
            y -= s2_lines * line_h_small

            # add blank space between sections
            y -= 0.5 * line_h_small

            # SECTION 3: 
            ax.text(0.0, y, "What you can use next:", ha='left', va='top', fontsize=12, weight='bold')
            y -= line_h_medium
            avail = []
            avail.append("• 2-D views: " + (", ".join(embeddings_available) if embeddings_available else "(none cached)") + " in adata.obsm")
            avail.append("• Graphs: " + (", ".join(graphs_available) if graphs_available else "(none cached)") + " in adata.obsp")
            if cluster_keys:
                avail.append("• Clustering results: " + ", ".join(cluster_keys[:8]) + (" ..." if len(cluster_keys) > 8 else "") + " in adata.obs")
            if best_geo_line:
                avail.append(best_geo_line)

            s3 = "\n".join(avail)
            ax.text(0.0, y, s3, ha='left', va='top', fontsize=10, linespacing=1.30, wrap=True)

            # (no extra tip box; Section 3 replaces the old footer tip as requested)

            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)
            
            # ===== PART 1: Geometry Preservation Benchmarks =====
            df_eval = adata.uns.get("topometry_representation_eval", None)

            fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
            ax = fig.add_axes([0.04, 0.12, 0.92, 0.80])
            ax.axis('off')

            # title = "Geometry Preservation Benchmarks"
            # fig.suptitle(title, fontsize=14, x=0.04, ha='left', y=0.99)
            fig.text(0.01, 0.98, "Geometry Preservation Benchmark", fontsize=14, weight='bold', va='center', ha='left')


            # --- Geometry-preservation explanatory legend ABOVE the table ---
            # Reserve a thin band near the top; keep it inside page margins.
            ax_top = fig.add_axes([0.04, 0.84, 0.92, 0.10])  # [left, bottom, width, height]
            ax_top.axis('off')
            legend_top = (
                "Geometry preservation compares the diffusion operator on the reference space, Pₓ (built on adata.X), "
                "to the operator induced by each representation, Pᵧ. \n When Pᵧ ≈ Pₓ, both global and local geometry are "
                "well preserved. TopoMetry's geometry preservation scores are:\n \n"
                "• PF1 — Sparse Neighborhood F1: overlap of the top-k transition supports per row; focuses on whether the same neighbors are kept in the operator (weights ignored).\n  \n"
                "• PJS — Row-wise Jensen–Shannon Similarity: compares the probability distributions of transitions for each cell; sensitive to how mass is redistributed.\n  \n"
                "• SP — Spectral Procrustes (R²): aligns multiscale diffusion coordinates (Φ_t) up to a rotation; captures coordinate-level consistency of the geometry.\n  \n"
                "Scores range from 0 to 1.0. Higher is better for all scores."
            )

            ax_top.text(0.0, 0.5, legend_top, ha='left', va='center', fontsize=11, wrap=True)
            if df_eval is None or (hasattr(df_eval, "empty") and df_eval.empty):
                ax.text(0.0, 0.9, "No evaluation results found.", fontsize=11, va='top')
            else:
                # ---- Normalize df into a tidy (reps × metrics) numeric table ----
                _df = df_eval.copy()

                # Two common shapes:
                # (A) wide-by-rep with metric rows → columns = representations, index = metric names
                # (B) long/tidy with 'representation' column and metric columns
                if "representation" in _df.columns:
                    # Keep only known metric columns; map friendly names with arrows later
                    metric_cols = [c for c in ["PF1","PJS","SP"]
                                if c in _df.columns]
                    reps = _df["representation"].astype(str).tolist()
                    M = _df.set_index("representation")[metric_cols]  # reps × metrics
                else:
                    # Assume rows are metric names; columns are representations
                    # Make reps × metrics
                    M = _df.T.copy()
                    # Try to keep a consistent metric order if present
                    metric_cols = [c for c in ["PF1","PJS","SP"]
                                if c in M.columns]

                # If some expected metrics are missing, keep whatever is there
                if not metric_cols:
                    metric_cols = list(M.columns)

                # Coerce to floats where possible
                for c in metric_cols:
                    M[c] = pd.to_numeric(M[c], errors="coerce")
                M = M[metric_cols]

                # Pretty row labels with arrows (↑/↓)
                row_labels_map = {
                    "PF1":             "PF1 (↑)",
                    "PJS":             "PJS (↑)",
                    "SP":              "SP (↑)",
                }
                display_rows = [row_labels_map.get(c, c) for c in metric_cols]

                # Determine best-per-column (bold the max for ↑, min for ↓)
                # Build a boolean mask same shape as table: best_mask[i_row, j_col]
                is_high_better = np.array([("(↓)" not in row_labels_map.get(c, "")) for c in metric_cols], dtype=bool)
                data_vals = M.values.astype(float)
                best_mask = np.zeros_like(data_vals, dtype=bool)
                for j in range(data_vals.shape[0]):  # iterate rows? — careful: M is reps × metrics
                    pass
                # Correct orientation: table shows reps as rows and metrics as columns.
                # data_vals shape = (n_reps, n_metrics). Iterate each metric (column).
                best_mask = np.zeros_like(data_vals, dtype=bool)
                for j in range(data_vals.shape[1]):
                    col = data_vals[:, j]
                    if np.all(~np.isfinite(col)):
                        continue
                    if is_high_better[j]:
                        target = np.nanmax(col)
                        winners = np.isclose(col, target, equal_nan=False)
                    else:
                        target = np.nanmin(col)
                        winners = np.isclose(col, target, equal_nan=False)
                    best_mask[:, j] = winners

                # Format numbers
                fmt_vals = np.empty_like(data_vals, dtype=object)
                for i in range(data_vals.shape[0]):
                    for j in range(data_vals.shape[1]):
                        v = data_vals[i, j]
                        fmt_vals[i, j] = f"{v:.3f}" if np.isfinite(v) else "n/a"

                # Build table (representations as rows; metrics as columns with arrows)
                reps = [str(r) for r in M.index.tolist()]
                col_headers = display_rows

                # Ensure column order and shorten long headers for display
                cols_full = ["representation", "PF1", "PJS", "SP"]
                # short_map = {
                #     "eigenvalue_w1": "eigW1",
                #     "subspace_cos":  "subspaceCos",
                # }
                present = [c for c in cols_full if c in df_eval.columns]
                show_df = df_eval.loc[:, present].copy()
                # show_df.columns = [short_map.get(c, c) for c in show_df.columns]

                # Pretty formatting for numeric columns in [0,1]
                num_cols = [c for c in show_df.columns if c != "representation"]
                for c in num_cols:
                    show_df[c] = show_df[c].map(lambda x: f"{x:0.3f}" if pd.notnull(x) else "n/a")

                # Draw table, forcing it to stay within page borders
                ax.set_position([0.02, 0.18, 0.98, 0.64])  # left, bottom, width, height — tighter band
                tbl = ax.table(
                    cellText=show_df.values,
                    colLabels=[f"$\\bf{{{h}}}$" for h in show_df.columns],  # bold headers
                    loc='center',
                    cellLoc='center',
                    colLoc='center',
                    colWidths=[1.0 / len(show_df.columns)] * len(show_df.columns),  # fit to axes width
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(8)  # slightly smaller to avoid clipping
                #tbl.scale(0.9, 1.15)  # leaner horizontally, modest row height

                # Light styling
                for (row, col), cell in tbl.get_celld().items():
                    if row == 0:
                        cell.set_facecolor('#f0f0f0')
                        cell.set_edgecolor('#cccccc')
                        cell.set_linewidth(0.8)
                    else:
                        cell.set_edgecolor('#dddddd')
                        cell.set_linewidth(0.5)
                try:
                    for j, col_name in enumerate(show_df.columns):
                        if col_name == "representation":
                            continue
                        # numeric values for this column
                        vals = pd.to_numeric(df_eval[present[j]], errors='coerce') if present[j] in df_eval.columns else pd.to_numeric(show_df[col_name], errors='coerce')
                        if vals.notna().any():
                            higher_is_better = {"PF1","PJS","SP"}
                            if col_name in higher_is_better:
                                idx = int(vals.idxmax())
                            else:
                                continue
                            # +1 for header row
                            cell = tbl[(idx + 1, j)]
                            cell.set_text_props(fontweight='bold')
                except Exception:
                    pass


                # Expanded legend (wrapped)
                ax2 = fig.add_axes([0.04, 0.05, 0.92, 0.09])  # taller band; keep inside margins
                ax2.axis('off')
                legend = (
                    "• PF1 — Sparse Neighborhood F1: overlap of the top-k transition supports per row; focuses on whether the same neighbors are kept in the operator (weights ignored).\n"
                    "\n• PJS — Row-wise Jensen–Shannon Similarity: compares the probability distributions of transitions for each cell; sensitive to how mass is redistributed.\n"
                    "\n• SP — Spectral Procrustes (R²): aligns multiscale diffusion coordinates (Φ_t) up to a rotation; captures coordinate-level consistency of the geometry.\n"
                )
                ax2.text(0.0, 0.5, legend, va='center', fontsize=11, wrap=True)

            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)

            # ===== PART 2 : EIGENSPECTRUM / I.D. (1×4) =====
            fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
            gs = fig.add_gridspec(2, 2, left=0.04, right=0.98, top=0.92, bottom=0.12, wspace=0.28, hspace=0.35)
            # Row 1: spectrum & diff
            ax_curve = fig.add_subplot(gs[0, 0]); ax_diff = fig.add_subplot(gs[0, 1])
            evals_ms = _eigvals_from_tg(tg, variant='msDM')
            _decay_plot_axes_original(ax_curve, ax_diff, evals_ms, title="Eigenspectrum & Eigengap")

            # ID histograms (original style)
            ax_fsa = fig.add_subplot(gs[1, 0]); ax_mle = fig.add_subplot(gs[1, 1])
            id_est = adata.uns.get('intrinsic_dim_estimator', None)
            plot_id_histograms(ax_fsa, ax_mle, id_est)

            #fig.suptitle("Scaffold eigenspectrum / intrinsic dimensionality", y=0.98, fontsize=14)
            fig.text(0.02, 0.98, "Scaffold eigenspectrum and intrinsic dimensionality estimates", fontsize=16, weight='bold', va='center', ha='left')
            pdf.savefig(fig, dpi=dpi); plt.close(fig)

            # ===== PART 3 : CLUSTERING (1x3/2×3 grid) =====
            show_keys = _pick_three(res_keys)
            ncols = max(1, len(show_keys))
            for variant in variant_order:
                fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
                gs = fig.add_gridspec(2, ncols, left=0.04, right=0.98, top=0.92, bottom=0.12, wspace=0.15, hspace=0.20)
                # Row 1: TopoMAP
                for j, k in enumerate(show_keys):
                    ax = fig.add_subplot(gs[0, j])
                    _embedding(ax, k, basis_name='TopoMAP', variant=variant, title=f"{k}", legend_loc='on data', legend_fontsize=7, legend_fontoutline=2)
                # Row 2: TopoPaCMAP
                for j, k in enumerate(show_keys):
                    ax = fig.add_subplot(gs[1, j])
                    _embedding(ax, k, basis_name='TopoPaCMAP', variant=variant, title=f"{k}", legend_loc='on data', legend_fontsize=7, legend_fontoutline=2)
                #fig.suptitle(f"Clustering across resolutions — {variant}", y=0.98, fontsize=14)
                fig.text(0.02, 0.98, f"Clustering across resolutions — {variant}", fontsize=14, weight='bold', va='center', ha='left')
                pdf.savefig(fig, dpi=dpi); plt.close(fig)


            # ===== PART 4 : RIEMANN DIAGNOSTICS: one page per embedding (1×3 + bottom text) =====
            # Ensure we're in non-interactive mode during report build
            plt.ioff()

            # Pick a label key for coloring inside plot_riemann_diagnostics (if present)
            rk_lab_key = None
            for k in [labels_key_for_page_titles, 'topo_clusters', 'cell_type', 'leiden']:
                if k and (k in adata.obs):
                    rk_lab_key = k
                    break

            # Canonical order of embeddings; render a page for each one that exists and is 2-D
            _riem_order = ['X_TopoMAP', 'X_msTopoMAP', 'X_TopoPaCMAP', 'X_msTopoPaCMAP']
            for _obsm in _riem_order:
                if _obsm not in adata.obsm:
                    continue
                try:
                    arr = np.asarray(adata.obsm[_obsm])
                    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] != adata.n_obs:
                        continue
                except Exception:
                    continue

                # Let the helper build the 1×3 panel figure for this embedding.
                # NOTE: plot_riemann_diagnostics should NOT close the figure it creates.
                _ret = plot_riemann_diagnostics(
                    adata,
                    tg,
                    proj_key=_obsm,
                    groupby=(rk_lab_key or 'topo_clusters'),
                    do_all=False,
                    verbose=False,
                    show=False,
                    title_fontsize=16
                )
                # Get the figure object (supports versions that return fig or not)
                fig = _ret if hasattr(_ret, 'savefig') else plt.gcf()
                try:
                    # 1) Fit the page size
                    fig.set_size_inches(a4_landscape_inches)

                    # 2) Reserve margins so subplots don’t overcrop each other
                    #    Leave ~18% at bottom for the explanatory text; modest top margin for suptitle.
                    fig.tight_layout(rect=[0.04, 0.22, 0.98, 0.92])

                    # 3) Tame the suptitle size/position if present (created by plot_riemann_diagnostics)
                    # if getattr(fig, "_suptitle", None) is not None:
                    #     fig._suptitle.set_fontsize(14)
                    #     fig._suptitle.set_y(0.96)
                    fig.text(0.02, 0.98, f"Riemannian diagnostics - {_obsm[2:]}", fontsize=14, weight='bold', va='center', ha='left')


                    # 4) Explanatory text: place it at the bottom band (use fig.text to avoid axes overlap)
                    guide = (
                        "How to read these Riemann-metric panels:\n"
                        "• Localized indicatrices — Each ellipse summarizes how distances are stretched or squashed around a point. "
                        "Long ellipses indicate a preferred direction of variation (anisotropy); small round ellipses indicate locally "
                        "uniform structure. Comparing anisotropy across clusters can reveal biological trends.\n"
                        "• Global indicatrices (overlay) — A coarse grid of ellipses shows the overall deformation field on the embedding. "
                        "The background encodes the centered log-determinant of the metric (blue = contraction, red = expansion). "
                        "Consistent colors/ellipses often align with transitions (e.g., differentiation) or boundaries between states.\n"
                        "• Local contraction/expansion — Points colored by the same deformation score highlight compressed (blue) or dilated (red) regions, "
                        "useful for spotting bottlenecks, hubs, or spread-out manifolds in the cellular landscape."
                    )
                    # left=0.04, baseline ~0.12 from bottom; anchor at top of band
                    fig.text(0.08, 0.18, guide, ha='left', va='top', fontsize=11, wrap=True)

                    ax_exp = fig.add_axes([0.035, 0.07, 0.93, 0.16])
                    ax_exp.axis('off')
                    pdf.savefig(fig, dpi=dpi)
                finally:
                    plt.close(fig)

            # ===== PART 5 : I.D. EMBEDDINGS (2×4 grid) =====
            # Row 1: TopoMAPs colored by ID low/high for FSA and MLE (if present)
            def _find_id_keys(prefix: str):
                ks = [c for c in adata.obs.columns if c.startswith(prefix)]
                if not ks: return (None, None)
                nums = []
                for c in ks:
                    try: nums.append(int(c.split('k',1)[1]))
                    except Exception: nums.append(9999)
                order = np.argsort(nums)
                low = ks[order[0]]
                high = ks[order[-1]]
                return (low, high) if low != high else (low, None)
            
            for variant in variant_order:
                fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
                gs = fig.add_gridspec(2, 4, left=0.04, right=0.98, top=0.92, bottom=0.12, wspace=0.28, hspace=0.35)

                fsa_low_key, fsa_high_key = _find_id_keys('id_fsa_k')
                mle_low_key, mle_high_key = _find_id_keys('id_mle_k')

                ax = fig.add_subplot(gs[0, 0]); _embedding(ax, fsa_low_key, 'TopoMAP', variant, title=f'FSA id ({fsa_low_key})', colorbar_loc=None, cmap='Reds')
                ax = fig.add_subplot(gs[0, 1]); _embedding(ax, fsa_high_key, 'TopoMAP', variant, title=f'FSA id ({fsa_high_key})', colorbar_loc=None, cmap='Reds')
                ax = fig.add_subplot(gs[0, 2]); _embedding(ax, mle_low_key, 'TopoMAP', variant, title=f'MLE id ({mle_low_key})', colorbar_loc=None, cmap='Reds')
                ax = fig.add_subplot(gs[0, 3]); _embedding(ax, mle_high_key, 'TopoMAP', variant, title=f'MLE id ({mle_high_key})', colorbar_loc=None, cmap='Reds')

                ax = fig.add_subplot(gs[1, 0]); _embedding(ax, fsa_low_key, 'TopoPaCMAP', variant, title=f'FSA id ({fsa_low_key})', colorbar_loc=None, cmap='Reds')
                ax = fig.add_subplot(gs[1, 1]); _embedding(ax, fsa_high_key, 'TopoPaCMAP', variant, title=f'FSA id ({fsa_high_key})', colorbar_loc=None, cmap='Reds')
                ax = fig.add_subplot(gs[1, 2]); _embedding(ax, mle_low_key, 'TopoPaCMAP', variant, title=f'MLE id ({mle_low_key})', colorbar_loc=None, cmap='Reds')
                ax = fig.add_subplot(gs[1, 3]); _embedding(ax, mle_high_key, 'TopoPaCMAP', variant, title=f'MLE id ({mle_high_key})', colorbar_loc=None, cmap='Reds')

                #fig.suptitle("Eigenspectrum and intrinsic dimensionality", y=0.98, fontsize=12)
                fig.text(0.02, 0.98, "Intrinsic dimensionality across cells", fontsize=14, weight='bold', va='center', ha='left')
                plt.tight_layout()
                pdf.savefig(fig, dpi=dpi); plt.close(fig)

            # ===== PART 6: SPECTRAL SELECTIVITY (2×4) =====
            for variant in variant_order:
                fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
                # Leave a bit more bottom margin to fit the legend band.
                gs = fig.add_gridspec(
                    2, 4,
                    left=0.04, right=0.98, top=0.92, bottom=0.14,
                    wspace=0.25, hspace=0.30
                )

                # Row 1 • TopoMAP
                ax = fig.add_subplot(gs[0, 0]); _embedding(ax, 'spectral_EAS',      'TopoMAP',    variant, title='EAS',      cmap='Reds', colorbar_loc=None)
                ax = fig.add_subplot(gs[0, 1]); _embedding(ax, 'spectral_RayScore', 'TopoMAP',    variant, title='RayScore', cmap='Reds', colorbar_loc=None)
                ax = fig.add_subplot(gs[0, 2]); _embedding(ax, 'spectral_LAC',      'TopoMAP',    variant, title='LAC',      cmap='Reds', colorbar_loc=None)
                ax = fig.add_subplot(gs[0, 3]); _embedding(ax, 'spectral_radius',   'TopoMAP',    variant, title='Radius',   cmap='Reds', colorbar_loc=None)

                # Row 2 • TopoPaCMAP
                ax = fig.add_subplot(gs[1, 0]); _embedding(ax, 'spectral_EAS',      'TopoPaCMAP', variant, title='EAS',      cmap='Reds', colorbar_loc=None)
                ax = fig.add_subplot(gs[1, 1]); _embedding(ax, 'spectral_RayScore', 'TopoPaCMAP', variant, title='RayScore', cmap='Reds', colorbar_loc=None)
                ax = fig.add_subplot(gs[1, 2]); _embedding(ax, 'spectral_LAC',      'TopoPaCMAP', variant, title='LAC',      cmap='Reds', colorbar_loc=None)
                ax = fig.add_subplot(gs[1, 3]); _embedding(ax, 'spectral_radius',   'TopoPaCMAP', variant, title='Radius',   cmap='Reds', colorbar_loc=None)

                # Bottom legend band (wrapped text; small font; sits under the grid)
                leg_ax = fig.add_axes([0.04, 0.045, 0.92, 0.06])  # x, y, w, h (fractions of figure)
                leg_ax.axis('off')
                legend_text = (
                    "EAS (Entropy-based Axis Selectivity): in [0,1]; higher means each cell’s energy is concentrated on a single spectral axis. "
                    "Computed from squared, standardized scaffold coordinates with eigenvalue weights (default λ/(1−λ)).\n"
                    "RayScore: highlights coherent radial progressions along a dominant axis; defined as sigmoid(neighborhood radial z-score) × EAS. "
                    "Large values indicate ray-like trajectories pointing outward; axis sign is stored separately.\n"
                    "LAC (Local Axial Coherence): fraction of local variance explained by the first principal component (EVR₁) within k-NN; "
                    "near 1.0 indicates locally 1-D structure aligned with a single axis.\n"
                    "Radius (spectral radius): Euclidean norm of standardized scaffold coordinates (‖Z‖₂); a proxy for distance from the origin in spectral space, "
                    "often correlating with progress along diffusion time."
                )
                leg_ax.text(0.0, 0.5, legend_text, ha='left', va='center', fontsize=11, wrap=True)

                #fig.suptitle(f"Spectral selectivity — {variant}", y=0.98, fontsize=12)
                if variant == 'msDM':
                    variant_title = '(multiscale)'
                else:
                    variant_title = ''
                fig.text(0.02, 0.98, f"Spectral selectivity {variant_title}", fontsize=14, weight='bold', va='center', ha='left')

                pdf.savefig(fig, dpi=dpi); plt.close(fig)


            # ===== PART 7: IMPUTATION =====
            fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
            fig.text(0.02, 0.98, f"Imputation: example gene, QC and statistics", fontsize=14, weight='bold', va='center', ha='left')
            gs = fig.add_gridspec(2, 3, left=0.04, right=0.98, top=0.88, bottom=0.18, wspace=0.28, hspace=0.38)
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

            iqc = adata.uns.get("imputation_qc", {})
            corr_raw = iqc.get("corr_raw", None)
            corr_imp = iqc.get("corr_imp_best", None)
            genes = iqc.get("heatmap_genes", None)

            def _plot_heatmap(ax, C, title):
                ax.set_title(title, fontsize=10)
                if C is None:
                    ax.axis('off'); ax.text(0.5, 0.5, "N/A", ha='center', va='center'); return
                im = ax.imshow(C, vmin=-1, vmax=1, cmap='coolwarm', interpolation='nearest', aspect='auto')
                ax.tick_params(axis='x', labelsize=6); ax.tick_params(axis='y', labelsize=6)
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
                ax.tick_params(axis='y', labelsize=0)
                ax.legend(frameon=False, fontsize=8)
            else:
                ax.axis('off'); ax.text(0.5, 0.5, "No QC stats", ha='center', va='center')

            ax_exp = fig.add_axes([0.06, 0.06, 0.88, 0.06]); ax_exp.axis('off')
            ax_exp.text(0.0, 0.5,
                        "Imputation uses diffusion (P^t) on the refined TopoMetry graph to denoise expression. \n"
                        "QC compares mean absolute gene–gene correlations against null (per-gene permutations) across t. \n"
                        "Best t minimizes the empirical null p-value (ties broken by max z-score).",
                        fontsize=11, va='center')
            pdf.savefig(fig, dpi=dpi); plt.close(fig)

            for tmp in ['_gene_raw','_gene_imputed']:
                if tmp in adata.obs: del adata.obs[tmp]

            # ===== PART 8: SIMULATED DISEASE-STATE FILTERING =====
            keys_for_signals = list(signal_plot_keys) if signal_plot_keys else []
            if not keys_for_signals:
                rng = np.random.default_rng(7)
                cluster_key = None
                keys = ['topo_clusters'] + [c for c in adata.obs.columns if c.startswith('topo_clusters_res')]
                if len(keys)>1:
                    if keys[-1] in adata.obs:
                        cluster_key = keys[-1]
                else:
                    for k in keys:
                        if k in adata.obs:
                            cluster_key = k
                            break
                if cluster_key is None:
                    adata.obs['_all'] = pd.Categorical(['all'] * adata.n_obs)
                    cluster_key = '_all'
                labels = adata.obs[cluster_key].astype('category')
                cats = list(labels.cat.categories)
                pick = rng.choice(cats, size=min(5, len(cats)), replace=False)
                mask = np.zeros(adata.n_obs, dtype=bool)
                for c in pick:
                    idx = np.where(labels.values == c)[0]
                    if idx.size == 0: continue
                    ksel = int(round(0.1 * idx.size))
                    chosen = rng.choice(idx, size=max(1, min(ksel, idx.size)), replace=False)
                    mask[chosen] = True
                sim_key = "_simulated_state_for_example"
                adata.obs[sim_key] = pd.Categorical(np.where(mask, 'diseased', 'healthy'))
                keys_for_signals = [sim_key]
                _cleanup_sim_key_after = sim_key
            else:
                _cleanup_sim_key_after = None

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
                gs = fig.add_gridspec(1, 3, left=0.04, right=0.98, top=0.95, bottom=0.10, wspace=0.1, hspace=0.35)
                ax = fig.add_subplot(gs[0, 0]); _embedding(ax, sig_key,       'TopoMAP', 'DM', title='Simulated disease state', legend_loc=None)
                ax = fig.add_subplot(gs[0, 1]); _embedding(ax, '_gf_cat_raw', 'TopoMAP', 'DM', title='Noisy categorical readout', cmap='coolwarm', colorbar_loc=None)
                ax = fig.add_subplot(gs[0, 2]); _embedding(ax, '_gf_cat_flt', 'TopoMAP', 'DM', title='Graph-filtered readout',   cmap='coolwarm', colorbar_loc=None)
                plt.tight_layout()
                fig.text(0.02, 0.98, f"Simulated disease-state filtering (example)", fontsize=14, weight='bold', va='center', ha='left')
                #fig.suptitle(f"Simulated disease-state filtering")
                pdf.savefig(fig, dpi=dpi); plt.close(fig)

                for tmp in ['_gf_cat_raw','_gf_cat_flt','_gf_rand_raw','_gf_rand_flt']:
                    if tmp in adata.obs: del adata.obs[tmp]

            if _cleanup_sim_key_after and (_cleanup_sim_key_after in adata.obs):
                del adata.obs[_cleanup_sim_key_after]

            # ===== PART 9: PURE NOISE FILTERING CONTROL =====
            P = tg.P_of_msZ
            evecs, evals = None, None
            try:
                key_e = 'msDM with ' + tg.base_kernel_version
                evecs, evals = tg.EigenbasisDict[key_e].results(return_evals=True)
            except Exception:
                pass

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

            spec_raw = np.array([]); spec_flt = np.array([])
            gtv_raw = np.nan; gtv_flt = np.nan; spec_energy_raw = np.nan; spec_energy_flt = np.nan
            try:
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
            fig.text(0.02, 0.98, f"Simulated disease-state filtering (controls)", fontsize=14, weight='bold', va='center', ha='left')
            gs = fig.add_gridspec(2, 3, height_ratios=[4.0, 1.4],
                                left=0.04, right=0.98, top=0.88, bottom=0.10, wspace=0.25, hspace=0.25)
            ax0 = fig.add_subplot(gs[0, 0]); _embedding(ax0, '_gf_null_mean', 'TopoMAP', 'DM', title='Filtered pure-noise: mean', cmap='coolwarm')
            ax1 = fig.add_subplot(gs[0, 1]); _embedding(ax1, '_gf_null_std',  'TopoMAP', 'DM', title='Filtered pure-noise: std',  cmap='viridis')
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
                "• The filtered pure-noise mean/std visualizes diffusion smoothing under a null model.\n"
                "• Graph Total Variation (GTV) decreases after filtering (smoother signals).\n"
                "• Spectral energy shifts towards low-frequency modes after diffusion.\n"
                f"GTV raw: {_fmt(gtv_raw)} | GTV filtered: {_fmt(gtv_flt)} | Δ: {_fmt(gtv_raw - gtv_flt)} | "
                f"Spectral energy (low+mid): raw={_fmt(spec_energy_raw)}, filtered={_fmt(spec_energy_flt)}"
            )
            axt.text(0.0, 1.0, txt, fontsize=11, va='top')
            pdf.savefig(fig, dpi=dpi); plt.close(fig)

            for tmp in ['_gf_null_mean','_gf_null_std']:
                if tmp in adata.obs: del adata.obs[tmp]

            # ===== PART 10: PSEUDOTIME =====
            # res = _componentwise_pseudotime_colors(
            #     adata, tg, cluster_key=lab_key if lab_key else 'topo_clusters'
            # )
            # fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
            # ax = fig.add_axes([0.06, 0.10, 0.88, 0.82])
            # Yb = adata.obsm.get('X_msTopoMAP', None)
            # if Yb is None:
            #     Yb = adata.obsm.get('X_TopoMAP', None)
            # if res is not None and Yb is not None:
            #     pt_key, pt_color_key, n_comp = res
            #     cols = adata.obs[pt_color_key].astype(str).values
            #     ax.scatter(Yb[:, 0], Yb[:, 1], s=6, c=cols, linewidths=0, alpha=0.95)
            #     ax.set_title(f"Pseudotime within components (n={n_comp})", fontsize=12)
            #     ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
            # else:
            #     ax.axis('off'); ax.text(0.5, 0.5, "Pseudotime colors unavailable", ha='center', va='center')
            # pdf.savefig(fig, dpi=dpi); plt.close(fig)






        return pdf_path


    # ----------------------------------------------------
    # Convenience wrapper: run + plot in one call
    # ----------------------------------------------------

    def run_and_report(
        adata: AnnData,
        tg: TopOGraph | None = None,
        *,
        # --- Pass-through analysis knobs (same defaults as run_topometry_analysis) ---
        base_knn: int = 30,
        graph_knn: int = 30,
        min_eigs: int = 100,
        projections: tuple[str, ...] = ("MAP", "PaCMAP"),
        base_metric: str = "cosine",
        graph_metric: str = "euclidean",
        graph_kernel_version: str = "bw_adaptive",
        diff_t: int = 1,
        n_jobs: int = -1,
        verbosity: int = 0,
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
        # --- Report / I/O ---
        output_dir: str = ".",
        filename: str = "topometry_report.pdf",
        dpi: int = 300,
        a4_landscape_inches: tuple[float, float] = (11.69, 8.27),
        gene_for_imputation: str | None = None,
        labels_key_for_page_titles: str | None = None,
        # >>> NEW: graph filtering report controls <<<
        signal_plot_keys: list[str] | None = None,
    ):
        """
        Run the full TopOMetry analysis pipeline and emit a consolidated PDF report.

        This is a convenience wrapper around `run_topometry_analysis(...)` followed by
        `plot_topometry_report(...)`, forwarding the relevant options to each stage.

        Parameters
        ----------
        adata : AnnData
            Input dataset; will be populated with TopOMetry outputs.
        tg : TopOGraph or None, default None
            Existing model to reuse or None to fit anew.
        ... (see arguments; pass-through to analysis / report steps)

        Returns
        -------
        tg : TopOGraph
            The fitted/reused model used in the analysis.
        pdf_path : str
            Path to the generated PDF report.
        """

        tg = run_topometry_analysis(
            adata,
            tg,
            base_knn=base_knn,
            graph_knn=graph_knn,
            min_eigs=min_eigs,
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
            projections=projections,
            leiden_key_base=leiden_key_base,
            leiden_resolutions=leiden_resolutions,
            leiden_primary_index=leiden_primary_index,
            spec_weight_mode=spec_weight_mode,
            spec_k_neighbors=spec_k_neighbors,
            spec_smooth_P=spec_smooth_P,
            spec_smooth_t=spec_smooth_t,
            riem_center=riem_center,
            riem_diffusion_t=riem_diffusion_t,
            riem_diffusion_op=riem_diffusion_op,
            riem_normalize=riem_normalize,
            riem_clip_percentile=riem_clip_percentile,
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
# Old functions kept for backwards compatibility
# ----------------------------------------------------


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