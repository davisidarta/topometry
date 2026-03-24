# Wrapper functions for single-cell analysis with Scanpy and TopoMetry
# Author: David S Oliveira <david.oliveira(at)dpag(dot)ox(dot)ac(dot)uk>
# All of these functions call scanpy and thus require it for working
# However, I opted not to include it as a hard-dependency as not all users are interested in single-cell analysis
#
from __future__ import annotations
from typing import Any, Dict, Optional
import os
import numpy as np
import pandas as pd
import time
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.sparse import issparse, csr_matrix
from scipy.stats import gaussian_kde
from scipy.sparse.csgraph import connected_components
import colorsys
from matplotlib.backends.backend_pdf import PdfPages
import textwrap
from matplotlib.collections import PathCollection
import matplotlib.patheffects as patheffects
from adjustText import adjust_text

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

    def plot_id_histograms(adata, ax_fsa=None, ax_mle=None, id_est=None, figsize=(10, 4), dpi=100):
        if ax_fsa is None or ax_mle is None:
            fig, (ax_fsa, ax_mle) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        # accept either an IntrinsicDim object or a {'local_id':..., 'global_id':...} dict
        if id_est is None:
            id_est = adata.uns.get('intrinsic_dim_estimator', None)
        if id_est is None:
            local = {}; global_ = {}
        elif isinstance(id_est, dict):
            local  = id_est.get("local_id", {}) or {}
            global_ = id_est.get("global_id", {}) or {}
        else:
            local  = getattr(id_est, "local_id", {}) or {}
            global_ = getattr(id_est, "global_id", {}) or {}

        def _one(ax, method_name: str):
            if (method_name not in local) or (len(local[method_name]) == 0):
                ax.axis('off'); ax.text(0.5,0.5,"N/A", ha='center', va='center'); return
            for key in local[method_name].keys():
                x = local[method_name][key]
                gi = None
                try:
                    gi = int((global_.get(method_name, {}) or {}).get(key))
                except Exception:
                    pass
                label = f'k = {key}    ( estim.i.d. = {gi if gi is not None else "n/a"} )'
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

    def _h5ad_safe(x):
        """Recursively convert Python containers to AnnData/HDF5-writable types."""
        import numpy as np, pandas as pd
        from collections.abc import Mapping

        # pass through common encodable types
        if isinstance(x, (pd.DataFrame, np.ndarray, str, float, int, bool)) or x is None:
            return x
        # numpy scalars -> python
        if isinstance(x, (np.floating, np.integer, np.bool_)):
            return x.item()
        # dict-like: sanitize values and keys
        if isinstance(x, Mapping):
            return {str(k): _h5ad_safe(v) for k, v in x.items()}
        # lists/tuples/sets -> lists
        if isinstance(x, (list, tuple, set)):
            return [_h5ad_safe(v) for v in x]
        # last resort: string representation (avoids crashing the writer)
        return str(x)


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
        - Text annotation color in each cell flips based on the cell's value
        relative to half the column's max to keep labels readable.
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
            M, reps, cols, ax=ax, cmap="viridis", cbarlabel="Score (0-1)"
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
            print(f"[plot] Saved heatmap -> {out_png}")
        plt.close(fig)


    # =========================
    # STEPWISE API (new)
    # =========================

    def filter_cells(
        adata,
        qc_metrics,
        percentile=5,
    ):
        """
        Filter cells by trimming extremes of QC metrics in `AnnData.obs`.

        For each metric in `qc_metrics`, cells in the bottom `percentile` and top
        `percentile` (by that metric) are removed. The final keep mask is the
        intersection across metrics (a cell must pass all metrics to be kept).

        Parameters
        ----------
        AnnData : anndata.AnnData
            Input annotated matrix.
        qc_metrics : list
            List of keys in `AnnData.obs` to use for filtering.
        percentile : float, default 5
            Percentile cutoff on each tail. Must be in (0, 50).

        Returns
        -------
        anndata.AnnData
            Filtered copy of the input AnnData.
        """
        if not isinstance(adata, AnnData):
            raise TypeError("adata must be an AnnData object.")
        qc_metrics = list(qc_metrics) if qc_metrics is not None else []
        p = float(percentile)
        if p <= 0.0 or p >= 50.0:
            raise ValueError("percentile must be in (0, 50).")
        if len(qc_metrics) == 0:
            return adata.copy()

        keep = np.ones(adata.n_obs, dtype=bool)

        for key in qc_metrics:
            if key not in adata.obs:
                raise KeyError("QC metric not found in adata.obs: %r" % key)

            x = adata.obs[key]

            if pd.api.types.is_numeric_dtype(x) or pd.api.types.is_bool_dtype(x):
                v = np.asarray(x, dtype=float)
            else:
                # Robust conversion for categorical/object/string
                if pd.api.types.is_categorical_dtype(x):
                    codes = x.cat.codes.to_numpy(dtype=float)
                else:
                    codes = pd.Categorical(x).codes.astype(float)
                codes[codes < 0] = np.nan
                v = codes

            lo = np.nanpercentile(v, p)
            hi = np.nanpercentile(v, 100.0 - p)

            keep &= (v > lo) & (v < hi)

        return adata[keep].copy()



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
        Standardize an AnnData matrix with Scanpy-style HVG selection and scaling.

        This prepares `.X` for TopoMetry's scaffold construction: counts are
        normalized and log-transformed, highly-variable genes are selected, and
        scaled values are written back to `.X` (also stored in `.layers['scaled']`).

        Parameters
        ----------
        AnnData : anndata.AnnData
            Input annotated matrix; `.X` is treated as counts/UMIs on entry.
        normalize : bool, default True
            Whether to library-size normalize `.X` with `sc.pp.normalize_total`.
        log : bool, default True
            Apply `sc.pp.log1p` to `.X` after normalization.
        target_sum : float, default 1e4
            Target total counts per cell for `normalize_total`.
        min_mean : float, default 0.0125
            Lower bound for mean expression when selecting HVGs.
        max_mean : float, default 3.0
            Upper bound for mean expression when selecting HVGs.
        min_disp : float, default 0.5
            Lower bound for dispersion when selecting HVGs.
        max_value : float, default 10
            Clip value passed to `sc.pp.scale` when `scale` is True.
        save_to_raw : bool, default True
            If True, snapshot the full matrix to `.raw` before HVG subsetting.
        plot_hvg : bool, default False
            If True, plot HVG diagnostics via `sc.pl.highly_variable_genes`.
        scale : bool, default True
            If True, writes a dense copy to `.layers['scaled']`, scales it, and
            mirrors the scaled values into `.X`.
        n_top_genes : int, default 3000
            Number of HVGs to retain.
        flavor : {"seurat", "cell_ranger", "seurat_v3"}, default "seurat_v3"
            Scanpy HVG flavor.
        **kwargs
            Forwarded to `sc.pp.highly_variable_genes`.

        Returns
        -------
        anndata.AnnData
            Copy of the input, subset to HVGs with normalized/logged/scaled
            values in `.X` and helper layers populated.

        Notes
        -----
        - Adds `.layers["counts"]` before normalization for reproducibility.
        - Does not modify the original object in-place; a copy is returned.

        Examples
        --------
        >>> import numpy as np, anndata as ad, topo as tp
        >>> adata = ad.AnnData(np.abs(np.random.randn(5, 3)))
        >>> processed = tp.sc.preprocess(adata, n_top_genes=2, save_to_raw=False)
        >>> processed.X.shape
        (5, 2)
        """

        # 1) Keep a copy of raw counts before any normalization/log
        AnnData.layers["counts"] = AnnData.X.copy()

        # 2) Library-size normalization (on X)
        if normalize:
            sc.pp.normalize_total(AnnData, target_sum=target_sum)

        # 3) Log1p transform (on X)
        if log:
            sc.pp.log1p(AnnData)

        # 4) HVGs on raw counts layer (Seurat v3-style)
        sc.pp.highly_variable_genes(
            AnnData,
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
                sc.pl.highly_variable_genes(AnnData, layer="counts")
            except TypeError:
                # older Scanpy versions may not support layer= in the plot call
                sc.pl.highly_variable_genes(AnnData)

        # 5) Save full matrix snapshot to .raw (optional)
        if save_to_raw:
            AnnData.raw = AnnData.copy()

        # 6) Subset to HVGs
        AnnData = AnnData[:, AnnData.var.highly_variable].copy()

        # 7) Scale into .layers['scaled'] then make X == scaled
        if scale:
            # stash a dense copy into 'scaled'
            Xdense = AnnData.X.toarray() if sp.issparse(AnnData.X) else np.asarray(AnnData.X)
            AnnData.layers["scaled"] = Xdense.copy()

            # Some Scanpy versions don't support `layer=` in scale; handle both.
            try:
                sc.pp.scale(AnnData, layer="scaled", max_value=max_value)
                # ensure X follows the scaled layer
                AnnData.X = AnnData.layers["scaled"]
            except TypeError:
                # fallback: temporarily operate on X
                AnnData.X = AnnData.layers["scaled"].copy()
                sc.pp.scale(AnnData, max_value=max_value)
                AnnData.layers["scaled"] = AnnData.X.copy()

        else:
            # even if not scaling, keep X consistent with the (possibly log-normalized) values
            pass

        return AnnData.copy()


    def fit_adata(
        adata: AnnData,
        tg: TopOGraph | None = None,
        *,
        projections: tuple[str, ...] = ("MAP", "PaCMAP"),
        do_leiden: bool = True,
        leiden_key_base: str = "topo_clusters",
        leiden_resolutions: list[float] | tuple[float, ...] = (0.2,0.8),
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
            # size mismatch -> refit
            elif (getattr(tg, "n", None) != adata.n_obs) or (getattr(tg, "m", None) != adata.n_vars):
                need_refit = True

        if need_refit:
            tg.fit(adata.X)

        # Normalise leiden_resolutions: accept a bare scalar (e.g. 0.4) or any iterable.
        if isinstance(leiden_resolutions, (int, float)):
            leiden_resolutions = (float(leiden_resolutions),)
        else:
            leiden_resolutions = tuple(leiden_resolutions)
        # Clamp primary index so a single-resolution call never raises IndexError.
        leiden_primary_index = min(leiden_primary_index, len(leiden_resolutions) - 1)

        # (1) store scaffolds (TopOGraph.spectral_scaffold already returns UoM aggregates if enabled)
        adata.obsm["X_ms_spectral_scaffold"] = tg.spectral_scaffold(multiscale=True)[:, :getattr(tg, "_scaffold_components_ms", tg.n_eigs)]
        adata.obsm["X_spectral_scaffold"]    = tg.spectral_scaffold(multiscale=False)[:, :getattr(tg, "_scaffold_components_dm", tg.n_eigs)]

        # (2) store projections if requested/available (unchanged loop; properties now resolve UoM)
        if projections is not None:
            if isinstance(projections, str):
                projections = (projections,)
            pretty_map = {"MAP": "TopoMAP", "PaCMAP": "TopoPaCMAP"}
            prop_map = {"MAP": {False: "TopoMAP", True: "msTopoMAP"},
                        "PaCMAP": {False: "TopoPaCMAP", True: "msTopoPaCMAP"}}

            def _get_projection_if_available(tg, method: str, multiscale: bool):
                prop = prop_map[method][multiscale]
                try:
                    return getattr(tg, prop)
                except AttributeError:
                    return None

            def _ensure_projection(tg, method: str, multiscale: bool):
                Y = _get_projection_if_available(tg, method, multiscale)
                if Y is None:
                    tg.project(projection_method=method, multiscale=multiscale)
                    Y = _get_projection_if_available(tg, method, multiscale)
                return Y

            for proj in projections:
                if proj not in prop_map:
                    continue
                pretty = pretty_map.get(proj, proj)
                Y_ms = _ensure_projection(tg, proj, multiscale=True)
                if Y_ms is not None:
                    adata.obsm[f"X_ms{pretty}"] = Y_ms
                Y_dm = _ensure_projection(tg, proj, multiscale=False)
                if Y_dm is not None:
                    adata.obsm[f"X_{pretty}"] = Y_dm

        try:
            if getattr(tg, "uom_enabled", False) and getattr(tg, "uom_comp_labels_", None) is not None:
                adata.obs["topometry_component"] = pd.Categorical(tg.uom_comp_labels_.astype(str))
        except Exception:
            pass

        def _csr(A):
            return A if issparse(A) else csr_matrix(A)

        def _do_leiden(
            adata,
            tg,
            P,
            *,
            leiden_key_base: str,
            leiden_resolutions: list[float] | tuple[float, ...],
            leiden_primary_index: int,
            neighbors_key: str,
            out_connectivities_key: str,
            out_distances_key: str,
        ):
            import scanpy as sc
            # expose to obsp
            adata.obsp[out_connectivities_key] = P
            adata.obsp[out_distances_key] = P
            # reuse neighbors slot if compatible
            need_fit = True
            if neighbors_key in adata.uns:
                conn = adata.uns[neighbors_key].get("connectivities", None)
                if conn is not None and getattr(conn, "shape", None) == P.shape:
                    need_fit = False
            if need_fit:
                use_rep = None
                if "X_ms_spectral_scaffold" in adata.obsm_keys():
                    use_rep = "X_ms_spectral_scaffold"
                elif "X_spectral_scaffold" in adata.obsm_keys():
                    use_rep = "X_spectral_scaffold"
                sc.pp.neighbors(
                    adata,
                    use_rep=use_rep,
                    n_neighbors=2,
                    method="umap",
                    key_added=neighbors_key,
                )
            adata.uns[neighbors_key]["connectivities"] = P
            adata.uns[neighbors_key]["distances"] = P
            for res in leiden_resolutions:
                key = f"{leiden_key_base}_res{res:g}"
                sc.tl.leiden(adata, resolution=res, adjacency=P, key_added=key)
            primary = f"{leiden_key_base}_res{leiden_resolutions[leiden_primary_index]:g}"
            adata.obs[leiden_key_base] = adata.obs[primary].astype("category")

        # single unified path (UoM returns block-diagonal via properties)
        if do_leiden:
            P_dm = _csr(tg.P_of_Z)
            _do_leiden(
                adata, tg, P_dm,
                leiden_key_base=leiden_key_base,
                leiden_resolutions=leiden_resolutions,
                leiden_primary_index=leiden_primary_index,
                neighbors_key="_topo_tmp_dm",
                out_connectivities_key="topometry_connectivities",
                out_distances_key="topometry_distances",
            )
            P_ms = getattr(tg, "P_of_msZ", None)
            if P_ms is not None:
                P_ms = _csr(P_ms)
                _do_leiden(
                    adata, tg, P_ms,
                    leiden_key_base=f"{leiden_key_base}_ms",
                    leiden_resolutions=leiden_resolutions,
                    leiden_primary_index=leiden_primary_index,
                    neighbors_key="_topo_tmp_ms",
                    out_connectivities_key="topometry_connectivities_ms",
                    out_distances_key="topometry_distances_ms",
                )

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

        Returns
        -------
        pandas.DataFrame or None
            If `return_df` is True, a table with PF1, PJS, SP, eigenvalue Wasserstein
            distance, and subspace cosine similarity per representation; otherwise None.

        Notes
        -----
        - PF1, PJS, and SP are geometry-preservation scores (higher is better).
        - Operators are rebuilt per representation using the requested `metric` and `n_neighbors`.

        Examples
        --------
        >>> import numpy as np, anndata as ad, topo as tp
        >>> adata = ad.AnnData(np.random.randn(20, 5))
        >>> tg = tp.TopOGraph().fit(adata.X)
        >>> tg.project()  # adds TopoMAP layouts used for evaluation
        >>> tp.sc._safe_set_obsm(adata, \"X_TopoMAP\", tg.TopoMAP)
        >>> df = tp.sc.evaluate_representations(adata, tg, return_df=True, print_results=False, plot_results=False)
        >>> set(df.columns) >= {'PF1', 'PJS', 'SP'}
        True
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
        rows.append(row_from_dict("PF1@k   (set overlap, higher is better)", {k: parts_all[k]['PF1']       for k in obsm_keys}))
        rows.append(row_from_dict("PJS     (1 - JS rows, higher is better)", {k: parts_all[k]['PJS']       for k in obsm_keys}))
        rows.append(row_from_dict("SP      (Procrustes R^2, higher is better)", {k: parts_all[k]['SP']        for k in obsm_keys}))

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

    def calculate_deformation_on_projection(adata, tg, proj_key='TopoMAP', diffusion_t=1, **kwargs):
            """Compute and store a deformation metric (centered log det(G)) for a given projection.
            This quantifies local volume changes induced by the projection relative to the original graph geometry.
            The metric is stored in `adata.obs['deformation_{proj_key}']`
            
            Parameters
            ----------
            adata : AnnData
                Annotated data matrix with projections in adata.obsm.
            tg : TopoGraph
                TopoGraph object fitted on `adata` with base_kernel and Laplacian.
            proj_key : str, list of str, default 'TopoMAP'
                Key in adata.obsm (without the 'X_' prefix) to use as the 2D embedding for computing deformation.
                If a list is provided, computes deformation for each key and stores in separate columns.
            diffusion_t : int, default 1
                Diffusion time parameter for smoothing the deformation metric; higher values yield smoother estimates.
            """
            from topo.eval.rmetric import calculate_deformation
            L = tg.graph_kernel.L
            # check if proj_key is a list; if not, make it a single-item list for uniform processing
            if isinstance(proj_key, str):
                proj_keys = [proj_key]
            else:
                proj_keys = list(proj_key)
            for proj_key in proj_keys:
                # Normalise: accept keys with or without 'X_' prefix
                if 'X_' + proj_key in adata.obsm:
                    obsm_key = 'X_' + proj_key
                elif proj_key in adata.obsm:
                    obsm_key = proj_key
                else:
                    raise KeyError(f"X_{proj_key} not found in adata.obsm")
                Y = adata.obsm[obsm_key]
                # Compute deformation once (centered log det(G)), optionally smoothed
                deform_vals, (dmin, dmax) = calculate_deformation(
                    Y, L,
                    diffusion_t=diffusion_t,
                    diffusion_op=getattr(tg.base_kernel, "P", None),
                    return_limits=True,
                    **kwargs
                )
                # store metric
                adata.obs[f'deformation_{proj_key}'] = deform_vals
                # store limits for potential use in plotting
                adata.uns[f'deformation_{proj_key}_limits'] = [float(dmin), float(dmax)]


    def plot_riemann_diagnostics(
        adata,
        tg,
        proj_key='X_TopoMAP',
        groupby='topo_clusters',
        diffusion_t=1,
        n_plot='10%',
        scale_gain=1.0,
        ellipse_alpha=0.15,
        point_size=6,
        do_all=False,
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
        diffusion_t : int, default 1
            Diffusion time for smoothing the deformation metric; higher values yield smoother estimates
        n_plot : int or str, default '10%'
            Number of localized indicatrices to plot, or a percentage string (e.g. '10%')
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
        None (plots are shown and metrics saved in adata)

        """
        from topo.eval.rmetric import (
        RiemannMetric,
        plot_riemann_metric_localized,
        plot_riemann_metric_global,
        calculate_deformation,
        )
        if isinstance(n_plot, str) and n_plot.endswith('%'):
            try:
                pct = float(n_plot[:-1]) / 100.0
                n_localized = int(len(adata) * pct)
            except ValueError:
                raise ValueError(f"Invalid percentage string for n_plot: {n_plot}")
        elif isinstance(n_plot, int):
            n_localized = n_plot
        elif isinstance(n_plot, float) and 0 < n_plot < 1:
            n_localized = int(len(adata) * n_plot)
        else:
            raise ValueError(f"Invalid value for n_plot: {n_plot}. Must be int, float in (0,1), or percentage string like '10%'.")
        # Normalise: accept both 'TopoMAP' and 'X_TopoMAP' as proj_key.
        # obsm_key  → the actual key in adata.obsm  (with    'X_' prefix)
        # basis_key → the key for sc.pl.embedding   (without 'X_' prefix)
        obsm_key  = proj_key if proj_key.startswith('X_') else ('X_' + proj_key)
        basis_key = proj_key[2:] if proj_key.startswith('X_') else proj_key
        if obsm_key not in adata.obsm:
            raise KeyError(f"{proj_key} not found in adata.obsm")
        if not hasattr(tg, 'base_kernel') or not hasattr(tg.base_kernel, 'L'):
            raise ValueError("tg must have a fitted base_kernel with attribute L (Laplacian)")
        
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

        L = tg.graph_kernel.L

        # Initialise per-tg cache (survives across calls, keyed by basis_key + Y fingerprint)
        if not hasattr(tg, '_riemann_cache'):
            tg._riemann_cache = {}

        def _make_plot(Y, L, basis_key, show=show):
            # basis_key is WITHOUT the 'X_' prefix (e.g. 'TopoMAP'), for use with
            # sc.pl.embedding and adata.obs key names.

            # --- Cached geometry computations (keyed by basis_key + Y fingerprint) ---
            Y_fp = (Y.shape, float(Y[0, 0]), float(Y[-1, -1]))  # lightweight fingerprint
            cache_key = (basis_key, Y_fp)
            _cache = tg._riemann_cache.get(cache_key, None)

            if _cache is not None:
                if verbose:
                    print(f"  [riemann] Using cached geometry for '{basis_key}'")
                deform_vals = _cache['deform_vals']
                dmin        = _cache['dmin']
                dmax        = _cache['dmax']
                G           = _cache['G']
                lam         = _cache['lam']
            else:
                if verbose:
                    print(f"  [riemann] Computing geometry for '{basis_key}' (will be cached)...")
                deform_vals, (dmin, dmax) = calculate_deformation(
                    Y, L,
                    center="median",
                    diffusion_t=8,
                    diffusion_op=getattr(tg.base_kernel, "P", None),
                    normalize="symmetric",
                    clip_percentile=2.0,
                    return_limits=True,
                )
                G   = RiemannMetric(Y, L).get_rmetric()
                lam = np.linalg.eigvalsh(G)
                lam = np.clip(lam, 1e-12, None)
                tg._riemann_cache[cache_key] = {
                    'deform_vals': deform_vals,
                    'dmin': dmin, 'dmax': dmax,
                    'G': G, 'lam': lam,
                }

            adata.obs[f'deformation_{basis_key}'] = deform_vals

            fig, axs = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
            # plot using the mapped colors directly (do not pass cmap)
            plot_riemann_metric_localized(
                Y, L,
                n_plot=n_localized,
                scale_mode="logdet",
                scale_gain=scale_gain,
                alpha=ellipse_alpha,
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

            # (b) Global indicatrices (thinned grid-averaged ellipses) overlaid on the embedding,
            #     colored by local contraction/expansion using the shared color scale
            sc.pl.embedding(
                adata,
                basis=basis_key,
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
                cmap="seismic",
                vmin=dmin, vmax=dmax,               # keep color scale consistent with panel (d)
                min_sep_factor=1.1,                 # reduce ellipse overlap
                choose_strong_first=True,
                deformation_vals=deform_vals,       # reuse computed deformation
            )
            plt.gca().set_aspect('equal'); plt.tight_layout()
            axs[1].set_title("Global indicatrices (C/E overlay)", fontsize=title_fontsize)

            # (c) Metric-derived scalar maps (anisotropy and log-det(G))
            # G and lam come from the cache computed above.
            adata.obs[f'metric_anisotropy_{basis_key}'] = np.log(lam[:, -1] / lam[:, 0])
            adata.obs[f'metric_logdetG_{basis_key}'] = np.sum(np.log(lam), axis=1)

            # (d) CONTRACTION vs EXPANSION PANEL (points), using the same deformation and limits
            # Try to suppress colorbar via Scanpy arg (newer versions)
            _axes_before = list(plt.gcf().axes)
            try:
                sc.pl.embedding(
                    adata,
                    basis=basis_key,
                    color=[f'deformation_{basis_key}'],
                    cmap='seismic',
                    wspace=0.25,
                    show=False,
                    frameon=False,
                    ax=axs[2],
                    vmin=dmin,
                    vmax=dmax,
                    colorbar_loc=None,   # <-- preferred (if supported)
                )
            except TypeError:
                # Older Scanpy: draw normally, then remove any colorbar axes added
                sc.pl.embedding(
                    adata,
                    basis=basis_key,
                    color=[f'deformation_{basis_key}'],
                    cmap='seismic',
                    wspace=0.25,
                    show=False,
                    frameon=False,
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
            proj_names = []
            for name in adata.obsm_keys():
                Yc = np.asarray(adata.obsm[name])
                if Yc.ndim == 2 and Yc.shape[1] == 2:
                    proj_names.append(name)
            for proj_name in proj_names:
                Y = np.asarray(adata.obsm[proj_name])
                # proj_name from obsm_keys() already has 'X_' prefix; strip for _make_plot
                bk = proj_name[2:] if proj_name.startswith('X_') else proj_name
                if verbose:
                    print(f"Riemannian diagnostics for projection '{proj_name}'")
                _fig = _make_plot(Y, L, bk, show=show)
                if (not show) and (_fig is not None):
                    plt.close(_fig)
            return None
        else:
            Y = np.asarray(adata.obsm[obsm_key])
            if verbose: print(f"Riemannian diagnostics for projection '{proj_key}'")
            return _make_plot(Y, L, basis_key, show=show)


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
        # NEW:
        evaluate_snapshots: bool = False,
        grid_search: bool = False,
        min_dist_grid = (0.2, 0.6, 1.0),
        spread_grid = (0.8, 1.2, 1.6),
        initial_alpha_grid = (0.4, 1.0, 1.6),
        # evaluation knobs (used when evaluate_snapshots=True or grid_search=True)
        eval_metric: str = "euclidean",
        eval_n_neighbors: int = 30,
        eval_backend: str = "hnswlib",
        eval_jobs: int = -1,
        eval_times = (1, 2, 4),
        eval_r: int = 32,
        eval_k_for_pf1: int = None,
        eval_symmetric_hint: bool = True,
    ):
        """
        Creates an animated GIF showing the evolution of MAP optimization, with optional
        on-the-fly evaluation of geometry-preservation metrics and/or a lightweight
        grid search for MAP hyperparameters.

        At regular checkpoints collected during `TopOGraph.project(..., save_every=...)`,
        the current 2-D embedding is drawn and colored by a categorical/numeric label
        from `adata.obs` or by a gene if `groupby` matches a variable name. Snapshots are
        then stitched into an animation. If requested, each snapshot is scored against
        the reference Markov operator from the base kernel and the resulting metrics are
        overlaid on the frame.

        Parameters
        ----------
        adata : AnnData
            Source of colors/labels; also used for gene coloring when `groupby` is a gene.
        tg : TopOGraph
            Fitted model containing MAP snapshots (multiscale or single-scale).
        groupby : str, default "topo_clusters"
            Column in `adata.obs` (categorical or numeric) or a gene in `adata.var_names`.
        num_iters : int, default 600
            Total iterations to visualize (clips to available snapshots).
        save_every : int, default 10
            Snapshot frequency used during optimization; used to index frames.
        dpi : int, default 120
            DPI for frames in the resulting GIF.
        multiscale : bool, default True
            If True, visualize multiscale MAP (msMAP) snapshots; otherwise single-scale MAP.
        fps : int, default 20
            Frames per second for the GIF.
        point_size : float, default 3.0
            Scatter marker size.
        filename : str or None, default None
            Output path (e.g., "map_optimization.gif"). If None, a name is auto-chosen.
        cmap : str, default "inferno"
            Colormap for numeric coloring in case `groupby` is continuous or a gene.

        evaluate_snapshots : bool, default False
            If True, compute geometry-preservation scores (PF1, PJS, SP and composite TP)
            for each snapshot and overlay them on frames. Uses the evaluation knobs below.
        grid_search : bool, default False
            If True, run a small grid search over MAP hyperparameters (min_dist, spread,
            initial_alpha) via `tg.find_ideal_projection(...)` prior to rendering. When
            enabled, snapshots are also evaluated and annotated.
        min_dist_grid : tuple of float, default (0.2, 0.6, 1.0)
            Candidate `min_dist` values for the grid search.
        spread_grid : tuple of float, default (0.8, 1.2, 1.6)
            Candidate `spread` values for the grid search.
        initial_alpha_grid : tuple of float, default (0.4, 1.0, 1.6)
            Candidate initial learning rates for the grid search.

        eval_metric : {"euclidean", ...}, default "euclidean"
            Distance metric used to build snapshot neighborhoods for scoring.
        eval_n_neighbors : int, default 30
            Local neighborhood size for evaluation graphs.
        eval_backend : {"hnswlib","pynndescent","bruteforce"}, default "hnswlib"
            Nearest-neighbor backend for evaluation graphs.
        eval_jobs : int, default -1
            Parallelism for neighbor search during evaluation.
        eval_times : tuple of int, default (1, 2, 4)
            Diffusion times used in composite TopoScore (TP) computation.
        eval_r : int, default 32
            Spectral rank used when approximating diffusion operators for scoring.
        eval_k_for_pf1 : int or None, default None
            Optional top-k used in PF1 computation; if None, uses `eval_n_neighbors`.
        eval_symmetric_hint : bool, default True
            Whether to treat operators as (near) symmetric to speed up evaluation.

        Returns
        -------
        str
            Path to the saved GIF.

        Notes
        -----
        - Requires snapshots to have been collected during optimization
          (see `TopOGraph.project(..., save_every=...)`). If missing, this function
          will run a projection pass with `include_init_snapshot=True`.
        - When `groupby` is categorical, Scanpy palettes in
          `adata.uns[f"{groupby}_colors"]` are honored if present.
        - Snapshot evaluation uses the base operator from `tg.base_kernel.P` as the
          reference graph unless otherwise overridden inside the model.
        """

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        n = adata.n_obs

        # 1) Colors
        if groupby in adata.var_names:
            gidx = adata.var_names.get_loc(groupby)
            x = adata.X[:, gidx]
            x = x.toarray().ravel() if hasattr(x, "toarray") else np.asarray(x).ravel()
            vmin, vmax = np.nanpercentile(x, [1, 99])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = float(np.nanmin(x)), float(np.nanmax(x))
            t = (x - vmin) / (vmax - vmin + 1e-12)
            colors = plt.get_cmap(cmap)(np.clip(t, 0, 1))
        elif groupby in adata.obs.columns:
            series = adata.obs[groupby]
            if pd.api.types.is_categorical_dtype(series):
                cats = series.cat.categories
                codes = series.cat.codes.to_numpy()
                palette_key = f"{groupby}_colors"
                if palette_key in adata.uns and len(adata.uns[palette_key]) >= len(cats):
                    palette = list(adata.uns[palette_key])
                else:
                    _cmap = plt.get_cmap("tab20")
                    palette = [mcolors.to_hex(_cmap(i % 20)) for i in range(len(cats))]
                    adata.uns[palette_key] = palette
                neutral = (0.7, 0.7, 0.7, 0.6)
                colors = np.empty((n, 4), dtype=float)
                for i, c in enumerate(codes):
                    colors[i] = mcolors.to_rgba(palette[c]) if (c >= 0 and c < len(palette)) else neutral
            else:
                vals = np.asarray(series.values, float).ravel()
                vmin, vmax = np.nanpercentile(vals, [1, 99])
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
                t = (vals - vmin) / (vmax - vmin + 1e-12)
                colors = plt.get_cmap("viridis")(np.clip(t, 0, 1))
        else:
            colors = np.tile(np.array([0.15, 0.15, 0.15, 0.85], dtype=float), (n, 1))

        # 2) Align color order to tg snapshot order when possible
        tg_names = None
        for attr in ("obs_names_", "_obs_names", "obs_index_", "_obs_index", "cell_names_", "_cell_names"):
            if hasattr(tg, attr) and getattr(tg, attr) is not None:
                tg_names = pd.Index(getattr(tg, attr))
                break
        if tg_names is None:
            for attr in ("indices_", "_indices", "_order", "order_", "_fit_indices"):
                if hasattr(tg, attr) and getattr(tg, attr) is not None:
                    idx = np.asarray(getattr(tg, attr))
                    if np.issubdtype(idx.dtype, np.integer) and idx.ndim == 1 and idx.size == n:
                        colors = colors[idx]
                        break
        if tg_names is not None:
            inv = pd.Index(adata.obs_names).get_indexer(pd.Index(tg_names))
            if (inv >= 0).all():
                colors = colors[inv]

        # 3) Grid-search if requested (will also annotate snapshots with metrics)
        if bool(grid_search):
            _ = tg.find_ideal_projection(
                min_dist_grid=list(min_dist_grid),
                spread_grid=list(spread_grid),
                initial_alpha_grid=list(initial_alpha_grid),
                multiscale=bool(multiscale),
                num_iters=int(num_iters),
                save_every=int(save_every),
                metric=eval_metric,
                n_neighbors=int(eval_n_neighbors),
                backend=eval_backend,
                n_jobs=int(eval_jobs),
                times=tuple(eval_times),
                r=int(eval_r),
                k_for_pf1=eval_k_for_pf1,
                symmetric_hint=bool(eval_symmetric_hint),
                verbosity=1,
            )

        # 4) Otherwise, ensure snapshots exist
        primary_attr = "msTopoMAP_snapshots" if multiscale else "TopoMAP_snapshots"
        legacy_attr = "msmap_snapshots" if multiscale else "map_snapshots"
        snapshots = getattr(tg, primary_attr, None) or getattr(tg, legacy_attr, None)

        if not snapshots or len(snapshots) < 2:
            tg.project(
                projection_method="MAP",
                multiscale=bool(multiscale),
                num_iters=int(num_iters),
                save_every=int(save_every),
                include_init_snapshot=True,
            )
            snapshots = getattr(tg, primary_attr, None) or getattr(tg, legacy_attr, None)

        if not snapshots:
            raise RuntimeError("No MAP snapshots available to render.")

        # 5) If evaluation requested and we did not run grid_search, annotate snapshots now
        if bool(evaluate_snapshots) and not bool(grid_search):
            # Attach metrics per snapshot so TopOGraph.visualize_optimization can overlay them
            from scipy.sparse import csr_matrix, issparse
            from topo.eval.topo_metrics import topo_preserve_score, get_P

            PX_ref = tg.base_kernel.P
            if not issparse(PX_ref):
                PX_ref = csr_matrix(PX_ref)

            for snap in snapshots:
                Ysnap = snap["embedding"]
                PY = get_P(
                    Ysnap,
                    metric=eval_metric,
                    n_neighbors=int(eval_n_neighbors),
                    backend=eval_backend,
                    n_jobs=int(eval_jobs),
                )
                if not issparse(PY):
                    PY = csr_matrix(PY)
                score, parts = topo_preserve_score(
                    PX_ref,
                    PY,
                    times=tuple(eval_times),
                    r=int(eval_r),
                    symmetric_hint=bool(eval_symmetric_hint),
                    k_for_pf1=eval_k_for_pf1,
                )
                snap["metrics"] = {
                    "TP": float(score),
                    "PF1": float(parts.get("PF1", np.nan)),
                    "PJS": float(parts.get("PJS", np.nan)),
                    "SP": float(parts.get("SP", np.nan)),
                }

        # 6) Render via model method; overlay_metrics if we evaluated
        out_path = tg.visualize_optimization(
            num_iters=len(snapshots) * int(save_every),
            save_every=int(save_every),
            dpi=int(dpi),
            color=colors,
            multiscale=multiscale,
            fps=int(fps),
            point_size=float(point_size),
            filename=filename,
            overlay_metrics=bool(evaluate_snapshots),
        )
        return out_path

    # def visualize_optimization(
    #     adata,
    #     tg,
    #     groupby: str = "topo_clusters",
    #     num_iters: int = 600,
    #     save_every: int = 10,
    #     dpi: int = 120,
    #     *,
    #     multiscale: bool = True,
    #     fps: int = 20,
    #     point_size: float = 3.0,
    #     filename: str = None,
    #     cmap: str = "inferno",
    # ):
    #     """
    #     Create an animated GIF showing the evolution of MAP optimization.

    #     At regular checkpoints collected during `TopOGraph.project(..., save_every=...)`,
    #     draw the current 2-D embedding colored by a categorical/numeric label, or by
    #     a gene if `groupby` matches a variable name. Combines the snapshots into an
    #     animation.

    #     Parameters
    #     ----------
    #     adata : AnnData
    #         Source of colors/labels; also used for gene coloring when `groupby` is a gene.
    #     tg : TopOGraph
    #         Fitted model containing MAP snapshots (ms or DM).
    #     groupby : str, default "topo_clusters"
    #         Column in `adata.obs` (categorical or numeric) or a gene in `adata.var_names`.
    #     num_iters : int, default 600
    #         Total iterations to visualize (clips to available snapshots).
    #     save_every : int, default 10
    #         Snapshot frequency used during optimization; used to index frames.
    #     dpi : int, default 120
    #         DPI for frames in the resulting GIF.
    #     multiscale : bool, default True
    #         If True, visualize msMAP snapshots; otherwise DM MAP snapshots.
    #     fps : int, default 20
    #         Frames per second for the GIF.
    #     point_size : float, default 3.0
    #         Scatter marker size.
    #     filename : str or None, default None
    #         Output path (e.g., "map_optimization.gif"). If None, a name is auto-chosen.
    #     cmap : str, default "inferno"
    #         Colormap for numeric coloring.

    #     Returns
    #     -------
    #     filename : str
    #         Path to the saved GIF.

    #     Notes
    #     -----
    #     - Requires that snapshots were collected during optimization (see `TopOGraph.project`).
    #     - Preserves Scanpy categorical palettes when present in `adata.uns[f"{groupby}_colors"]`.
    #     """
    #     import numpy as np
    #     import pandas as pd
    #     import matplotlib.pyplot as plt
    #     import matplotlib.colors as mcolors

    #     n = adata.n_obs

    #     # 1) Colors
    #     if groupby in adata.var_names:
    #         gidx = adata.var_names.get_loc(groupby)
    #         x = adata.X[:, gidx]
    #         x = x.toarray().ravel() if hasattr(x, "toarray") else np.asarray(x).ravel()
    #         vmin, vmax = np.nanpercentile(x, [1, 99])
    #         if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
    #             vmin, vmax = float(np.nanmin(x)), float(np.nanmax(x))
    #         t = (x - vmin) / (vmax - vmin + 1e-12)
    #         colors = plt.get_cmap(cmap)(np.clip(t, 0, 1))
    #     elif groupby in adata.obs.columns:
    #         series = adata.obs[groupby]
    #         if pd.api.types.is_categorical_dtype(series):
    #             cats = series.cat.categories
    #             codes = series.cat.codes.to_numpy()
    #             palette_key = f"{groupby}_colors"
    #             if palette_key in adata.uns and len(adata.uns[palette_key]) >= len(cats):
    #                 palette = list(adata.uns[palette_key])
    #             else:
    #                 _cmap = plt.get_cmap("tab20")
    #                 palette = [mcolors.to_hex(_cmap(i % 20)) for i in range(len(cats))]
    #                 adata.uns[palette_key] = palette
    #             neutral = (0.7, 0.7, 0.7, 0.6)
    #             colors = np.empty((n, 4), dtype=float)
    #             for i, c in enumerate(codes):
    #                 colors[i] = mcolors.to_rgba(palette[c]) if (c >= 0 and c < len(palette)) else neutral
    #         else:
    #             vals = np.asarray(series.values, float).ravel()
    #             vmin, vmax = np.nanpercentile(vals, [1, 99])
    #             if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
    #                 vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    #             t = (vals - vmin) / (vmax - vmin + 1e-12)
    #             colors = plt.get_cmap("viridis")(np.clip(t, 0, 1))
    #     else:
    #         colors = np.tile(np.array([0.15, 0.15, 0.15, 0.85], dtype=float), (n, 1))

    #     # 2) Align color order to tg snapshot order when possible
    #     tg_names = None
    #     for attr in ("obs_names_", "_obs_names", "obs_index_", "_obs_index", "cell_names_", "_cell_names"):
    #         if hasattr(tg, attr) and getattr(tg, attr) is not None:
    #             tg_names = pd.Index(getattr(tg, attr))
    #             break
    #     if tg_names is None:
    #         for attr in ("indices_", "_indices", "_order", "order_", "_fit_indices"):
    #             if hasattr(tg, attr) and getattr(tg, attr) is not None:
    #                 idx = np.asarray(getattr(tg, attr))
    #                 if np.issubdtype(idx.dtype, np.integer) and idx.ndim == 1 and idx.size == n:
    #                     colors = colors[idx]
    #                     break
    #     if tg_names is not None:
    #         inv = pd.Index(adata.obs_names).get_indexer(pd.Index(tg_names))
    #         if (inv >= 0).all():
    #             colors = colors[inv]

    #     # 3) Acquire or generate snapshots; require >=2 frames
    #     primary_attr = "msTopoMAP_snapshots" if multiscale else "TopoMAP_snapshots"
    #     legacy_attr = "msmap_snapshots" if multiscale else "map_snapshots"
    #     snapshots = getattr(tg, primary_attr, None) or getattr(tg, legacy_attr, None)

    #     if not snapshots or len(snapshots) < 2:
    #         tg.project(
    #             projection_method="MAP",
    #             multiscale=bool(multiscale),
    #             num_iters=int(num_iters),
    #             save_every=int(save_every),
    #             include_init_snapshot=True,
    #         )
    #         snapshots = getattr(tg, primary_attr, None) or getattr(tg, legacy_attr, None)

    #     if not snapshots or len(snapshots) < 1:
    #         raise RuntimeError("No MAP snapshots available to render.")

    #     # 4) Render via model method
    #     out_path = tg.visualize_optimization(
    #         num_iters=len(snapshots) * int(save_every),
    #         save_every=int(save_every),
    #         dpi=int(dpi),
    #         color=colors,
    #         multiscale=multiscale,
    #         fps=int(fps),
    #         point_size=float(point_size),
    #         filename=filename,
    #     )
    #     return out_path



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
            id_details = getattr(tg, "_id_details", None)
            if id_details is not None:
                adata.uns['topometry_id_details'] = _h5ad_safe(id_details)
            adata.uns[f'topometry_id_global_{tg.id_method}'] = float(tg.global_id) if tg.global_id is not None else None
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
            if 'id_est' in locals() and id_est is not None:
                id_summary = {
                    "local_id": getattr(id_est, "local_id", None),
                    "global_id": getattr(id_est, "global_id", None),
                }
                adata.uns['intrinsic_dim_estimator'] = _h5ad_safe(id_summary)
        except Exception as e:
            print(f"[TopoMetry] IntrinsicDim estimation skipped: {e}")


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
        Graph-filter a user-provided signal using TopoMetry's Markov operators.

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
            - "auto": if input looks like a probability (0-1), keep range; otherwise unit-scale to 0-1.
            - "unit": force min-max to 0-1 after filtering.
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
        - Works with dense or sparse operators; computation is vector-matrix.

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
                    L=tg.graph_kernel.L,
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
                lims = riem.get('limits', None)
                if lims is not None:
                    try:
                        lo, hi = float(lims[0]), float(lims[1])
                        adata.uns['metric_limits'][key] = np.array([lo, hi], dtype=float)
                    except Exception:
                        adata.uns['metric_limits'][key] = None
                else:
                    adata.uns['metric_limits'][key] = None
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
        evals = tg.EigenbasisDict[key].eigenvalues                  # includes lambda_0
        # choose k (drop the trivial first eigenpair)
        k_use = int(min(64, Z_full.shape[1] - 1)) if Z_full.shape[1] > 1 else 1
        # weights: lambda/(1 - lambda), like in tg.pseudotime
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
            print(f"[TopoMetry] Imputation failed at best_t={best_t}: {e}")

        adata.uns["imputation_qc"] = {
            "t_grid": [int(t) for t in t_grid],
            "stats": df_stats,
            "best_t": best_t,
            "heatmap_genes": top_genes,
            "corr_raw": corr_raw,
            "corr_imp_best": corr_by_t.get(best_t, None),
        }


    def find_ppv_markers(
        AnnData,
        groupby,
        use_raw=False,
        pval_thr=0.05,
        logfc_thr=0.5,
        min_frac_in_cluster=0.5,
        return_df=False,
        logreg_tol=1e-8,
        logreg_max_iter=300,
        n_jobs=1
    ):
        """
        Compute PPV-ranked marker genes per group using:
        - Wilcoxon DE for p-values / logFC
        - Logistic regression (logreg) for a secondary ranking signal
        and write results back to `.uns` in Scanpy's rank_genes_groups format.

        The resulting key can be consumed by:
        - `sc.get.rank_genes_groups_df(AnnData, key=...)`
        - `sc.pl.rank_genes_groups_dotplot(AnnData, key=...)`

        Parameters
        ----------
        AnnData : anndata.AnnData
            Input AnnData. Must contain `AnnData.obs[groupby]`.
        groupby : str
            Grouping key in `AnnData.obs`.
        use_raw : bool, default True
            Use `AnnData.raw.X` (and `.raw.var_names`) for expression when available.
        pval_thr : float, default 0.05
            Threshold on adjusted p-values from Wilcoxon (`pvals_adj`).
        logfc_thr : float, default 0.5
            Threshold on log fold-changes from Wilcoxon (`logfoldchanges`).
        min_frac_in_cluster : float, default 0.5
            Minimum fraction of cells in a group expressing the gene (>0) required.
        return_df : bool, default False
            If True, also return a tidy DataFrame of per-(group,gene) statistics.
        logreg_tol: float, default 1e-4
            Tolerance for sklearn.linear_model.LogisticRegression stopping criteria.
        logreg_max_iter: int, default 300
            Maximum number of iterations taken for the LogisticRegression solver to converge.
        n_jobs: int, default 1
            Number of CPU cores used when parallelizing.

        Returns
        -------
        dict
            keys:
            - 'key_added': str, the `.uns` key for the PPV-ranked results
            - 'uns_key_wilcoxon': str, Wilcoxon DE key
            - 'uns_key_logreg': str, LogReg DE key
            - 'uns_key_reordered': str|None, if `{groupby}_logreg_X` was reordered in-place
            - 'ppv': dict[group -> dict[gene -> ppv]]
            - 'ranked_markers': dict[group -> list[gene]]
            - 'candidates': dict[group -> set[gene]]
            - 'df': pandas.DataFrame (only if return_df=True)
        """
        if not isinstance(AnnData, AnnData):
            raise TypeError("AnnData must be an anndata.AnnData")
        if groupby not in AnnData.obs:
            raise KeyError("groupby key not found in AnnData.obs: %r" % groupby)
        if not (0.0 < float(min_frac_in_cluster) <= 1.0):
            raise ValueError("min_frac_in_cluster must be in (0, 1].")

        labels = AnnData.obs[groupby].astype("category")
        clusters = list(labels.cat.categories)
        labels_arr = labels.to_numpy()

        # ---- Run (or reuse) DE: Wilcoxon + LogReg ----
        wilcoxon_key = "%s_wilcoxon_X" % groupby
        logreg_key = "%s_logreg_X" % groupby

        if wilcoxon_key not in AnnData.uns:
            sc.tl.rank_genes_groups(
                AnnData,
                groupby=groupby,
                method="wilcoxon",
                use_raw=bool(use_raw),
                key_added=wilcoxon_key,
            )

        if logreg_key not in AnnData.uns:
            sc.tl.rank_genes_groups(
                AnnData,
                groupby=groupby,
                method="logreg",
                use_raw=bool(use_raw),
                key_added=logreg_key,
            )

        # ---- Collect Wilcoxon stats (pvals/logfc) ----
        wdf = sc.get.rank_genes_groups_df(AnnData, key=wilcoxon_key, group=None)
        if wdf is None or len(wdf) == 0:
            raise ValueError("Wilcoxon rank_genes_groups produced no results under key: %r" % wilcoxon_key)

        if "pvals_adj" in wdf.columns:
            wdf = wdf[wdf["pvals_adj"] < float(pval_thr)]
        if "logfoldchanges" in wdf.columns:
            wdf = wdf[wdf["logfoldchanges"] > float(logfc_thr)]
        if wdf is None or len(wdf) == 0:
            raise ValueError("No candidate genes after filtering Wilcoxon by pval/logfc thresholds.")

        # Map (group,gene) -> wilcoxon stats
        wdf["group"] = wdf["group"].astype(str)
        wdf["names"] = wdf["names"].astype(str)

        w_stats = {}
        for _, r in wdf.iterrows():
            g = str(r["group"])
            gene = str(r["names"])
            if g not in set(map(str, clusters)):
                continue
            w_stats[(g, gene)] = {
                "pvals": float(r["pvals"]) if "pvals" in r and pd.notnull(r["pvals"]) else np.nan,
                "pvals_adj": float(r["pvals_adj"]) if "pvals_adj" in r and pd.notnull(r["pvals_adj"]) else np.nan,
                "logfoldchanges": float(r["logfoldchanges"]) if "logfoldchanges" in r and pd.notnull(r["logfoldchanges"]) else np.nan,
            }

        # Candidates per group
        cluster_candidates = {str(c): set() for c in clusters}
        for (g, gene) in w_stats.keys():
            cluster_candidates[g].add(gene)

        # ---- Collect LogReg scores for tie-breaking ----
        ldf = sc.get.rank_genes_groups_df(AnnData, key=logreg_key, group=None)
        logreg_scores = {}
        if ldf is not None and len(ldf) > 0 and "scores" in ldf.columns:
            ldf["group"] = ldf["group"].astype(str)
            ldf["names"] = ldf["names"].astype(str)
            for _, r in ldf.iterrows():
                g = str(r["group"])
                gene = str(r["names"])
                if g not in cluster_candidates:
                    continue
                if pd.isnull(r["scores"]):
                    continue
                logreg_scores[(g, gene)] = float(r["scores"])

        # ---- Expression matrix selection ----
        if AnnData.raw is not None and bool(use_raw):
            X_expr = AnnData.raw.X
            var_names = AnnData.raw.var_names.astype(str)
        else:
            X_expr = AnnData.X
            var_names = AnnData.var_names.astype(str)

        if issparse(X_expr):
            X_expr = X_expr.tocsr()

        # ---- Compute PPV + pct_in/out using expressed>0 for candidate genes only ----
        all_candidate_genes = sorted(set().union(*cluster_candidates.values()))
        gene_to_idx = {g: i for i, g in enumerate(var_names) if g in set(all_candidate_genes)}
        present_genes = [g for g in all_candidate_genes if g in gene_to_idx]
        if len(present_genes) == 0:
            raise ValueError("None of the candidate genes are present in `.var_names` / `.raw.var_names`.")

        gene_idx = np.array([gene_to_idx[g] for g in present_genes], dtype=int)

        if issparse(X_expr):
            bool_expr = (X_expr[:, gene_idx] > 0)
            total_expr = np.asarray(bool_expr.sum(axis=0)).ravel().astype(int)
        else:
            sub_expr = np.asarray(X_expr[:, gene_idx])
            bool_expr = (sub_expr > 0)
            total_expr = bool_expr.sum(axis=0).astype(int)

        # Precompute not-in-group denominators
        n_total = int(AnnData.n_obs)

        cluster_ppv = {str(c): {} for c in clusters}
        ranked_markers = {str(c): [] for c in clusters}

        rows = []  # optional tidy DF

        for c in clusters:
            c = str(c)
            idx_in = np.where(labels_arr.astype(str) == c)[0]
            if idx_in.size == 0:
                continue
            n_in = int(idx_in.size)
            n_out = n_total - n_in

            if issparse(bool_expr):
                n_in_cluster = np.asarray(bool_expr[idx_in, :].sum(axis=0)).ravel().astype(int)
            else:
                n_in_cluster = bool_expr[idx_in, :].sum(axis=0).astype(int)

            frac_in = n_in_cluster / float(n_in)

            # Compute outside counts efficiently
            if issparse(bool_expr):
                n_out_cluster = total_expr - n_in_cluster
            else:
                n_out_cluster = total_expr - n_in_cluster

            pct_out = (n_out_cluster / float(n_out)) if n_out > 0 else np.zeros_like(frac_in, dtype=float)
            pct_in = frac_in

            cand = cluster_candidates.get(c, set())
            scores = {}

            for j, gene in enumerate(present_genes):
                if gene not in cand:
                    continue
                if total_expr[j] == 0:
                    continue
                if frac_in[j] < float(min_frac_in_cluster):
                    continue

                ppv = float(n_in_cluster[j]) / float(total_expr[j])
                scores[gene] = ppv
                cluster_ppv[c][gene] = ppv

                ws = w_stats.get((c, gene), {"pvals": np.nan, "pvals_adj": np.nan, "logfoldchanges": np.nan})
                rows.append(
                    {
                        "group": c,
                        "names": gene,
                        "ppv": ppv,
                        "pvals": ws["pvals"],
                        "pvals_adj": ws["pvals_adj"],
                        "logfoldchanges": ws["logfoldchanges"],
                        "pct_nz_group": float(pct_in[j]),
                        "pct_nz_reference": float(pct_out[j]),
                        "logreg_score": float(logreg_scores.get((c, gene), np.nan)),
                    }
                )

            # Sort: PPV desc, then logreg score desc (if available), then gene name
            def _key(gene):
                return (
                    -float(scores.get(gene, -np.inf)),
                    -float(logreg_scores.get((c, gene), -np.inf)) if (c, gene) in logreg_scores else 0.0,
                    gene,
                )

            ranked = sorted(scores.keys(), key=_key)
            ranked_markers[c] = ranked

        # ---- Build Scanpy rank_genes_groups-like structure under a new key ----
        key_added = "%s_ppv" % groupby

        # Determine per-group gene lists (ensure consistent length)
        max_len = max((len(ranked_markers[str(c)]) for c in clusters), default=0)
        if max_len == 0:
            raise ValueError("No PPV markers found after applying min_frac_in_cluster and DE thresholds.")

        # Build recarrays with fields named by groups (Scanpy convention in many versions)
        fields = tuple([str(i) for i in range(len(clusters))])

        def _make_recarr(fill_value):
            arr = np.empty((max_len,), dtype=[(f, object) for f in fields])
            for f in fields:
                arr[f] = fill_value
            return arr

        names_arr = _make_recarr("")
        scores_arr = _make_recarr(np.nan)          # use PPV as 'scores'
        pvals_arr = _make_recarr(np.nan)
        pvals_adj_arr = _make_recarr(np.nan)
        logfc_arr = _make_recarr(np.nan)
        pct_in_arr = _make_recarr(np.nan)          # pct_nz_group
        pct_out_arr = _make_recarr(np.nan)         # pct_nz_reference

        # Index tidy DF for fast lookup
        df_all = pd.DataFrame(rows) if len(rows) > 0 else pd.DataFrame(
            columns=["group","names","ppv","pvals","pvals_adj","logfoldchanges","pct_nz_group","pct_nz_reference","logreg_score"]
        )
        df_idx = None
        if len(df_all) > 0:
            df_idx = df_all.set_index(["group", "names"], drop=False)

        for gi, c in enumerate(clusters):
            c = str(c)
            field = fields[gi]
            genes = ranked_markers.get(c, [])
            if len(genes) == 0:
                continue

            # Fill in
            for k, gene in enumerate(genes[:max_len]):
                names_arr[field][k] = gene
                if df_idx is not None and (c, gene) in df_idx.index:
                    r = df_idx.loc[(c, gene)]
                    # r can be Series or DataFrame if duplicate; handle robustly
                    if isinstance(r, pd.DataFrame):
                        r = r.iloc[0]
                    scores_arr[field][k] = float(r["ppv"])
                    pvals_arr[field][k] = float(r["pvals"]) if pd.notnull(r["pvals"]) else np.nan
                    pvals_adj_arr[field][k] = float(r["pvals_adj"]) if pd.notnull(r["pvals_adj"]) else np.nan
                    logfc_arr[field][k] = float(r["logfoldchanges"]) if pd.notnull(r["logfoldchanges"]) else np.nan
                    pct_in_arr[field][k] = float(r["pct_nz_group"]) if pd.notnull(r["pct_nz_group"]) else np.nan
                    pct_out_arr[field][k] = float(r["pct_nz_reference"]) if pd.notnull(r["pct_nz_reference"]) else np.nan

        AnnData.uns[key_added] = {
            "params": {
                "groupby": groupby,
                "reference": "rest",
                "method": "ppv",
                "use_raw": bool(use_raw),
                "pval_thr": float(pval_thr),
                "logfc_thr": float(logfc_thr),
                "min_frac_in_cluster": float(min_frac_in_cluster),
                "source_wilcoxon_key": wilcoxon_key,
                "source_logreg_key": logreg_key,
            },
            "names": names_arr,
            "scores": scores_arr,               # Scanpy expects 'scores'
            "pvals": pvals_arr,
            "pvals_adj": pvals_adj_arr,
            "logfoldchanges": logfc_arr,
            "pct_nz_group": pct_in_arr,
            "pct_nz_reference": pct_out_arr,
        }

        # ---- Optionally reorder the existing logreg key in-place using PPV order ----
        uns_key_reordered = None
        if logreg_key in AnnData.uns and isinstance(AnnData.uns[logreg_key], dict) and "names" in AnnData.uns[logreg_key]:
            try:
                rg = AnnData.uns[logreg_key]
                names0 = rg["names"]
                group_fields = names0.dtype.names
                if group_fields is not None and len(group_fields) == len(clusters):
                    n_genes_all = names0.shape[0]
                    for group_idx, cname in enumerate(map(str, clusters)):
                        field = group_fields[group_idx]
                        orig = [str(x) for x in list(names0[field])]
                        ppv_rank = ranked_markers.get(cname, [])
                        if not ppv_rank:
                            continue
                        seen = set()
                        new_list = []
                        for g in ppv_rank:
                            if g and g not in seen:
                                new_list.append(g)
                                seen.add(g)
                        for g in orig:
                            if g and g not in seen:
                                new_list.append(g)
                                seen.add(g)
                        if len(new_list) < n_genes_all:
                            new_list.extend([""] * (n_genes_all - len(new_list)))
                        elif len(new_list) > n_genes_all:
                            new_list = new_list[:n_genes_all]
                        names0[field] = np.array(new_list, dtype=object)
                    AnnData.uns[logreg_key]["names"] = names0
                    uns_key_reordered = logreg_key
            except Exception:
                uns_key_reordered = None

        out = {
            "key_added": key_added,
            "uns_key_wilcoxon": wilcoxon_key,
            "uns_key_logreg": logreg_key,
            "uns_key_reordered": uns_key_reordered,
            "ppv": cluster_ppv,
            "ranked_markers": ranked_markers,
            "candidates": cluster_candidates,
        }

        if return_df:
            out["df"] = df_all

        return out



    def repel_annotation_labels(
        adata,
        groupby,
        basis="TopoMAP",
        exclude=(),
        ax=None,
        adjust_kwargs=dict(arrowprops=dict(
                arrowstyle="-",
                color="black",
                lw=0.5,
                alpha=0.7,
            ),
        ),
        text_kwargs=dict(
            fontsize=16,
            fontweight="bold",
            color="black",
            path_effects=[patheffects.withStroke(linewidth=3.0, foreground="white")])
        ):
        """
        Repel group labels on a Scanpy 2-D embedding by placing them at group medians and using adjustText for repulsion.
        Use for creating publication-quality figures.

        Parameters
        ----------
        adata : AnnData
            Input dataset with the embedding in `adata.obsm[f"X_{basis}"]` and group labels in `adata.obs[groupby]`.
        groupby : str
            Column in `adata.obs` defining groups for which to place labels.
        basis : str, default "TopoMAP"
            Which embedding in `adata.obsm` to use for coordinates (without the "X_" prefix).
        exclude : iterable of group labels to exclude from labeling, default ()
            Groups in `adata.obs[groupby]` to skip when placing labels.
        ax : matplotlib.axes.Axes or None, default None
            Optional axes to plot on; if None, a new figure and axes are created.
        adjust_kwargs : dict, default dict(arrowprops=dict(
            Additional keyword arguments passed to `adjust_text` for fine-tuning repulsion behavior.
        text_kwargs : dict, default dict(...))
            Keyword arguments for styling the text labels (passed to `ax.text`).
        
        Example
        -------
        ```python

        fig, ax = plt.subplots(1, 1, figsize=(6,6))  

        sc.pl.embedding(adata, basis="TopoMAP", color="topo_clusters", show=False, ax=ax) 

        tp.sc.repel_annotation_labels(adata, groupby='topo_clusters', basis='TopoPaCMAP', ax=ax,
         adjust_kwargs=dict(arrowprops=dict(
                arrowstyle="-",
                color="red",
                lw=0.5,
                alpha=0.7,            ),
        ),
          text_kwargs=dict(fontsize=12,
            fontweight='bold',
              color='black',
                path_effects=[patheffects.withStroke(linewidth=3.0, foreground='white')]
                )
            )
        fig = ax.get_figure()
        fig.tight_layout()
        plt.show()
        ```


        """

        if adjust_kwargs is None:
            adjust_kwargs = {}
        if text_kwargs is None:
            text_kwargs = {}

        coords = np.asarray(adata.obsm[f"X_{basis}"])
        groups = adata.obs[groupby]

        # compute medians per group using a boolean mask
        medians = {}
        for g in groups.cat.categories if hasattr(groups, "cat") else groups.unique():
            if g in exclude:
                continue
            mask = (groups.values == g)
            if not np.any(mask):
                continue
            medians[g] = np.median(coords[mask, :], axis=0)

        if ax is None:
            fig, ax = plt.subplots()

        # find the main scatter artist for repulsion
        scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
        scatter = scatters[0] if scatters else None

        texts = []
        for k, (x, y) in medians.items():
            t = ax.text(x, y, k, zorder=10, **text_kwargs)
            
            texts.append(t)

        base_adjust_kwargs = dict(
            autoalign="xy",
            only_move={"points": "xy", "text": "xy"},
            expand_points=(1.3, 1.5),
            expand_text=(1.1, 1.3),
            force_points=0.5,
            force_text=0.8,
            lim=500,
        )
        base_adjust_kwargs.update(adjust_kwargs or {})

        if scatter is not None:
            adjust_text(texts, add_objects=[scatter], **base_adjust_kwargs)
        else:
            adjust_text(texts, **base_adjust_kwargs)

        return texts




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
        End-to-end TopoMetry pipeline on an AnnData object.

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
            The same object, enriched with TopoMetry outputs.
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

        return tg




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
        Generate a multi-page A4-landscape PDF summarizing a TopoMetry run.

        Pages typically include:
        -  Dataset overview and QC.
        -  Dual scaffolds and requested 2-D projections (colored by categories/signals).
        -  Clustering summaries (if computed).
        -  Spectral selectivity / alignment.
        -  Riemann diagnostics overlays and deformation maps.
        -  Imputation QC (and optional gene example).
        -  Optional graph-filtering visuals controlled by *filtering_* knobs.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix with TopoMetry results.
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

            # A single wide axis with page margins; we'll stack text blocks manually top->bottom.
            ax = fig.add_axes([0.06, 0.08, 0.88, 0.84])  # left, bottom, width, height
            ax.axis('off')

            # ----- Title -----
            fig.text(0.05, 0.95, "TopoMetry - summary", fontsize=16, weight='bold', va='center', ha='left')

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

            # Geometry preservation "best"
            best_geo_line = None
            try:
                df_eval = adata.uns.get('topometry_representation_eval', None)
                if df_eval is not None and not df_eval.empty:
                    best_idx = int(np.nanargmax(df_eval['TopoPreserve'].values))
                    best_rep = df_eval.iloc[best_idx]['representation']
                    best_geo_line = f"-  Best geometry preservation (TopoPreserve): {best_rep}"
            except Exception:
                pass
            if best_geo_line is None:
                try:
                    gtbl = adata.uns.get('geometry_metrics_table', None)
                    if gtbl is not None and 'Composite' in gtbl:
                        best_rep = gtbl['Composite'].idxmax()
                        best_geo_line = f"-  Best geometry preservation (composite): {best_rep}"
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
                "   5) uses the scaffold and refined graph to produce faithful 2-D views (TopoMAP / TopoPaCMAP) that "
                "preserve global and local structure better than direct PCA/UMAP in many datasets.\n \n"
                "   -  TopoMetry also quantifies distortions in low-dimensional representations and provides intuitive diagnostic plots.\n \n"
                "   -  Extra tools: intrinsic dimensionality estimation, spectral selectivity (axes linked to biology),"
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
                f"-  Cells x genes: {n_cells} x {n_genes}\n"
                f"-  Global ID ({tg.id_method}): {int(tg.global_id)}\n"
                f"-  spectral scaffold size: {int(tg.n_eigs)}\n"
                "\n"
                "Hyperparameters\n"
                f"-  k-nearest neighbors for base graph: {base_knn} / metric: {base_metric} / kernel version: {_safe(bk_ver)}\n"
                f"-  k-nearest neighbors for refined graph: {graph_knn} / metric: {graph_metric} / kernel version: {_safe(gk_ver)}\n"
            )
            ax.text(0.0, y, s2, ha='left', va='top', fontsize=11, linespacing=1.30, wrap=True)
            s2_lines = s2.count("\n") + 2
            y -= s2_lines * line_h_small

            # add blank space between sections
            y -= 0.5 * line_h_small

            # SECTION 3: 
            ax.text(0.0, y, "What you can use next:", ha='left', va='top', fontsize=12, weight='bold')
            y -= line_h_medium
            avail = []
            avail.append("-  Spectral scaffold coordinates: in `adata.obsm['X_spectral_scaffold']` and `adata.obsm['X_multiscale_scaffold']`")
            avail.append("  - Construct your own neighborhood graphs on these scaffolds using `sc.pp.neighbors(adata, use_rep='X_spectral_scaffold')`" \
            " to generate custom UMAPs, clusters, etc." )
            avail.append("  - For RNA velocity analyses, use `scv.pp.moments(adata, use_rep='X_msDM with bw_adaptive', n_neighbors=10)` ." )
            avail.append("-  2-D layouts: " + (", ".join(embeddings_available) if embeddings_available else "(none cached)") + " in adata.obsm")
            avail.append("  - Visualize gene expression and metadata in `adata.obs` using `sc.pl.embedding(adata, basis='TopoMAP',...)` ." )
            avail.append("-  Graphs: " + (", ".join(graphs_available) if graphs_available else "(none cached)") + " in adata.obsp")
            avail.append("-  Imputed data: : imputed/smoothed signals stored in `adata.layers['topo_imputation']`." )
            if cluster_keys:
                avail.append("-  Clustering results: " + ", ".join(cluster_keys[:8]) + (" ..." if len(cluster_keys) > 8 else "") + " in adata.obs")
            if best_geo_line:
                avail.append(best_geo_line)

            s3 = "\n".join(avail)
            ax.text(0.0, y, s3, ha='left', va='top', fontsize=11, linespacing=1.30, wrap=True)

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
            ax_top = fig.add_axes([0.04, 0.7, 0.92, 0.20])  # [left, bottom, width, height]
            ax_top.axis('off')
            legend_top = (
                "Geometry preservation compares the diffusion operator on the reference space, P_x (built on adata.X), "
                "to the operator induced by each representation, P_y. When P_y is close to P_x, both global and local geometry are "
                "well preserved. TopoMetry's geometry preservation scores are:\n \n"
                "-  PF1 - Sparse Neighborhood F1: overlap of the top-k transition supports per row; focuses on whether the same neighbors are kept in the operator (weights ignored).\n  \n"
                "-  PJS - Row-wise Jensen-Shannon Similarity: compares the probability distributions of transitions for each cell; sensitive to how mass is redistributed.\n  \n"
                "-  SP - Spectral Procrustes (R^2): aligns multiscale diffusion coordinates (Phi_t) up to a rotation; captures coordinate-level consistency of the geometry.\n  \n"
                "Scores range from 0 to 1.0. Higher is better for all scores."
            )

            ax_top.text(0.0, 0.5, legend_top, ha='left', va='center', fontsize=11, wrap=True)
            if df_eval is None or (hasattr(df_eval, "empty") and df_eval.empty):
                ax.text(0.0, 0.9, "No evaluation results found.", fontsize=11, va='top')
            else:
                # ---- Normalize df into a tidy (reps x metrics) numeric table ----
                _df = df_eval.copy()

                # Two common shapes:
                # (A) wide-by-rep with metric rows -> columns = representations, index = metric names
                # (B) long/tidy with 'representation' column and metric columns
                if "representation" in _df.columns:
                    # Keep only known metric columns; map friendly names with arrows later
                    metric_cols = [c for c in ["PF1","PJS","SP"]
                                if c in _df.columns]
                    reps = _df["representation"].astype(str).tolist()
                    M = _df.set_index("representation")[metric_cols]  # reps x metrics
                else:
                    # Assume rows are metric names; columns are representations
                    # Make reps x metrics
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

                # Pretty row labels with direction hints (higher/lower)
                row_labels_map = {
                    "PF1":             "PF1 (higher)",
                    "PJS":             "PJS (higher)",
                    "SP":              "SP (higher)",
                }
                display_rows = [row_labels_map.get(c, c) for c in metric_cols]

                # Determine best-per-column (bold the max for "higher", min for "lower")
                # Build a boolean mask same shape as table: best_mask[i_row, j_col]
                is_high_better = np.array([("lower" not in row_labels_map.get(c, "").lower()) for c in metric_cols], dtype=bool)
                data_vals = M.values.astype(float)
                best_mask = np.zeros_like(data_vals, dtype=bool)
                for j in range(data_vals.shape[0]):  # iterate rows? - careful: M is reps x metrics
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
                ax.set_position([0.02, 0.18, 0.98, 0.64])  # left, bottom, width, height - tighter band
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


            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)

            # ===== PART 2 : EIGENSPECTRUM / I.D. (1x4) =====
            fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
            gs = fig.add_gridspec(2, 2, left=0.04, right=0.98, top=0.92, bottom=0.12, wspace=0.28, hspace=0.35)
            # Row 1: spectrum & diff
            ax_curve = fig.add_subplot(gs[0, 0]); ax_diff = fig.add_subplot(gs[0, 1])
            evals_ms = _eigvals_from_tg(tg, variant='msDM')
            _decay_plot_axes_original(ax_curve, ax_diff, evals_ms, title="Eigenspectrum & Eigengap")

            # ID histograms (original style)
            ax_fsa = fig.add_subplot(gs[1, 0]); ax_mle = fig.add_subplot(gs[1, 1])
            id_est = adata.uns.get('intrinsic_dim_estimator', None)
            plot_id_histograms(adata, ax_fsa, ax_mle, id_est)

            #fig.suptitle("Scaffold eigenspectrum / intrinsic dimensionality", y=0.98, fontsize=14)
            fig.text(0.02, 0.98, "Scaffold eigenspectrum and intrinsic dimensionality estimates", fontsize=16, weight='bold', va='center', ha='left')
            pdf.savefig(fig, dpi=dpi); plt.close(fig)

            # ===== PART 3 : CLUSTERING (1x3/2x3 grid) =====
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
                #fig.suptitle(f"Clustering across resolutions - {variant}", y=0.98, fontsize=14)
                fig.text(0.02, 0.98, f"Clustering across resolutions - {variant}", fontsize=14, weight='bold', va='center', ha='left')
                pdf.savefig(fig, dpi=dpi); plt.close(fig)


            # ===== PART 4 : RIEMANN DIAGNOSTICS: one page per embedding (1x3 + bottom text) =====
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

                # Let the helper build the 1x3 panel figure for this embedding.
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

                    # 2) Reserve margins so subplots don't overcrop each other
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
                        "-  Localized indicatrices - Each ellipse summarizes how distances are stretched or squashed around a point. "
                        "Long ellipses indicate a preferred direction of variation (anisotropy); small round ellipses indicate locally "
                        "uniform structure. Comparing anisotropy across clusters can reveal biological trends.\n"
                        "-  Global indicatrices (overlay) - A coarse grid of ellipses shows the overall deformation field on the embedding. "
                        "The background encodes the centered log-determinant of the metric (blue = contraction, red = expansion). "
                        "Consistent colors/ellipses often align with transitions (e.g., differentiation) or boundaries between states.\n"
                        "-  Local contraction/expansion - Points colored by the same deformation score highlight compressed (blue) or dilated (red) regions, "
                        "useful for spotting bottlenecks, hubs, or spread-out manifolds in the cellular landscape."
                    )
                    # left=0.04, baseline ~0.12 from bottom; anchor at top of band
                    fig.text(0.08, 0.18, guide, ha='left', va='top', fontsize=11, wrap=True)

                    ax_exp = fig.add_axes([0.035, 0.07, 0.93, 0.16])
                    ax_exp.axis('off')
                    pdf.savefig(fig, dpi=dpi)
                finally:
                    plt.close(fig)

            # ===== PART 5 : I.D. EMBEDDINGS (2x4 grid) =====
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

            # ===== PART 6: SPECTRAL SELECTIVITY (2x4) =====
            for variant in variant_order:
                fig = plt.figure(figsize=a4_landscape_inches, dpi=dpi)
                # Leave a bit more bottom margin to fit the legend band.
                gs = fig.add_gridspec(
                    2, 4,
                    left=0.04, right=0.98, top=0.92, bottom=0.14,
                    wspace=0.25, hspace=0.30
                )

                # Row 1 -  TopoMAP
                ax = fig.add_subplot(gs[0, 0]); _embedding(ax, 'spectral_EAS',      'TopoMAP',    variant, title='EAS',      cmap='Reds', colorbar_loc=None)
                ax = fig.add_subplot(gs[0, 1]); _embedding(ax, 'spectral_RayScore', 'TopoMAP',    variant, title='RayScore', cmap='Reds', colorbar_loc=None)
                ax = fig.add_subplot(gs[0, 2]); _embedding(ax, 'spectral_LAC',      'TopoMAP',    variant, title='LAC',      cmap='Reds', colorbar_loc=None)
                ax = fig.add_subplot(gs[0, 3]); _embedding(ax, 'spectral_radius',   'TopoMAP',    variant, title='Radius',   cmap='Reds', colorbar_loc=None)

                # Row 2 -  TopoPaCMAP
                ax = fig.add_subplot(gs[1, 0]); _embedding(ax, 'spectral_EAS',      'TopoPaCMAP', variant, title='EAS',      cmap='Reds', colorbar_loc=None)
                ax = fig.add_subplot(gs[1, 1]); _embedding(ax, 'spectral_RayScore', 'TopoPaCMAP', variant, title='RayScore', cmap='Reds', colorbar_loc=None)
                ax = fig.add_subplot(gs[1, 2]); _embedding(ax, 'spectral_LAC',      'TopoPaCMAP', variant, title='LAC',      cmap='Reds', colorbar_loc=None)
                ax = fig.add_subplot(gs[1, 3]); _embedding(ax, 'spectral_radius',   'TopoPaCMAP', variant, title='Radius',   cmap='Reds', colorbar_loc=None)

                # Bottom legend band (wrapped text; small font; sits under the grid)
                leg_ax = fig.add_axes([0.04, 0.045, 0.92, 0.06])  # x, y, w, h (fractions of figure)
                leg_ax.axis('off')
                legend_text = (
                    "-  EAS (Entropy-based Axis Selectivity): in [0,1]; higher means each cell's energy is concentrated on a single spectral axis. "
                    "Computed from squared, standardized scaffold coordinates with eigenvalue weights (default lambda/(1 - lambda)).\n"
                    "-  RayScore: highlights coherent radial progressions along a dominant axis; defined as sigmoid(neighborhood radial z-score) * EAS. "
                    "Large values indicate ray-like trajectories pointing outward; axis sign is stored separately.\n"
                    "-  LAC (Local Axial Coherence): fraction of local variance explained by the first principal component (EVR_1) within k-NN; "
                    "near 1.0 indicates locally 1-D structure aligned with a single axis.\n"
                    "-  Radius (spectral radius): Euclidean norm of standardized scaffold coordinates (||Z||_2); a proxy for distance from the origin in spectral space, "
                    "often correlating with progress along diffusion time."
                )
                leg_ax.text(0.0, 0.5, legend_text, ha='left', va='center', fontsize=9, wrap=True)

                #fig.suptitle(f"Spectral selectivity - {variant}", y=0.98, fontsize=12)
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

            ax_h1 = fig.add_subplot(gs[1, 0]); _plot_heatmap(ax_h1, corr_raw, "Gene-gene corr (raw)")
            ax_h2 = fig.add_subplot(gs[1, 1]); _plot_heatmap(ax_h2, corr_imp, "Gene-gene corr (imputed @ best t)")

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
                        "QC compares mean absolute gene-gene correlations against null (per-gene permutations) across t. \n"
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
                "-  The filtered pure-noise mean/std visualizes diffusion smoothing under a null model.\n"
                "-  Graph Total Variation (GTV) decreases after filtering (smoother signals).\n"
                "-  Spectral energy shifts towards low-frequency modes after diffusion.\n"
                f"GTV raw: {_fmt(gtv_raw)} | GTV filtered: {_fmt(gtv_flt)} | delta: {_fmt(gtv_raw - gtv_flt)} | "
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
        Run the full TopoMetry analysis pipeline and emit a consolidated PDF report.

        This is a convenience wrapper around `run_topometry_analysis(...)` followed by
        `plot_topometry_report(...)`, forwarding the relevant options to each stage.

        Parameters
        ----------
        adata : AnnData
            Input dataset; will be populated with TopoMetry outputs.
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
    # Integration methods
    # ----------------------------------------------------



    # -----------------------
    # Internal helpers
    # -----------------------
    def _ensure_rep(adata: "AnnData", use_rep: Optional[str], *, n_pcs: int = 50) -> str:
        """Ensure a representation exists and return its key.
        If `use_rep` is None or 'X_pca', computes PCA when missing.
        """
        key = use_rep or "X_pca"
        if key == "X_pca" and "X_pca" not in adata.obsm:
            sc.tl.pca(adata, n_comps=int(n_pcs))
        if key not in adata.obsm and key != "X_pca":
            raise KeyError(f"`use_rep='{key}'` not found in adata.obsm")
        return key

    def _graph_backup(adata: "AnnData", method: str):
        """Copy current graph slots to namespaced backups in .obsp.
        Avoids putting large sparse matrices into .uns to keep h5ad IO safe.
        """
        if "connectivities" in adata.obsp:
            adata.obsp[f"connectivities__backup_before_{method}"] = adata.obsp["connectivities"].copy()
        if "distances" in adata.obsp:
            adata.obsp[f"distances__backup_before_{method}"] = adata.obsp["distances"].copy()

    def _persist_meta(adata: "AnnData", method: str, params: Dict[str, Any]):
        d = adata.uns.setdefault("topometry_integration", {})
        md = d.setdefault(method, {})
        # store only JSON-serializable fields
        md["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        md["params"] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                         for k, v in dict(params).items()}

    def _neighbors_on_rep(adata: "AnnData", rep_key: str, *, method: str, n_neighbors: int = 30, metric: str = "euclidean"):
        key_added = f"neighbors_{method}"
        sc.pp.neighbors(adata, use_rep=rep_key, n_neighbors=int(n_neighbors), metric=metric, key_added=key_added)
        # method-specific graph copies for convenience
        adata.obsp[f"connectivities_{method}"] = adata.uns[key_added]["connectivities"].copy()
        adata.obsp[f"distances_{method}"] = adata.uns[key_added]["distances"].copy()
        return key_added


    # inside the _HAVE_SCANPY loop
    # -----------------------
    # Public API
    # -----------------------

    # local import (same package)
    from topo.topograph import TopOGraph  # adjust relative import if needed

    def _temp_obsm(adata, key: str, arr):
        """Context manager: temporarily attach an array to adata.obsm[key]."""
        class _Ctx:
            def __init__(self, adata, key, arr):
                self.adata, self.key = adata, key
                self.had = key in adata.obsm
                self.prev = adata.obsm[key] if self.had else None
                self.arr = arr
            def __enter__(self):
                self.adata.obsm[self.key] = self.arr
                return self.key
            def __exit__(self, exc_type, exc, tb):
                if self.had:
                    self.adata.obsm[self.key] = self.prev
                else:
                    del self.adata.obsm[self.key]
        return _Ctx(adata, key, arr)

    def _asarray_X(adata):
        X = adata.X
        if sp.issparse(X):
            return X.toarray()
        return np.asarray(X)

    def _ensure_tg(adata, tg: Optional["TopOGraph"]):
        """Use provided TopOGraph or create a new one (no PCA ever)."""
        if tg is not None:
            return tg
        # Fresh TopOGraph: we'll pass precomputed graphs (base_metric='precomputed')
        return TopOGraph(
            base_metric='precomputed',
            graph_metric='euclidean',
            base_knn=30,
            graph_knn=30,
            random_state=42,
            verbosity=0,
        )

    # --- robust unwrap for scanpy.external.pp.mnn_correct outputs ---
    def _unwrap_anndata(obj):
        """
        Scanpy/mnnpy wrappers sometimes return nested (tuple/list) structures.
        This walks down until we hit an AnnData-like object (has .obs_names).
        """
        cur = obj
        while isinstance(cur, (list, tuple)):
            if len(cur) == 0:
                raise ValueError("Empty MNN result.")
            cur = cur[0]
        if not hasattr(cur, "obs_names"):
            raise TypeError("MNN result does not contain an AnnData-like object.")
        return cur

    def _store_layouts(adata, tg: "TopOGraph", suffix: str):
        """Save TopoMAP/TopoPaCMAP under method-specific, capitalized keys."""
        try:
            tg.project(projection_method="MAP", multiscale=False)
        except Exception:
            pass
        try:
            tg.project(projection_method="PaCMAP", multiscale=False)
        except Exception:
            pass
        if hasattr(tg, "TopoMAP"):
            adata.obsm[f"X_TopoMAP_{suffix}"] = np.array(tg.TopoMAP, copy=True)
        if hasattr(tg, "TopoPaCMAP"):
            adata.obsm[f"X_TopoPaCMAP_{suffix}"] = np.array(tg.TopoPaCMAP, copy=True)

    def _persist_meta_integration(adata, method: str, extra: dict):
        _persist_meta(adata, method, extra)

    def _knn_from_topo(X: np.ndarray, k: int, metric: str, n_jobs: int = -1, backend: str = "hnswlib"):
        """
        Build a kNN graph using topo.base.ann.kNN and return a CSR adjacency (symmetric).
        topo.base.ann.kNN returns a distance-weighted CSR; we keep it as-is since
        TopOGraph kernels expect distances (TopOGraph handles symmetrization/expansion).
        """
        from topo.base import ann as _ann
        knn = _ann.kNN(
            X,
            n_neighbors=int(k),
            metric=metric,
            n_jobs=n_jobs,
            backend=backend,          # respects TopOGraph backend choice
            return_instance=False,
            verbose=False,
        )
        # Ensure CSR & float32
        knn = knn.tocsr().astype(np.float32, copy=False)
        return knn

    def integrate_bbknn(
        adata: "AnnData",
        *,
        batch_key: str,
        neighbors_within_batch: int = 3,
        tg: "TopOGraph" = None,
        **kwargs,
    ) -> "AnnData":
        """
        BBKNN-backed integration grounded on TopoGraph (no PCA).
        - Build a **base** batch-balanced neighbor graph on X using BBKNN -> P_of_X (tg).
        - Compute spectral scaffold (eigs) in tg.
        - Build **refined** batch-balanced graph on the (ms)scaffold using BBKNN.
        - Compute TopoMAP/TopoPaCMAP, store with suffix '_BBKNN'.

        Fix: match TopoGraph's n_neighbors to the **actual degree** of the BBKNN graph
        to avoid adaptive-bandwidth median index errors.
        """
        try:
            import bbknn  # noqa: F401
        except ImportError:
            raise ImportError("BBKNN integration requires `bbknn` (pip install bbknn).")

        tg = _ensure_tg(adata, tg)

        # ---------- Base graph on X via BBKNN ----------
        X = _asarray_X(adata)
        with _temp_obsm(adata, "_bbknn_X", X):
            sc.external.pp.bbknn(
                adata,
                batch_key=batch_key,
                use_rep="_bbknn_X",
                neighbors_within_batch=int(neighbors_within_batch),
                **kwargs,
            )
        if "connectivities" not in adata.obsp:
            raise RuntimeError("BBKNN failed to populate `.obsp['connectivities']`.")

        knn_X = adata.obsp["connectivities"].tocsr().astype(np.float32, copy=True)
        degX = np.diff(knn_X.indptr)
        k_base_eff = int(max(3, degX.min()))  # must be <= min degree to avoid index errors

        # Fit TopOGraph from precomputed base graph using the effective k
        tg.base_metric = "precomputed"
        prev_base_knn = getattr(tg, "base_knn", None)
        tg.base_knn = k_base_eff
        tg.fit(X=knn_X)  # computes P_of_X, eigenbasis, initial refined graphs
        if prev_base_knn is not None:
            tg.base_knn = prev_base_knn  # restore if you want to keep original hyperparameter

        # ---------- Refined graph on spectral scaffold via BBKNN ----------
        msZ = tg.spectral_scaffold(multiscale=True)
        with _temp_obsm(adata, "_bbknn_msZ", msZ):
            sc.external.pp.bbknn(
                adata,
                batch_key=batch_key,
                use_rep="_bbknn_msZ",
                neighbors_within_batch=int(neighbors_within_batch),
                **kwargs,
            )
        knn_msZ = adata.obsp["connectivities"].tocsr().astype(np.float32, copy=True)
        deg_msZ = np.diff(knn_msZ.indptr)
        k_msZ_eff = int(max(3, deg_msZ.min()))

        tg._knn_msZ = knn_msZ
        tg._kernel_msZ, _ = tg._compute_kernel_from_version_knn(
            tg._knn_msZ, k_msZ_eff, tg.graph_kernel_version, tg.GraphKernelDict,
            suffix=" from msDM (BBKNN)", low_memory=tg.low_memory, base=False, data_for_expansion=msZ
        )

        # ---------- Optional: refined graph on fixed-time DM (Z) via BBKNN ----------
        Z = tg.spectral_scaffold(multiscale=False)
        with _temp_obsm(adata, "_bbknn_Z", Z):
            sc.external.pp.bbknn(
                adata,
                batch_key=batch_key,
                use_rep="_bbknn_Z",
                neighbors_within_batch=int(neighbors_within_batch),
                **kwargs,
            )
        tg._knn_Z = adata.obsp["connectivities"].tocsr().astype(np.float32, copy=True)
        deg_Z = np.diff(tg._knn_Z.indptr)
        k_Z_eff = int(max(3, deg_Z.min()))

        tg._kernel_Z, _ = tg._compute_kernel_from_version_knn(
            tg._knn_Z, k_Z_eff, tg.graph_kernel_version, tg.GraphKernelDict,
            suffix=" from DM (BBKNN)", low_memory=tg.low_memory, base=False, data_for_expansion=Z
        )

        # ---------- Projections & storage (TopOMAP / TopoPaCMAP) ----------
        _store_layouts(adata, tg, "BBKNN")
        # Also write capitalized TopoMetry-style keys expected by plotting code
        if hasattr(tg, "TopoMAP"):
            adata.obsm["X_TopoMAP_BBKNN"] = np.array(tg.TopoMAP, copy=True)
        if hasattr(tg, "TopoPaCMAP"):
            adata.obsm["X_TopoPaCMAP_BBKNN"] = np.array(tg.TopoPaCMAP, copy=True)

        # Optional: keep operators for inspection
        adata.obsp["P_of_X_BBKNN"] = tg.P_of_X
        adata.obsp["P_of_msZ_BBKNN"] = tg.P_of_msZ
        adata.obsp["P_of_Z_BBKNN"] = tg.P_of_Z

        _persist_meta_integration(
            adata,
            "bbknn",
            {
                "batch_key": batch_key,
                "neighbors_within_batch": int(neighbors_within_batch),
                "k_base_eff": k_base_eff,
                "k_msZ_eff": k_msZ_eff,
                "k_Z_eff": k_Z_eff,
            },
        )
        return adata



    def integrate_mnn(
        adata: "AnnData",
        *,
        batch_key: str,
        var_subset: Optional[list[str]] = None,
        n_jobs: Optional[int] = None,
        tg: "TopOGraph" = None,
        **kwargs,
    ) -> "AnnData":
        """
        MNN-backed integration on TopoGraph (no PCA).
        Workflow:
        1) Correct X via scanpy.external.pp.mnn_correct.
        2) Base graph on corrected X -> P_of_X (tg).
        3) Refined graph on (ms)Z computed by TopoGraph.
        4) Store TopoMAP/TopoPaCMAP with suffix '_MNN'.
        """
        res = sc.external.pp.mnn_correct(
            adata, batch_key=batch_key, var_subset=var_subset, n_jobs=n_jobs, **kwargs
        )
        adata_corr = _unwrap_anndata(res)

        # Align to original obs order robustly (don't assume same/contiguous)
        idx = adata_corr.obs_names.get_indexer(adata.obs_names)
        if np.any(idx < 0):
            raise ValueError("MNN corrected AnnData is missing some original cells.")
        Xcorr = adata_corr.X[idx]
        adata.layers["X_corrected_mnn"] = Xcorr.copy()

        tg = _ensure_tg(adata, tg)

        # Base graph from corrected X (plain kNN on corrected expression)
        X_base = adata.layers["X_corrected_mnn"]
        if sp.issparse(X_base):
            X_base = X_base.toarray()
        tg.base_metric = "cosine"
        tg.fit(X=X_base)

        # Refined graph already built by tg.fit(); compute/store layouts
        _store_layouts(adata, tg, "MNN")
        adata.obsm["X_spectral_scaffold_MNN"] = tg.spectral_scaffold(multiscale=True)
        adata.obsp["P_of_X_MNN"] = tg.P_of_X
        adata.obsp["P_of_msZ_MNN"] = tg.P_of_msZ
        adata.obsp["P_of_Z_MNN"] = tg.P_of_Z

        _persist_meta_integration(
            adata, "mnn",
            {"batch_key": batch_key, "var_subset": None if var_subset is None else len(var_subset)}
        )
        return adata



    def integrate_scanorama(
        adata: "AnnData",
        *,
        batch_key: str,
        out_dim: Optional[int] = None,
        tg: "TopOGraph" = None,
        multiscale: bool = True,
        **kwargs,
    ) -> "AnnData":
        """
        Scanorama correction on spectral scaffold (never PCA):
        - Fit tg on X if needed (to get base operator).
        - Run scanorama on a TEMP batch-sorted view of the scaffold to avoid the
            'non-contiguous batches' error, then restore original order.
        - Rebuild refined kernel on corrected scaffold and project.

        Note: Some Scanorama versions don't support `out_dim`. We therefore never
        pass it to the wrapper, and if `out_dim` is provided we truncate after.
        """
        tg = _ensure_tg(adata, tg)
        if tg.base_kernel is None and tg.base_knn_graph is None:
            tg.base_metric = "cosine"
            tg.fit(X=adata.X)

        Z = tg.spectral_scaffold(multiscale=multiscale)
        key_in = "_scanorama_scaffold"

        # --- sort by batch to ensure contiguity for the wrapper ---
        b = np.asarray(adata.obs[batch_key]).astype(str)
        order = np.argsort(b)
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)

        A_sorted = adata[order].copy()
        with _temp_obsm(A_sorted, key_in, Z[order]):
            # IMPORTANT: do NOT pass out_dim; older scanorama.assemble() rejects it
            sc.external.pp.scanorama_integrate(
                A_sorted, key=batch_key, basis=key_in, **kwargs
            )

        # retrieve corrected scaffold and unsort back
        if "X_scanorama" in A_sorted.obsm:
            Zc_sorted = A_sorted.obsm["X_scanorama"]
        else:
            Zc_sorted = A_sorted.obsm[key_in]
        Zc = np.asarray(Zc_sorted)[inv]

        # If the caller asked for a specific dimensionality, truncate here
        if out_dim is not None and Zc.shape[1] > int(out_dim):
            Zc = Zc[:, :int(out_dim)]

        # Rebuild refined kernel on corrected scaffold using Topo's ANN kNN
        knn_ref = _knn_from_topo(
            Zc,
            k=tg.graph_knn,
            metric=tg.graph_metric,
            n_jobs=getattr(tg, "n_jobs", -1),
            backend=getattr(tg, "backend", "hnswlib"),
        )

        if multiscale:
            tg._knn_msZ = knn_ref
            tg._kernel_msZ, _ = tg._compute_kernel_from_version_knn(
                tg._knn_msZ, tg.graph_knn, tg.graph_kernel_version, tg.GraphKernelDict,
                suffix=" (scanorama msZ)", low_memory=tg.low_memory, base=False, data_for_expansion=Zc
            )
        else:
            tg._knn_Z = knn_ref
            tg._kernel_Z, _ = tg._compute_kernel_from_version_knn(
                tg._knn_Z, tg.graph_knn, tg.graph_kernel_version, tg.GraphKernelDict,
                suffix=" (scanorama Z)", low_memory=tg.low_memory, base=False, data_for_expansion=Zc
            )

        adata.obsm["X_spectral_scaffold_scanorama"] = Zc
        _store_layouts(adata, tg, "scanorama")
        adata.obsp["P_of_msZ_scanorama"] = tg.P_of_msZ
        adata.obsp["P_of_Z_scanorama"] = tg.P_of_Z
        _persist_meta_integration(adata, "scanorama", {"batch_key": batch_key, "multiscale": bool(multiscale), "out_dim": out_dim})
        return adata

    def integrate_harmony(
        adata: "AnnData",
        *,
        batch_key: str,
        tg: "TopOGraph" = None,
        multiscale: bool = False,
        adjusted_basis_key: str = "_harmony_scaffold",
        # --- improvements / knobs ---
        standardize: bool = True,
        whiten: bool = True,
        coarse_pass: bool = True,
        coarse_params: Optional[dict] = None,
        fine_params: Optional[dict] = None,
        reference_values: Optional[list] = None,
        blend_bbknn: bool = False,
        bbknn_neighbors_within_batch: int = 2,
        blend_alpha: float = 0.3,
        target_min_degree: Optional[int] = None,
        report_metrics: bool = True,
        **kwargs,
    ) -> "AnnData":
        """
        Harmony correction on the TopoMetry spectral scaffold (no PCA).

        Pipeline (scaffold-centric):
        1) Ensure TopOGraph P(X) is fitted on adata.X (never PCA).
        2) Build scaffold Z (fixed-time) and/or msZ (coarse multiscale).
        3) Standardize (and optionally rotate-whiten) the scaffold(s).
        4) Optional coarse Harmony pass on msZ; then fine Harmony pass on Z (warm-start via concatenation).
        5) Rebuild refined kNN from corrected scaffold Zc using Topo's ANN kNN.
        6) Optional blend with a light BBKNN graph on Zc.
        7) Optional enforce a minimum degree by increasing k once.
        8) Rebuild kernels and project TopoMAP / TopoPaCMAP.
        9) (Optional) record light diagnostics (batch-mixing@k before/after).

        Parameters
        ----------
        standardize : bool
            Z-score columns of the scaffold before Harmony.
        whiten : bool
            Apply rotation-only whitening (U from SVD) to standardized scaffold(s).
        coarse_pass : bool
            Run a coarse Harmony pass on msZ before the fine pass on Z.
        coarse_params, fine_params : dict
            Harmony parameters for coarse/fine passes. If None, sensible defaults are used.
        reference_values : list[str] or None
            Optional reference batch names for Harmony (anchors correction).
        blend_bbknn : bool
            If True, blend refined kNN on Zc with a BBKNN graph built on Zc to reduce seams.
        bbknn_neighbors_within_batch : int
            BBKNN's neighbors_within_batch used when blending.
        blend_alpha : float
            Blend weight in [0,1]; final_knn = (1 - alpha) * kNN + alpha * BBKNN.
        target_min_degree : int or None
            If set, rebuild the kNN once with k = max(graph_knn, target_min_degree) to avoid under-connected cells.
        report_metrics : bool
            Compute simple batch-mixing@k before/after and store in .uns['topometry_integration']['harmony']['metrics'].
        """
        try:
            import harmonypy  # noqa: F401
        except Exception:
            # Scanpy wrapper will error if Harmony is actually missing
            pass

        # ---- local helpers ----
        def _zscore(A):
            if not standardize:
                return A
            m = A.mean(0)
            s = A.std(0) + 1e-8
            return (A - m) / s

        def _rot_whiten(A):
            if not whiten:
                return A
            U, _, _ = np.linalg.svd(np.nan_to_num(A), full_matrices=False)
            return U  # rotation-only whitening (preserves scale in a stable way)

        def _batch_mixing_at_k(emb, labels, k):
            # fraction of neighbors with a batch != self batch
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(k + 1, max(2, emb.shape[0] - 1)), metric="euclidean").fit(emb)
            inds = nn.kneighbors(emb, return_distance=False)[:, 1:]
            labs = np.asarray(labels)
            return float((labs[inds] != labs[:, None]).mean())

        # ---- ensure TopOGraph base operator ----
        tg = _ensure_tg(adata, tg)
        if tg.base_kernel is None and tg.base_knn_graph is None:
            tg.base_metric = "cosine"
            tg.fit(X=adata.X)

        # ---- get scaffold(s) and standardize/whiten ----
        Z = tg.spectral_scaffold(multiscale=False)      # fixed-time
        msZ = tg.spectral_scaffold(multiscale=True)     # coarse multiscale

        Zw = _rot_whiten(_zscore(Z))
        msZw = _rot_whiten(_zscore(msZ))

        # ---- default Harmony params ----
        if coarse_params is None:
            coarse_params = dict(
                nclust=min(100, max(30, int(np.sqrt(max(100, adata.n_obs) / 100)))),
                theta=4.0,
                lambda_=4.0,
                max_iter_harmony=30,
                max_iter_kmeans=20,
                epsilon_harmony=1e-3,
                epsilon_cluster=1e-3,
            )
        if fine_params is None:
            fine_params = dict(
                nclust=min(80, max(30, int(np.sqrt(max(100, adata.n_obs) / 120)))),
                theta=2.0,
                lambda_=2.0,
                max_iter_harmony=25,
                max_iter_kmeans=20,
                epsilon_harmony=1e-3,
                epsilon_cluster=1e-3,
            )

        # Optionally include reference batches
        if reference_values is not None:
            coarse_params = {**coarse_params, "reference_values": reference_values}
            fine_params = {**fine_params, "reference_values": reference_values}

        # ---- metrics (pre) ----
        metrics = {}
        if report_metrics:
            labels = adata.obs[batch_key].astype(str).values
            try:
                metrics["mixing_pre@{}".format(tg.graph_knn)] = _batch_mixing_at_k(Zw, labels, tg.graph_knn)
            except Exception:
                pass

        # ---- Harmony coarse pass on msZ (optional) ----
        if coarse_pass:
            ms_key = "_harmony_msZ_basis"
            with _temp_obsm(adata, ms_key, msZw):
                sc.external.pp.harmony_integrate(adata, key=batch_key, basis=ms_key, **coarse_params)
            msZc = adata.obsm.get(ms_key, None)
            if msZc is None:
                # some versions may rename; try common fallbacks
                msZc = adata.obsm.get("_X_harmony", None) or adata.obsm.get("X_pca_harmony", None)
            if msZc is None:
                raise KeyError("Harmony (coarse) did not return an adjusted basis on msZ.")
            msZc = np.asarray(msZc)
            # warm-start by concatenation with fine scaffold
            fine_basis = np.c_[Zw, msZc]
        else:
            fine_basis = Zw

        # ---- Harmony fine pass on Z (warm-started by coarse if used) ----
        with _temp_obsm(adata, adjusted_basis_key, fine_basis):
            sc.external.pp.harmony_integrate(adata, key=batch_key, basis=adjusted_basis_key, **fine_params)

        Zc = adata.obsm.get(adjusted_basis_key, None)
        if Zc is None:
            Zc = adata.obsm.get("_X_harmony", None) or adata.obsm.get("X_pca_harmony", None)
            if Zc is None:
                raise KeyError("Harmony (fine) did not return an adjusted basis on the scaffold.")
        Zc = np.asarray(Zc)

        # If we concatenated with msZc, truncate back to original Z dims
        if Zc.shape[1] > Z.shape[1]:
            Zc = Zc[:, : Z.shape[1]]

        # ---- refined kNN on corrected scaffold using Topo's ANN ----
        def _ref_knn(Z_embed, k):
            return _knn_from_topo(
                Z_embed,
                k=k,
                metric=tg.graph_metric,
                n_jobs=getattr(tg, "n_jobs", -1),
                backend=getattr(tg, "backend", "hnswlib"),
            )

        k_use = int(tg.graph_knn)
        knn_ref = _ref_knn(Zc, k_use)

        # Optional: blend with a light BBKNN on Zc to erase residual seams
        if blend_bbknn:
            tmp_key = "_bbknn_on_Zc"
            with _temp_obsm(adata, tmp_key, Zc):
                sc.external.pp.bbknn(
                    adata,
                    batch_key=batch_key,
                    use_rep=tmp_key,
                    neighbors_within_batch=int(bbknn_neighbors_within_batch),
                )
            if "connectivities" in adata.obsp:
                knn_bb = adata.obsp["connectivities"].tocsr().astype(np.float32, copy=False)
                # convex blend; ensure shapes match
                if knn_bb.shape == knn_ref.shape:
                    knn_ref = (1.0 - float(blend_alpha)) * knn_ref + float(blend_alpha) * knn_bb

        # Optional: enforce minimum degree by increasing k once
        if target_min_degree is not None and target_min_degree > 0:
            deg = np.diff(knn_ref.indptr)
            if int(deg.min()) < int(target_min_degree):
                k_new = max(k_use, int(target_min_degree))
                knn_ref = _ref_knn(Zc, k_new)

        # ---- install refined kernel on tg ----
        if multiscale:
            tg._knn_msZ = knn_ref
            tg._kernel_msZ, _ = tg._compute_kernel_from_version_knn(
                tg._knn_msZ,
                tg.graph_knn,
                tg.graph_kernel_version,
                tg.GraphKernelDict,
                suffix=" (harmony msZ)",
                low_memory=tg.low_memory,
                base=False,
                data_for_expansion=Zc,
            )
        else:
            tg._knn_Z = knn_ref
            tg._kernel_Z, _ = tg._compute_kernel_from_version_knn(
                tg._knn_Z,
                tg.graph_knn,
                tg.graph_kernel_version,
                tg.GraphKernelDict,
                suffix=" (harmony Z)",
                low_memory=tg.low_memory,
                base=False,
                data_for_expansion=Zc,
            )

        # ---- store outputs ----
        adata.obsm["X_spectral_scaffold_harmony"] = Zc
        _store_layouts(adata, tg, "harmony")  # writes X_TopoMAP_harmony / X_TopoPaCMAP_harmony
        adata.obsp["P_of_msZ_harmony"] = tg.P_of_msZ
        adata.obsp["P_of_Z_harmony"] = tg.P_of_Z

        # ---- metrics (post) ----
        if report_metrics:
            try:
                labels = adata.obs[batch_key].astype(str).values
                metrics["mixing_post@{}".format(tg.graph_knn)] = _batch_mixing_at_k(Zc, labels, tg.graph_knn)
            except Exception:
                pass
            # persist
            if "topometry_integration" not in adata.uns:
                adata.uns["topometry_integration"] = {}
            adata.uns["topometry_integration"].setdefault("harmony", {})
            adata.uns["topometry_integration"]["harmony"]["metrics"] = metrics

        _persist_meta_integration(
            adata,
            "harmony",
            {
                "batch_key": batch_key,
                "multiscale": bool(multiscale),
                "standardize": bool(standardize),
                "whiten": bool(whiten),
                "coarse_pass": bool(coarse_pass),
                "blend_bbknn": bool(blend_bbknn),
                "bbknn_neighbors_within_batch": int(bbknn_neighbors_within_batch),
                "blend_alpha": float(blend_alpha),
                "target_min_degree": (None if target_min_degree is None else int(target_min_degree)),
                "reference_values": reference_values,
                "coarse_params": coarse_params,
                "fine_params": fine_params,
            },
        )
        return adata



    # def integrate_harmony(
    #     adata: "AnnData",
    #     *,
    #     batch_key: str,
    #     tg: "TopOGraph" = None,
    #     multiscale: bool = True,
    #     adjusted_basis_key: str = "_harmony_scaffold",
    #     **kwargs,
    # ) -> "AnnData":
    #     """
    #     Harmony correction on spectral scaffold:
    #     - Fit tg on X if needed.
    #     - Run harmony on scaffold via temp obsm key.
    #     - Rebuild refined kernel from corrected scaffold using a robust kNN CSR.
    #     """
    #     try:
    #         import harmonypy  # noqa: F401
    #     except Exception:
    #         pass

    #     tg = _ensure_tg(adata, tg)
    #     if tg.base_kernel is None and tg.base_knn_graph is None:
    #         tg.base_metric = "cosine"
    #         tg.fit(X=adata.X)

    #     Z = tg.spectral_scaffold(multiscale=multiscale)
    #     Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
        
    #     # optional whitening
    #     U, S, _ = np.linalg.svd(np.nan_to_num(Zs), full_matrices=False)
    #     Zw = (U * 1.0)  # pure-rotation whitening (or use U/S for full whitening)
    #     # prefer rotation-only whitening to avoid over-amplifying noise


    #     with _temp_obsm(adata, adjusted_basis_key, Zw):
    #         sc.external.pp.harmony_integrate(adata, key=batch_key, basis=adjusted_basis_key, **kwargs)

    #     if adjusted_basis_key not in adata.obsm:
    #         Zc = adata.obsm.get("_X_harmony", None) or adata.obsm.get("X_pca_harmony", None)
    #         if Zc is None:
    #             raise KeyError("Harmony did not return an adjusted basis on the scaffold.")
    #     else:
    #         Zc = adata.obsm[adjusted_basis_key]
    #     Zc = np.asarray(Zc)

    #     # Rebuild refined kernel on corrected scaffold
    #     knn_ref = _knn_from_topo(
    #                         Zc,
    #                         k=tg.graph_knn,
    #                         metric=tg.graph_metric,
    #                         n_jobs=getattr(tg, "n_jobs", -1),
    #                         backend=getattr(tg, "backend", "hnswlib"),
    #                     )
    #     if multiscale:
    #         tg._knn_msZ = knn_ref
    #         tg._kernel_msZ, _ = tg._compute_kernel_from_version_knn(
    #             tg._knn_msZ, tg.graph_knn, tg.graph_kernel_version, tg.GraphKernelDict,
    #             suffix=" (harmony msZ)", low_memory=tg.low_memory, base=False, data_for_expansion=Zc
    #         )
    #     else:
    #         tg._knn_Z = knn_ref
    #         tg._kernel_Z, _ = tg._compute_kernel_from_version_knn(
    #             tg._knn_Z, tg.graph_knn, tg.graph_kernel_version, tg.GraphKernelDict,
    #             suffix=" (harmony Z)", low_memory=tg.low_memory, base=False, data_for_expansion=Zc
    #         )

    #     adata.obsm["X_spectral_scaffold_harmony"] = Zc
    #     _store_layouts(adata, tg, "harmony")
    #     adata.obsp["P_of_msZ_harmony"] = tg.P_of_msZ
    #     adata.obsp["P_of_Z_harmony"] = tg.P_of_Z

    #     _persist_meta_integration(adata, "harmony", {"batch_key": batch_key, "multiscale": bool(multiscale)})
    #     return adata






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

        A wrapper around TopoMetry's topological workflow. Clustering is performed
        with the leiden algorithm on TopoMetry's topological graphs. This wrapper takes an AnnData object containing a
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



    def pca_explained_variance_by_hvg(adata, title='PCA explained variance', n_pcs=200, gene_number_range=[1000, 2000, 3000, 'Default'], figsize=(8, 6), sup_title_fontsize=30, title_fontsize=20, return_dicts=False):
        """
        Plots and saves the explained variance by PCA with varying numbers of highly variable genes, including a default option.

        Parameters
        ----------
        adata: AnnData
            The target AnnData object.

        output_path: str
            Path to save the plot.

        title: str (optional, default 'PCA explained variance').

        n_pcs: int (optional, default 200).
            Number of principal components to use.

        gene_number_range: list of int or 'Default' (optional, default [250, 1000, 3000, 'Default']).
            List of numbers of highly variable genes to test, including 'Default'.

        figsize: tuple of int (optional, default (12,6)).

        sup_title_fontsize: int (optional, default 20).

        title_fontsize: int (optional, default 16).

        return_dicts: bool (optional, default False).
            Whether to return explained covariance ratio and singular values dictionaries.

        Returns
        -------
        Saves a plot as a .tif file. If `return_dicts=True`, also returns a tuple of dictionaries (explained_cov_ratio, singular_values).
        """
        from sklearn.decomposition import PCA
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        explained_cov_ratio = {}
        singular_values = {}
        for n_genes in gene_number_range:
            if n_genes == 'Default':
                sc.pp.highly_variable_genes(adata)
            else:
                sc.pp.highly_variable_genes(adata, n_top_genes=n_genes)
            adata_sub = adata[:, adata.var.highly_variable].copy()
            sc.pp.scale(adata_sub, max_value=10)
            pca = PCA(n_components=n_pcs)
            pca.fit(adata_sub.X)
            adata_sub.obsm['X_pca'] = pca.transform(adata_sub.X)
            explained_cov_ratio[str(n_genes)] = pca.explained_variance_ratio_
            singular_values[str(n_genes)] = pca.singular_values_

        # Create subplots for eigenspectrum and cumulative explained variance
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
        fig.subplots_adjust(left=0.1, right=0.98, wspace=0.3)

        plt.suptitle(title, fontsize=sup_title_fontsize)

        # Plot eigenspectrum
        axes[0].set_title('Eigenspectrum', fontsize=title_fontsize)
        axes[0].set_xlabel('Principal component', fontsize=title_fontsize - 6)
        axes[0].set_ylabel('Singular values', fontsize=title_fontsize - 6)

        # Plot cumulative explained variance
        axes[1].set_title('Total explained variance', fontsize=title_fontsize)
        axes[1].set_xlabel('Principal component', fontsize=title_fontsize - 6)
        axes[1].set_ylabel('Cumulative explained variance', fontsize=title_fontsize - 6)

        for j, gene_number in enumerate(gene_number_range):
            axes[0].plot(singular_values[str(gene_number)], label='{} genes'.format(gene_number), color=colors[j])
            axes[1].plot(explained_cov_ratio[str(gene_number)].cumsum(), label='{} genes'.format(gene_number), color=colors[j])

        axes[0].legend(title='Number of HVGs', fontsize=14)
        axes[1].legend(title='Number of HVGs', fontsize=14)

        plt.tight_layout()
        plt.show()

        if return_dicts:
            return explained_cov_ratio, singular_values


    # =====================================================================
    # Scaffold feature modes: gene–component association matrix
    # =====================================================================

    def _stationary_distribution_sc(P, *, maxiter=10_000, tol=1e-12, dtype=np.float64):
        """
        Compute the stationary distribution of a (sparse) Markov operator via power
        iteration on P^T. Falls back to uniform if iteration does not converge.
        """
        P = P.tocsr() if sp.issparse(P) else sp.csr_matrix(P)
        n = P.shape[0]
        pi = np.full(n, 1.0 / n, dtype=dtype)
        PT = P.T.tocsr()
        for _ in range(int(maxiter)):
            pi_new = PT @ pi
            s = float(pi_new.sum())
            if not np.isfinite(s) or s <= 0:
                break
            pi_new /= s
            if float(np.linalg.norm(pi_new - pi, ord=1)) < float(tol):
                pi = pi_new
                break
            pi = pi_new
        pi = np.clip(pi, 0.0, None)
        s = float(pi.sum())
        return (pi / s) if s > 0 else np.full(n, 1.0 / n, dtype=dtype)


    def calculate_feature_modes(
        adata: "AnnData",
        tg: "TopOGraph",
        return_results: bool = False,
        *,
        multiscale: bool = True,
        use_scaffold_components: bool = True,
        operator: str = "X",
        standardize: str = "corr",
        weight: bool = True,
        center_weighted: bool = True,
        check_basis: bool = True,
        basis_atol: float = 1e-2,
        basis_rtol: float = 1e-2,
        tail_q: float = 0.95,
        topk: int = 50,
        eps: float = 1e-12,
        dtype=np.float32,
        chunk_size: int = 2048,
    ):
        """
        Compute a genes × scaffold-components association matrix (feature loadings).

        Associates each gene profile with each TopoMetry scaffold component via
        a geometry-consistent inner product under the stationary distribution π of
        a chosen Markov operator, optionally transformed for discovery.

        Parameters
        ----------
        adata : AnnData
            Must contain `adata.X` (scaled expression, same cell order as used to
            fit `tg`). Genes are rows of `adata.var`.
        tg : TopOGraph
            Fitted TopoMetry model.
        return_results : bool, default False
            If True, return the resulting DataFrame (genes × components).
        multiscale : bool, default True
            If True, use `tg.spectral_scaffold(multiscale=True)` (msDM).
        use_scaffold_components : bool, default True
            Truncate scaffold to `tg._scaffold_components_ms` / `_dm` if available.
        operator : {'X','Z','msZ'}, default 'X'
            Which diffusion operator to use for the stationary measure π.
        standardize : str, default 'corr'
            Scoring mode. Options:

            * ``'raw'``             – raw weighted inner products X^T diag(π) Ψ.
            * ``'corr'``            – π-weighted correlation in [-1, 1] (signed).
            * ``'abs_corr'``        – |corr| in [0, 1] (magnitude only).
            * ``'r2'``              – corr² in [0, 1] (energy-like).
            * ``'corr_pow0.5'``     – sign(corr)·|corr|^0.5 in [-1, 1] (boosts mid-range).
            * ``'corr_atanh'``      – arctanh(corr) normalized to [-1, 1] (expands near ±1).
            * ``'corr_rank'``       – signed percentile rank in [-1, 1] (magnitude-free).
            * ``'abs_corr_tailq95'``– |corr| / column q95 clipped to [0, 1] (outlier-robust).
            * ``'topk'``            – signed corr with top-k genes kept, rest zeroed.
        weight : bool, default True
            Use stationary π; if False, use uniform weights.
        center_weighted : bool, default True
            π-center X and Ψ before correlation computation (prevents constant-mode
            leakage).
        check_basis : bool, default True
            Run Gram-matrix diagnostics on Ψ under <·,·>_π.
        basis_atol, basis_rtol : float
            Tolerances for orthonormality warning.
        tail_q : float, default 0.95
            Quantile used by ``'abs_corr_tailq95'``.
        topk : int, default 50
            Number of top genes per component kept in ``'topk'`` mode.
        eps : float, default 1e-12
            Numerical stability floor.
        dtype : numpy dtype, default np.float32
            Stored output dtype.
        chunk_size : int, default 2048
            Genes processed per chunk to limit peak memory.

        Returns
        -------
        pandas.DataFrame or None
            If `return_results=True`, genes × components DataFrame; else None.

        Side Effects
        ------------
        * ``adata.varm[store_key]`` – (n_genes, n_components) array.
        * ``adata.uns[store_key + '_meta']`` – metadata dict.

        Store key pattern: ``feature_modes_{prefix}_{op}_{score_name}``
        where ``prefix`` is ``ms`` (multiscale) or ``dm`` (fixed-time DM),
        and ``op`` is ``operator.lower()``.
        """
        import warnings as _warn

        valid_modes = {
            "raw", "corr", "abs_corr", "r2",
            "corr_pow0.5", "corr_atanh", "corr_rank",
            "abs_corr_tailq95", "topk",
        }
        standardize = str(standardize).lower()
        if standardize not in valid_modes:
            raise ValueError(
                f"`standardize` must be one of {sorted(valid_modes)}. Got {standardize!r}."
            )

        # ---- 1) Expression matrix X ----------------------------------------
        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64, order="C")   # (n_cells, n_genes)

        # ---- 2) Scaffold Ψ --------------------------------------------------
        Psi = tg.spectral_scaffold(multiscale=bool(multiscale))
        Psi = np.asarray(Psi, dtype=np.float64, order="C")  # (n_cells, k)

        n = Psi.shape[0]
        if X.shape[0] != n:
            raise ValueError(
                f"adata.X has {X.shape[0]} cells but scaffold has {n}."
            )

        # truncate scaffold to automated sizing if available
        k = None
        if use_scaffold_components:
            attr = "_scaffold_components_ms" if multiscale else "_scaffold_components_dm"
            k = getattr(tg, attr, None)
            if k is not None:
                k = int(k)
        if k is None:
            k = int(Psi.shape[1])
        k = int(min(k, Psi.shape[1]))
        Psi = Psi[:, :k]

        # ---- 3) Operator kernel & π -----------------------------------------
        # Both DM and msDM are eigenbases of the base kernel (P_of_X).
        # operator='X'   → use base kernel (natural for both multiscale and DM scaffold)
        # operator='Z'   → use refined DM kernel (measure adapted to DM geometry)
        # operator='msZ' → use refined msDM kernel (measure adapted to msDM geometry)
        op = str(operator).lower().strip()
        if op == "x":
            _kernel_obj = tg.base_kernel
        elif op == "z":
            _kernel_obj = tg._kernel_Z
        elif op == "msz":
            _kernel_obj = tg._kernel_msZ
        else:
            raise ValueError("`operator` must be one of {'X','Z','msZ'}.")

        # Mild consistency hints (not errors — all combinations are valid).
        if op == "z" and multiscale:
            _warn.warn(
                "operator='Z' with multiscale=True: the DM-kernel measure will be used "
                "for a multiscale scaffold. Consider operator='msZ' for a measure "
                "consistent with the msDM geometry, or operator='X' (default) for the "
                "base-kernel measure.",
                UserWarning, stacklevel=2,
            )
        if op == "msz" and not multiscale:
            _warn.warn(
                "operator='msZ' with multiscale=False: the msDM-kernel measure will be "
                "used for a fixed-time DM scaffold. Consider operator='Z' for a "
                "measure consistent with the DM geometry, or operator='X' (default).",
                UserWarning, stacklevel=2,
            )

        if weight:
            # π ∝ kernel degree d_i = Σ_j K_ij — the stationary distribution of the
            # underlying asymmetric row-stochastic diffusion operator D^{-1}K.
            # This avoids power-iteration on the internally symmetrized .P matrix,
            # which is not row-stochastic and would yield an ill-defined stationary.
            try:
                _K = _kernel_obj.K
                _d = np.asarray(_K.sum(axis=1) if sp.issparse(_K) else np.asarray(_K).sum(axis=1)).ravel()
                _d = np.clip(_d.astype(np.float64), 0.0, None)
                _s = float(_d.sum())
                pi = _d / _s if (_s > 0 and np.all(np.isfinite(_d))) else np.full(n, 1.0 / n, dtype=np.float64)
            except Exception:
                pi = np.full(n, 1.0 / n, dtype=np.float64)
        else:
            pi = np.full(n, 1.0 / n, dtype=np.float64)

        # sanity
        if not np.all(np.isfinite(Psi)):
            raise ValueError("Scaffold Ψ contains non-finite values.")
        pi = np.clip(pi, 0.0, None)
        _s = float(pi.sum())
        if not np.isfinite(_s) or _s <= 0:
            pi = np.full(n, 1.0 / n, dtype=np.float64)
        else:
            pi /= _s

        # ---- 4) Basis diagnostics (Gram matrix under <·,·>_π) ---------------
        diag_report: Dict[str, Any] = {}
        if check_basis:
            Gram = Psi.T @ (pi[:, None] * Psi)
            _diag = np.diag(Gram)
            _off  = Gram - np.diag(_diag)
            diag_report["gram_diag_mean"]    = float(np.mean(_diag))
            diag_report["gram_diag_min"]     = float(np.min(_diag))
            diag_report["gram_diag_max"]     = float(np.max(_diag))
            diag_report["gram_offdiag_maxabs"] = float(np.max(np.abs(_off))) if _off.size else 0.0
            if not np.allclose(Gram, np.eye(k), atol=basis_atol, rtol=basis_rtol):
                # Expected when π is degree-weighted and Ψ was computed for the
                # symmetrized operator (L2-orthonormal, not π-orthonormal).
                # Correlation-based scoring normalises per-gene and per-component,
                # so the result is well-defined regardless.
                diag_report["orthonormal_under_pi"] = False

        p = X.shape[1]  # n_genes
        cs = int(chunk_size)

        # ---- 5a) Raw mode (no correlation) -----------------------------------
        if standardize == "raw":
            WPsi = pi[:, None] * Psi   # (n, k)
            S = np.empty((p, k), dtype=np.float64)
            for j0 in range(0, p, cs):
                j1 = min(p, j0 + cs)
                S[j0:j1, :] = X[:, j0:j1].T @ WPsi
            score_name = "raw"

        # ---- 5b) Correlation-based modes (build corr first) ------------------
        else:
            mu_psi = (pi @ Psi) if center_weighted else np.zeros(k, dtype=np.float64)
            Psi_c  = Psi - mu_psi[None, :] if center_weighted else Psi
            denom_psi = np.sqrt(np.maximum((Psi_c * Psi_c).T @ pi, 0.0) + eps)
            denom_psi = np.where(denom_psi > 0, denom_psi, 1.0)
            WPsi_c = pi[:, None] * Psi_c   # (n, k)

            corr = np.empty((p, k), dtype=np.float64)
            for j0 in range(0, p, cs):
                j1 = min(p, j0 + cs)
                Xb = X[:, j0:j1].copy()    # (n, chunk)
                if center_weighted:
                    Xb -= (pi @ Xb)[None, :]
                numer   = Xb.T @ WPsi_c    # (chunk, k)
                denom_x = np.sqrt(np.maximum((Xb * Xb).T @ pi, 0.0) + eps)
                denom_x = np.where(denom_x > 0, denom_x, 1.0)
                corr[j0:j1, :] = np.clip(
                    numer / (denom_x[:, None] * denom_psi[None, :] + eps),
                    -1.0, 1.0,
                )

            # apply the requested transformation
            if standardize == "corr":
                S = corr
                score_name = "corr"

            elif standardize == "abs_corr":
                S = np.abs(corr)
                score_name = "abs_corr"

            elif standardize == "r2":
                S = corr ** 2
                score_name = "r2"

            elif standardize == "corr_pow0.5":
                S = np.sign(corr) * np.sqrt(np.abs(corr))
                score_name = "corr_pow0.5"

            elif standardize == "corr_atanh":
                # arctanh is defined on (-1, 1); clip slightly inside
                _c = np.clip(corr, -1.0 + 1e-7, 1.0 - 1e-7)
                raw_atanh = np.arctanh(_c)
                # normalize each column to [-1, 1] by its max absolute value
                col_max = np.max(np.abs(raw_atanh), axis=0, keepdims=True)
                col_max = np.where(col_max > 0, col_max, 1.0)
                S = raw_atanh / col_max
                score_name = "corr_atanh"

            elif standardize == "corr_rank":
                # per-column signed percentile rank mapped to [-1, 1]
                S = np.empty_like(corr)
                for j in range(k):
                    col  = corr[:, j]
                    rank = np.argsort(np.argsort(np.abs(col)))  # rank by magnitude
                    pct  = rank / max(p - 1, 1)                 # [0, 1]
                    S[:, j] = np.sign(col) * pct                # [-1, 1]
                score_name = "corr_rank"

            elif standardize == "abs_corr_tailq95":
                q_int = int(round(tail_q * 100))
                abs_c = np.abs(corr)
                col_q = np.quantile(abs_c, tail_q, axis=0)      # (k,)
                col_q = np.where(col_q > eps, col_q, 1.0)
                S = np.clip(abs_c / col_q[None, :], 0.0, 1.0)
                score_name = f"abs_corr_tailq{q_int}"

            elif standardize == "topk":
                S = np.zeros_like(corr)
                _topk = int(max(1, min(topk, p)))
                for j in range(k):
                    col     = corr[:, j]
                    top_idx = np.argpartition(np.abs(col), -_topk)[-_topk:]
                    S[top_idx, j] = col[top_idx]
                score_name = f"topk{_topk}"

        # ---- 6) Store -------------------------------------------------------
        S = S.astype(dtype, copy=False)
        # prefix "ms"/"dm" matches adata.obsm key convention (X_ms_spectral_scaffold)
        prefix     = "ms" if multiscale else "dm"
        columns    = [f"SC_{i}" for i in range(k)]
        store_key  = f"feature_modes_{prefix}_{op}_{score_name}"

        adata.varm[store_key] = S
        adata.uns[f"{store_key}_meta"] = {
            "columns":          columns,
            "index":            "var_names",
            "multiscale":       bool(multiscale),
            "operator":         str(operator),
            "weight":           bool(weight),
            "center_weighted":  bool(center_weighted),
            "standardize":      str(standardize),
            "score":            str(score_name),
            "dtype":            str(np.dtype(dtype)),
            "tail_q":           float(tail_q),
            "topk":             int(topk),
            "basis_diagnostics": diag_report,
        }

        if return_results:
            return pd.DataFrame(S, index=adata.var_names, columns=columns)
        return None


    def plot_feature_modes(
        adata: "AnnData",
        components_to_plot=range(0, 11),
        n_top_features: int = 3,
        cmap: str = "bwr",
        show_colorbar: bool = True,
        fontsize: int = 14,
        ax=None,
        show: bool = True,
        store_key: Optional[str] = None,
    ):
        """
        Visualize a gene × scaffold-components association heatmap.

        Selects the top genes per component by score magnitude and displays them
        as a color-coded matrix.  The color scale is chosen automatically based
        on the stored score type (signed or unsigned).

        Parameters
        ----------
        adata : AnnData
            Must contain at least one ``feature_modes_*`` entry in ``.varm``
            (produced by :func:`calculate_feature_modes`).
        components_to_plot : iterable of int or str, default range(0, 11)
            Which components to include (integer indices or column-name strings).
            Pass ``None`` to plot the first 8.
        n_top_features : int, default 3
            Top genes to show per component (selected by |score| magnitude).
        cmap : str, default 'bwr'
            Colormap; 'bwr' works well for signed scores; 'viridis'/'plasma'
            for unsigned.
        show_colorbar : bool, default True
            Whether to attach a colorbar.
        fontsize : int, default 14
            Tick-label and colorbar font size.
        ax : matplotlib.Axes or None, default None
            Axes to draw into; a new figure is created if None.
        show : bool, default True
            If True, call ``plt.show()`` and return None; else return the Axes.
        store_key : str or None, default None
            Explicit key in ``adata.varm``.  If None, the most recently added
            ``feature_modes_*`` key is used.

        Returns
        -------
        matplotlib.Axes or None
        """
        # ---- key selection --------------------------------------------------
        if store_key is None:
            keys = [k for k in adata.varm.keys() if str(k).startswith("feature_modes_")]
            if not keys:
                raise KeyError(
                    "No feature-modes matrix found in adata.varm. "
                    "Run tp.sc.calculate_feature_modes(adata, tg) first."
                )
            store_key = keys[-1]
        elif store_key not in adata.varm:
            raise KeyError(f"store_key {store_key!r} not found in adata.varm.")

        A_mat = np.asarray(adata.varm[store_key])
        meta  = adata.uns.get(f"{store_key}_meta", {})
        cols  = list(meta.get("columns", [f"mode_{i+1}" for i in range(A_mat.shape[1])]))
        idx   = list(adata.var_names)

        # ---- component selection -------------------------------------------
        if components_to_plot is None:
            comps = cols[:8]
        else:
            comps_in = list(components_to_plot)
            if not comps_in:
                comps = cols[:6]
            elif isinstance(comps_in[0], (int, np.integer)):
                comps = [cols[int(i)] for i in comps_in if 0 <= int(i) < len(cols)]
            else:
                comps = [str(c) for c in comps_in]
        col_to_j = {c: j for j, c in enumerate(cols)}
        comps    = [c for c in comps if c in col_to_j]
        js       = [col_to_j[c] for c in comps]
        if not js:
            raise ValueError("No valid components selected from components_to_plot.")

        # ---- top genes per component (by |score|) --------------------------
        n_top  = int(n_top_features)
        genes  = []
        seen   = set()
        for j in js:
            v       = A_mat[:, j]
            top_idx = np.argsort(np.abs(v))[::-1][:n_top]
            for ii in top_idx:
                g = idx[int(ii)]
                if g not in seen:
                    seen.add(g)
                    genes.append(g)

        row_to_i = {g: i for i, g in enumerate(idx)}
        rows     = [row_to_i[g] for g in genes]
        M        = A_mat[np.ix_(rows, js)]   # (n_genes_selected, n_comps)

        # ---- color scale ---------------------------------------------------
        score = meta.get("score", "")
        _SIGNED_SCORES = {"corr", "corr_pow0.5", "corr_atanh", "corr_rank", "raw"}
        # topk keeps signs; abs_corr_tailq*, abs_corr, r2 are unsigned
        if score in _SIGNED_SCORES or score.startswith("topk"):
            vmin, vmax = -1.0, 1.0
        elif score == "raw":
            vmin, vmax = None, None   # use data range
        else:
            vmin, vmax = 0.0, 1.0

        # ---- plot ----------------------------------------------------------
        if ax is None:
            _, ax = plt.subplots(
                figsize=(0.6 * len(comps) + 6, 0.18 * len(genes) + 4)
            )

        im = ax.imshow(
            M, aspect="auto", interpolation="nearest",
            vmin=vmin, vmax=vmax, cmap=cmap,
        )
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.set_label(score or "score", fontsize=fontsize)

        ax.set_xticks(np.arange(len(comps)))
        ax.set_xticklabels(comps, rotation=45, ha="right", fontsize=fontsize)
        ax.set_yticks(np.arange(len(genes)))
        ax.set_yticklabels(genes, fontsize=fontsize)
        ax.grid(False)
        ax.figure.tight_layout()

        if show:
            plt.show()
            return None
        return ax


    def highlight_embedding(
        adata,
        basis,
        target,
        groupby=None,
        dot_size=10,
        circle_size_factor=1,
        linewidth=2,
        ax=None,
        show=True,
        palette='Reds',
        use_raw=True,
        **kwargs
    ):
        """
        Highlight the densest region of a given target on a low-dimensional embedding.

        Computes a kernel density estimate (KDE) over the coordinates of cells
        associated with `target` — either expressing a gene or belonging to a
        categorical group — identifies the peak-density coordinate, and overlays
        an auto-scaled circle on the embedding plot.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix.
        basis : str
            Embedding basis key. Coordinates are read from ``adata.obsm['X_' + basis]``.
        target : str
            Either a gene name (present in ``adata.var_names`` or ``adata.raw.var_names``)
            or a single category within ``adata.obs[groupby]``.
            - Gene mode: KDE is weighted by expression values; only cells with
              expression > 0 are included.
            - Category mode: unweighted KDE over all cells belonging to the category.
        groupby : str, optional
            Column in ``adata.obs`` to use for categorical targeting. Required when
            ``target`` is not a gene.
        dot_size : int, default 10
            Dot size passed to ``sc.pl.embedding``.
        circle_size_factor : float, default 1
            Scalar multiplier applied to the auto-computed circle radius, allowing
            manual fine-tuning without changing the density-based anchor.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If None, a new (6, 6) figure is created.
        show : bool, default True
            If True, calls ``plt.show()``. If False, returns the Axes object —
            consistent with scanpy's ``show`` convention.
        palette : str or list, default 'Reds'
            Color palette passed to ``sc.pl.embedding``.
        use_raw : bool, default True
            Whether to use ``adata.raw`` for gene expression lookup (gene mode only).
        **kwargs
            Additional keyword arguments forwarded to ``sc.pl.embedding``
            (e.g., ``title``, ``na_color``, ``legend_loc``).

        Returns
        -------
        matplotlib.axes.Axes or None
            Returns the Axes if ``show=False``, otherwise None.

        Raises
        ------
        ValueError
            If the embedding key is missing, ``target`` cannot be resolved,
            ``groupby`` is missing when required, or too few cells are found.
        """
        obsm_key = f"X_{basis}"
        if obsm_key not in adata.obsm:
            raise ValueError(f"'{obsm_key}' not found in adata.obsm.")

        all_coords = adata.obsm[obsm_key]

        # ------------------------------------------------------------------ #
        # Resolve target: gene or categorical                                  #
        # ------------------------------------------------------------------ #
        raw_var_names = adata.raw.var_names if (use_raw and adata.raw is not None) else None

        if target in adata.var_names:
            is_gene = True
            expr = adata[:, target].X
            expr = expr.toarray().flatten() if hasattr(expr, 'toarray') else np.array(expr).flatten()

        elif raw_var_names is not None and target in raw_var_names:
            is_gene = True
            expr = adata.raw[:, target].X
            expr = expr.toarray().flatten() if hasattr(expr, 'toarray') else np.array(expr).flatten()

        else:
            is_gene = False
            if groupby is None:
                raise ValueError(f"'{target}' is not a gene. Provide `groupby` to treat it as a categorical target.")
            if groupby not in adata.obs.columns:
                raise ValueError(f"'{groupby}' not found in adata.obs.")
            if target not in adata.obs[groupby].cat.categories:
                raise ValueError(f"'{target}' not found in adata.obs['{groupby}'].")

        # ------------------------------------------------------------------ #
        # Extract coordinates and optional weights for KDE                    #
        # ------------------------------------------------------------------ #
        if is_gene:
            mask = expr > 0
            coords = all_coords[mask]
            weights = expr[mask]
            color_key, groups_arg = target, None
        else:
            mask = (adata.obs[groupby] == target).values
            coords = all_coords[mask]
            weights = None
            color_key, groups_arg = groupby, [target]

        if coords.shape[0] < 2:
            raise ValueError(f"Fewer than 2 cells found for target '{target}'. Cannot compute KDE.")

        # ------------------------------------------------------------------ #
        # KDE: find peak-density coordinate                                   #
        # ------------------------------------------------------------------ #
        kde = gaussian_kde(coords.T, weights=weights)
        densities = kde(coords.T)
        cx, cy = coords[densities.argmax()]

        # Auto-radius: 50th percentile of distances from peak among target cells,
        # scaled by circle_size_factor for manual adjustment
        dists = np.hypot(coords[:, 0] - cx, coords[:, 1] - cy)
        radius = np.percentile(dists, 50) * circle_size_factor

        # ------------------------------------------------------------------ #
        # Plot                                                                 #
        # ------------------------------------------------------------------ #
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 6))

        kwargs.setdefault('na_color', '#BEBEBE')
        kwargs.setdefault('legend_loc', None)
        kwargs.setdefault('frameon', False)
        kwargs.setdefault('title', '')

        sc.pl.embedding(
            adata,
            basis=basis,
            color=color_key,
            groups=groups_arg,
            size=dot_size,
            palette=palette,
            ax=ax,
            show=False,
            use_raw=use_raw if is_gene else False,
            **kwargs
        )

        circle = plt.Circle((cx, cy), radius, color='k', fill=False,
                             linewidth=linewidth, clip_on=False)
        ax.add_patch(circle)

        if show:
            plt.show()
        else:
            return ax    
            
    # =======================================================================
    # CCA-Anchor Batch Integration (Seurat v3-style)
    # =======================================================================
    #
    # Python implementation of the Seurat v3 integration workflow
    # (Stuart et al., 2019, Cell), without PCA.  CCA replaces PCA as the
    # shared low-dimensional space for anchor finding.  Correction is
    # applied symmetrically in log-normalised expression space (both
    # datasets move to a midpoint).  Corrected values are clamped >= 0.
    #
    # No PCA anywhere.  No sklearn.neighbors.  hnswlib for all kNN.
    # All new code: private helpers + one public function.
    # =======================================================================

    # ── hnswlib helpers ────────────────────────────────────────────────────

    def _resolve_n_jobs(n_jobs: int) -> int:
        """Resolve n_jobs: -1 → all cores, 0 → 1, positive → as-is."""
        if n_jobs <= 0:
            from multiprocessing import cpu_count
            return cpu_count()
        return n_jobs

    def _build_hnsw_index(
        data: np.ndarray,
        space: str = "l2",
        M: int = 60,
        ef_construction: int = 200,
        n_jobs: int = 1,
        seed: int = 0,
    ):
        """Build an HNSW index. *data* must be C-contiguous float32."""
        import hnswlib
        n, d = data.shape
        n_threads = _resolve_n_jobs(n_jobs)
        idx = hnswlib.Index(space=space, dim=d)
        idx.init_index(max_elements=n, ef_construction=ef_construction,
                       M=M, random_seed=seed)
        idx.set_num_threads(n_threads)
        idx.add_items(np.ascontiguousarray(data.astype(np.float32)))
        return idx

    def _query_hnsw(
        index,
        queries: np.ndarray,
        k: int,
        ef: int | None = None,
        n_jobs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Query HNSW index.  Returns (labels, distances), shape (m, k)."""
        if ef is None:
            ef = max(k * 2, 50)
        index.set_ef(ef)
        index.set_num_threads(_resolve_n_jobs(n_jobs))
        labels, dists = index.knn_query(
            np.ascontiguousarray(queries.astype(np.float32)), k=k)
        return labels.astype(np.int32), dists.astype(np.float32)

    # ── Hyperparameter estimators ──────────────────────────────────────────

    def _estimate_n_features(adatas: list, n_features):
        shared = set(adatas[0].var_names)
        for a in adatas[1:]:
            shared &= set(a.var_names)
        n_shared = len(shared)
        if n_features is not None:
            assert n_features > 0
            return min(n_features, n_shared)
        return max(200, min(3000, n_shared))

    def _estimate_k_anchor(n_a: int, n_b: int, k_anchor):
        if k_anchor is not None:
            return max(2, k_anchor)
        # sqrt(n_min)/2 scales gracefully; cap at 200 (generous but bounded)
        k = int(round(np.sqrt(min(n_a, n_b)) / 2))
        return max(3, min(200, k))

    def _estimate_k_filter(n_a: int, n_b: int, k_filter):
        if k_filter is not None:
            return k_filter  # 0 = disabled
        # sqrt(n_ref) targets ~0.5-1% of the reference for typical datasets.
        # Capped at 200 (matching Seurat default) to stay selective.
        n_ref = max(n_a, n_b)
        k = int(round(np.sqrt(n_ref)))
        return max(30, min(200, k))

    def _estimate_k_score(k_anchor: int, k_score):
        if k_score is not None:
            return max(k_anchor, k_score)
        # 6× k_anchor for robust SNN overlap; cap at 200
        return max(k_anchor, min(200, k_anchor * 6))

    def _estimate_k_weight(k_anchor: int, n_anchors: int, k_weight):
        if k_weight is not None:
            return min(k_weight, n_anchors)
        return min(max(k_anchor * 20, 100), n_anchors)

    def _estimate_sd_bandwidth(raw_w_b: np.ndarray, sd_bandwidth):
        import warnings as _w
        if sd_bandwidth is not None:
            return float(sd_bandwidth)
        pos = raw_w_b[raw_w_b > 0]
        if pos.size == 0 or np.median(pos) < 1e-4:
            _w.warn("Degenerate raw_w; using sd_bandwidth=1.0")
            return 1.0
        median_rw = float(np.median(pos))
        return float(2.0 * np.sqrt(np.log(2.0) / median_rw))

    # ── Normalization heuristic ───────────────────────────────────────────

    def _detect_normalization(X: np.ndarray) -> str:
        """Classify expression matrix normalization state.

        Returns
        -------
        'raw_counts'      : integer counts, needs CPM + log1p
        'pre_normalized'  : non-integer large values (TPM/FPKM), needs log1p only
        'lognorm'         : already log-normalized (max <= 20), pass through
        """
        xmax = float(X.max())
        if xmax <= 20:
            return 'lognorm'
        if np.issubdtype(X.dtype, np.integer):
            return 'raw_counts'
        nz = X[X > 0]
        if nz.size > 10_000:
            rng = np.random.default_rng(0)
            nz = rng.choice(nz, size=10_000, replace=False)
        if bool(np.all(nz == np.floor(nz))):
            return 'raw_counts'
        # Non-integer values with max > 20 → pre-normalized (TPM/FPKM)
        return 'pre_normalized'

    def _is_raw_counts(X: np.ndarray) -> bool:
        """Return True if X appears to be raw integer count data."""
        return _detect_normalization(X) == 'raw_counts'

    # ── Feature selection ──────────────────────────────────────────────────

    def _select_integration_features(
        adatas: list,
        n_features: int,
    ) -> list[str]:
        """Select integration genes via Seurat v3-style ranked HVG union."""
        import warnings as _w

        shared = set(adatas[0].var_names)
        for a in adatas[1:]:
            shared &= set(a.var_names)
        shared = sorted(shared)
        if len(shared) < 50:
            raise ValueError(
                f"Datasets share only {len(shared)} genes (minimum 50).")

        hvg_per = []
        disp_per = []
        for a in adatas:
            a_copy = a[:, shared].copy()
            X_check = a_copy.X
            if sp.issparse(X_check):
                X_check = X_check.toarray()
            norm_state = _detect_normalization(X_check)
            if norm_state == 'raw_counts':
                flavor = "seurat_v3"
                disp_col = "variances_norm"
            elif norm_state == 'pre_normalized':
                # Pre-normalized (TPM/FPKM): apply log1p before HVG
                a_copy.X = sp.csr_matrix(np.log1p(X_check).astype(np.float32))
                flavor = "seurat"
                disp_col = "dispersions_norm"
            else:
                flavor = "seurat"
                disp_col = "dispersions_norm"
            try:
                sc.pp.highly_variable_genes(
                    a_copy, flavor=flavor,
                    n_top_genes=min(n_features, len(shared)),
                    inplace=True)
            except Exception:
                sc.pp.highly_variable_genes(
                    a_copy, flavor="seurat",
                    n_top_genes=min(n_features, len(shared)),
                    inplace=True)
                disp_col = "dispersions_norm"

            hvg_set = set(a_copy.var_names[a_copy.var["highly_variable"]])
            hvg_per.append(hvg_set)
            if disp_col in a_copy.var.columns:
                disp = dict(zip(a_copy.var_names, a_copy.var[disp_col]))
            elif "dispersions" in a_copy.var.columns:
                disp = dict(zip(a_copy.var_names, a_copy.var["dispersions"]))
            else:
                disp = {g: 0.0 for g in shared}
            disp_per.append(disp)

        gene_counts = {}
        gene_disp = {}
        for g in shared:
            gene_counts[g] = sum(1 for s in hvg_per if g in s)
            gene_disp[g] = np.mean([d.get(g, 0.0) for d in disp_per])
        ranked = sorted(shared,
                        key=lambda g: (-gene_counts[g], -gene_disp[g], g))
        result = ranked[:min(n_features, len(shared))]
        assert all(g in set(a.var_names) for a in adatas for g in result)
        return result

    # ── Matrix preparation ─────────────────────────────────────────────────

    def _prepare_matrices(
        adata,
        features: list[str],
        batch_name: str = "",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (X_lognorm, X_scaled) for one dataset."""
        import warnings as _w
        X_sub = adata[:, features].X
        if sp.issparse(X_sub):
            X_sub = X_sub.toarray()
        X_sub = np.asarray(X_sub, dtype=np.float32)

        norm_state = _detect_normalization(X_sub)
        if norm_state == 'raw_counts':
            row_sums = X_sub.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            X_lognorm = np.log1p(X_sub / row_sums * 1e4).astype(np.float32)
            _w.warn(
                f"Batch '{batch_name}': raw counts detected; "
                "applying log1p(CPM/1e4).",
                UserWarning, stacklevel=3)
        elif norm_state == 'pre_normalized':
            # TPM/FPKM: already size-normalized, just needs log1p
            X_lognorm = np.log1p(X_sub).astype(np.float32)
            _w.warn(
                f"Batch '{batch_name}': pre-normalized values detected "
                "(max={float(X_sub.max()):.0f}, non-integer); applying log1p.",
                UserWarning, stacklevel=3)
        else:
            X_lognorm = X_sub.astype(np.float32)
        X_lognorm = np.ascontiguousarray(X_lognorm)

        mu = X_lognorm.mean(axis=0)
        sigma = X_lognorm.std(axis=0, ddof=1)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        X_scaled = np.clip((X_lognorm - mu) / sigma, -10.0, 10.0)
        X_scaled = np.ascontiguousarray(X_scaled.astype(np.float32))
        return X_lognorm, X_scaled

    def _rescale_merged(X_lognorm: np.ndarray) -> np.ndarray:
        """Z-score a merged lognorm matrix for the next CCA step."""
        mu = X_lognorm.mean(axis=0)
        sigma = X_lognorm.std(axis=0, ddof=1)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        return np.ascontiguousarray(
            np.clip((X_lognorm - mu) / sigma, -10.0, 10.0).astype(np.float32))

    # ── CCA ────────────────────────────────────────────────────────────────

    def _compute_cca(
        X_a_scaled: np.ndarray,
        X_b_scaled: np.ndarray,
        n_components: int,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Diagonalised CCA via truncated SVD of the cell-cell cross-product.

        CCA formulation (Python cells x genes convention):
            C = X_a_scaled @ X_b_scaled.T    shape (n_a, n_b)
        Works for any n_a, n_b (no requirement that batch sizes match).

        Gene loadings are recovered by back-projection from the RAW
        (un-normalised) left singular vectors BEFORE row-normalisation is
        applied to cc_a.  Computing U_k after normalisation distorts the
        loading magnitudes.

        Returns
        -------
        cc_a : (n_a, k) L2-normalised CCA coordinates for A
        cc_b : (n_b, k) L2-normalised CCA coordinates for B
        U_k  : (p, k)   gene loadings, column-normalised.
               Computed from raw U BEFORE row-normalising cc_a.
               Passed to _filter_anchors_hd and stored for query mapping.

        where k = min(n_components, min(n_a, n_b) - 1).
        """
        from sklearn.utils.extmath import randomized_svd
        n_a, p = X_a_scaled.shape
        n_b = X_b_scaled.shape[0]
        k = min(n_components, min(n_a, n_b) - 1)
        if k < 1:
            raise ValueError(
                f"Cannot compute CCA: min(n_a={n_a}, n_b={n_b}) - 1 < 1.")

        # Cell-cell cross-product — valid for any n_a, n_b
        C = X_a_scaled @ X_b_scaled.T
        U_raw, _, Vt_raw = randomized_svd(
            C, n_components=k, n_iter=4, random_state=seed)

        # Step 1: gene loadings from RAW singular vectors (before row-norm)
        U_k = X_a_scaled.T @ U_raw                          # (p, k)
        col_norms = np.linalg.norm(U_k, axis=0, keepdims=True)
        U_k /= np.where(col_norms < 1e-12, 1.0, col_norms)
        U_k = np.ascontiguousarray(U_k.astype(np.float32))

        # Step 2: row-normalise CCA cell coordinates
        cc_a = U_raw.astype(np.float32)
        cc_b = Vt_raw.T.astype(np.float32)
        cc_a /= (np.linalg.norm(cc_a, axis=1, keepdims=True) + 1e-12)
        cc_b /= (np.linalg.norm(cc_b, axis=1, keepdims=True) + 1e-12)

        return (np.ascontiguousarray(cc_a),
                np.ascontiguousarray(cc_b),
                U_k)

    # ── MNN anchors ────────────────────────────────────────────────────────

    def _find_mnn_in_cca(
        cc_a: np.ndarray,
        cc_b: np.ndarray,
        k: int,
        n_jobs: int = 1,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mutual nearest neighbours in CCA space via hnswlib."""
        n_jobs = _resolve_n_jobs(n_jobs)
        n_a, n_b = len(cc_a), len(cc_b)
        k_a = min(k, n_b - 1)
        k_b = min(k, n_a - 1)
        if k_a < 1 or k_b < 1:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        if n_jobs > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_b = ex.submit(_build_hnsw_index, cc_b, "l2", 60, 200, 1, seed)
                fut_a = ex.submit(_build_hnsw_index, cc_a, "l2", 60, 200, 1, seed)
                idx_b_hnsw = fut_b.result()
                idx_a_hnsw = fut_a.result()
        else:
            idx_b_hnsw = _build_hnsw_index(cc_b, "l2", 60, 200, 1, seed)
            idx_a_hnsw = _build_hnsw_index(cc_a, "l2", 60, 200, 1, seed)

        idx_a2b, _ = _query_hnsw(idx_b_hnsw, cc_a, k=k_a, n_jobs=n_jobs)
        idx_b2a, _ = _query_hnsw(idx_a_hnsw, cc_b, k=k_b, n_jobs=n_jobs)

        # Vectorised MNN via sparse boolean AND
        rows_a = np.repeat(np.arange(n_a, dtype=np.int32), k_a)
        cols_a = idx_a2b.ravel().astype(np.int32)
        A2B = sp.csr_matrix(
            (np.ones(len(rows_a), dtype=np.bool_), (rows_a, cols_a)),
            shape=(n_a, n_b))

        rows_b = np.repeat(np.arange(n_b, dtype=np.int32), k_b)
        cols_b = idx_b2a.ravel().astype(np.int32)
        B2A = sp.csr_matrix(
            (np.ones(len(rows_b), dtype=np.bool_), (rows_b, cols_b)),
            shape=(n_b, n_a))

        mnn = A2B.multiply(B2A.T)
        anchor_a, anchor_b = mnn.nonzero()
        return np.asarray(anchor_a, dtype=np.int32), np.asarray(anchor_b, dtype=np.int32)

    # ── HD filter ──────────────────────────────────────────────────────────

    def _filter_anchors_hd(
        anchor_a: np.ndarray,
        anchor_b: np.ndarray,
        X_a_scaled: np.ndarray,
        X_b_scaled: np.ndarray,
        U_k: np.ndarray,
        k_filter: int,
        n_jobs: int = 1,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Discard CCA anchors not supported in CCA-loading-weighted feature space."""
        if k_filter == 0 or len(anchor_a) == 0:
            return anchor_a, anchor_b

        p = X_a_scaled.shape[1]
        weights = np.abs(U_k).sum(axis=1)
        weights /= (weights.sum() + 1e-12)

        max_feat = min(200, p)
        top_idx = np.argsort(weights)[-max_feat:]

        X_a_top = X_a_scaled[:, top_idx] * weights[top_idx][None, :]
        X_b_top = X_b_scaled[:, top_idx] * weights[top_idx][None, :]

        norms_a = np.linalg.norm(X_a_top, axis=1, keepdims=True)
        norms_b = np.linalg.norm(X_b_top, axis=1, keepdims=True)
        X_a_top = X_a_top / np.where(norms_a < 1e-12, 1.0, norms_a)
        X_b_top = X_b_top / np.where(norms_b < 1e-12, 1.0, norms_b)
        X_a_top = np.ascontiguousarray(X_a_top.astype(np.float32))
        X_b_top = np.ascontiguousarray(X_b_top.astype(np.float32))

        k_filter = min(k_filter, X_a_top.shape[0] - 1)
        if k_filter < 1:
            return anchor_a, anchor_b

        unique_b, inv = np.unique(anchor_b, return_inverse=True)
        idx_a_hnsw = _build_hnsw_index(X_a_top, "l2", 60, 200, n_jobs, seed)
        labels, _ = _query_hnsw(idx_a_hnsw, X_b_top[unique_b],
                                 k=k_filter, n_jobs=n_jobs)
        labels_per_anchor = labels[inv]
        keep = (labels_per_anchor == anchor_a[:, None]).any(axis=1)
        return anchor_a[keep], anchor_b[keep]

    # ── SNN scoring ────────────────────────────────────────────────────────

    def _score_anchors(
        anchor_a: np.ndarray,
        anchor_b: np.ndarray,
        cc_a: np.ndarray,
        cc_b: np.ndarray,
        k_score: int,
        n_jobs: int = 1,
        seed: int = 0,
    ) -> np.ndarray:
        """SNN-overlap scores rescaled to [0, 1]."""
        n_jobs = _resolve_n_jobs(n_jobs)
        n_a, n_b = len(cc_a), len(cc_b)
        k_a = min(k_score, n_a - 1)
        k_b = min(k_score, n_b - 1)
        if k_a < 1 or k_b < 1 or len(anchor_a) == 0:
            return np.ones(len(anchor_a), dtype=np.float32)

        # Within-dataset kNN
        if n_jobs > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_a = ex.submit(_build_hnsw_index, cc_a, "l2", 60, 200, 1, seed)
                fut_b = ex.submit(_build_hnsw_index, cc_b, "l2", 60, 200, 1, seed)
                idx_a_h = fut_a.result()
                idx_b_h = fut_b.result()
        else:
            idx_a_h = _build_hnsw_index(cc_a, "l2", 60, 200, 1, seed)
            idx_b_h = _build_hnsw_index(cc_b, "l2", 60, 200, 1, seed)

        N_a_idx, _ = _query_hnsw(idx_a_h, cc_a, k=k_a, n_jobs=n_jobs)
        N_b_idx, _ = _query_hnsw(idx_b_h, cc_b, k=k_b, n_jobs=n_jobs)

        # Sparse neighbourhood matrices
        SNN_A = sp.csr_matrix(
            (np.ones(n_a * k_a, dtype=np.float32),
             (np.repeat(np.arange(n_a, dtype=np.int32), k_a),
              N_a_idx.ravel().astype(np.int32))),
            shape=(n_a, n_a))

        SNN_B = sp.csr_matrix(
            (np.ones(n_b * k_b, dtype=np.float32),
             (np.repeat(np.arange(n_b, dtype=np.int32), k_b),
              N_b_idx.ravel().astype(np.int32))),
            shape=(n_b, n_b))

        ANC_B2A = sp.csr_matrix(
            (np.ones(len(anchor_a), dtype=np.float32),
             (anchor_b.astype(np.int32), anchor_a.astype(np.int32))),
            shape=(n_b, n_a))

        # REACH via unique anchor_b rows
        unique_anc_b, inv_anc_b = np.unique(anchor_b, return_inverse=True)
        REACH_rows = SNN_B[unique_anc_b] @ ANC_B2A
        REACH_at_anchors = REACH_rows[inv_anc_b]

        # Chunked overlap
        chunk_size = min(10_000, len(anchor_a))
        raw_scores = np.zeros(len(anchor_a), dtype=np.float32)
        for s in range(0, len(anchor_a), chunk_size):
            e = min(s + chunk_size, len(anchor_a))
            ov = SNN_A[anchor_a[s:e]].multiply(REACH_at_anchors[s:e])
            raw_scores[s:e] = np.asarray(ov.sum(axis=1)).ravel()
        raw_scores /= (2.0 * k_score)

        # Rescale
        q01 = float(np.quantile(raw_scores, 0.01))
        q90 = float(np.quantile(raw_scores, 0.90))
        if q90 > q01:
            scores = np.clip((raw_scores - q01) / (q90 - q01), 0.0, 1.0)
        else:
            scores = np.ones(len(raw_scores), dtype=np.float32)
        return scores.astype(np.float32)

    # ── Symmetric correction ───────────────────────────────────────────────

    def _compute_correction_vectors(
        X_a_lognorm: np.ndarray,
        X_b_lognorm: np.ndarray,
        anchor_a: np.ndarray,
        anchor_b: np.ndarray,
        scores: np.ndarray,
        cc_a: np.ndarray,
        cc_b: np.ndarray,
        k_weight: int,
        sd_bandwidth,
        n_jobs: int = 1,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Symmetric per-cell correction in log-normalised expression space."""
        n_jobs = _resolve_n_jobs(n_jobs)
        n_a = X_a_lognorm.shape[0]
        n_b = X_b_lognorm.shape[0]
        p = X_a_lognorm.shape[1]
        n_anc = len(anchor_a)

        # Midpoint batch vectors
        midpoint_vecs = (X_a_lognorm[anchor_a] - X_b_lognorm[anchor_b]) / 2.0

        # Anchor positions in CCA space
        anchor_pos_b = np.ascontiguousarray(cc_b[anchor_b].astype(np.float32))
        anchor_pos_a = np.ascontiguousarray(cc_a[anchor_a].astype(np.float32))

        k_eff = min(k_weight, n_anc)

        # Build HNSW on anchor positions
        if n_jobs > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_b = ex.submit(_build_hnsw_index, anchor_pos_b, "l2", 60, 200, 1, seed)
                fut_a = ex.submit(_build_hnsw_index, anchor_pos_a, "l2", 60, 200, 1, seed)
                idx_anc_b = fut_b.result()
                idx_anc_a = fut_a.result()
        else:
            idx_anc_b = _build_hnsw_index(anchor_pos_b, "l2", 60, 200, 1, seed)
            idx_anc_a = _build_hnsw_index(anchor_pos_a, "l2", 60, 200, 1, seed)

        idx_b, dist_b = _query_hnsw(idx_anc_b, cc_b, k=k_eff, n_jobs=n_jobs)
        idx_a, dist_a = _query_hnsw(idx_anc_a, cc_a, k=k_eff, n_jobs=n_jobs)

        # Raw weights
        d_k_b = dist_b[:, -1:] + 1e-10
        raw_w_b = np.clip(1.0 - dist_b / d_k_b, 0.0, 1.0) * scores[idx_b]

        d_k_a = dist_a[:, -1:] + 1e-10
        raw_w_a = np.clip(1.0 - dist_a / d_k_a, 0.0, 1.0) * scores[idx_a]

        # Estimate sd
        sd = _estimate_sd_bandwidth(raw_w_b, sd_bandwidth)

        # Gaussian kernel
        sd_val = 2.0 / sd
        kernel_b = 1.0 - np.exp(-raw_w_b / sd_val**2)
        kernel_a = 1.0 - np.exp(-raw_w_a / sd_val**2)

        # Row-normalise
        kernel_b /= (kernel_b.sum(axis=1, keepdims=True) + 1e-12)
        kernel_a /= (kernel_a.sum(axis=1, keepdims=True) + 1e-12)

        # Weighted correction (chunked einsum)
        corr_b = np.zeros((n_b, p), dtype=np.float32)
        chunk = max(500, n_b // max(1, n_jobs))
        for s in range(0, n_b, chunk):
            e = min(s + chunk, n_b)
            corr_b[s:e] = np.einsum(
                'ck,ckp->cp',
                kernel_b[s:e],
                midpoint_vecs[idx_b[s:e]])

        corr_a = np.zeros((n_a, p), dtype=np.float32)
        chunk = max(500, n_a // max(1, n_jobs))
        for s in range(0, n_a, chunk):
            e = min(s + chunk, n_a)
            corr_a[s:e] = np.einsum(
                'ck,ckp->cp',
                kernel_a[s:e],
                -midpoint_vecs[idx_a[s:e]])

        return corr_b, corr_a, sd

    # ── Guide tree ─────────────────────────────────────────────────────────

    def _build_guide_tree(
        adatas: list,
        features: list[str],
        n_components: int,
        n_jobs: int,
        seed: int,
    ) -> list[tuple[int, int]]:
        """Data-driven pairwise merge order for N > 2 datasets."""
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform

        n_jobs = _resolve_n_jobs(n_jobs)
        N = len(adatas)
        features_low = features[:min(500, len(features))]

        def _pair_sim(i, j):
            Xi_ln, Xi_sc = _prepare_matrices(adatas[i], features_low)
            Xj_ln, Xj_sc = _prepare_matrices(adatas[j], features_low)
            nc = min(10, n_components)
            try:
                cc_i, cc_j, _ = _compute_cca(Xi_sc, Xj_sc, nc, seed)
                anc_i, anc_j = _find_mnn_in_cca(cc_i, cc_j, k=3,
                                                   n_jobs=1, seed=seed)
                return len(anc_i) / max(1, min(len(Xi_sc), len(Xj_sc)))
            except Exception:
                return 0.0

        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        if n_jobs > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                sim_values = list(ex.map(lambda p: _pair_sim(*p), pairs))
        else:
            sim_values = [_pair_sim(i, j) for i, j in pairs]

        D = np.zeros((N, N), dtype=np.float64)
        for (i, j), sim in zip(pairs, sim_values):
            d = 1.0 / (sim + 1e-6)
            D[i, j] = d
            D[j, i] = d

        Z = linkage(squareform(D), method="average")
        return [(int(Z[i, 0]), int(Z[i, 1])) for i in range(len(Z))]

    def _node_name(node_id: int, batch_names: list[str]) -> str:
        if node_id < len(batch_names):
            return batch_names[node_id]
        return f"merged_node_{node_id}"

    # ── Public API ─────────────────────────────────────────────────────────

    def run_cca_integration(
        data,
        batch_key: str | None = None,
        n_features: int | None = None,
        n_components: int = 30,
        k_anchor: int | None = None,
        k_filter: int | None = None,
        k_score: int | None = None,
        k_weight: int | None = None,
        sd_bandwidth: float | None = None,
        scale_output: bool = True,
        n_jobs: int = 1,
        layer: str | None = None,
        seed: int = 0,
        _fixed_features: list | None = None,
    ):
        """CCA-anchor batch correction for scRNA-seq data.

        Python implementation of the Seurat v3 integration workflow
        (Stuart et al., 2019, Cell), without PCA. CCA replaces PCA as the
        shared low-dimensional space for anchor finding. Correction is
        applied in log-normalised expression space.

        Symmetric correction: both datasets move to a shared midpoint at
        each merge step, so no single dataset acts as a fixed reference.
        Corrected values are clamped to >= 0 (log-normalised values cannot
        be negative).

        Known approximation: z-scoring is applied per dataset independently
        before CCA, approximating the global cross-dataset covariance. This
        matches Seurat's behaviour.

        Parameters
        ----------
        data : AnnData or list[AnnData]
            Single AnnData (requires batch_key) or list of per-batch AnnData.
        batch_key : str, optional
            .obs column identifying batches. Required for single AnnData input.
        n_features : int, optional
            Number of integration genes. Default None -> max(200, min(3000,
            n_shared_genes)).
        n_components : int
            CCA dimensions. Default 30 (same as Seurat).
        k_anchor : int, optional
            k for MNN search. Default None -> sqrt(n_min)/2, clamped [2, 20].
        k_filter : int, optional
            HD-support filter k. Set 0 to disable. Default None ->
            0.2 * n_min, clamped [50, 500].
        k_score : int, optional
            SNN-scoring k. Default None -> clamp(k_anchor*6, k_anchor, 100).
        k_weight : int, optional
            Correction smoothing k. Default None -> min(max(k_anchor*20, 100),
            n_anchors).
        sd_bandwidth : float, optional
            Gaussian kernel bandwidth. Default None -> calibrated from raw_w.
        n_jobs : int
            Threads for hnswlib. Default 1.
        scale_output : bool
            If True (default), z-score the corrected expression and store it
            in .X (ready for TopOGraph / tp.sc.fit_adata). The corrected
            log-normalised expression is preserved in .layers["corrected"].
            If False, .X contains the corrected log-normalised expression
            directly.
        layer : str, optional
            Use adata.layers[layer] instead of adata.X.
        seed : int
            Random seed.

        Returns
        -------
        AnnData
            .X : z-scored corrected expression if scale_output=True,
                 otherwise corrected log-normalised (CSR float32, >= 0)
            .layers["corrected"] : corrected log-normalised (if scale_output)
            .layers["original"] : uncorrected log-normalised (CSR float32)
            .obs["batch"] : batch labels
            .var_names : n_features integration genes only
            .uns["cca_integration"] : run log

        Notes
        -----
        Output covers ONLY the n_features integration genes. Use
        layers["original"] for uncorrected values.
        No PCA. No TopoMetry graph required. Same-modality only.
        """
        import anndata as ad
        import warnings

        # ── 0. Coerce input ────────────────────────────────────────────
        if isinstance(data, AnnData):
            if batch_key is None:
                raise ValueError("batch_key required for single AnnData input.")
            if batch_key not in data.obs.columns:
                raise ValueError(f"'{batch_key}' not found in adata.obs.")
            if layer is not None:
                data_copy = data.copy()
                data_copy.X = data_copy.layers[layer]
                data = data_copy
            batch_cats = sorted(data.obs[batch_key].astype(str).unique())
            adatas = [data[data.obs[batch_key].astype(str) == b].copy()
                      for b in batch_cats]
            batch_names = batch_cats
        else:
            adatas = []
            batch_names = []
            for i, a in enumerate(data):
                a2 = a.copy()
                if layer is not None:
                    a2.X = a2.layers[layer]
                adatas.append(a2)
                if "batch" in a2.obs.columns and a2.obs["batch"].nunique() == 1:
                    batch_names.append(str(a2.obs["batch"].iloc[0]))
                else:
                    batch_names.append(f"batch_{i}")

        N = len(adatas)
        if N < 2:
            raise ValueError("Need at least 2 batches; got 1.")

        # ── Guardrails ─────────────────────────────────────────────────
        for i, a in enumerate(adatas):
            if a.n_obs < 20:
                raise ValueError(
                    f"Batch '{batch_names[i]}' has {a.n_obs} cells (minimum 20).")

        all_obs = [name for a in adatas for name in a.obs_names]
        if len(set(all_obs)) < len(all_obs):
            warnings.warn(
                "Non-unique obs_names across batches. Appending '_{batch}' suffix.",
                UserWarning)
            new_adatas = []
            for a, bname in zip(adatas, batch_names):
                a2 = a.copy()
                a2.obs_names = [f"{n}_{bname}" for n in a2.obs_names]
                new_adatas.append(a2)
            adatas = new_adatas

        # ── 1. Feature selection ───────────────────────────────────────
        if _fixed_features is not None:
            features = list(_fixed_features)
            # Validate all genes present in all datasets
            for i, a in enumerate(adatas):
                missing = [g for g in features if g not in set(a.var_names)]
                if missing:
                    raise ValueError(
                        f"Batch '{batch_names[i]}' is missing {len(missing)} "
                        f"of the {len(features)} fixed features.")
        else:
            # Check if prepare_for_integration pre-computed HVGs
            pre_feats = adatas[0].uns.get("integration_features", None)
            if pre_feats is not None:
                # Validate all batches carry the same pre-computed feature list
                ok = all(
                    set(a.uns.get("integration_features", [])) == set(pre_feats)
                    for a in adatas)
                if ok:
                    features = list(pre_feats)
                else:
                    warnings.warn(
                        "Inconsistent 'integration_features' across batches; "
                        "re-running HVG selection.", UserWarning, stacklevel=2)
                    n_feat = _estimate_n_features(adatas, n_features)
                    features = _select_integration_features(adatas, n_feat)
            else:
                n_feat = _estimate_n_features(adatas, n_features)
                features = _select_integration_features(adatas, n_feat)
        n_feat = len(features)

        # ── 2. Prepare matrices ────────────────────────────────────────
        Xs_lognorm = []
        Xs_scaled = []
        for i, (a, bname) in enumerate(zip(adatas, batch_names)):
            ln, sc_m = _prepare_matrices(a, features, batch_name=bname)
            Xs_lognorm.append(ln)
            Xs_scaled.append(sc_m)

        # ── 3. Guide tree or trivial order ─────────────────────────────
        if N == 2:
            merge_order = [(0, 1)]
        else:
            merge_order = _build_guide_tree(adatas, features, n_components,
                                            n_jobs, seed)

        # ── 4. Merge loop ──────────────────────────────────────────────
        log_raw_anchors = []
        log_filt_anchors = []
        log_k_anchor = []
        log_k_filter = []
        log_k_score = []
        log_k_weight = []
        log_sd = []

        # Reference fields for query mapping (captured in success path)
        last_U_k = None
        last_cc_ref = None
        last_mu_ref = None
        last_sigma_ref = None

        nodes = {
            i: (Xs_lognorm[i],
                Xs_scaled[i],
                Xs_lognorm[i].copy(),
                [batch_names[i]] * adatas[i].n_obs,
                adatas[i])
            for i in range(N)
        }
        next_node = N

        for left_id, right_id in merge_order:
            X_a_ln, X_a_sc, X_a_orig, labels_a, adata_a = nodes[left_id]
            X_b_ln, X_b_sc, X_b_orig, labels_b, adata_b = nodes[right_id]
            n_a, n_b = len(X_a_ln), len(X_b_ln)

            ka = _estimate_k_anchor(n_a, n_b, k_anchor)
            kf = _estimate_k_filter(n_a, n_b, k_filter)
            ks = _estimate_k_score(ka, k_score)
            log_k_anchor.append(ka)
            log_k_filter.append(kf)
            log_k_score.append(ks)

            # CCA
            cc_a, cc_b, U_k = _compute_cca(X_a_sc, X_b_sc, n_components, seed)

            # MNN
            anc_a, anc_b = _find_mnn_in_cca(cc_a, cc_b, ka, n_jobs, seed)
            log_raw_anchors.append(len(anc_a))

            X_orig_merged = np.vstack([X_a_orig, X_b_orig])

            # Zero-anchor fallback (before HD filter)
            if len(anc_a) == 0:
                warnings.warn(
                    f"No anchors found between '{_node_name(left_id, batch_names)}'"
                    f" and '{_node_name(right_id, batch_names)}'. "
                    "Concatenating without correction.",
                    UserWarning, stacklevel=2)
                log_filt_anchors.append(0)
                log_k_weight.append(0)
                log_sd.append(float("nan"))
                X_merged_ln = np.vstack([X_a_ln, X_b_ln])
                X_merged_sc = _rescale_merged(X_merged_ln)
                nodes[next_node] = (X_merged_ln, X_merged_sc, X_orig_merged,
                                    labels_a + labels_b,
                                    ad.concat([adata_a, adata_b]))
                next_node += 1
                continue

            # HD filter
            anc_a, anc_b = _filter_anchors_hd(
                anc_a, anc_b, X_a_sc, X_b_sc, U_k, kf, n_jobs, seed)
            log_filt_anchors.append(len(anc_a))

            # Zero-anchor fallback (after HD filter)
            if len(anc_a) == 0:
                warnings.warn(
                    "All anchors removed by HD filter between "
                    f"'{_node_name(left_id, batch_names)}' and "
                    f"'{_node_name(right_id, batch_names)}'. "
                    "Concatenating without correction.",
                    UserWarning, stacklevel=2)
                log_k_weight.append(0)
                log_sd.append(float("nan"))
                X_merged_ln = np.vstack([X_a_ln, X_b_ln])
                X_merged_sc = _rescale_merged(X_merged_ln)
                nodes[next_node] = (X_merged_ln, X_merged_sc, X_orig_merged,
                                    labels_a + labels_b,
                                    ad.concat([adata_a, adata_b]))
                next_node += 1
                continue

            # Score
            scores = _score_anchors(anc_a, anc_b, cc_a, cc_b, ks,
                                    n_jobs, seed)

            # k_weight
            kw = _estimate_k_weight(ka, len(anc_a), k_weight)
            log_k_weight.append(kw)

            # Capture U_k for query mapping (mu/sigma/cc_ref computed after correction)
            last_U_k = U_k

            # Symmetric correction
            corr_b, corr_a, sd_used = _compute_correction_vectors(
                X_a_ln, X_b_ln, anc_a, anc_b, scores,
                cc_a, cc_b, kw, sd_bandwidth, n_jobs, seed)
            log_sd.append(sd_used)

            # Apply and clamp
            X_a_ln_c = np.maximum(X_a_ln + corr_a, 0.0)
            X_b_ln_c = np.maximum(X_b_ln + corr_b, 0.0)

            X_merged_ln = np.vstack([X_a_ln_c, X_b_ln_c])
            X_merged_sc = _rescale_merged(X_merged_ln)

            # Capture reference fields for query mapping:
            # Use FULL merged statistics and re-project ALL cells through U_k.
            # This ensures cc_ref and future query projections use the same
            # z-scoring and the same gene-loading path.
            last_mu_ref = X_merged_ln.mean(axis=0).astype(np.float32)
            last_sigma_ref = X_merged_ln.std(axis=0, ddof=1).astype(np.float32)
            last_sigma_ref = np.where(last_sigma_ref < 1e-8, 1.0, last_sigma_ref)
            # Re-project all merged cells through U_k (same path query will use)
            last_cc_ref = (X_merged_sc @ last_U_k).astype(np.float32)
            row_norms = np.linalg.norm(last_cc_ref, axis=1, keepdims=True)
            last_cc_ref /= np.where(row_norms < 1e-12, 1.0, row_norms)
            last_cc_ref = np.ascontiguousarray(last_cc_ref)

            nodes[next_node] = (
                X_merged_ln, X_merged_sc, X_orig_merged,
                labels_a + labels_b,
                ad.concat([adata_a, adata_b]))
            next_node += 1

        # ── 5. Assemble output ─────────────────────────────────────────
        root_id = next_node - 1
        X_final, _, X_original, all_labels, adt_final = nodes[root_id]

        adata_out = adt_final[:, features].copy()
        adata_out.X = sp.csr_matrix(X_final.astype(np.float32))
        adata_out.obs["batch"] = all_labels
        adata_out.layers["original"] = sp.csr_matrix(
            X_original.astype(np.float32))

        # ── Store reference fields for query mapping ──────────────────
        if last_U_k is not None:
            adata_out.varm["cca_loadings"] = last_U_k
            adata_out.obsm["X_cca"] = last_cc_ref.astype(np.float32)
        else:
            import warnings
            warnings.warn(
                "No successful merge step found anchors; reference fields "
                "(varm['cca_loadings'], obsm['X_cca'], ref_mu, ref_sigma) "
                "were not stored. Query mapping will not be available.",
                UserWarning, stacklevel=2)

        # ── 6. Run log ─────────────────────────────────────────────────
        adata_out.uns["cca_integration"] = {
            "n_datasets": N,
            "n_features": n_feat,
            "features": features,
            "n_components": n_components,
            "merge_order": merge_order,
            "batch_names": batch_names,
            "correction_mode": "symmetric",
            "seed": seed,
            "n_jobs": n_jobs,
            "k_anchor_per_merge": log_k_anchor,
            "k_filter_per_merge": log_k_filter,
            "k_score_per_merge": log_k_score,
            "k_weight_per_merge": log_k_weight,
            "sd_bandwidth_per_merge": log_sd,
            "n_anchors_raw_per_merge": log_raw_anchors,
            "n_anchors_filt_per_merge": log_filt_anchors,
            "user_n_features": n_features,
            "user_k_anchor": k_anchor,
            "user_k_filter": k_filter,
            "user_k_score": k_score,
            "user_k_weight": k_weight,
            "user_sd_bandwidth": sd_bandwidth,
        }
        if last_mu_ref is not None:
            adata_out.uns["cca_integration"]["ref_mu"] = last_mu_ref
            adata_out.uns["cca_integration"]["ref_sigma"] = last_sigma_ref

        # ── 7. Optional z-score scaling ────────────────────────────────
        if scale_output:
            X_ln = adata_out.X
            if sp.issparse(X_ln):
                X_ln = X_ln.toarray()
            X_ln = np.asarray(X_ln, dtype=np.float32)
            adata_out.layers["corrected"] = sp.csr_matrix(X_ln)
            mu_s = X_ln.mean(axis=0)
            sigma_s = X_ln.std(axis=0, ddof=1)
            sigma_s = np.where(sigma_s < 1e-8, 1.0, sigma_s)
            X_sc = np.clip((X_ln - mu_s) / sigma_s, -10.0, 10.0).astype(np.float32)
            adata_out.X = X_sc

        return adata_out
    # =======================================================================
    # CCA Query Mapping — save/load reference + map new queries
    # =======================================================================

    def _validate_reference_fields(adata_ref) -> None:
        """Raise ValueError if required reference fields are absent."""
        missing = []
        for key in ["cca_loadings"]:
            if key not in adata_ref.varm:
                missing.append(f"varm['{key}']")
        for key in ["X_cca"]:
            if key not in adata_ref.obsm:
                missing.append(f"obsm['{key}']")
        for key in ["ref_mu", "ref_sigma"]:
            if key not in adata_ref.uns.get("cca_integration", {}):
                missing.append(f"uns['cca_integration']['{key}']")
        if missing:
            raise ValueError(
                f"Reference AnnData is missing required fields: {missing}. "
                "Save and load via save_cca_reference / load_cca_reference.")

    def save_cca_reference(
        adata_int,
        path,
    ) -> None:
        """Save a completed CCA integration result as a reusable mapping reference.

        Parameters
        ----------
        adata_int : AnnData
            Output of run_cca_integration().
        path : str or Path
            Output path (should end in '.h5ad').
        """
        from pathlib import Path as _Path
        _validate_reference_fields(adata_int)
        adata_int.write_h5ad(_Path(path))

    def load_cca_reference(path):
        """Load a saved CCA reference from an .h5ad file.

        Parameters
        ----------
        path : str or Path
            Path to a .h5ad file saved by save_cca_reference().

        Returns
        -------
        AnnData with the reference fields validated.
        """
        from pathlib import Path as _Path
        import anndata as ad
        path = _Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Reference file not found: {path}")
        adata_ref = ad.read_h5ad(path)
        _validate_reference_fields(adata_ref)
        return adata_ref

    # ── Query correction helpers ───────────────────────────────────────────

    def _compute_query_correction(
        X_ref_lognorm: np.ndarray,
        query_lognorm: np.ndarray,
        anc_ref: np.ndarray,
        anc_query: np.ndarray,
        scores: np.ndarray,
        embed_ref: np.ndarray,
        embed_query: np.ndarray,
        k_weight: int,
        sd_bandwidth,
        n_jobs: int = 1,
        seed: int = 0,
    ) -> tuple[np.ndarray, float]:
        """One-sided correction: compute correction vector for each query cell.

        Batch vectors point from query toward reference (full step, not midpoint).
        embed_ref/embed_query are the embedding used for weight kNN (can be
        CCA space or z-scored expression space).
        """
        n_jobs = _resolve_n_jobs(n_jobs)
        n_q, p = query_lognorm.shape

        batch_vecs = (X_ref_lognorm[anc_ref] -
                      query_lognorm[anc_query]).astype(np.float32)

        anchor_pos_ref = np.ascontiguousarray(
            embed_ref[anc_ref].astype(np.float32))
        k_eff = min(k_weight, len(anc_ref))
        idx_anchors = _build_hnsw_index(anchor_pos_ref, seed=seed)
        idx_q, dist_q = _query_hnsw(
            idx_anchors, embed_query, k=k_eff, n_jobs=n_jobs)

        d_k_q = dist_q[:, -1:] + 1e-10
        raw_w_q = np.clip(1.0 - dist_q / d_k_q, 0.0, 1.0) * scores[idx_q]

        sd = _estimate_sd_bandwidth(raw_w_q, sd_bandwidth)
        sd_val = 2.0 / sd
        kernel_q = 1.0 - np.exp(-raw_w_q / sd_val**2)
        kernel_q /= (kernel_q.sum(axis=1, keepdims=True) + 1e-12)

        correction_query = np.zeros((n_q, p), dtype=np.float32)
        chunk = max(500, n_q // max(1, n_jobs))
        for s in range(0, n_q, chunk):
            e = min(s + chunk, n_q)
            correction_query[s:e] = np.einsum(
                'ck,ckp->cp',
                kernel_q[s:e],
                batch_vecs[idx_q[s:e]])

        return correction_query, sd

    def _align_query_to_features(query_adata, features: list) -> "AnnData":
        """Return a copy of query_adata aligned to `features`, zero-imputing missing genes."""
        import anndata as ad
        n_q = query_adata.n_obs
        p = len(features)
        X_aligned = np.zeros((n_q, p), dtype=np.float32)
        query_gene_index = {g: i for i, g in enumerate(query_adata.var_names)}
        for ref_idx, gene in enumerate(features):
            if gene in query_gene_index:
                q_idx = query_gene_index[gene]
                col = query_adata.X[:, q_idx]
                if sp.issparse(col):
                    col = col.toarray().ravel()
                X_aligned[:, ref_idx] = np.asarray(col, dtype=np.float32)

        out = ad.AnnData(
            X=sp.csr_matrix(X_aligned),
            obs=query_adata.obs.copy(),
            var=pd.DataFrame(index=features))
        if "batch" not in out.obs.columns:
            out.obs["batch"] = "query"
        return out

    def _assemble_query_output(
        query_adata,
        X_corrected: np.ndarray,
        X_original: np.ndarray,
        features: list,
        n_anchors_raw: int,
        n_anchors_filt: int,
        ka: int, kf: int, ks: int, kw: int,
        sd_used: float,
        mode: str,
        reference,
    ):
        """Build the output AnnData for mode='query_only'."""
        import anndata as ad

        obs = query_adata.obs.copy()
        if "batch" not in obs.columns:
            obs["batch"] = "query"

        out = ad.AnnData(
            X=sp.csr_matrix(X_corrected.astype(np.float32)),
            obs=obs,
            var=pd.DataFrame(index=features))
        out.layers["original"] = sp.csr_matrix(X_original.astype(np.float32))
        out.uns["cca_mapping"] = {
            "mode": mode,
            "n_ref_cells": reference.n_obs,
            "n_query_cells": query_adata.n_obs,
            "n_features": len(features),
            "features": features,
            "n_anchors_raw": n_anchors_raw,
            "n_anchors_filt": n_anchors_filt,
            "k_anchor_used": ka,
            "k_filter_used": kf,
            "k_score_used": ks,
            "k_weight_used": kw,
            "sd_bandwidth_used": sd_used,
            "ref_batch_names": list(reference.uns.get(
                "cca_integration", {}).get("batch_names", [])),
            "projection_method": "back_projected_U_k",
        }
        return out

    # ── Map query only ─────────────────────────────────────────────────────

    def _map_query_only(
        query_adata,
        query_lognorm,
        query_scaled_perquery,
        query_scaled_refnorm,
        X_ref_lognorm,
        cc_ref,
        U_k,
        mu_ref,
        sigma_ref,
        features,
        reference,
        k_anchor, k_filter, k_score, k_weight, sd_bandwidth,
        n_jobs, seed,
    ):
        """One-sided correction: only query cells move toward the reference.

        Query cells are projected into the reference CCA space via U_k
        (gene loadings from the reference integration). Anchors, scoring,
        and correction weights all operate in this shared CCA space.
        Correction vectors are computed in lognorm expression space.
        """
        import warnings
        n_q = query_lognorm.shape[0]
        n_ref = X_ref_lognorm.shape[0]

        # Project query into reference CCA space via U_k
        cc_query = (query_scaled_refnorm @ U_k).astype(np.float32)
        norms_q = np.linalg.norm(cc_query, axis=1, keepdims=True)
        cc_query /= np.where(norms_q < 1e-12, 1.0, norms_q)
        cc_query = np.ascontiguousarray(cc_query)

        ka = _estimate_k_anchor(n_q, n_ref, k_anchor)
        kf = _estimate_k_filter(n_q, n_ref, k_filter)
        ks = _estimate_k_score(ka, k_score)

        # MNN anchors in CCA space (ref=A, query=B)
        anc_ref, anc_query = _find_mnn_in_cca(
            cc_ref, cc_query, ka, n_jobs, seed)
        n_anchors_raw = len(anc_ref)

        # HD filter using CCA-loading-weighted feature space
        X_ref_scaled = np.ascontiguousarray(
            np.clip((X_ref_lognorm - mu_ref) / sigma_ref, -10.0, 10.0
                    ).astype(np.float32))
        if n_anchors_raw > 0 and kf > 0:
            anc_ref, anc_query = _filter_anchors_hd(
                anc_ref, anc_query,
                X_ref_scaled, query_scaled_perquery,
                U_k, kf, n_jobs, seed)
        n_anchors_filt = len(anc_ref)

        # Zero-anchor fallback
        if n_anchors_filt == 0:
            warnings.warn(
                "No anchors found between query and reference. "
                "Returning query with no correction applied.",
                UserWarning, stacklevel=3)
            return _assemble_query_output(
                query_adata, query_lognorm, query_lognorm,
                features, n_anchors_raw, n_anchors_filt,
                ka, kf, ks, 0, float("nan"), mode="query_only",
                reference=reference)

        # SNN scoring in CCA space
        scores = _score_anchors(
            anc_ref, anc_query, cc_ref, cc_query, ks, n_jobs, seed)

        kw = _estimate_k_weight(ka, n_anchors_filt, k_weight)

        # One-sided correction (batch vectors in lognorm space,
        # weight kNN in CCA space)
        correction_query, sd_used = _compute_query_correction(
            X_ref_lognorm, query_lognorm,
            anc_ref, anc_query, scores,
            cc_ref, cc_query,
            kw, sd_bandwidth, n_jobs, seed)

        query_corrected = np.maximum(query_lognorm + correction_query, 0.0)

        return _assemble_query_output(
            query_adata, query_corrected, query_lognorm,
            features, n_anchors_raw, n_anchors_filt,
            ka, kf, ks, kw, sd_used, mode="query_only",
            reference=reference)

    # ── Full reintegration ─────────────────────────────────────────────────

    def _map_full_reintegration(
        query_adata,
        reference,
        features: list,
        n_components: int,
        k_anchor, k_filter, k_score, k_weight, sd_bandwidth,
        n_jobs, seed,
    ):
        """Full symmetric re-integration of reference + query."""
        import anndata as ad

        ref_sub = reference[:, features].copy()
        # Ensure ref_sub.X is lognorm (not z-scored)
        if "corrected" in reference.layers:
            ref_sub.X = reference[:, features].layers["corrected"].copy()
        ref_sub.obs["batch"] = reference.obs.get("batch",
            pd.Series(["reference"] * reference.n_obs,
                       index=reference.obs_names))

        query_sub = _align_query_to_features(query_adata, features)

        result = run_cca_integration(
            [ref_sub, query_sub],
            n_features=len(features),
            n_components=n_components,
            k_anchor=k_anchor,
            k_filter=k_filter,
            k_score=k_score,
            k_weight=k_weight,
            sd_bandwidth=sd_bandwidth,
            scale_output=True,
            n_jobs=n_jobs,
            seed=seed,
            _fixed_features=features,
        )
        result.uns["cca_integration"]["mapping_mode"] = "full_reintegration"
        return result

    # ── Main public function ───────────────────────────────────────────────

    def _map_single_query(
        query,
        reference,
        mode,
        min_shared_features,
        k_anchor, k_filter, k_score, k_weight, sd_bandwidth,
        n_jobs, seed,
    ):
        """Map a single query AnnData against a reference. Internal helper."""
        from pathlib import Path as _Path
        import warnings

        # Extract reference components
        features = list(reference.var_names)
        n_features = len(features)
        n_components = reference.uns["cca_integration"]["n_components"]
        U_k = np.asarray(reference.varm["cca_loadings"], dtype=np.float32)
        cc_ref = np.asarray(reference.obsm["X_cca"], dtype=np.float32)
        mu_ref = np.asarray(
            reference.uns["cca_integration"]["ref_mu"], dtype=np.float32)
        sigma_ref = np.asarray(
            reference.uns["cca_integration"]["ref_sigma"], dtype=np.float32)
        sigma_ref = np.where(sigma_ref < 1e-8, 1.0, sigma_ref)

        if "corrected" in reference.layers:
            _ref_src = reference.layers["corrected"]
        else:
            _ref_src = reference.X
        X_ref_lognorm = (_ref_src.toarray()
                         if sp.issparse(_ref_src)
                         else np.asarray(_ref_src, dtype=np.float32))

        # Feature alignment
        query_genes = set(query.var_names)
        shared = [g for g in features if g in query_genes]
        coverage = len(shared) / len(features)

        if coverage < min_shared_features:
            raise ValueError(
                f"Query covers only {coverage:.1%} of reference integration "
                f"genes ({len(shared)}/{len(features)}). Minimum is "
                f"{min_shared_features:.1%}.")

        missing_in_query = [g for g in features if g not in query_genes]
        if missing_in_query:
            warnings.warn(
                f"Query is missing {len(missing_in_query)}/{len(features)} "
                f"reference genes. Missing genes will be zero-imputed. "
                f"Coverage: {coverage:.1%}.",
                UserWarning, stacklevel=3)

        # Build aligned query expression matrix
        X_sub = np.zeros((query.n_obs, n_features), dtype=np.float32)
        query_gene_index = {g: i for i, g in enumerate(query.var_names)}
        for ref_idx, gene in enumerate(features):
            if gene in query_gene_index:
                q_idx = query_gene_index[gene]
                col = query.X[:, q_idx]
                if sp.issparse(col):
                    col = col.toarray().ravel()
                X_sub[:, ref_idx] = np.asarray(col, dtype=np.float32)

        # Prepare query matrices
        norm_state = _detect_normalization(X_sub)
        if norm_state == "raw_counts":
            row_sums = X_sub.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            query_lognorm = np.log1p(
                X_sub / row_sums * 1e4).astype(np.float32)
            warnings.warn(
                "Query: raw counts detected; applying log1p(CPM/1e4).",
                UserWarning, stacklevel=3)
        elif norm_state == "pre_normalized":
            query_lognorm = np.log1p(X_sub).astype(np.float32)
            warnings.warn(
                f"Query: pre-normalised values detected "
                f"(max={float(X_sub.max()):.0f}); applying log1p.",
                UserWarning, stacklevel=3)
        else:
            query_lognorm = X_sub.astype(np.float32)
        query_lognorm = np.ascontiguousarray(query_lognorm)

        mu_q = query_lognorm.mean(axis=0)
        sigma_q = query_lognorm.std(axis=0, ddof=1)
        sigma_q = np.where(sigma_q < 1e-8, 1.0, sigma_q)
        query_scaled_perquery = np.ascontiguousarray(
            np.clip((query_lognorm - mu_q) / sigma_q, -10.0, 10.0
                    ).astype(np.float32))
        query_scaled_refnorm = np.ascontiguousarray(
            np.clip((query_lognorm - mu_ref) / sigma_ref, -10.0, 10.0
                    ).astype(np.float32))

        if mode == "query_only":
            return _map_query_only(
                query, query_lognorm, query_scaled_perquery,
                query_scaled_refnorm,
                X_ref_lognorm, cc_ref, U_k, mu_ref, sigma_ref, features,
                reference, k_anchor, k_filter, k_score, k_weight,
                sd_bandwidth, n_jobs, seed)
        else:
            return _map_full_reintegration(
                query, reference, features, n_components,
                k_anchor, k_filter, k_score, k_weight, sd_bandwidth,
                n_jobs, seed)

    def map_to_cca_reference(
        query,
        reference,
        mode: str = "query_only",
        mapping_order: list | None = None,
        min_shared_features: float = 0.8,
        k_anchor: int | None = None,
        k_filter: int | None = None,
        k_score: int | None = None,
        k_weight: int | None = None,
        sd_bandwidth: float | None = None,
        n_jobs: int = 1,
        seed: int = 0,
        sequential_topometry: bool = False,
        topometry_kwargs: dict | None = None,
        return_intermediates: bool = False,
    ):
        """Correct one or more query datasets against a saved CCA reference.

        Parameters
        ----------
        query : AnnData or list[AnnData]
            The new dataset(s) to map.  When a list is provided, queries are
            mapped sequentially.
        reference : AnnData or str or Path
            A completed integration result from ``run_cca_integration()``.
        mode : str
            ``'query_only'``: only query cells are corrected (reference frozen).
            ``'full_reintegration'``: both re-corrected symmetrically.
        mapping_order : list of int, optional
            Indices into ``query`` (when query is a list) specifying the
            order in which to map datasets.  Use the output of
            :func:`find_mapping_order`.  If None, uses the list order as-is.
        min_shared_features : float
            Minimum fraction of reference genes present in query.  Default 0.8.
        k_anchor, k_filter, k_score, k_weight, sd_bandwidth
            Override adaptive hyperparameter estimates.  Default None (auto).
        n_jobs : int
            Threads for hnswlib.  Default 1.
        seed : int
            Random seed.  Default 0.
        sequential_topometry : bool
            If True and query is a list, run :func:`fit_adata` on the growing
            atlas after each query is mapped.  Stores TopoMetry scaffolds and
            projections in the intermediate atlas objects.  Default False.
        topometry_kwargs : dict, optional
            Extra keyword arguments forwarded to :func:`fit_adata` when
            ``sequential_topometry=True``.
        return_intermediates : bool
            If True, return a tuple ``(atlas, intermediates)`` where
            *intermediates* is an ordered dict mapping step labels to the
            atlas AnnData after each mapping step.  Default False.

        Returns
        -------
        AnnData
            Final atlas (reference + all corrected queries, concatenated).
            If ``scale_output=True`` was used in ``run_cca_integration``, the
            corrected expression values are z-scored.
        tuple(AnnData, dict)
            Only when ``return_intermediates=True``: ``(atlas, steps)`` where
            *steps* is ``{'step_0': atlas_after_first_query, ...}``.

        Notes
        -----
        When mapping multiple queries **always** maps against the original
        *reference* (with CCA fields intact).  The growing atlas (reference +
        previously mapped queries) is used only for z-scoring and for
        :func:`fit_adata` when ``sequential_topometry=True``.
        """
        import anndata as ad
        import warnings
        from pathlib import Path as _Path

        # ── Load / validate reference ──────────────────────────────────────
        if isinstance(reference, (str, _Path)):
            reference = load_cca_reference(reference)
        else:
            _validate_reference_fields(reference)

        if mode not in ("query_only", "full_reintegration"):
            raise ValueError(
                f"mode must be 'query_only' or 'full_reintegration', "
                f"got '{mode}'.")

        # ── Single-query fast path ─────────────────────────────────────────
        if isinstance(query, AnnData):
            result = _map_single_query(
                query, reference, mode, min_shared_features,
                k_anchor, k_filter, k_score, k_weight, sd_bandwidth,
                n_jobs, seed)
            if return_intermediates:
                return result, {"step_0": result}
            return result

        # ── Multi-query sequential path ────────────────────────────────────
        queries = list(query)
        if not queries:
            raise ValueError("query list is empty.")

        if mapping_order is not None:
            if sorted(mapping_order) != list(range(len(queries))):
                raise ValueError(
                    "mapping_order must be a permutation of "
                    f"range({len(queries)}).")
            ordered = [queries[i] for i in mapping_order]
        else:
            ordered = queries
            mapping_order = list(range(len(queries)))

        if topometry_kwargs is None:
            topometry_kwargs = {}

        # The reference with CCA fields is kept frozen for all mapping steps
        adata_ref_map = reference

        # For atlas building, always use lognorm .X so that the concat is
        # on a consistent scale before z-scoring.  When scale_output=True
        # in run_cca_integration, .X is z-scored and lognorm lives in
        # layers["corrected"].  We need lognorm here so that the whole
        # atlas (ref + queries) can be re-z-scored uniformly.
        ref_for_atlas = reference.copy()
        if "corrected" in ref_for_atlas.layers:
            _ln = ref_for_atlas.layers["corrected"]
            ref_for_atlas.X = (_ln if not sp.issparse(_ln)
                               else _ln)   # keep as-is (sparse or dense)

        def _zscale_atlas(adata_c):
            """Z-score .X inplace; preserve lognorm in layers['corrected']."""
            X = adata_c.X
            if sp.issparse(X):
                X = X.toarray()
            X = np.asarray(X, dtype=np.float32)
            adata_c.layers["corrected"] = sp.csr_matrix(X)
            mu = X.mean(axis=0)
            sigma = X.std(axis=0, ddof=1)
            sigma = np.where(sigma < 1e-8, 1.0, sigma)
            adata_c.X = np.clip(
                (X - mu) / sigma, -10.0, 10.0).astype(np.float32)
            return adata_c

        atlas_parts = [ref_for_atlas]  # lognorm .X for the reference
        intermediates = {}

        for step_idx, q in enumerate(ordered):
            step_label = f"step_{step_idx}"

            mapped = _map_single_query(
                q, adata_ref_map, mode, min_shared_features,
                k_anchor, k_filter, k_score, k_weight, sd_bandwidth,
                n_jobs, seed)
            atlas_parts.append(mapped)   # mapped.X is already lognorm

            # Build current atlas (reference + all mapped so far), z-scored
            current_atlas = ad.concat(
                atlas_parts,
                join="inner",
                label=None,
                uns_merge="first",
            )
            _zscale_atlas(current_atlas)

            if sequential_topometry:
                fit_adata(current_atlas, **topometry_kwargs)

            if return_intermediates:
                intermediates[step_label] = current_atlas.copy()

        # Final atlas: re-concat and z-score (avoids duplicating last step)
        final_atlas = ad.concat(
            atlas_parts,
            join="inner",
            label=None,
            uns_merge="first",
        )
        _zscale_atlas(final_atlas)

        if return_intermediates:
            return final_atlas, intermediates
        return final_atlas
    # =======================================================================
    # Integration Quality Metrics
    # =======================================================================

    def knn_purity(
        adata,
        label_key: str = "cell_type",
        embedding_key: str | None = None,
        k: int = 30,
        n_jobs: int = 2,
        seed: int = 0,
    ) -> float:
        """kNN purity: fraction of each cell's k nearest neighbors sharing its label.

        Higher = better biological preservation (max = 1.0).

        Parameters
        ----------
        adata : AnnData
            Must contain the embedding in .obsm[embedding_key] or .X.
        label_key : str
            Column in .obs for the label (e.g. cell type).
        embedding_key : str or None
            Key in .obsm. If None, uses .X (densified if sparse).
        k : int
            Number of neighbors.
        n_jobs : int
            Threads for hnswlib.
        seed : int
            Random seed.

        Returns
        -------
        float : mean kNN purity across all cells.
        """
        Z = _get_embedding(adata, embedding_key)
        labels = np.asarray(adata.obs[label_key].values)
        idx = _knn_indices(Z, k, n_jobs, seed)
        nbr_labels = labels[idx]
        purity = (nbr_labels == labels[:, None]).mean(axis=1)
        return float(purity.mean())

    def knn_mixing(
        adata,
        batch_key: str = "batch",
        embedding_key: str | None = None,
        k: int = 30,
        n_jobs: int = 2,
        seed: int = 0,
    ) -> float:
        """kNN mixing: fraction of each cell's k nearest neighbors from a different batch.

        Higher = better batch mixing (max = 1.0).

        Parameters
        ----------
        adata : AnnData
            Must contain the embedding in .obsm[embedding_key] or .X.
        batch_key : str
            Column in .obs identifying batches.
        embedding_key : str or None
            Key in .obsm. If None, uses .X.
        k : int
            Number of neighbors.
        n_jobs : int
            Threads for hnswlib.
        seed : int
            Random seed.

        Returns
        -------
        float : mean kNN mixing across all cells.
        """
        Z = _get_embedding(adata, embedding_key)
        batches = np.asarray(adata.obs[batch_key].values)
        idx = _knn_indices(Z, k, n_jobs, seed)
        nbr_batches = batches[idx]
        mixing = (nbr_batches != batches[:, None]).mean(axis=1)
        return float(mixing.mean())

    def ilisi(
        adata,
        batch_key: str = "batch",
        embedding_key: str | None = None,
        k: int = 30,
        n_jobs: int = 2,
        seed: int = 0,
        return_per_cell: bool = False,
    ):
        """Integration LISI (iLISI): local inverse Simpson's index on batch labels.

        Higher = better batch mixing (max = number of batches).

        Parameters
        ----------
        adata : AnnData
        batch_key : str
            Column in .obs identifying batches.
        embedding_key : str or None
            Key in .obsm. If None, uses .X.
        k : int
            Number of neighbors.
        n_jobs : int
            Threads for hnswlib.
        seed : int
            Random seed.
        return_per_cell : bool
            If True, return the full per-cell array instead of the median.

        Returns
        -------
        float (median iLISI) or np.ndarray (per-cell) if return_per_cell.
        """
        Z = _get_embedding(adata, embedding_key)
        labels = np.asarray(adata.obs[batch_key].values)
        vals = _compute_lisi_values(Z, labels, k, n_jobs, seed)
        if return_per_cell:
            return vals
        return float(np.median(vals))

    def clisi(
        adata,
        label_key: str = "cell_type",
        embedding_key: str | None = None,
        k: int = 30,
        n_jobs: int = 2,
        seed: int = 0,
        return_per_cell: bool = False,
    ):
        """Cell-type LISI (cLISI): local inverse Simpson's index on cell-type labels.

        Lower = better cell-type preservation (min = 1.0).

        Parameters
        ----------
        adata : AnnData
        label_key : str
            Column in .obs for cell-type labels.
        embedding_key : str or None
            Key in .obsm. If None, uses .X.
        k : int
            Number of neighbors.
        n_jobs : int
            Threads for hnswlib.
        seed : int
            Random seed.
        return_per_cell : bool
            If True, return the full per-cell array instead of the median.

        Returns
        -------
        float (median cLISI) or np.ndarray (per-cell) if return_per_cell.
        """
        Z = _get_embedding(adata, embedding_key)
        labels = np.asarray(adata.obs[label_key].values)
        vals = _compute_lisi_values(Z, labels, k, n_jobs, seed)
        if return_per_cell:
            return vals
        return float(np.median(vals))

    def cluster_agreement(
        adata,
        label_key: str = "cell_type",
        cluster_key: str = "topo_clusters",
    ) -> dict:
        """Compute ARI and NMI between cluster assignments and ground-truth labels.

        Parameters
        ----------
        adata : AnnData
        label_key : str
            Ground-truth label column in .obs.
        cluster_key : str
            Cluster assignment column in .obs.

        Returns
        -------
        dict with keys 'ari' and 'nmi'.
        """
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        labels = adata.obs[label_key].values
        clusters = adata.obs[cluster_key].values
        return {
            "ari": float(adjusted_rand_score(labels, clusters)),
            "nmi": float(normalized_mutual_info_score(labels, clusters)),
        }

    def compute_integration_metrics(
        adata_corrected,
        adata_uncorrected = None,
        metrics: list | None = None,
        batch_key: str = "batch",
        cell_type_key: str | None = None,
        cluster_key: str = "topo_clusters",
        embedding_key: str | None = None,
        k: int = 30,
        n_jobs: int = 2,
        seed: int = 0,
    ) -> dict:
        """Compute a suite of integration quality metrics.

        Parameters
        ----------
        adata_corrected : AnnData
            The integrated/corrected AnnData.
        adata_uncorrected : AnnData, optional
            If provided, also compute metrics on the uncorrected data and
            return a side-by-side comparison.
        metrics : list of str, optional
            Which metrics to compute. Default: all available.
            Options: 'knn_purity', 'knn_mixing', 'ilisi', 'clisi', 'ari', 'nmi'.
        batch_key : str
            Column in .obs identifying batches.
        cell_type_key : str or None
            Column in .obs for cell-type labels. If None, cell-type-dependent
            metrics (knn_purity, clisi, ari, nmi) are skipped.
        cluster_key : str
            Column in .obs for cluster assignments (used by ari/nmi).
        embedding_key : str or None
            Key in .obsm for the embedding. If None, uses .X.
            If the key 'X_ms_spectral_scaffold' exists in .obsm, it is used
            by default for corrected data.
        k : int
            Number of neighbors for kNN-based metrics.
        n_jobs : int
            Threads for hnswlib.
        seed : int
            Random seed.

        Returns
        -------
        dict : keys are metric names, values are floats.
            If adata_uncorrected is provided, keys are prefixed with
            'corrected_' and 'uncorrected_', plus 'delta_' for differences.
        """
        all_metrics = ['knn_purity', 'knn_mixing', 'ilisi', 'clisi', 'ari', 'nmi']
        if metrics is None:
            metrics = all_metrics
        metrics = [m for m in metrics if m in all_metrics]

        ct_metrics = {'knn_purity', 'clisi', 'ari', 'nmi'}

        def _compute_for(ad, emb_key):
            # Auto-detect scaffold if no explicit key
            ek = emb_key
            if ek is None and 'X_ms_spectral_scaffold' in ad.obsm:
                ek = 'X_ms_spectral_scaffold'
            elif ek is None and 'X_spectral_scaffold' in ad.obsm:
                ek = 'X_spectral_scaffold'

            result = {}
            if 'knn_purity' in metrics and cell_type_key is not None:
                result['knn_purity'] = knn_purity(
                    ad, label_key=cell_type_key, embedding_key=ek,
                    k=k, n_jobs=n_jobs, seed=seed)
            if 'knn_mixing' in metrics:
                result['knn_mixing'] = knn_mixing(
                    ad, batch_key=batch_key, embedding_key=ek,
                    k=k, n_jobs=n_jobs, seed=seed)
            if 'ilisi' in metrics:
                result['ilisi'] = ilisi(
                    ad, batch_key=batch_key, embedding_key=ek,
                    k=k, n_jobs=n_jobs, seed=seed)
            if 'clisi' in metrics and cell_type_key is not None:
                result['clisi'] = clisi(
                    ad, label_key=cell_type_key, embedding_key=ek,
                    k=k, n_jobs=n_jobs, seed=seed)
            if 'ari' in metrics and cell_type_key is not None and cluster_key in ad.obs.columns:
                result['ari'] = cluster_agreement(
                    ad, label_key=cell_type_key, cluster_key=cluster_key)['ari']
            if 'nmi' in metrics and cell_type_key is not None and cluster_key in ad.obs.columns:
                result['nmi'] = cluster_agreement(
                    ad, label_key=cell_type_key, cluster_key=cluster_key)['nmi']
            return result

        corr_results = _compute_for(adata_corrected, embedding_key)

        if adata_uncorrected is None:
            return corr_results

        uncorr_results = _compute_for(adata_uncorrected, embedding_key)

        combined = {}
        for m in corr_results:
            combined[f"corrected_{m}"] = corr_results[m]
        for m in uncorr_results:
            combined[f"uncorrected_{m}"] = uncorr_results[m]
        for m in set(corr_results) & set(uncorr_results):
            combined[f"delta_{m}"] = corr_results[m] - uncorr_results[m]
        return combined

    # ── Metrics internals ──────────────────────────────────────────────────

    def _get_embedding(adata, embedding_key):
        """Extract dense float32 embedding from adata."""
        if embedding_key is not None:
            Z = np.asarray(adata.obsm[embedding_key], dtype=np.float32)
        else:
            Z = adata.X
            if sp.issparse(Z):
                Z = Z.toarray()
            Z = np.asarray(Z, dtype=np.float32)
        return np.ascontiguousarray(Z)

    def _knn_indices(Z, k, n_jobs, seed):
        """Build hnswlib index on Z, return (n, k) neighbor indices (excluding self)."""
        n = Z.shape[0]
        k_eff = min(k, n - 1)
        idx_hnsw = _build_hnsw_index(Z, space="l2", n_jobs=n_jobs, seed=seed)
        labels, _ = _query_hnsw(idx_hnsw, Z, k=k_eff + 1, n_jobs=n_jobs)
        return labels[:, 1:k_eff + 1]

    def _compute_lisi_values(Z, labels, k, n_jobs, seed):
        """Compute per-cell LISI values with Gaussian kernel weighting."""
        n = Z.shape[0]
        k_eff = min(k, n - 1)
        idx_hnsw = _build_hnsw_index(Z, space="l2", n_jobs=n_jobs, seed=seed)
        labels_nn, dists = _query_hnsw(idx_hnsw, Z, k=k_eff + 1, n_jobs=n_jobs)
        labels_nn = labels_nn[:, 1:k_eff + 1]
        dists = dists[:, 1:k_eff + 1]

        categories = np.unique(labels)
        n_cats = len(categories)
        cat_to_idx = {c: i for i, c in enumerate(categories)}
        lab_idx = np.array([cat_to_idx[l] for l in labels], dtype=np.int32)

        nbr_labels = lab_idx[labels_nn]

        sigma = np.median(dists, axis=1, keepdims=True)
        sigma = np.where(sigma < 1e-10, 1.0, sigma)
        weights = np.exp(-dists / sigma)
        weights /= weights.sum(axis=1, keepdims=True)

        lisi_values = np.zeros(n, dtype=np.float64)
        for i in range(n):
            props = np.zeros(n_cats)
            for j in range(k_eff):
                props[nbr_labels[i, j]] += weights[i, j]
            simpson = np.sum(props ** 2)
            lisi_values[i] = 1.0 / max(simpson, 1e-12)

        return lisi_values

    # =======================================================================
    # High-level integration API
    # =======================================================================

    def prepare_for_integration(
        data,
        batch_key: str | None = None,
        input_type: str = "auto",
        layer: str | None = None,
        select_hvgs: bool = True,
        n_hvgs: int = 3000,
        hvg_flavor: str = "seurat_v3",
        scale_max_val: float = 10.0,
    ) -> None:
        """Normalize and optionally select integration HVGs in-place.

        Call this on your data **before** ``run_cca_integration`` to ensure
        ``.X`` contains a clean log-normalised expression matrix.

        Parameters
        ----------
        data : AnnData or list[AnnData]
            Input data.  Modifications are applied in-place.
        batch_key : str, optional
            ``.obs`` column identifying batches.  Used for per-batch HVG
            selection when *data* is a single AnnData.
        input_type : {'auto', 'counts', 'lognorm'}
            ``'auto'``: auto-detect normalisation state.
            ``'counts'``: raw integer counts → CPM + log1p.
            ``'lognorm'``: data is already log-normalised, no-op.
        layer : str, optional
            Use ``adata.layers[layer]`` as the expression source instead of
            ``.X``.  After processing, ``.X`` will contain the lognorm matrix.
        select_hvgs : bool
            If True (default), run per-batch HVG selection and store the
            selected gene list in ``adata.uns['integration_features']`` and a
            boolean flag in ``adata.var['integration_hvg']``.
        n_hvgs : int
            Maximum number of integration HVGs.  Default 3000.
        hvg_flavor : str
            Scanpy HVG flavour used in ``_select_integration_features``.
            Default ``'seurat_v3'``.
        scale_max_val : float
            Unused placeholder retained for API consistency.

        Returns
        -------
        None  (modifies data in-place).

        Notes
        -----
        * Raw counts are normalised to CPM/1e4 then log1p-transformed.
        * Pre-normalised values (TPM/FPKM) receive log1p only.
        * The original expression is preserved in ``adata.layers['counts']``
          (if not already present).
        * After this call, ``adata.X`` is the lognorm matrix and
          ``adata.layers['lognorm']`` is a copy of it.
        * If ``select_hvgs=True``, ``run_cca_integration`` will automatically
          use the pre-computed ``uns['integration_features']`` instead of
          running HVG selection again.
        """
        import warnings

        # ── Coerce to list of AnnDatas ─────────────────────────────────────
        if isinstance(data, AnnData):
            adatas_main = [data]
            if batch_key is not None and batch_key in data.obs.columns:
                batch_cats = sorted(data.obs[batch_key].astype(str).unique())
                batch_views = [data[data.obs[batch_key].astype(str) == b]
                               for b in batch_cats]
            else:
                batch_views = [data]
        elif isinstance(data, list):
            adatas_main = data
            batch_views = data
        else:
            raise TypeError("data must be AnnData or list[AnnData].")

        # ── Normalize each AnnData inplace ─────────────────────────────────
        def _norm_inplace(adata):
            X = adata.layers[layer] if layer is not None else adata.X
            if sp.issparse(X):
                X = X.toarray()
            X = np.asarray(X, dtype=np.float32)

            if input_type == "auto":
                norm_state = _detect_normalization(X)
            elif input_type == "counts":
                norm_state = "raw_counts"
            elif input_type == "lognorm":
                norm_state = "lognorm"
            else:
                raise ValueError(
                    "input_type must be 'auto', 'counts', or 'lognorm'.")

            if norm_state == "raw_counts":
                if "counts" not in adata.layers:
                    adata.layers["counts"] = adata.X.copy()
                row_sums = X.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums == 0, 1.0, row_sums)
                X_ln = np.log1p(X / row_sums * 1e4).astype(np.float32)
                adata.X = sp.csr_matrix(X_ln)
                adata.layers["lognorm"] = sp.csr_matrix(X_ln)
            elif norm_state == "pre_normalized":
                warnings.warn(
                    f"Pre-normalised values detected (max={float(X.max()):.0f},"
                    " non-integer); applying log1p only.",
                    UserWarning, stacklevel=3)
                if "counts" not in adata.layers:
                    adata.layers["counts"] = adata.X.copy()
                X_ln = np.log1p(X).astype(np.float32)
                adata.X = sp.csr_matrix(X_ln)
                adata.layers["lognorm"] = sp.csr_matrix(X_ln)
            else:
                # Already lognorm — record a copy
                adata.layers["lognorm"] = adata.X.copy()

        for adata in adatas_main:
            _norm_inplace(adata)

        # ── HVG selection across batches ───────────────────────────────────
        if select_hvgs:
            n_feat = _estimate_n_features(batch_views, n_hvgs)
            try:
                features = _select_integration_features(batch_views, n_feat)
            except Exception as e:
                warnings.warn(
                    f"HVG selection failed ({e}); skipping.", UserWarning)
                return
            for adata in adatas_main:
                adata.uns["integration_features"] = features
                adata.var["integration_hvg"] = adata.var_names.isin(
                    set(features))

    # ── compute_all_integration_metrics ────────────────────────────────────

    def compute_all_integration_metrics(
        adata_dict: dict,
        batch_key: str = "batch",
        cell_type_key: str | None = None,
        cluster_key: str = "topo_clusters",
        metrics: list | None = None,
        embedding_key: str | None = None,
        k: int = 30,
        n_jobs: int = -1,
        seed: int = 0,
    ):
        """Compute integration quality metrics for multiple AnnData objects.

        Convenience wrapper around individual metric functions that accepts a
        labelled dictionary of AnnData objects and returns a
        :class:`pandas.DataFrame`.

        Parameters
        ----------
        adata_dict : dict[str, AnnData]
            Mapping of ``{label: AnnData}``.  Each value is scored
            independently.  Example::

                {
                    "uncorrected": adata_raw,
                    "integrated":  adata_int,
                    "step_1":      adata_step1,
                }

        batch_key : str
            ``.obs`` column identifying batches.  Default ``'batch'``.
        cell_type_key : str or None
            ``.obs`` column for cell-type labels.  If None, cell-type-
            dependent metrics (``knn_purity``, ``clisi``, ``ari``, ``nmi``)
            are skipped automatically.
        cluster_key : str
            ``.obs`` column for cluster assignments (used by ``ari``/``nmi``).
            Default ``'topo_clusters'``.
        metrics : list of str, optional
            Subset of ``['knn_purity', 'knn_mixing', 'ilisi', 'clisi',
            'ari', 'nmi']``.  Default: all available.
        embedding_key : str or None
            ``.obsm`` key for the embedding.  If None, uses ``'X_ms_spectral_scaffold'``
            when present, then ``'X_spectral_scaffold'``, else ``.X``.
        k : int
            Number of neighbours for kNN-based metrics.  Default 30.
        n_jobs : int
            Threads for hnswlib.  Default ``-1`` (all cores).
        seed : int
            Random seed.

        Returns
        -------
        pandas.DataFrame
            Rows = metric names, columns = dataset labels.
            Missing metrics (e.g. cell-type metrics when no label key is
            provided) are represented as ``NaN``.
        """
        all_metrics = [
            "knn_purity", "knn_mixing", "ilisi", "clisi", "ari", "nmi"]
        if metrics is None:
            metrics = all_metrics
        metrics = [m for m in metrics if m in all_metrics]

        def _score_one(adata):
            ek = embedding_key
            if ek is None and "X_ms_spectral_scaffold" in adata.obsm:
                ek = "X_ms_spectral_scaffold"
            elif ek is None and "X_spectral_scaffold" in adata.obsm:
                ek = "X_spectral_scaffold"

            Z = _get_embedding(adata, ek)
            row = {}

            if "knn_mixing" in metrics and batch_key in adata.obs.columns:
                batches = np.asarray(adata.obs[batch_key].values)
                idx = _knn_indices(Z, k, n_jobs, seed)
                row["knn_mixing"] = float(
                    (batches[idx] != batches[:, None]).mean())
            if "ilisi" in metrics and batch_key in adata.obs.columns:
                row["ilisi"] = float(np.median(
                    _compute_lisi_values(
                        Z, np.asarray(adata.obs[batch_key].values),
                        k, n_jobs, seed)))
            if (cell_type_key is not None
                    and cell_type_key in adata.obs.columns):
                ct = np.asarray(adata.obs[cell_type_key].values)
                idx = _knn_indices(Z, k, n_jobs, seed)
                if "knn_purity" in metrics:
                    row["knn_purity"] = float(
                        (ct[idx] == ct[:, None]).mean())
                if "clisi" in metrics:
                    row["clisi"] = float(np.median(
                        _compute_lisi_values(Z, ct, k, n_jobs, seed)))
                if cluster_key in adata.obs.columns:
                    from sklearn.metrics import (
                        adjusted_rand_score,
                        normalized_mutual_info_score,
                    )
                    cl = adata.obs[cluster_key].values
                    if "ari" in metrics:
                        row["ari"] = float(adjusted_rand_score(ct, cl))
                    if "nmi" in metrics:
                        row["nmi"] = float(
                            normalized_mutual_info_score(ct, cl))
            return row

        results = {}
        for label, adata in adata_dict.items():
            results[label] = _score_one(adata)

        # Build DataFrame with consistent index and NaN for missing metrics
        present = sorted(
            {m for row in results.values() for m in row},
            key=lambda m: all_metrics.index(m) if m in all_metrics else 99)
        return pd.DataFrame(results, index=present)

    # ── prepare_for_mapping ────────────────────────────────────────────────

    def prepare_for_mapping(
        adata_query,
        adata_ref,
        batch_key: str = "batch",
        input_type: str = "auto",
        layer: str | None = None,
    ) -> None:
        """Normalize query data in-place for mapping to a CCA reference.

        Applies the same normalisation logic as :func:`prepare_for_integration`
        but validates feature overlap with the reference and ensures a
        ``batch`` label exists in ``.obs``.

        Parameters
        ----------
        adata_query : AnnData or list[AnnData]
            Query dataset(s) to prepare.  Modified in-place.
        adata_ref : AnnData
            The completed CCA integration reference (output of
            ``run_cca_integration``).  Used to check feature overlap.
        batch_key : str
            ``.obs`` column for the batch label.  If absent, one is added.
            Default ``'batch'``.
        input_type : {'auto', 'counts', 'lognorm'}
            Normalisation mode.
        layer : str or None
            Use ``adata.layers[layer]`` as source.

        Returns
        -------
        None  (modifies data in-place).
        """
        import warnings

        queries = ([adata_query] if isinstance(adata_query, AnnData)
                   else list(adata_query))
        ref_features = set(adata_ref.var_names)
        n_ref_feat = len(ref_features)

        for i, q in enumerate(queries):
            X = q.layers[layer] if layer is not None else q.X
            if sp.issparse(X):
                X = X.toarray()
            X = np.asarray(X, dtype=np.float32)

            if input_type == "auto":
                norm_state = _detect_normalization(X)
            elif input_type == "counts":
                norm_state = "raw_counts"
            elif input_type == "lognorm":
                norm_state = "lognorm"
            else:
                raise ValueError(
                    "input_type must be 'auto', 'counts', or 'lognorm'.")

            if norm_state == "raw_counts":
                if "counts" not in q.layers:
                    q.layers["counts"] = q.X.copy()
                row_sums = X.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums == 0, 1.0, row_sums)
                X_ln = np.log1p(X / row_sums * 1e4).astype(np.float32)
                q.X = sp.csr_matrix(X_ln)
                q.layers["lognorm"] = sp.csr_matrix(X_ln)
            elif norm_state == "pre_normalized":
                warnings.warn(
                    f"Query {i}: pre-normalised values (max="
                    f"{float(X.max()):.0f}); applying log1p only.",
                    UserWarning, stacklevel=2)
                if "counts" not in q.layers:
                    q.layers["counts"] = q.X.copy()
                X_ln = np.log1p(X).astype(np.float32)
                q.X = sp.csr_matrix(X_ln)
                q.layers["lognorm"] = sp.csr_matrix(X_ln)
            else:
                q.layers["lognorm"] = q.X.copy()

            # Check feature overlap
            shared = ref_features & set(q.var_names)
            coverage = len(shared) / n_ref_feat if n_ref_feat > 0 else 0.0
            if coverage < 0.5:
                warnings.warn(
                    f"Query {i}: only {coverage:.1%} overlap with reference "
                    f"features ({len(shared)}/{n_ref_feat}). Mapping quality "
                    "may be poor.", UserWarning, stacklevel=2)

            # Ensure batch label exists
            if batch_key not in q.obs.columns:
                default = f"query_{i}" if len(queries) > 1 else "query"
                q.obs[batch_key] = default

    # ── find_mapping_order ─────────────────────────────────────────────────

    def find_mapping_order(
        adata_ref,
        adata_queries: list,
        n_components: int = 10,
        k: int = 5,
        n_jobs: int = -1,
        seed: int = 0,
    ) -> list:
        """Determine the optimal sequential mapping order for query datasets.

        Uses anchor count between each query and the reference as a similarity
        proxy.  Queries with more anchors are mapped first (they are more
        similar to the reference and produce a better-scaffolded atlas for
        subsequent queries).

        Parameters
        ----------
        adata_ref : AnnData
            The CCA integration reference (output of ``run_cca_integration``).
        adata_queries : list[AnnData]
            Query datasets to order.
        n_components : int
            CCA dimensions for the similarity probe.  Default 10.
        k : int
            MNN k for anchor counting.  Default 5.
        n_jobs : int
            Threads.  ``-1`` uses all cores.  Default ``-1``.
        seed : int
            Random seed.  Default 0.

        Returns
        -------
        list of int
            Indices into ``adata_queries``, sorted from most to least similar
            to the reference (map in this order for best results).

        Examples
        --------
        >>> order = tp.sc.find_mapping_order(adata_ref, queries)
        >>> adata_atlas, steps = tp.sc.map_to_cca_reference(
        ...     queries, adata_ref, mapping_order=order,
        ...     return_intermediates=True)
        """
        import warnings

        n_jobs_resolved = _resolve_n_jobs(n_jobs)
        features = list(adata_ref.var_names)
        features_probe = features[: min(500, len(features))]

        def _count_anchors(i):
            try:
                q = adata_queries[i]
                q_genes = set(q.var_names)
                feat_shared = [g for g in features_probe if g in q_genes]
                if len(feat_shared) < 20:
                    return 0
                _, X_r_sc = _prepare_matrices(adata_ref, feat_shared)
                _, X_q_sc = _prepare_matrices(q, feat_shared)
                nc = min(n_components, min(len(X_r_sc), len(X_q_sc)) - 1)
                if nc < 1:
                    return 0
                cc_r, cc_q, _ = _compute_cca(X_r_sc, X_q_sc, nc, seed)
                k_eff = min(k, min(len(cc_r), len(cc_q)) - 1)
                if k_eff < 1:
                    return 0
                anc_r, _ = _find_mnn_in_cca(
                    cc_r, cc_q, k=k_eff, n_jobs=1, seed=seed)
                return len(anc_r)
            except Exception:
                return 0

        if n_jobs_resolved > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=n_jobs_resolved) as ex:
                counts = list(ex.map(
                    _count_anchors, range(len(adata_queries))))
        else:
            counts = [_count_anchors(i) for i in range(len(adata_queries))]

        return sorted(range(len(adata_queries)), key=lambda i: -counts[i])
