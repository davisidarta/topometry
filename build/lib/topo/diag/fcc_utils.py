# These are functions for assessing structure preservation during dimensionality reduction
# They were introduced by Cody Heiser and Ken Lau in their paper
# 'A Quantitative Framework for Evaluating Single-Cell
# Data Structure Preservation by Dimensionality
# Reduction Techniques' - https://doi.org/10.1016/j.celrep.2020.107576
# And initially made available at https://github.com/KenLauLab/DR-structure-preservation
# They are implemented here with some minor modifications
# Because a license has not been estabilished, this code is bound by TopOMetry MIT license
"""
Utility functions for dimensionality reduction structural preservation analysis
"""
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from ot import wasserstein_1d
from scipy.spatial.distance import cdist, pdist
from scipy.stats import pearsonr
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize

import seaborn as sns

sc.set_figure_params(dpi=90, color_map="viridis")
sns.set(style="white")


def arcsinh(adata, layer=None, norm="l1", scale=1000):
    """
    Returns arcsinh-normalized values for each element in anndata counts matrix
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object
    layer : str, optional (default=None)
        name of layer to perform arcsinh-normalization on. if None, use `adata.X`
    norm : str {"l1","l2"}, optional (default="l1")
        normalization strategy prior to arcsinh transform. None=do not normalize data.
        "l1"=divide each count by sum of counts for each cell. "l2"=divide each count
        by sqrt of sum of squares of counts for cell.
    scale : float, optional (default=1000)
        factor to multiply normalized counts by
    Returns
    -------
    `adata` is edited in place to add `adata.layers["arcsinh_norm"]`
    """
    if layer is None:
        mat = adata.X
    else:
        mat = adata.layers[layer]

    adata.layers["arcsinh_norm"] = np.arcsinh(normalize(mat, axis=1, norm=norm) * scale)


def knn_graph(dist_matrix, k, adata, save_rep="knn"):
    """
    Builds simple binary k-nearest neighbor graph and add to anndata object
    Parameters
    ----------
    dist_matrix : np.array
        distance matrix to calculate knn graph for (i.e. `pdist(adata.obsm["X_pca"])`)
    k : int
        number of nearest neighbors to determine
    adata : anndata.AnnData
        AnnData object to add resulting graph to (in `.uns` slot)
    save_rep : str, optional (default="knn")
        name of `.uns` key to save knn graph to within adata
    Returns
    -------
    `adata` is edited in place, adding knn graph to `adata.uns[save_rep]`
    """
    adata.uns[save_rep] = {
        "graph": kneighbors_graph(
            dist_matrix, k, mode="connectivity", include_self=False, n_jobs=-1
        ).toarray(),
        "k": k,
    }


def subset_uns_by_ID(adata, uns_keys, obs_col, IDs):
    """
    Subsets symmetrical distance matrices and knn graphs in `adata.uns` by one or more
    IDs defined in `adata.obs`
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object
    uns_keys : list of str
        list of keys in `adata.uns` to subset. new `adata.uns` keys will be saved with
        ID appended to name (i.e. `adata.uns["knn"]` -> `adata.uns["knn_ID1"]`)
    obs_col : str
        name of column in `adata.obs` to use as cell IDs (i.e. "louvain")
    IDs : list of str
        list of IDs to include in subset
    Returns
    -------
    `adata` is edited in place, adding new `.uns` keys for each ID
    """
    for key in uns_keys:
        tmp = adata.uns[key][
            adata.obs[obs_col].isin(IDs), :
        ]  # subset symmetrical uns matrix along axis 0
        tmp = tmp[
            :, adata.obs[obs_col].isin(IDs)
        ]  # subset symmetrical uns matrix along axis 1

        adata.uns[
            "{}_{}".format(key, "_".join([str(x) for x in IDs]))
        ] = tmp  # save new .uns key by appending IDs to original key name


def find_centroids(adata, use_rep, obs_col="louvain"):
    """
    Finds cluster centroids
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object
    use_rep : str
        "X" or `adata.obsm` key containing space to calculate centroids in
        (i.e. "X_pca")
    obs_col "str, optional (default="louvain")
        `adata.obs` column name containing cluster IDs
    Returns
    -------
    `adata` is edited in place, adding `adata.uns["{}_centroids"]`,
    `adata.uns["{}_centroid_distances"]`, and `adata.uns["{}_centroid_MST"]`
    containing centroid coordinates, distance matrix between all centoids, and a
    minimum spanning tree graph between the centroids, respectively
    """
    # calculate centroids
    clu_names = adata.obs[obs_col].unique().astype(str)
    if use_rep == "X":
        adata.uns["{}_centroids".format(use_rep)] = np.array(
            [
                np.mean(adata.X[adata.obs[obs_col].astype(str) == clu, :], axis=0)
                for clu in clu_names
            ]
        )
    else:
        adata.uns["{}_centroids".format(use_rep)] = np.array(
            [
                np.mean(
                    adata.obsm[use_rep][adata.obs[obs_col].astype(str) == clu, :],
                    axis=0,
                )
                for clu in clu_names
            ]
        )
    # calculate distances between all centroids
    adata.uns["{}_centroid_distances".format(use_rep)] = cdist(
        adata.uns["{}_centroids".format(use_rep)],
        adata.uns["{}_centroids".format(use_rep)],
    )
    # build networkx minimum spanning tree between centroids
    G = nx.from_numpy_matrix(adata.uns["{}_centroid_distances".format(use_rep)])
    G = nx.relabel_nodes(G, mapping=dict(zip(list(G.nodes), clu_names)), copy=True)
    adata.uns["{}_centroid_MST".format(use_rep)] = nx.minimum_spanning_tree(G)


# dimensionality reduction plotting class #
class DR_plot:
    """
    Class defining pretty plots of dimension-reduced embeddings such as PCA, t-SNE,
    and UMAP
    Attributes
    ----------
    .fig : matplotlib.figure
        the figure object on which data will be plotted
    .ax : matplotlib.axes.ax
        the axes within `self.fig`
    .cmap : matplotlib.pyplot.cmap
        color map to use for plotting; default="plasma"
    Methods
    -------
    .plot()
        utility plotting function that can be passed any numpy array in the `data`
        parameter
    .plot_IDs()
        plots one or more cluster IDs on top of an `.obsm` from an AnnData object
    .plot_centroids()
        plots cluster centroids defined using `find_centroids()` function on AnnData
        object
    """

    def __init__(self, dim_name="dim", figsize=(5, 5), ax_labels=True):
        """
        Initializes `DR_plot` class
        Parameters
        ----------
        dim_name : str, optional (default="dim")
            how to label axes ("dim 1" on x and "dim 2" on y by default)
        figsize : tuple of float, optional (default=(5,5))
            size of resulting figure in inches
        ax_labels : bool, optional (default=True)
            draw arrows and dimension names in lower left corner of plot
        Returns
        -------
        Initializes `self.fig` and `self.ax` according to input specs
        """
        self.fig, self.ax = plt.subplots(1, figsize=figsize)
        self.cmap = plt.get_cmap("plasma")

        if ax_labels:
            plt.xlabel("{} 1".format(dim_name), fontsize=14)
            self.ax.xaxis.set_label_coords(0.2, -0.025)
            plt.ylabel("{} 2".format(dim_name), fontsize=14)
            self.ax.yaxis.set_label_coords(-0.025, 0.2)

            plt.annotate(
                "",
                textcoords="axes fraction",
                xycoords="axes fraction",
                xy=(-0.006, 0),
                xytext=(0.2, 0),
                arrowprops=dict(arrowstyle="<-", lw=2, color="black"),
            )
            plt.annotate(
                "",
                textcoords="axes fraction",
                xycoords="axes fraction",
                xy=(0, -0.006),
                xytext=(0, 0.2),
                arrowprops=dict(arrowstyle="<-", lw=2, color="black"),
            )

        plt.tick_params(labelbottom=False, labelleft=False)
        sns.despine(left=True, bottom=True)
        plt.tight_layout()

    def plot(self, data, color, pt_size=75, legend=None, save_to=None):
        """
        General plotting function for dimensionality reduction outputs with cute
        arrows and labels
        Parameters
        ----------
        data : np.array
            array containing variables in columns and observations in rows
        color : list
            list of length `nrow(data)` to determine how points should be colored (ie.
            `adata.obs["louvain"].values` to color by "louvain" cluster categories)
        pt_size : float, optional (default=75)
            size of points in plot
        legend : str {"full","brief"}, optional (default=None)
            string describing the legend size. None for no legend
        save_to : str, optional (default=None)
            path to `.png` file to save output. do not save if None
        Returns
        -------
        `self.fig`, `self.ax` edited; plot saved to `.png` file if `save_to` is not
        None
        """
        sns.scatterplot(
            x=data[:, 0],
            y=data[:, 1],
            s=pt_size,
            alpha=0.7,
            hue=color,
            legend=legend,
            edgecolor="none",
            ax=self.ax,
        )

        if legend is not None:
            plt.legend(
                bbox_to_anchor=(1, 1, 0.2, 0.2),
                loc="lower left",
                frameon=False,
                fontsize="small",
            )

        if save_to is None:
            return
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=1000)

    def plot_IDs(
        self, adata, use_rep, obs_col="leiden", IDs="all", pt_size=75, save_to=None
    ):
        """
        General plotting function for dimensionality reduction outputs with
        categorical colors (i.e. "leiden" or "louvain") and cute arrows and labels
        Parameters
        ----------
        adata : anndata.AnnData
            object to pull dimensionality reduction from
        use_rep : str
            `adata.obsm` key to plot from (i.e. "X_pca")
        obs_col : str, optional (default="leiden")
            name of column in `adata.obs` to use as cell IDs (i.e. "leiden")
        IDs : list of str, optional (default="all")
            list of IDs to plot, graying out cells not assigned to those IDs. if
            "all", show all ID categories.
        pt_size : float, optional (default=75)
            size of points in plot
        save_to : str, optional (default=None)
            path to `.png` file to save output. do not save if None
        Returns
        -------
        `self.fig`, `self.ax` edited; plot saved to `.png` file if `save_to` is not
        None
        """
        plotter = adata.obsm[use_rep]
        clu_names = adata.obs[obs_col].unique().astype(str)

        # use existing scanpy colors, if applicable
        if obs_col == "leiden":
            colors = [
                adata.uns["leiden_colors"][x]
                for x in adata.obs.leiden.unique().astype(int)
            ]
        elif obs_col == "louvain":
            colors = [
                adata.uns["louvain_colors"][x]
                for x in adata.obs.leiden.unique().astype(int)
            ]
        # otherwise, get new color mapping from obs_col using self.cmap
        else:
            colors = self.cmap(np.linspace(0, 1, len(clu_names)))

        cdict = dict(zip(clu_names, colors))

        if IDs == "all":
            self.ax.scatter(
                x=plotter[:, 0],
                y=plotter[:, 1],
                s=pt_size,
                alpha=0.7,
                c=[cdict[x] for x in adata.obs[obs_col].astype(str)],
                edgecolor="none",
            )

        else:
            sns.scatterplot(
                x=plotter[-adata.obs[obs_col].isin(IDs), 0],
                y=plotter[-adata.obs[obs_col].isin(IDs), 1],
                ax=self.ax,
                s=pt_size,
                alpha=0.1,
                color="gray",
                legend=False,
                edgecolor="none",
            )
            plt.scatter(
                x=plotter[adata.obs[obs_col].isin(IDs), 0],
                y=plotter[adata.obs[obs_col].isin(IDs), 1],
                s=pt_size,
                alpha=0.7,
                c=[
                    cdict[x]
                    for x in adata.obs.loc[
                        adata.obs[obs_col].isin(IDs), obs_col
                    ].astype(str)
                ],
                edgecolor="none",
            )

        if save_to is None:
            return
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=1000)

    def plot_centroids(
        self,
        adata,
        use_rep,
        obs_col="leiden",
        ctr_size=300,
        pt_size=75,
        draw_edges=True,
        highlight_edges=False,
        save_to=None,
    ):
        """
        General plotting function for cluster centroid graph and MST
        (i.e. from "leiden" or "louvain") and cute arrows and labels
        Parameters
        ----------
        adata : anndata.AnnData
            object to pull dimensionality reduction from
        use_rep : str
            `adata.obsm` key to plot from (i.e. "X_pca")
        obs_col : str, optional (default="leiden")
            name of column in `adata.obs` to use as cell IDs (i.e. "leiden")
        ctr_size : float, optional (default=300)
            size of centroid points in plot
        pt_size : float, optional (default=75)
            size of points in plot
        draw_edges : bool, optional (default=True)
            draw edges of minimum spanning tree between all centroids
        highlight_edges : list of int, optional (default=False)
            list of edge IDs as tuples to highlight in red on plot. e.g.
            `set(adata.uns['X_tsne_centroid_MST'].edges).difference(set(adata.uns['X_umap_centroid_MST'].edges))`
            with output {(0,3), (0,7)} says that edges from centroid 0 to 3 and 0 to 7
            are found in 'X_tsne_centroids' but not in 'X_umap_centroids'. highlight
            the edges to show this.
        save_to : str, optional (default=None)
            path to `.png` file to save output. do not save if None
        Returns
        -------
        `self.fig`, `self.ax` edited; plot saved to `.png` file if `save_to` is not
        None
        """
        clu_names = adata.obs[obs_col].unique().astype(str)
        # use existing scanpy colors, if applicable
        if obs_col == "leiden":
            colors = [
                adata.uns["leiden_colors"][x]
                for x in adata.obs.leiden.unique().astype(int)
            ]
        elif obs_col == "louvain":
            colors = [
                adata.uns["louvain_colors"][x]
                for x in adata.obs.leiden.unique().astype(int)
            ]
        # otherwise, get new color mapping from obs_col using self.cmap
        else:
            colors = self.cmap(np.linspace(0, 1, len(clu_names)))

        # draw points in embedding first
        sns.scatterplot(
            x=adata.obsm[use_rep][:, 0],
            y=adata.obsm[use_rep][:, 1],
            ax=self.ax,
            s=pt_size,
            alpha=0.1,
            color="gray",
            legend=False,
            edgecolor="none",
        )

        # draw MST edges if desired, otherwise just draw centroids
        if not draw_edges:
            self.ax.scatter(
                x=adata.uns["{}_centroids".format(use_rep)][:, 0],
                y=adata.uns["{}_centroids".format(use_rep)][:, 1],
                s=ctr_size,
                c=colors,
                edgecolor="none",
            )
        else:
            pos = dict(zip(clu_names, adata.uns["{}_centroids".format(use_rep)][:, :2]))
            nx.draw_networkx(
                adata.uns["{}_centroid_MST".format(use_rep)],
                pos=pos,
                ax=self.ax,
                with_labels=False,
                width=2,
                node_size=ctr_size,
                node_color=colors,
            )
            # highlight edges if desired
            if highlight_edges:
                nx.draw_networkx_edges(
                    adata.uns["{}_centroid_MST".format(use_rep)],
                    pos=pos,
                    ax=self.ax,
                    edgelist=highlight_edges,
                    width=5,
                    edge_color="red",
                )

        if save_to is None:
            return
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=1000)



def distance_stats(pre, post, downsample=False, verbose=True):
    """
    Tests for correlation between Euclidean cell-cell distances before and after
    transformation by a function or DR algorithm.
    Parameters
    ----------
    pre : np.array
        vector of unique distances (pdist()) or distance matrix of shape (n_cells,
        m_cells), i.e. (cdist()) before transformation/projection
    post : np.array
        vector of unique distances (pdist()) or distance matrix of shape (n_cells,
        m_cells), i.e. (cdist()) after transformation/projection
    downsample : int, optional (default=False)
        number of distances to downsample to. maximum of 50M (~10k cells, if
        symmetrical) is recommended for performance.
    verbose : bool, optional (default=True)
        print progress statements to console
    Returns
    -------
    pre : np.array
        vector of normalized unique distances (pdist()) or distance matrix of shape
        (n_cells, m_cells), before transformation/projection
    post : np.array
        vector of normalized unique distances (pdist()) or distance matrix of shape
        (n_cells, m_cells), after transformation/projection
    corr_stats : list
        output of `pearsonr()` function correlating the two normalized unique distance
        vectors
    EMD : float
        output of `wasserstein_1d()` function calculating the Earth Mover's Distance
        between the two normalized unique distance vectors
    1) performs Pearson correlation of distance distributions
    2) normalizes unique distances using min-max standardization for each dataset
    3) calculates Wasserstein or Earth-Mover's Distance for normalized distance
    distributions between datasets
    """
    # make sure the number of cells in each matrix is the same
    assert (
        pre.shape == post.shape
    ), 'Matrices contain different number of distances.\n{} in "pre"\n{} in "post"\n'.format(
        pre.shape[0], post.shape[0]
    )

    # if distance matrix (mA x mB, result of cdist), flatten to unique cell-cell distances
    if pre.ndim == 2:
        if verbose:
            print("Flattening pre-transformation distance matrix into 1D array...")
        # if symmetric, only keep unique values (above diagonal)
        if np.allclose(pre, pre.T, rtol=1e-05, atol=1e-08):
            pre = pre[np.triu_indices(n=pre.shape[0], k=1)]
        # otherwise, flatten all distances
        else:
            pre = pre.flatten()

    # if distance matrix (mA x mB, result of cdist), flatten to unique cell-cell distances
    if post.ndim == 2:
        if verbose:
            print("Flattening post-transformation distance matrix into 1D array...")
        # if symmetric, only keep unique values (above diagonal)
        if np.allclose(post, post.T, rtol=1e-05, atol=1e-08):
            post = post[np.triu_indices(n=post.shape[0], k=1)]
        # otherwise, flatten all distances
        else:
            post = post.flatten()

    # if dataset is large, randomly downsample to reasonable number of distances for calculation
    if downsample:
        assert downsample < len(
            pre
        ), "Must provide downsample value smaller than total number of cell-cell distances provided in pre and post"
        if verbose:
            print("Downsampling to {} total cell-cell distances...".format(downsample))
        idx = np.random.choice(np.arange(len(pre)), downsample, replace=False)
        pre = pre[idx]
        post = post[idx]

    # calculate correlation coefficient using Pearson correlation
    if verbose:
        print("Correlating distances")
    corr_stats = pearsonr(x=pre, y=post)

    # min-max normalization for fair comparison of probability distributions
    if verbose:
        print("Normalizing unique distances")
    pre -= pre.min()
    pre /= pre.ptp()

    post -= post.min()
    post /= post.ptp()

    # calculate EMD for the distance matrices
    # by default, downsample to 50M distances to speed processing time,
    # since this function often breaks with larger distributions
    if verbose:
        print("Calculating Earth-Mover's Distance between distributions")
    if len(pre) > 50000000:
        idx = np.random.choice(np.arange(len(pre)), 50000000, replace=False)
        pre_EMD = pre[idx]
        post_EMD = post[idx]
        EMD = wasserstein_1d(pre_EMD, post_EMD)
    else:
        EMD = wasserstein_1d(pre, post)

    return pre, post, corr_stats, EMD


def knn_preservation(pre, post):
    """
    Tests for k-nearest neighbor preservation (%) before and after transformation by a
    function or DR algorithm.
    Parameters
    ----------
    pre : np.array
        knn graph of shape (n_cells, n_cells) before transformation/projection
    post : np.array
        knn graph of shape (n_cells, n_cells) after transformation/projection
    Returns
    -------
    knn_pres : float
        knn preservation expressed as a percentage out of 100 %
    """
    # make sure the number of cells in each matrix is the same
    assert (
        pre.shape == post.shape
    ), 'Matrices contain different number of cells.\n{} in "pre"\n{} in "post"\n'.format(
        pre.shape[0], post.shape[0]
    )
    return np.round(
        (np.isclose(pre, post, rtol=1e-05, atol=1e-08).sum() / (pre.shape[0] ** 2))
        * 100,
        4,
    )


def structure_preservation_sc(
    adata,
    latent,
    native="X",
    metric="euclidean",
    k=30,
    downsample=False,
    verbose=True,
    force_recalc=False,
):
    """
    Wrapper function for full structural preservation workflow applied to `scanpy`
    AnnData object
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with latent space to test in `.obsm` slot, and native
        (reference) space in `.X` or `.obsm`
    latent : str
        `adata.obsm` key that contains low-dimensional latent space for testing
    native : str, optional (default="X")
        `adata.obsm` key or `.X` containing high-dimensional native space, which
        should be direct input to dimension reduction that generated latent `.obsm`
        for fair comparison. default "X", which uses `adata.X`.
    metric : str {"chebyshev","cityblock","euclidean","minkowski","mahalanobis",
    "seuclidean"}, optional (default="euclidean")
        distance metric to use
    k : int, optional (default=30)
        number of nearest neighbors to test preservation
    downsample : int, optional (default=False)
        number of distances to downsample to. maximum of 50M (~10k cells, if
        symmetrical) is recommended for performance.
    verbose : bool, optional (default=True)
        print progress statements to console
    force_recalc : bool, optional (default=False)
        if True, recalculate all distances and neighbor graphs, regardless of their
        presence in `adata`
    Returns
    -------
    corr_stats : list
        output of `pearsonr()` function correlating the two normalized unique distance
        vectors
    EMD : float
        output of `wasserstein_1d()` function calculating the Earth Mover's Distance
        between the two normalized unique distance vectors
    knn_pres : float
        knn preservation expressed as a percentage out of 100 %
    """
    # 0) determine native space according to argument
    if native == "X":
        native_space = adata.X.copy()
    else:
        native_space = adata.obsm[native].copy()

    # 1) calculate unique cell-cell distances
    if (
        "{}_distances".format(native) not in adata.uns.keys() or force_recalc
    ):  # check for existence in AnnData to prevent re-work
        if verbose:
            print("Calculating unique distances for native space, {}".format(native))
        adata.uns["{}_distances".format(native)] = cdist(
            native_space, native_space, metric=metric
        )

    if (
        "{}_distances".format(latent) not in adata.uns.keys() or force_recalc
    ):  # check for existence in AnnData to prevent re-work
        if verbose:
            print("Calculating unique distances for latent space, {}".format(latent))
        adata.uns["{}_distances".format(latent)] = cdist(
            adata.obsm[latent], adata.obsm[latent], metric=metric
        )

    # 2) get correlation and EMD values, and return normalized distance vectors for plotting distributions
    (
        adata.uns["{}_norm_distances".format(native)],
        adata.uns["{}_norm_distances".format(latent)],
        corr_stats,
        EMD,
    ) = distance_stats(
        pre=adata.uns["{}_distances".format(native)].copy(),
        post=adata.uns["{}_distances".format(latent)].copy(),
        verbose=verbose,
        downsample=downsample,
    )

    # 3) determine neighbors
    if (
        "{}_neighbors".format(native) not in adata.uns.keys() or force_recalc
    ):  # check for existence in AnnData to prevent re-work
        if verbose:
            print(
                "{}-nearest neighbor calculation for native space, {}".format(k, native)
            )
        knn_graph(
            adata.uns["{}_distances".format(native)],
            k=k,
            adata=adata,
            save_rep="{}_knn".format(native),
        )

    if (
        "{}_neighbors".format(latent) not in adata.uns.keys() or force_recalc
    ):  # check for existence in AnnData to prevent re-work
        if verbose:
            print(
                "{}-nearest neighbor calculation for latent space, {}".format(k, latent)
            )
        knn_graph(
            adata.uns["{}_distances".format(latent)],
            k=k,
            adata=adata,
            save_rep="{}_knn".format(latent),
        )

    # 4) calculate neighbor preservation
    if verbose:
        print("Determining nearest neighbor preservation")
    if (
        adata.uns["{}_knn".format(native)]["k"]
        != adata.uns["{}_knn".format(latent)]["k"]
    ):
        warnings.warn(
            'Warning: Nearest-neighbor graphs constructed with different k values. k={} in "{}_neighbors", while k={} in "{}_neighbors". Consider re-generating neighbors graphs by setting force_recalc=True.'.format(
                adata.uns["{}_knn".format(native)]["k"],
                native,
                adata.uns["{}_knn".format(latent)]["k"],
                latent,
            )
        )
    knn_pres = knn_preservation(
        pre=adata.uns["{}_knn".format(native)]["graph"],
        post=adata.uns["{}_knn".format(latent)]["graph"],
    )

    if verbose:
        print("Done!")
    return corr_stats, EMD, knn_pres



class SP_plot:
    """
    Class defining pretty plots for structural evaluation of dimension-reduced
    embeddings such as PCA, t-SNE, and UMAP
    Attributes
    ----------
    .figsize : tuple of float
        the size of the figure object on which data will be plotted
    .fig : matplotlib.figure
        the figure object on which data will be plotted
    .ax : matplotlib.axes.ax
        the axes within `self.fig`
    .palette : sns.cubehelix_palette()
        color palette to use for coloring `seaborn` plots
    .cmap : matplotlib.pyplot.cmap
        color map to use for plotting; default="cubehelix" from `seaborn`
    .pre : np.array
        flattened vector of normalized, unique cell-cell distances
        "pre-transformation". upper triangle of cell-cell distance matrix, flattened
        to vector of shape ((n_cells^2)/2)-n_cells.
    .post : np.array
        flattened vector of normalized, unique cell-cell distances
        "post-transformation". upper triangle of cell-cell distance matrix, flattened
        to vector of shape ((n_cells^2)/2)-n_cells.
    .labels : list of str
        name of pre- and post-transformation spaces for legend (plot_cell_distances,
        plot_distributions, plot_cumulative_distributions) or axis labels
        (plot_distance_correlation, joint_plot_distance_correlation) as list of two
        strings. False to exclude labels.
    Methods
    -------
    .plot_cell_distances()
        plots all unique cell-cell distances before and after some transformation
    .plot_distributions()
        plots probability distributions for all unique cell-cell distances before and
        after some transformation
    .plot_cumulative_distributions()
        plots cumulative probability distributions for all unique cell-cell distances
        before and after some transformation
    .plot_distance_correlation()
        plots correlation of all unique cell-cell distances before and after some
        transformation
    .joint_plot_distance_correlation()
        plots correlation of all unique cell-cell distances before and after some
        transformation. includes marginal plots of each distribution.
    """

    def __init__(
        self, pre_norm, post_norm, figsize=(4, 4), labels=["Native", "Latent"]
    ):
        """
        Initializes SP plot class
        Parameters
        ----------
        pre_norm : np.array
            flattened vector of normalized, unique cell-cell distances
            "pre-transformation". upper triangle of cell-cell distance matrix, flattened
            to vector of shape ((n_cells^2)/2)-n_cells.
        post_norm : np.array
            flattened vector of normalized, unique cell-cell distances
            "post-transformation". upper triangle of cell-cell distance matrix, flattened
            to vector of shape ((n_cells^2)/2)-n_cells.
        figsize : tuple of float, optional (default=(4,4))
            the size of the figure object on which data will be plotted
        labels : list of str, optional (default=["Native","Latent"])
            name of pre- and post-transformation spaces for legend (plot_cell_distances,
            plot_distributions, plot_cumulative_distributions) or axis labels
            (plot_distance_correlation, joint_plot_distance_correlation) as list of two
            strings. False to exclude labels.
        Returns
        -------
        Initializes `self.fig` and `self.ax` according to input specs
        """
        self.figsize = figsize
        self.fig, self.ax = plt.subplots(1, figsize=self.figsize)
        self.palette = sns.cubehelix_palette()
        self.cmap = sns.cubehelix_palette(as_cmap=True)
        self.pre = pre_norm
        self.post = post_norm
        self.labels = labels

        plt.tick_params(labelbottom=False, labelleft=False)
        sns.despine()
        plt.tight_layout()

    def plot_cell_distances(self, legend=True, save_to=None):
        """
        Plots all unique cell-cell distances before and after some transformation
        Parameters
        ----------
        legend : bool, optional (default=True)
            display legend on plot
        save_to : str, optional (default=None)
            path to `.png` file to save output. do not save if None
        Returns
        -------
        `self.fig`, `self.ax` edited; plot saved to `.png` file if `save_to` is not
        None
        """
        plt.plot(self.pre, alpha=0.7, label=self.labels[0], color=self.palette[-1])
        plt.plot(self.post, alpha=0.7, label=self.labels[1], color=self.palette[2])
        if legend:
            plt.legend(loc="best", fontsize="xx-large")
        else:
            plt.legend()
            self.ax.legend().remove()

        if save_to is None:
            return
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=1000)

    def plot_distributions(self, legend=True, save_to=None):
        """
        Plots probability distributions for all unique cell-cell distances before and
        after some transformation
        Parameters
        ----------
        legend : bool, optional (default=True)
            display legend on plot
        save_to : str, optional (default=None)
            path to `.png` file to save output. do not save if None
        Returns
        -------
        `self.fig`, `self.ax` edited; plot saved to `.png` file if `save_to` is not
        None
        """
        sns.distplot(
            self.pre, hist=False, kde=True, label=self.labels[0], color=self.palette[-1]
        )
        sns.distplot(
            self.post, hist=False, kde=True, label=self.labels[1], color=self.palette[2]
        )
        if legend:
            plt.legend(loc="best", fontsize="xx-large")
        else:
            plt.legend()
            self.ax.legend().remove()

        if save_to is None:
            return
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=1000)

    def plot_cumulative_distributions(self, legend=True, save_to=None):
        """
        Plots cumulative probability distributions for all unique cell-cell distances
        before and after some transformation
        Parameters
        ----------
        legend : bool, optional (default=True)
            display legend on plot
        save_to : str, optional (default=None)
            path to `.png` file to save output. do not save if None
        Returns
        -------
        `self.fig`, `self.ax` edited; plot saved to `.png` file if `save_to` is not
        None
        """
        num_bins = int(len(self.pre) / 100)
        pre_counts, pre_bin_edges = np.histogram(self.pre, bins=num_bins)
        pre_cdf = np.cumsum(pre_counts)
        post_counts, post_bin_edges = np.histogram(self.post, bins=num_bins)
        post_cdf = np.cumsum(post_counts)
        plt.plot(
            pre_bin_edges[1:],
            pre_cdf / pre_cdf[-1],
            label=self.labels[0],
            color=self.palette[-1],
        )
        plt.plot(
            post_bin_edges[1:],
            post_cdf / post_cdf[-1],
            label=self.labels[1],
            color=self.palette[2],
        )
        if legend:
            plt.legend(loc="lower right", fontsize="xx-large")
        else:
            plt.legend()
            self.ax.legend().remove()

        if save_to is None:
            return
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=1000)

    def plot_distance_correlation(self, save_to=None):
        """
        Plots correlation of all unique cell-cell distances before and after some
        transformation
        Parameters
        ----------
        save_to : str, optional (default=None)
            path to `.png` file to save output. do not save if None
        Returns
        -------
        `self.fig`, `self.ax` edited; plot saved to `.png` file if `save_to` is not
        None
        """
        plt.hist2d(x=self.pre, y=self.post, bins=50, cmap=self.cmap)
        plt.plot(
            np.linspace(max(min(self.pre), min(self.post)), 1, 100),
            np.linspace(max(min(self.pre), min(self.post)), 1, 100),
            linestyle="dashed",
            color=self.palette[-1],
        )  # plot identity line as reference for regression
        if self.labels:
            plt.xlabel(self.labels[0], fontsize="xx-large", color=self.palette[-1])
            plt.ylabel(self.labels[1], fontsize="xx-large", color=self.palette[2])

        if save_to is None:
            return
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=1000)

    def joint_plot_distance_correlation(self, save_to=None):
        """
        Plots correlation of all unique cell-cell distances before and after some
        transformation. includes marginal plots of each distribution.
        Parameters
        ----------
        save_to : str, optional (default=None)
            path to `.png` file to save output. do not save if None
        Returns
        -------
        `self.fig`, `self.ax` edited; plot saved to `.png` file if `save_to` is not
        None
        """
        plt.close()  # close matplotlib figure from __init__() and start over with seaborn.JointGrid()
        self.fig = sns.JointGrid(
            x=self.pre, y=self.post, space=0, height=self.figsize[0]
        )
        self.fig.plot_joint(plt.hist2d, bins=50, cmap=self.cmap)
        sns.kdeplot(
            x=self.pre,
            color=self.palette[-1],
            shade=False,
            bw_method=0.01,
            ax=self.fig.ax_marg_x,
        )
        sns.kdeplot(
            y=self.post,
            color=self.palette[2],
            shade=False,
            bw_method=0.01,
            ax=self.fig.ax_marg_y,
        )
        self.fig.ax_joint.plot(
            np.linspace(max(min(self.pre), min(self.post)), 1, 100),
            np.linspace(max(min(self.pre), min(self.post)), 1, 100),
            linestyle="dashed",
            color=self.palette[-1],
        )  # plot identity line as reference for regression
        if self.labels:
            plt.xlabel(self.labels[0], fontsize="xx-large", color=self.palette[-1])
            plt.ylabel(self.labels[1], fontsize="xx-large", color=self.palette[2])

        plt.tick_params(labelbottom=False, labelleft=False)

        if save_to is None:
            return
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=1000)



def cluster_arrangement_sc(
    adata,
    pre,
    post,
    obs_col,
    IDs,
    ID_names=None,
    figsize=(4, 4),
    legend=True,
    ax_labels=["Native", "Latent"],
):
    """
    Determines pairwise distance preservation between 3 IDs from `adata.obs[obs_col]`
    Parameters
    ----------
    adata : anndata.AnnData
        anndata object to pull dimensionality reduction from
    pre : np.array
        matrix to subset as pre-transformation (i.e. `adata.X`)
    post : np.array
        matrix to subset as pre-transformation (i.e. `adata.obsm["X_pca"]`)
    obs_col : str
        name of column in `adata.obs` to use as cell IDs (i.e. "louvain")
    IDs : list of int (len==3)
        list of THREE ID indices to compare (i.e. [0,1,2])
    figsize : tuple of float, optional (default=(4,4))
        size of resulting figure
    legend : bool, optional (default=True)
        display legend on plot
    ax_labels : list of str (len==2), optional (default=["Native","Latent"])
        list of two strings for x and y axis labels, respectively. if False, exclude
        axis labels.
    Returns
    -------
    corr_stats : list
        list of outputs of `pearsonr()` function correlating the three normalized
        unique distance vectors in a pairwise fashion
    EMD : float
        list of outputs of `wasserstein_1d()` function calculating the Earth Mover's
        Distance between the three normalized unique distance vectors in a pairwise
        fashion
    Outputs jointplot with scatter of pairwise distance correlations, with marginal
    KDE plots showing density of each native and latent distance vector
    """
    # distance calculations for pre_obj
    dist_0_1 = cdist(
        pre[adata.obs[obs_col] == IDs[0]], pre[adata.obs[obs_col] == IDs[1]]
    ).flatten()
    dist_0_2 = cdist(
        pre[adata.obs[obs_col] == IDs[0]], pre[adata.obs[obs_col] == IDs[2]]
    ).flatten()
    dist_1_2 = cdist(
        pre[adata.obs[obs_col] == IDs[1]], pre[adata.obs[obs_col] == IDs[2]]
    ).flatten()
    # combine and min-max normalize
    dist = np.append(np.append(dist_0_1, dist_0_2), dist_1_2)
    dist -= dist.min()
    dist /= dist.ptp()
    # split normalized distances by cluster pair
    dist_norm_0_1 = dist[: dist_0_1.shape[0]]
    dist_norm_0_2 = dist[dist_0_1.shape[0] : dist_0_1.shape[0] + dist_0_2.shape[0]]
    dist_norm_1_2 = dist[dist_0_1.shape[0] + dist_0_2.shape[0] :]

    # distance calculations for post_obj
    post_0_1 = cdist(
        post[adata.obs[obs_col] == IDs[0]], post[adata.obs[obs_col] == IDs[1]]
    ).flatten()
    post_0_2 = cdist(
        post[adata.obs[obs_col] == IDs[0]], post[adata.obs[obs_col] == IDs[2]]
    ).flatten()
    post_1_2 = cdist(
        post[adata.obs[obs_col] == IDs[1]], post[adata.obs[obs_col] == IDs[2]]
    ).flatten()
    # combine and min-max normalize
    post = np.append(np.append(post_0_1, post_0_2), post_1_2)
    post -= post.min()
    post /= post.ptp()
    # split normalized distances by cluster pair
    post_norm_0_1 = post[: post_0_1.shape[0]]
    post_norm_0_2 = post[post_0_1.shape[0] : post_0_1.shape[0] + post_0_2.shape[0]]
    post_norm_1_2 = post[post_0_1.shape[0] + post_0_2.shape[0] :]

    # calculate EMD and Pearson correlation stats
    EMD = [
        wasserstein_1d(dist_norm_0_1, post_norm_0_1),
        wasserstein_1d(dist_norm_0_2, post_norm_0_2),
        wasserstein_1d(dist_norm_1_2, post_norm_1_2),
    ]
    corr_stats = [
        pearsonr(x=dist_0_1, y=post_0_1)[0],
        pearsonr(x=dist_0_2, y=post_0_2)[0],
        pearsonr(x=dist_1_2, y=post_1_2)[0],
    ]

    if ID_names is None:
        ID_names = IDs.copy()

    # generate jointplot
    g = sns.JointGrid(x=dist, y=post, space=0, height=figsize[0])
    g.plot_joint(plt.hist2d, bins=50, cmap=sns.cubehelix_palette(as_cmap=True))
    sns.kdeplot(
        dist_norm_0_1,
        shade=False,
        bw_method=0.01,
        ax=g.ax_marg_x,
        color="darkorange",
        label=ID_names[0] + " - " + ID_names[1],
        legend=legend,
    )
    sns.kdeplot(
        dist_norm_0_2,
        shade=False,
        bw_method=0.01,
        ax=g.ax_marg_x,
        color="darkgreen",
        label=ID_names[0] + " - " + ID_names[2],
        legend=legend,
    )
    sns.kdeplot(
        dist_norm_1_2,
        shade=False,
        bw_method=0.01,
        ax=g.ax_marg_x,
        color="darkred",
        label=ID_names[1] + " - " + ID_names[2],
        legend=legend,
    )
    if legend:
        g.ax_marg_x.legend(loc=(1.01, 0.1))

    sns.kdeplot(
        y=post_norm_0_1,
        shade=False,
        bw_method=0.01,
        color="darkorange",
        ax=g.ax_marg_y,
    )
    sns.kdeplot(
        y=post_norm_0_2,
        shade=False,
        bw_method=0.01,
        color="darkgreen",
        ax=g.ax_marg_y,
    )
    sns.kdeplot(
        y=post_norm_1_2,
        shade=False,
        bw_method=0.01,
        color="darkred",
        ax=g.ax_marg_y,
    )
    g.ax_joint.plot(
        np.linspace(max(dist.min(), post.min()), 1, 100),
        np.linspace(max(dist.min(), post.min()), 1, 100),
        linestyle="dashed",
        color=sns.cubehelix_palette()[-1],
    )  # plot identity line as reference for regression
    if ax_labels:
        plt.xlabel(ax_labels[0], fontsize="xx-large", color=sns.cubehelix_palette()[-1])
        plt.ylabel(ax_labels[1], fontsize="xx-large", color=sns.cubehelix_palette()[2])

    plt.tick_params(labelleft=False, labelbottom=False)

    return corr_stats, EMD