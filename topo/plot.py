import sys
import numba
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.neighbors import KDTree
from sklearn.utils import check_random_state
try:
    from matplotlib import cm
    import matplotlib.pyplot as plt
except ImportError:
    print("Matplotlib is required for the plotting functions.")
    sys.exit()

def decay_plot(evals, title=None, figsize=(9, 5), fontsize=14, label_fontsize=10):
    """
    Plot the eigenspectrum decay and its first derivatives.

    Parameters
    ----------
    evals : Eigenvalues to be visualized.

    title : Title of the plot.

    Returns
    -------

    A simple plot of the eigenspectrum decay.

    """
    fig = plt.figure(figsize=figsize)
    max_eigs = int(np.sum(evals > 0, axis=0))
    first_diff = np.diff(evals)
    sec_diff = np.diff(first_diff)
    eigengap = np.argmax(first_diff) + 1
    ax1 = fig.add_subplot(1, 2, 1)
    if title is not None:
        plt.suptitle(title, fontsize=fontsize)
    ax1.plot(range(0, len(evals)), evals, 'b')
    ax1.set_ylabel('Eigenvalues', fontsize=label_fontsize)
    ax1.set_xlabel('Eigenvectors', fontsize=label_fontsize)
    if max_eigs == len(evals):
        # Could not find a discrete eigengap crossing 0
        ax1.vlines(
            eigengap, plt.ylim()[0], plt.ylim()[1], linestyles="--", label='Eigengap'
        )
        plt.suptitle('Spectrum decay and eigengap (%i)' %
                      int(eigengap), fontsize=fontsize)
    else:
        ax1.vlines(
            max_eigs, plt.ylim()[0], plt.ylim()[1], linestyles="--", label='Eigengap'
        )
        plt.suptitle('Spectrum decay and eigengap (%i)' %
                      int(max_eigs), fontsize=fontsize)
    ax1.legend(prop={'size': 12}, fontsize=label_fontsize, loc='best')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_yscale('log')
    ax2.scatter(range(0, len(first_diff)), np.abs(first_diff))
    ax2.set_ylabel('Eigenvalues first derivatives (abs)', fontsize=label_fontsize)
    ax2.set_xlabel('Eigenvalues', fontsize=label_fontsize)
    if max_eigs == len(evals):
        # Could not find a discrete eigengap crossing 0
        ax2.vlines(
            eigengap, plt.ylim()[0], plt.ylim()[1], linestyles="--", label='Eigengap'
        )
    else:
        ax2.vlines(
            max_eigs, plt.ylim()[0], plt.ylim()[1], linestyles="--", label='Eigengap'
        )
    plt.tight_layout()
    return plt.show()

def scatter(res, labels=None, pt_size=5, marker='o', opacity=1, cmap='Spectral', **kwargs):
    """
    Basic scatter plot function.

    Parameters
    ----------

    labels
    pt_size
    marker
    opacity
    cmap

    Returns
    -------

    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_box_aspect(1) 
    ax.scatter(
        res[:, 0],
        res[:, 1],
        cmap=cmap,
        c=labels,
        s=pt_size,
        marker=marker,
        alpha=opacity,
        **kwargs)
    return plt.show()


def scatter3d(res, labels=None, pt_size=5, marker='o', opacity=1, cmap='Spectral'):
    if len(res[0]) != 3:
        return print('Expects array with 3 columns. Input has ' + str(int(len(res[0]))) + '.')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(res[:, 0], res[:, 1], res[:, 2],
               cmap=cmap,
               c=labels,
               s=pt_size,
               marker=marker,
               alpha=opacity)
    return plt.show()

def hyperboloid(emb, labels=None, pt_size=5, marker='o', opacity=1, cmap='Spectral'):
    x, y, z = two_to_3d_hyperboloid(emb)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z,
               cmap=cmap,
               c=labels,
               s=pt_size,
               marker=marker,
               alpha=opacity
               )
    ax.view_init(35, 80)
    ax.set_aspect("equal", adjustable="datalim")
    return plt.show()



def two_to_3d_hyperboloid(emb):
    x = emb[:, 0]
    y = emb[:, 1]
    z = np.sqrt(1 + np.sum(emb ** 2, axis=1))
    return x, y, z

def poincare(emb, labels=None, pt_size=5, marker='o', opacity=1, cmap='Spectral'):
    x = emb[:, 0]
    y = emb[:, 1]
    z = np.sqrt(1 + np.sum(emb ** 2, axis=1))
    disk_x = x / (1 + z)
    disk_y = y / (1 + z)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    boundary = plt.Circle((0, 0), 1, fc='none', ec='k')
    ax.add_artist(boundary)
    ax.scatter(disk_x, disk_y,
               cmap=cmap,
               c=labels,
               s=pt_size,
               marker=marker,
               alpha=opacity)
    ax.axis('off')
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_box_aspect(1) 
    return plt.show()

def sphere(emb, labels=None, pt_size=5, marker='o', opacity=1, cmap='Spectral'):
    x = np.sin(emb[:, 0]) * np.cos(emb[:, 1])
    y = np.sin(emb[:, 0]) * np.sin(emb[:, 1])
    z = np.cos(emb[:, 0])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z,
               cmap=cmap,
               c=labels,
               s=pt_size,
               marker=marker,
               alpha=opacity)
    return plt.show()

def sphere_projection(emb, labels=None, pt_size=5, marker='o', opacity=1, cmap='Spectral'):
    x = np.sin(emb[:, 0]) * np.cos(emb[:, 1])
    y = np.sin(emb[:, 0]) * np.sin(emb[:, 1])
    z = np.cos(emb[:, 0])
    x = np.arctan2(x, y)
    y = -np.arccos(z)
    plt.scatter(x, y,
                cmap=cmap,
                c=labels,
                s=pt_size,
                marker=marker,
                alpha=opacity)
    return plt.show()


def toroid(emb, R=3, r=1, labels=None, pt_size=5, marker='o', opacity=1, cmap='Spectral'):
    x = (R + r * np.cos(emb[:, 0])) * np.cos(emb[:, 1])
    y = (R + r * np.cos(emb[:, 0])) * np.sin(emb[:, 1])
    z = r * np.sin(emb[:, 0])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z,
               cmap=cmap,
               c=labels,
               s=pt_size,
               marker=marker,
               alpha=opacity)
    ax.set_zlim3d(-3, 3)
    ax.view_init(35, 70)
    return plt.show()


def draw_simple_ellipse(position, width, height, angle,
                        ax=None, from_size=0.1, to_size=0.5, n_ellipses=3,
                        alpha=0.1, color=None):
    ax = ax or plt.gca()
    angle = (angle / np.pi) * 180
    width, height = np.sqrt(width + 10e-4), np.sqrt(height + 10e-4)
    # Draw the Ellipse
    for nsig in np.linspace(from_size, to_size, n_ellipses):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, alpha=alpha, lw=0, color=color))


def gaussian_potential(emb, dims=[2, 3, 4],
                            labels=None, pt_size=5,
                            marker='o', opacity=1, cmap='Spectral'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, np.shape(labels.unique())[0]))
    for i in range(emb.shape[0]):
        pos = emb[i, :2]
        draw_simple_ellipse(pos, emb[i, dims[0]],
                            emb[i, dims[1]],
                            emb[i, dims[2]],
                            ax, n_ellipses=1,
                            color=colors[labels[i]],
                            from_size=1.0, to_size=1.0, alpha=0.01)

    ax.scatter(emb.T[0],
               emb.T[1],
               cmap=cmap,
               c=labels,
               s=pt_size,
               marker=marker,
               alpha=opacity)
    return plt.show()


@numba.njit(fastmath=True)
def eval_gaussian(x, pos=np.array([0, 0]), cov=np.eye(2, dtype=np.float32)):
    det = cov[0,0] * cov[1,1] - cov[0,1] * cov[1,0]
    if det > 1e-16:
        cov_inv = np.array([[cov[1,1], -cov[0,1]], [-cov[1,0], cov[0,0]]]) * 1.0 / det
        diff = x - pos
        m_dist = cov_inv[0,0] * diff[0]**2 - \
            (cov_inv[0,1] + cov_inv[1,0]) * diff[0] * diff[1] + \
            cov_inv[1,1] * diff[1]**2
        return (np.exp(-0.5 * m_dist)) / (2 * np.pi * np.sqrt(np.abs(det)))
    else:
        return 0.0

@numba.njit(fastmath=True)
def eval_density_at_point(x, embedding):
    result = 0.0
    for i in range(embedding.shape[0]):
        pos = embedding[i, :2]
        t = embedding[i, 4]
        U = np.array([[np.cos(t), np.sin(t)], [np.sin(t), -np.cos(t)]])
        cov = U @ np.diag(embedding[i, 2:4]) @ U
        result += eval_gaussian(x, pos=pos, cov=cov)
    return result


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


def create_density_plot(X, Y, embedding):
    Z = np.zeros_like(X)
    tree = KDTree(embedding[:, :2])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            nearby_points = embedding[tree.query_radius([[X[i,j],Y[i,j]]], r=2)[0]]
            Z[i, j] = eval_density_at_point(np.array([X[i,j],Y[i,j]]), nearby_points)
    return Z / Z.sum()

def plot_bases_scores(bases_scores, return_plot=True, figsize=(20,8), fontsize=20):
    keys = bases_scores.keys()
    values = bases_scores.values()
    cmap = get_cmap(len(keys), name='tab20')
    k_color = list()
    for k in np.arange(len(keys)):
        k_color.append(cmap(k))
    pca_vals = list()
    r_vals = list()
    for val in values:
        pca_vals.append(val[0])
        r_vals.append(val[1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Bases scores:', fontsize=fontsize)
    ax1.bar(keys, pca_vals, color=k_color)
    ax1.set_title('PCA loss', fontsize=fontsize)
    ax1.set_xticklabels(keys, fontsize=fontsize)
    ax2.bar(keys, r_vals, color=k_color)
    ax2.set_title('Geodesic Spearman R', fontsize=fontsize)
    ax2.set_xticklabels(keys, fontsize=fontsize)
    fig.tight_layout()
    if return_plot:
        return plt.show()
    else:
        return fig


def plot_graphs_scores(graphs_scores, return_plot=True, figsize=(20,8), fontsize=20):
    keys = graphs_scores.keys()
    values = graphs_scores.values()
    cmap = get_cmap(len(keys), name='tab20')
    k_color = list()
    for k in np.arange(len(keys)):
        k_color.append(cmap(k))

    fig, (ax1) = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle('Graphs scores:', fontsize=fontsize)
    ax1.bar(keys, values, color=k_color)
    ax1.set_title('Geodesic Spearman R', fontsize=fontsize)
    ax1.set_xticklabels(keys, fontsize=fontsize//2, rotation=90)
    fig.tight_layout()

    if return_plot:
        return plt.show()
    else:
        return fig


def plot_layouts_scores(layouts_scores, return_plot=True, figsize=(20,8), fontsize=20):
    keys = layouts_scores.keys()
    values = layouts_scores.values()
    cmap = get_cmap(len(keys), name='tab20')
    k_color = list()
    for k in np.arange(len(keys)):
        k_color.append(cmap(k))
    pca_vals = list()
    r_vals = list()
    for val in values:
        pca_vals.append(val[0])
        r_vals.append(val[1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Layouts scores:', fontsize=fontsize)
    ax1.bar(keys, pca_vals, color=k_color)
    ax1.set_title('PCA loss', fontsize=fontsize)
    ax1.set_xticklabels(keys, fontsize=fontsize//2, rotation=90)
    ax2.bar(keys, r_vals, color=k_color)
    ax2.set_title('Geodesic Spearman R', fontsize=fontsize)
    ax2.set_xticklabels(keys, fontsize=fontsize//2, rotation=90)
    fig.tight_layout()
    if return_plot:
        return plt.show()
    else:
        return fig




def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def plot_cov_ellipse(cov, pos, nstd=1, ax=None, **kwargs):
    #if ax is None:
    ax = plt.gca()
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(np.absolute(vals))
    ellip = Ellipse(pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_artist(ellip)
    return ellip



def plot_riemann_metric(emb, laplacian, H_emb=None, ax=None, n_plot=50, std=1, alpha=0.1, title='Riemannian metric', title_fontsize=10,
                        labels=None, pt_size=1, cmap='Spectral',  figsize=(8,8), random_state=None, **kwargs):
    """
    Plot Riemannian metric using ellipses. Adapted from Megaman (https://github.com/mmp2/megaman).

    Parameters
    ----------
    
    emb: numpy.ndarray
        Embedding matrix.
    
    laplacian: numpy.ndarray
       Graph Laplacian matrix. Should be provided if H_emb is not provided.
    
    H_emb: numpy.ndarray
        Embedding matrix of the H. Should be provided if laplacian is not provided.

    n_plot: int (optional, default 50)
        Number of ellipses to plot.

    std: int (optional, default 1)
        Standard deviation of the ellipses. This should be adjusted by hand for visualization purposes.

    labels: numpy.ndarray (optional, default None)
        Labels for the points.
    
    pt_size: int (optional, default 1)
        Size of the points.
    
    cmap: str (optional, default 'Spectral')
        Color map for the points.

    figsize: tuple (optional, default (8,8))
        Figure size.
    
    random_state: int (optional, default None)
        Random state for sampling points to plot ellipses of.

    kwargs: dict
        Additional arguments for matplotlib.

    References
    ----------
    "Non-linear dimensionality reduction: Riemannian metric estimation and
    the problem of geometric discovery",
    Dominique Perraul-Joncas, Marina Meila, arXiv:1305.7255


    """

    if H_emb is None:
        from topo.eval import RiemannMetric
        rmetric = RiemannMetric(emb, laplacian)
        H_emb = rmetric.get_dual_rmetric()

    N = np.shape(emb)[0]
    rng = check_random_state(random_state)
    sample_points = rng.choice(range(N), n_plot, replace=False)
    if ax == None:
        f, ax = plt.subplots(figsize=figsize)
    # ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_box_aspect(1)   # if an ellipse is a circle no distortion occured in particular directions
    if labels is not None:
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, np.shape(np.unique(labels))[0]))
        ax.scatter(emb[:, 0], emb[:, 1], s=pt_size, c=labels, cmap=cmap)
    else:
        ax.scatter(emb[:, 0], emb[:, 1], s=pt_size)
    for i in range(n_plot):
        ii = sample_points[i]
        cov = H_emb[ii, :, :]
        if labels is not None:
            plot_cov_ellipse(cov, emb[ii, :], nstd=std, ax=ax, edgecolor='none', color=colors[labels[ii]],
                             alpha=alpha)
        else:
            plot_cov_ellipse(cov, emb[ii, :], nstd=std, ax=ax, edgecolor='none',
                             alpha=alpha)
    return ax


def draw_edges(ax, data, kernel, color='black', **kwargs):
    for i in range(data.shape[0]-1):
        for j in range(i+1, data.shape[0]):
            affinity = kernel[i,j]
            if affinity > 0:
                ax.plot(data[[i,j],0], data[[i,j],1],
                        color=color, alpha=affinity, zorder=0, **kwargs)
    


def plot_scores(scores, return_plot=True, log=False, figsize=(20,8), fontsize=15, title='Eigenbasis local scores'):
    keys = scores.keys()
    values = scores.values()
    cmap = get_cmap(len(keys), name='tab20')
    k_color = list()
    for k in np.arange(len(keys)):
        k_color.append(cmap(k))
    fig, (ax1) = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle(title, fontsize=round(fontsize * 1.5))
    ax1.set_xticklabels(keys, fontsize=fontsize, rotation=90)
    ax1.bar(keys, values, color=k_color)
    if log:
        ax1.set_yscale('log')
    fig.tight_layout()
    if return_plot:
        return plt.show()
    else:
        return fig


def plot_all_scores(evaluation_dict, log=False, figsize=(20,8), fontsize=20):
    for key, value in evaluation_dict.items():
        plot_scores(value, figsize=figsize, log=log, fontsize=fontsize, title=key)


def plot_eigenvectors(eigenvectors, n_eigenvectors=10, labels=None, cmap='tab20', figsize=(23,2), fontsize=10, title='DC', **kwargs):
    plt.figure(figsize=figsize)
    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
    )
    plot_num = 1
    for i in range(0, n_eigenvectors):
        plt.subplot(1, n_eigenvectors, plot_num)
        plt.title(title+ ' ' + str(plot_num), fontsize=fontsize)
        plt.scatter(range(0, eigenvectors.shape[0]), eigenvectors[:,i], c=labels, cmap=cmap, **kwargs)
        plot_num += 1
        plt.xticks(())
        plt.yticks(())
    return plt.show()


def plot_dimensionality_histograms(local_id_dict, global_id_dict, bins=50, title = 'FSA', histtype='step', stacked=True, density=True, log=False, title_fontsize=22, legend_fontsize=15):
    fig, ax = plt.subplots(1,1)
    fig.set_figwidth(6)
    fig.set_figheight(8)
    for key in local_id_dict.keys():
        i=0
        x = local_id_dict[key]
        #
        # Make a multiple-histogram of data-sets with different length.
        label = 'k = ' + key + '    ( estim.i.d. = ' + str(int(global_id_dict[key])) + ' )'
        n, bins, patches  = ax.hist(x, bins=bins, histtype=histtype, stacked=stacked, density=density, log=log, label=label)
        sigma = np.std(x)
        mu = np.mean(x)
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
            np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
        i= i+1
    ax.set_title(title, fontsize=title_fontsize, pad=10)
    ax.legend(prop={'size': 12}, fontsize=legend_fontsize)
    ax.set_xlabel('Estimated intrinsic dimension', fontsize=legend_fontsize)
    ax.set_ylabel('Frequency', fontsize=legend_fontsize)
    ax.legend(prop={'size': 10})
    plt.show()

def plot_dimensionality_histograms_multiple(id_dict, bins=50, histtype='step', stacked=True, density=True, log=False,  title='I.D. estimates'):
    fig, ax = plt.subplots(1,1)
    # data
    for key in id_dict.keys():
        i=0
        x = id_dict[key]
        #
        # Make a multiple-histogram of data-sets with different length.
        n, bins, patches  = ax.hist(x, bins=bins, histtype=histtype, stacked=stacked, density=True, log=log, label=key)
        sigma = np.std(x)
        mu = np.mean(x)
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
            np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
        i= i+1
    ax.set_title(title)
    ax.legend(prop={'size': 10})
    fig.tight_layout()
    plt.show()


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", cbar_fontsize=12, shrink=0.6, cb_pad=0.3, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=shrink, pad=cb_pad, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=cbar_fontsize)

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, an_fontsize=8, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        from matplotlib import ticker
        valfmt = ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), fontsize=an_fontsize, **kw)
            texts.append(text)

    return texts