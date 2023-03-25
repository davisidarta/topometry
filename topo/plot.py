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

def decay_plot(evals, title=None, figsize=(9, 5), fontsize=10):
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
    eigengap = np.argmax(first_diff) + 1
    ax1 = fig.add_subplot(1, 2, 1)
    if title is not None:
        plt.suptitle(title, fontsize=fontsize)
    ax1.plot(range(0, len(evals)), evals, 'b')
    ax1.set_ylabel('Eigenvalues')
    ax1.set_xlabel('Eigenvectors')
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
    ax1.legend(loc='best')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(range(0, len(first_diff)), first_diff)
    ax2.set_ylabel('Eigenvalues first derivatives')
    ax2.set_xlabel('Eigenvalues')
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
    plt.gca().set_aspect('equal', 'datalim')
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
    plt.gca().set_aspect('auto', 'datalim')
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
    plt.gca().set_aspect('equal', 'datalim')
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
    width, height = np.sqrt(width + 10e-8), np.sqrt(height + 10e-8)
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
    if ax is None:
        ax = plt.gca()
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(np.absolute(vals))
    ellip = Ellipse(pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_artist(ellip)
    return ellip

def plot_riemann_metric(emb, laplacian=None, H_emb=None, n_plot=50, std=1, alpha=0.1, title=None, ax=None,
                        labels=None, pt_size=1, cmap='Spectral',  figsize=(12,12), random_state=None, **kwargs):
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
    if ax is None:
        f, ax = plt.subplots(figsize=figsize)
    if title is not None:
        ax.set_title(title)
    ax.grid(False)
    ax.set_aspect('equal', 'datalim')  # if an ellipse is a circle no distortion occured.
    if labels is not None:
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, np.shape(np.unique(labels))[0]))
        ax.scatter(emb[:, 0], emb[:, 1], s=pt_size, c=labels, cmap=cmap, **kwargs)
    else:
        ax.scatter(emb[:, 0], emb[:, 1], s=pt_size, **kwargs)
    for i in range(n_plot):
        ii = sample_points[i]
        cov = H_emb[ii, :, :]
        if labels is not None:
            plot_cov_ellipse(cov, emb[ii, :], nstd=std, ax=ax, edgecolor=None, color=colors[labels[ii]],
                             alpha=alpha)
        else:
            plot_cov_ellipse(cov, emb[ii, :], nstd=std, ax=ax, edgecolor=None,
                             alpha=alpha)
    plt.show()



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

