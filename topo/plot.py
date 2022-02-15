import sys
import numba
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.neighbors import KDTree
from matplotlib import cm
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Matplotlib is required for the plotting functions.")
    sys.exit()

def scatter(res, labels=None, pt_size=5, marker='o', opacity=1, cmap='Spectral'):
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
        alpha=opacity)
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

def plot_bases_scores(bases_scores, return_plot=True, figsize=(20,8)):
    keys = bases_scores.keys()
    values = bases_scores.values()
    cmap = get_cmap(len(keys), name='tab20')
    k_color = list()
    for k in np.arange(len(keys)):
        k_color.append(cmap(k))
    pca_vals = list()
    lap_vals = list()
    r_vals = list()
    t_vals = list()
    for val in values:
        pca_vals.append(val[0])
        lap_vals.append(val[1])
        r_vals.append(val[2])
        t_vals.append(val[3])

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
    fig.suptitle('Bases scores:')
    ax1.bar(keys, pca_vals, color=k_color)
    ax1.set_title('PCA loss')
    ax1.set_xticklabels(keys, rotation=90)
    ax2.bar(keys, lap_vals, color=k_color)
    ax2.set_title('LE loss')
    ax2.set_xticklabels(keys, rotation=90)
    ax3.bar(keys, r_vals, color=k_color)
    ax3.set_title('Geodesic Spearman R')
    ax3.set_xticklabels(keys, rotation=90)
    ax4.bar(keys, t_vals, color=k_color)
    ax4.set_title('Geodesic Kendall T')
    ax4.set_xticklabels(keys, rotation=90)
    fig.tight_layout()

    if return_plot:
        return plt.show()
    else:
        return fig

def plot_graphs_scores(graphs_scores, return_plot=True, figsize=(20,8)):
    keys = graphs_scores.keys()
    values = graphs_scores.values()
    cmap = get_cmap(len(keys), name='tab20')
    k_color = list()
    for k in np.arange(len(keys)):
        k_color.append(cmap(k))

    r_vals = list()
    t_vals = list()
    for val in values:
        r_vals.append(val[0])
        t_vals.append(val[1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Graphs scores:')
    ax1.bar(keys, r_vals, color=k_color)
    ax1.set_title('Geodesic Spearman R')
    ax1.set_xticklabels(keys, rotation=90)
    ax2.bar(keys, t_vals, color=k_color)
    ax2.set_title('Geodesic Kendall T')
    ax2.set_xticklabels(keys, rotation=90)
    fig.tight_layout()

    if return_plot:
        return plt.show()
    else:
        return fig


def plot_layouts_scores(layouts_scores, return_plot=True, figsize=(20,8)):
    keys = layouts_scores.keys()
    values = layouts_scores.values()
    cmap = get_cmap(len(keys), name='tab20')
    k_color = list()
    for k in np.arange(len(keys)):
        k_color.append(cmap(k))
    pca_vals = list()
    lap_vals = list()
    r_vals = list()
    t_vals = list()
    for val in values:
        pca_vals.append(val[0])
        lap_vals.append(val[1])
        r_vals.append(val[2])
        t_vals.append(val[3])

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
    fig.suptitle('Layouts scores:')
    ax1.bar(keys, pca_vals, color=k_color)
    ax1.set_title('PCA loss')
    ax1.set_xticklabels(keys, rotation=90)
    ax2.bar(keys, lap_vals, color=k_color)
    ax2.set_title('LE loss')
    ax2.set_xticklabels(keys, rotation=90)
    ax3.bar(keys, r_vals, color=k_color)
    ax3.set_title('Geodesic Spearman R')
    ax3.set_xticklabels(keys, rotation=90)
    ax4.bar(keys, t_vals, color=k_color)
    ax4.set_title('Geodesic Kendall T')
    ax4.set_xticklabels(keys, rotation=90)
    fig.tight_layout()
    if return_plot:
        return plt.show()
    else:
        return fig



def plot_all_layouts(TopOGraph, labels=None, pt_size=5, marker='o', opacity=1, cmap='Spectral'):
    n_bases = 0
    n_graphs = 0
    n_layouts = 0
    if TopOGraph.MSDiffMap is not None:
        n_bases = n_bases + 1
    if TopOGraph.CLapMap is not None:
        n_bases = n_bases + 1
    if TopOGraph.FuzzyLapMap is not None:
        n_bases = n_bases + 1
    if TopOGraph.DiffGraph is not None:
        n_graphs = n_graphs + 1
    if TopOGraph.FuzzyGraph is not None:
        n_graphs = n_graphs + 1
    if TopOGraph.CknnGraph is not None:
        n_graphs = n_graphs + 1
    if TopOGraph.tSNE_Y is not None:
        n_layouts = n_layouts + 1
    if TopOGraph.MAP_Y is not None:
        n_layouts = n_layouts + 1
    if TopOGraph.TriMAP_Y is not None:
        n_layouts = n_layouts + 1
    if TopOGraph.PaCMAP_Y is not None:
        n_layouts = n_layouts + 1
    if TopOGraph.MDE_Y is not None:
        n_layouts = n_layouts + 1
    embeddings = []
    emb_number = 0
    emb_list = list()
    for emb in embeddings:
        if emb is not None:
            emb_number = emb_number + 1
            emb_list.append(emb)
    fig, axes_tuple = plt.subplots(n_graphs, n_layouts)

    for i in range(len(emb_list)):
        axes_tuple[i].scatter(
            emb_list[i][:, 0],
            emb_list[i][:, 1],
            cmap=cmap,
            c=labels,
            s=pt_size,
            marker=marker,
            alpha=opacity)

    return plt.show()


