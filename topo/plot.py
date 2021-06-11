import sys

import numpy as np
from matplotlib.patches import Ellipse

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Matplotlib is required for the plotting functions.")
    sys.exit()

def scatter_plot(res, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.gca().set_aspect('equal', 'datalim')
    ax.scatter(
        res[:, 0],
        res[:, 1],
    **kwargs
    )
    return plt.show()

def scatter_3d_plot(res, labels=None, title=None, fontsize=18, pt_size=5, marker='o', opacity=1, cmap='Spectral',  **kwargs):
    if len(res[0]) != 3:
        return print('Expects array with 3 columns. Input has ' + str(int(len(res[0]))) + '.')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(res[:, 0], res[:, 1], res[:, 2],
               cmap=cmap,
               c=labels,
               s=pt_size,
               title=title,
               fontsize=fontsize,
               marker=marker,
               alpha=opacity,
               **kwargs)
    return plt.show()

def hyperboloid_3d_plot(emb, labels=None, title=None, fontsize=18, pt_size=5, marker='o', opacity=1, cmap='Spectral', **kwargs):
    x, y, z = hyperboloid_emb(emb)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z,
               cmap=cmap,
               c=labels,
               s=pt_size,
               title=title,
               fontsize=fontsize,
               marker=marker,
               alpha=opacity,
               **kwargs)
    ax.view_init(35, 80)
    plt.gca().set_aspect('equal', 'datalim')

    return plt.show()

def hyperboloid_emb(emb):
    x = emb[:, 0]
    y = emb[:, 1]
    z = np.sqrt(1 + np.sum(emb ** 2, axis=1))
    return x, y, z

def poincare_disk_plot(emb, labels=None, title=None, fontsize=18, pt_size=5, marker='o', opacity=1, cmap='Spectral', **kwargs):
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
               title=title,
               fontsize=fontsize,
               marker=marker,
               alpha=opacity,
               **kwargs)
    ax.axis('off')
    plt.gca().set_aspect('equal', 'datalim')
    return plt.show()

def sphere_3d_plot(emb, labels=None, title=None, fontsize=18, pt_size=5, marker='o', opacity=1, cmap='Spectral', **kwargs):
    x = np.sin(emb[:, 0]) * np.cos(emb[:, 1])
    y = np.sin(emb[:, 0]) * np.sin(emb[:, 1])
    z = np.cos(emb[:, 0])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z,
               cmap=cmap,
               c=labels,
               s=pt_size,
               title=title,
               fontsize=fontsize,
               marker=marker,
               alpha=opacity,
               **kwargs)
    return plt.show()

def sphere_projection(emb, labels=None, title=None, fontsize=18, pt_size=5, marker='o', opacity=1, cmap='Spectral', **kwargs):
    x = np.sin(emb[:, 0]) * np.cos(emb[:, 1])
    y = np.sin(emb[:, 0]) * np.sin(emb[:, 1])
    z = np.cos(emb[:, 0])
    x = np.arctan2(x, y)
    y = -np.arccos(z)
    plt.scatter(x, y,
                cmap=cmap,
                c=labels,
                s=pt_size,
                title=title,
                fontsize=fontsize,
                marker=marker,
                alpha=opacity,
                **kwargs)
    return plt.show()

def draw_simple_ellipse(position, width, height, angle,
                        ax=None, from_size=0.1, to_size=0.5, n_ellipses=3,
                        alpha=0.1, color=None,
                        **kwargs):
    ax = ax or plt.gca()
    angle = (angle / np.pi) * 180
    width, height = np.sqrt(width), np.sqrt(height)
    # Draw the Ellipse
    for nsig in np.linspace(from_size, to_size, n_ellipses):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, alpha=alpha, lw=0, color=color, **kwargs))


def toroid_3d_plot(emb, R=3, r=1, labels=None, title=None, fontsize=18, pt_size=5, marker='o', opacity=1, cmap='Spectral', **kwargs):
    x = (R + r * np.cos(emb[:, 0])) * np.cos(emb[:, 1])
    y = (R + r * np.cos(emb[:, 0])) * np.sin(emb[:, 1])
    z = r * np.sin(emb[:, 0])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z,
               cmap=cmap,
               c=labels,
               s=pt_size,
               title=title,
               fontsize=fontsize,
               marker=marker,
               alpha=opacity,
               **kwargs)
    ax.set_zlim3d(-3, 3)
    ax.view_init(35, 70)
    return plt.show()



def gaussian_potential_plot(emb, dims=[2, 3, 4],
                            labels=None, title=None, fontsize=18, pt_size=5,
                            marker='o', opacity=1, cmap='Spectral', **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(emb[0]):
        pos = emb[i, :2]
        draw_simple_ellipse(pos, emb[i, dims[0]],
                            emb[i, dims[1]],
                            emb[i, dims[2]],
                            ax, n_ellipses=1,
                            color=labels,
                            from_size=1.0, to_size=1.0, alpha=0.01)
    ax.scatter(emb.T[0],
               emb.T[1],
               cmap=cmap,
               c=labels,
               s=pt_size,
               title=title,
               fontsize=fontsize,
               marker=marker,
               alpha=opacity,
               **kwargs)
    return plt.show()
