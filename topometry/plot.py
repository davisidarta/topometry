import sys
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Matplotlib is required for the plotting functions.")
    sys.exit()

def scatter_plot(res, labels=None, title=None, fontsize=18, pt_size=5, marker='o', opacity=1, cmap='Spectral'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.gca().set_aspect('equal', 'datalim')
    ax.scatter(
        res[:, 0],
        res[:, 1],
    s=pt_size,
    c=labels,
    marker=marker,
    alpha=opacity,
    cmap=cmap
    )
    if title is not None:
        ax.title(title, fontsize=fontsize)
    return plt.show()

def scatter_plot_3d(res, labels, pt_size=5, marker='o', opacity=1, cmap='Spectral'):
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

def hyperboloid_plot(emb, labels=None, pt_size=None, marker='o', opacity=1, cmap='Spectral'):
    x = emb[:, 0]
    y = emb[:, 1]
    z = np.sqrt(1 + np.sum(emb ** 2, axis=1))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z,
               cmap=cmap,
               s=pt_size,
    c=labels,
    marker=marker,
    alpha=opacity)
    ax.view_init(35, 80)
    plt.gca().set_aspect('equal', 'datalim')

    return plt.show()

def poincare_disk_plot(emb, labels=None, pt_size=1, marker='o', opacity=1, cmap='Spectral'):
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


