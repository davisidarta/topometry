# Functions for using Minimum Distortion Embedding(MDE) for graph layout.
# The MDE algorithm was brilliantly coinceived by Akshay Agrawal in
# the monograph https://arxiv.org/abs/2103.02559
# and implemented in https://github.com/cvxgrp/pymde under the Apache 2.0 license

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from pymde import problem
from pymde import quadratic
from pymde import constraints
from pymde.functions import penalties, losses
from pymde import preprocess
import pymde
import igraph

# Some custom minimum-distortion-embedding problems

def IsomorphicMDE(data,
                  attractive_penalty,
                  repulsive_penalty,
                  embedding_dim=2,
                  constraint=None,
                  n_neighbors=None,
                  repulsive_fraction=None,
                  max_distance=None,
                  init='quadratic',
                  device='cpu',
                  verbose=False):
    """
    Construct an MDE problem designed to preserve local structure.
    This function constructs an MDE problem for preserving the
    local structure of original data. This MDE problem is well-suited for
    visualization (using ``embedding_dim`` 2 or 3), but can also be used to
    generate features for machine learning tasks (with ``embedding_dim`` = 10,
    50, or 100, for example). It yields embeddings in which similar items
    are near each other, and dissimilar items are not near each other.
    The original data can either be a data matrix, or a graph.
    Data matrices should be torch Tensors, NumPy arrays, or scipy sparse
    matrices; graphs should be instances of ``pymde.Graph``.
    The MDE problem uses distortion functions derived from weights (i.e.,
    penalties).
    To obtain an embedding, call the ``embed`` method on the returned ``MDE``
    object. To plot it, use ``pymde.plot``.
    .. code:: python3
        embedding = pymde.preserve_neighbors(data).embed()
        pymde.plot(embedding)
    Arguments
    ---------
    data: {torch.Tensor, numpy.ndarray, scipy.sparse matrix} or pymde.Graph
        The original data, a data matrix of shape ``(n_items, n_features)`` or
        a graph. Neighbors are computed using Euclidean distance if the data is
        a matrix, or the shortest-path metric if the data is a graph.
    embedding_dim: int
        The embedding dimension. Use 2 or 3 for visualization.
    attractive_penalty: pymde.Function class (or factory)
        Callable that constructs a distortion function, given positive
        weights. Typically one of the classes from ``pymde.penalties``,
        such as ``pymde.penalties.log1p``, ``pymde.penalties.Huber``, or
        ``pymde.penalties.Quadratic``.
    repulsive_penalty: pymde.Function class (or factory)
        Callable that constructs a distortion function, given negative
        weights. (If ``None``, only positive weights are used.) For example,
        ``pymde.penalties.Log`` or ``pymde.penalties.InversePower``.
    constraint: pymde.constraints.Constraint (optional)
        Embedding constraint, like ``pymde.Standardized()`` or
        ``pymde.Anchored(anchors, values)`` (or ``None``). Defaults to no
        constraint when a repulsive penalty is provided, otherwise defaults to
        ``pymde.Standardized()``.
    n_neighbors: int (optional)
        The number of nearest neighbors to compute for each row (item) of
        ``data``. A sensible value is chosen by default, depending on the
        number of items.
    repulsive_fraction: float (optional)
        How many repulsive edges to include, relative to the number
        of attractive edges. ``1`` means as many repulsive edges as attractive
        edges. The higher this number, the more uniformly spread out the
        embedding will be. Defaults to ``0.5`` for standardized embeddings, and
        ``1`` otherwise. (If ``repulsive_penalty`` is ``None``, this argument
        is ignored.)
    max_distance: float (optional)
        If not None, neighborhoods are restricted to have a radius
        no greater than ``max_distance``.
    init: str
        Initialization strategy; 'quadratic' or 'random'.
    device: str (optional)
        Device for the embedding (eg, 'cpu', 'cuda').
    verbose: bool
        If ``True``, print verbose output.
    Returns
    -------
    pymde.MDE
        A ``pymde.MDE`` object, based on the original data.
    """

    mde = pymde.preserve_neighbors(data=data,
                                   embedding_dim=embedding_dim, attractive_penalty=attractive_penalty,
                                   repulsive_penalty=repulsive_penalty, constraints=constraint,
                                   n_neighbors=n_neighbors, repulsive_fraction=repulsive_fraction,
                                   max_distance=max_distance,
                                   init=init,
                                   device=device,
                                   verbose=verbose)

    return mde




def IsometricMDE(data,
                  embedding_dim=2,
                  loss=losses.Absolute,
                  constraint=None,
                  max_distances=5e7,
                  device="cpu",
                  verbose=False):
    """
    Construct an MDE problem based on original distances.
    This function constructs an MDE problem for preserving pairwise
    distances between items. This can be useful for preserving the global
    structure of the data.
    The data can be specified with either a data matrix (a NumPy array, torch
    Tensor, or sparse matrix), or a ``pymde.Graph`` instance encoding the
    distances:
        A NumPy array, torch tensor, or sparse matrix is interpreted as a
        collection of feature vectors: each row gives the feature vector for an
        item. The original distances are the Euclidean distances between the
        feature vectors.
        A ``pymde.Graph`` instance is interpreted as encoding all (n_items
        choose 2) distances: the distance between i and j is taken to be the
        length of the shortest path connecting i and j.
    When the number of items n_items is large, the total number of pairs will
    be very large. When this happens, instead of computing all pairs of
    distances, this function will sample a subset uniformly at random. The
    maximum number of distances to compute is specified by the parameter
    ``max_distances``. Depending on how many items you have (and how much
    memory your machine has), you may need to adjust this parameter.
    To obtain an embedding, call the ``embed`` method on the returned object.
    To plot it, use ``pymde.plot``.
    For example:
    .. code:: python3
        embedding = pymde.preserve_distances(data).embed()
        pymde.plot(embedding)
    Arguments
    ---------
    data: {np.ndarray, torch.Tensor, scipy.sparse matrix} or pymde.Graph
        The original data, a data matrix of shape ``(n_items, n_features)`` or
        a graph.
    embedding_dim: int
        The embedding dimension.
    loss: pymde.Function class (or factory)
        Callable that constructs a distortion function, given
        original distances. Typically one of the classes defined in
        ``pymde.losses``, such as ``pymde.losses.Absolute``, or
        ``pymde.losses.WeightedQuadratic``.
    constraint: pymde.constraints.Constraint (optional)
        Embedding constraint, such as ``pymde.Standardized()`` or
        ``pymde.Anchored(anchors, values)`` (or ``None``). Defaults to no
        constraint. Note: when the constraint is ``pymde.Standardized()``,
        the original distances will be scaled by a constant (because the
        standardization constraint puts a limit on how large any one
        distance can be).
    max_distances: int
        Maximum number of distances to compute.
    device: str (optional)
        Device for the embedding (eg, 'cpu', 'cuda').
    verbose: bool
        If ``True``, print verbose output.
    Returns
    -------
    pymde.MDE
        A ``pymde.MDE`` instance, based on preserving the original distances.
    """
    mde = pymde.preserve_distances(data=data,
                                   embedding_dim=embedding_dim,
                                   loss=loss,
                                   constraint=None,
                                   max_distances=max_distances,
                                   device=device,
                                   verbose=verbose)
    return mde
