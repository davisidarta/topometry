import time

from numpy import random

try:
    from typing import Literal
except ImportError:
    try:
        from typing_extensions import Literal
    except ImportError:

        class LiteralMeta(type):
            def __getitem__(cls, values):
                if not isinstance(values, tuple):
                    values = (values,)
                return type('Literal_', (Literal,), dict(__args__=values))


        class Literal(metaclass=LiteralMeta):
            pass

# TriMAP

def TriMAP(X,
           init=None,
           n_dims=2,
           n_inliers=10,
           n_outliers=5,
           n_random=5,
           distance="euclidean",
           lr=1000.0,
           n_iters=400,
           triplets=None,
           weights=None,
           use_dist_matrix=False,
           knn_tuple=None,
           verbose=True,
           weight_adj=500.0,
           opt_method="dbd",
           return_seq=False):
    """
    Dimensionality Reduction Using Triplet Constraints
    Find a low-dimensional representation of the data by satisfying the sampled
    triplet constraints from the high-dimensional features.


    Inputs
    ------
    n_dims : Number of dimensions of the embedding (default = 2)

    n_inliers : Number of inlier points for triplet constraints (default = 10)

    n_outliers : Number of outlier points for triplet constraints (default = 5)

    n_random : Number of random triplet constraints per point (default = 5)

    distance : Distance measure ('euclidean' (default), 'manhattan', 'angular',
    'hamming')

    lr : Learning rate (default = 1000.0)

    n_iters : Number of iterations (default = 400)

    use_dist_matrix : X is the pairwise distances between points (default = False)

    knn_tuple : Use the pre-computed nearest-neighbors information in form of a
    tuple (knn_nbrs, knn_distances), needs also X to compute the embedding (default = None)


    opt_method : Optimization method ('sd': steepest descent,  'momentum': GD
    with momentum, 'dbd': GD with momentum delta-bar-delta (default))

    verbose : Print the progress report (default = True)

    weight_adj : Adjusting the weights using a non-linear transformation
    (default = 500.0)

    return_seq : Return the sequence of maps recorded every 10 iterations
    (default = False)
    """


    try:
        import trimap
        _have_trimap = True
    except ImportError('TriMAP is needed for this embedding. Install it with `pip install trimap`'):
        return print('TriMAP is needed for this embedding. Install it with `pip install trimap`')

    trimap_emb = trimap.TRIMAP(n_dims=n_dims,
                               n_inliers=n_inliers,
                               n_outliers=n_outliers,
                               n_random=n_random,
                               distance=distance,
                               lr=lr,
                               n_iters=n_iters,
                               triplets=triplets,
                               weights=weights,
                               use_dist_matrix=use_dist_matrix,
                               apply_pca=False,
                               knn_tuple=knn_tuple,
                               verbose=verbose,
                               weight_adj=weight_adj,
                               opt_method=opt_method,
                               return_seq=return_seq).fit_transform(X, init=init)



    return trimap_emb
