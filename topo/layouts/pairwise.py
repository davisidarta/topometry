import time
from numpy import random
# PaCMAP

def PaCMAP(data=None,
               init=None,
               n_dims=2,
               n_neighbors=10,
               MN_ratio=0.5,
               FP_ratio=2.0,
               pair_neighbors=None,
               pair_MN=None,
               pair_FP=None,
               distance="euclidean",
               lr=1.0,
               num_iters=450,
               verbose=False,
               intermediate=False):
    """
    Dimensionality Reduction Using Pairwise-controlled Manifold Approximation and Projectio

    Inputs
    ------
    data : np.array with the data to be reduced

    init : the initialization of the lower dimensional embedding. One of "pca" or "random", or a user-provided numpy ndarray with the shape (N, 2). Default to "random".

    n_dims :  the number of dimension of the output. Default to 2.

    n_neighbors : the number of neighbors considered in the k-Nearest Neighbor graph. Default to 10 for dataset whose
        sample size is smaller than 10000. For large dataset whose sample size (n) is larger than 10000, the default value
        is: 10 + 15 * (log10(n) - 4).

    MN_ratio :  the ratio of the number of mid-near pairs to the number of neighbors, n_MN = \lfloor n_neighbors * MN_ratio \rfloor .
     Default to 0.5.

    FP_ratio : the ratio of the number of further pairs to the number of neighbors, n_FP = \lfloor n_neighbors * FP_ratio \rfloor Default to 2.

    distance : Distance measure ('euclidean' (default), 'manhattan', 'angular',
    'hamming')


    lr : Optimization method ('sd': steepest descent,  'momentum': GD
    with momentum, 'dbd': GD with momentum delta-bar-delta (default))

    num_iters : number of iterations. Default to 450. 450 iterations is enough for most dataset to converge.

    pair_neighbors, pair_MN and pair_FP: pre-specified neighbor pairs, mid-near points, and further pairs. Allows user to use their own graphs. Default to None.

    verbose : controls verbosity (default False)

    intermediate : whether pacmap should also output the intermediate stages of the optimization process of the lower dimension embedding. If True, then the output will be a numpy array of the size (n, n_dims, 13), where each slice is a "screenshot" of the output embedding at a particular number of steps, from [0, 10, 30, 60, 100, 120, 140, 170, 200, 250, 300, 350, 450].

    random_state :
        RandomState object (default None)

    """


    try:
        import pacmap
        _have_pacmap = True
    except ImportError('TriMAP is needed for this embedding. Install it with `pip install trimap`'):
        return print('TriMAP is needed for this embedding. Install it with `pip install trimap`')

    pacmap_emb = pacmap.PaCMAP(n_dims=n_dims,
                               n_neighbors=n_neighbors,
                               MN_ratio=MN_ratio,
                               FP_ratio=FP_ratio,
                               pair_neighbors=pair_neighbors,
                               pair_MN=pair_MN,
                               pair_FP=pair_FP,
                               distance=distance,
                               lr=lr,
                               num_iters=num_iters,
                               verbose=verbose,
                               intermediate=intermediate).fit_transform(X=data, init=init)

    return pacmap_emb
