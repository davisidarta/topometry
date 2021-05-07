import numpy as np
from numpy import matlib as mb
from kneed import KneeLocator
import pandas as pd
import matplotlib.pyplot as plt

def multiscale(res,
                 n_eigs='knee',
                 verbose=True
                 ):
    """
    Learn multiscale maps from the diffusion basis.
    Parameters
    ----------
    verbose
    res: dict
        Results from the dbMAP framework. Expects dictionary containing numerical
        'EigenVectors' and 'EigenValues'.

    n_eigs: int or str (optional, default 'max')
        Number of eigenvectors to use. If 'max', expands to the maximum number of positive eigenvalues
        (reach of numerical precision), else to the maximum amount of computed components.
        If 'knee', uses Kneedle to find an optimal cutoff point, and expands it by ``expansion``.

    expansion: int (optional, default 30)
        Number of additional components to compute in search for a numerical eigengap.

    plot: bool (optional, default True)
        Whether to plot or not the scree plot of information entropy.

    verbose: bool
        Controls verbosity.
    Returns
    -------------

    np.ndarray containing the multiscale diffusion components.


    """
    if n_eigs == 'comp_gap':
        vals = np.array(res["EigenValues"])
        use_eigs = int(np.sum(vals > 0, axis=0))
        kn = KneeLocator(range(0, len(vals)), vals, S=50,
                     curve='convex', direction='decreasing', interp_method='polynomial')
    else:
        vals = np.positive(np.array(res["EigenValues"]))
        kn = KneeLocator(range(0, len(vals)), vals, S=50,
                     curve='convex', direction='decreasing', interp_method='polynomial')
        if n_eigs == 'knee':
            use_eigs = int(kn.knee)
    if isinstance(n_eigs, int):
        use_eigs = int(n_eigs)
    elif n_eigs == 'max':
        use_eigs = int(np.sum(vals > 0, axis=0))
    if not isinstance(use_eigs, int):
        raise Exception('Set `n_eigs` to either \'knee\', \'max\', \'comp_gap\' or an `int` value.')

    if use_eigs < 5:
        print('Raising n_eigs...')
        use_eigs = use_eigs + 5

    # Multiscale
    eigs_idx = list(range(1, int(use_eigs)))
    evals = np.positive(np.array(res["EigenValues"]))
    eig_vals = np.ravel(evals[eigs_idx])

    data = res['EigenVectors'].values[:, eigs_idx] * (eig_vals / (1 - eig_vals))
    data = pd.DataFrame(data, index=res['EigenVectors'].index)

    if verbose:
        if n_eigs == 'knee':
            print('Automatically selected and multiscaled ' + str(round(use_eigs)) +
                 ' among ' + str(int(np.sum(vals > 0, axis=0))) + ' diffusion components.')
        elif n_eigs == 'max':
            print('Multiscaled a maximum of ' + str(round(use_eigs)) +
                  ' computed diffusion components.')
        elif n_eigs == 'comp_gap':
            print('Multiscaled ' + str(round(use_eigs)) + ' diffusion components using '
                                                          'a discrete eigengap.')
        else:
            print('Multiscaled ' + str(round(use_eigs)) +
                  ' diffusion components.')
    data = np.array(data)
    return data, kn, use_eigs

