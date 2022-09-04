import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def multiscale(res,
               n_eigs='max',
               verbose=True
               ):
    """
    Learn multiscale maps from the diffusion basis.

    Parameters
    ----------
    res: dict
        Results from the Diffusor().fit_transform() method. Expects dictionary containing numerical
        'EigenVectors' and 'EigenValues'.

    n_eigs: int or str (optional, default 'max')
        Number of eigenvectors to use. If 'max', expands to the maximum number of positive eigenvalues
        (reach of numerical precision), else to the maximum amount of computed components.
        If 'knee', uses Kneedle to find an optimal cutoff point, and expands it by ``expansion``.
        If 'comp_gap', tries to find a discrete eigengap from the computation process.

    verbose: bool
        Controls verbosity.

    Returns
    -------------

    np.ndarray containing the multiscale diffusion components.


    """
    evals = np.array(res["EigenValues"])
    if n_eigs == 'knee':
        try:
            from kneed import KneeLocator
        except ImportError:
            raise Exception('Kneedle not installed!')
        kn = KneeLocator(range(0, len(evals)), evals, S=30,
                         curve='convex', direction='decreasing')
        if kn.knee is None:
            n_eigs = 'max'
        else:
            use_eigs = int(kn.knee)
    elif isinstance(n_eigs, int):
        use_eigs = int(n_eigs)
        if use_eigs < 5:
            if verbose:
                print('Raising n_eigs to maximum computed!')
            n_eigs = 'max'
    elif n_eigs == 'max':
        use_eigs = int(np.sum(evals > 0, axis=0))
    else:
        raise ValueError('n_eigs must be an integer or "max" or "knee"')

    # Multiscale
    eigs_idx = list(range(1, int(use_eigs)))
    eig_vals = np.ravel(evals[eigs_idx])

    data = res['EigenVectors'].values[:, eigs_idx] * \
        (eig_vals / (1 - eig_vals))
    data = pd.DataFrame(data, index=res['EigenVectors'].index)

    if verbose:
        if n_eigs == 'knee':
            print('Automatically selected and multiscaled ' + str(round(use_eigs)) +
                  ' among ' + str(int(np.sum(evals > 0, axis=0))) + ' diffusion components.')
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
    return data, use_eigs



