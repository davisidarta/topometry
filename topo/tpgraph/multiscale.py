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
        If 'comp_gap', tries to find a discrete eigengap from the computation process.

    verbose: bool
        Controls verbosity.

    Returns
    -------------

    np.ndarray containing the multiscale diffusion components.


    """
    use_eigs = len(np.array(res["EigenValues"])) // 2
    evals = np.array(res["EigenValues"])
    if isinstance(n_eigs, int):
        use_eigs = int(n_eigs)
    if use_eigs < 5:
        if verbose:
            print('Raising n_eigs to maximum computed!')
        n_eigs = 'max'
    if n_eigs == 'max':
        use_eigs = int(np.sum(evals > 0, axis=0))
    if not isinstance(use_eigs, int):
        raise Exception('Set `n_eigs` to either \'max\', \'comp_gap\' or an `int` value.')


    # Multiscale
    eigs_idx = list(range(1, int(use_eigs)))
    eig_vals = np.ravel(evals[eigs_idx])

    data = res['EigenVectors'].values[:, eigs_idx] * (eig_vals / (1 - eig_vals))
    data = pd.DataFrame(data, index=res['EigenVectors'].index)

    if verbose:
        if n_eigs == 'max':
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
