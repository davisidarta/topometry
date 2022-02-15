import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator


def multiscale(res,
                 n_eigs='max',
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
        If 'comp_gap', tries to find a discrete eigengap from the computation process.

    verbose: bool
        Controls verbosity.
    Returns
    -------------

    np.ndarray containing the multiscale diffusion components.


    """
    use_eigs = len(np.array(res["EigenValues"])) // 2
    evals = np.array(res["EigenValues"])
    kn = KneeLocator(range(0, len(evals)), evals, S=30,
                     curve='convex', direction='decreasing', interp_method='polynomial')
    if n_eigs == 'knee':
        if kn.knee is None:
            n_eigs = 'max'
        else:
            use_eigs = int(kn.knee)
    if isinstance(n_eigs, int):
        use_eigs = int(n_eigs)
    if use_eigs < 5:
        if verbose:
            print('Raising n_eigs to maximum computed!')
        n_eigs = 'max'
    if n_eigs == 'max':
        use_eigs = int(np.sum(evals > 0, axis=0))
    if not isinstance(use_eigs, int):
        raise Exception('Set `n_eigs` to either \'knee\', \'max\', \'comp_gap\' or an `int` value.')

    # Multiscale
    eigs_idx = list(range(1, int(use_eigs)))
    eig_vals = np.ravel(evals[eigs_idx])

    data = res['EigenVectors'].values[:, eigs_idx] * (eig_vals / (1 - eig_vals))
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
    return data, kn, use_eigs

def decay_plot(evals, n_eigs='knee', verbose=False, curve='convex'):
    """
    Visualize the information decay (scree plot).

    Parameters
    ----------
    evals: np.ndarray.
        Eigenvalues.

    n_eigs: int or str (optional, default 'knee')
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
    use_eigs = len(evals) // 2
    kn = KneeLocator(range(0, len(evals)), evals, S=30,
                     curve=curve, direction='decreasing', interp_method='polynomial')
    if n_eigs == 'knee':
        if kn.knee is None:
            n_eigs = 'max'
        else:
            use_eigs = int(kn.knee)
    if isinstance(n_eigs, int):
        use_eigs = int(n_eigs)
    if use_eigs < 5:
        if verbose:
            print('Raising n_eigs to maximum computed!')
        n_eigs = 'max'
    if n_eigs == 'max':
        use_eigs = int(np.sum(evals > 0, axis=0))
    if not isinstance(use_eigs, int):
        raise Exception('Set `n_eigs` to either \'knee\', \'max\', \'comp_gap\' or an `int` value.')
    eigs_idx = list(range(1, int(use_eigs)))
    eig_vals = np.ravel(evals[eigs_idx])
    kn = KneeLocator(range(0, len(eig_vals)), eig_vals, S=30,
                     curve=curve, direction='decreasing', interp_method='polynomial')


    if not isinstance(kn.knee, int):
        ax1 = plt.subplot(1, 1, 1)
        ax1.set_title('Spectrum decay and eigengap (%i)' % int(use_eigs))
        ax1.plot(kn.x, kn.y, 'b', label='data')
        ax1.set_ylabel('Eigenvalues')
        ax1.set_xlabel('Eigenvectors')
        ax1.vlines(
            use_eigs, plt.ylim()[0], plt.ylim()[1], linestyles="--", label='Eigenvalues'
        )
        ax1.legend(loc='best')
        plt.tight_layout()
    else:
        ax1 = plt.subplot(2, 1, 1)
        ax1.set_title('Spectrum decay and \'knee\' (%i)' % int(kn.knee))
        ax1.plot(kn.x, kn.y, 'b', label='data')
        ax1.set_ylabel('Eigenvalues')
        ax1.set_xlabel('Eigenvectors')
        ax1.vlines(
            kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles="--", label='Knee'
        )
        ax1.legend(loc='best')
        ax2 = plt.subplot(2, 1, 2)
        ax2.set_title('Curve analysis')
        ax2.plot(kn.x_normalized, kn.y_normalized, "b", label="normalized")
        ax2.plot(kn.x_difference, kn.y_difference, "r", label="differential")
        ax2.set_xticks(
            np.arange(kn.x_normalized.min(), kn.x_normalized.max() + 0.1, 0.1)
        )
        ax2.set_yticks(
            np.arange(kn.y_difference.min(), kn.y_normalized.max() + 0.1, 0.1)
        )
        ax2.vlines(
            kn.norm_knee,
            plt.ylim()[0],
            plt.ylim()[1],
            linestyles="--",
            label="Knee",
        )
        ax2.legend(loc="best")
        plt.tight_layout()

    return plt.show()