:py:mod:`topo.tpgraph.multiscale`
=================================

.. py:module:: topo.tpgraph.multiscale


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   topo.tpgraph.multiscale.multiscale
   topo.tpgraph.multiscale.decay_plot



.. py:function:: multiscale(res, n_eigs='max', verbose=True)

   Learn multiscale maps from the diffusion basis.
   :param verbose:
   :param res: Results from the dbMAP framework. Expects dictionary containing numerical
               'EigenVectors' and 'EigenValues'.
   :type res: dict
   :param n_eigs: Number of eigenvectors to use. If 'max', expands to the maximum number of positive eigenvalues
                  (reach of numerical precision), else to the maximum amount of computed components.
                  If 'knee', uses Kneedle to find an optimal cutoff point, and expands it by ``expansion``.
                  If 'comp_gap', tries to find a discrete eigengap from the computation process.
   :type n_eigs: int or str (optional, default 'max')
   :param verbose: Controls verbosity.
   :type verbose: bool

   :returns: *np.ndarray containing the multiscale diffusion components.*


.. py:function:: decay_plot(evals, n_eigs='knee', verbose=False, curve='convex')

   Visualize the information decay (scree plot).

   :param evals: Eigenvalues.
   :type evals: np.ndarray.
   :param n_eigs: Number of eigenvectors to use. If 'max', expands to the maximum number of positive eigenvalues
                  (reach of numerical precision), else to the maximum amount of computed components.
                  If 'knee', uses Kneedle to find an optimal cutoff point, and expands it by ``expansion``.
                  If 'comp_gap', tries to find a discrete eigengap from the computation process.
   :type n_eigs: int or str (optional, default 'knee')
   :param verbose: Controls verbosity.
   :type verbose: bool

   :returns: *np.ndarray containing the multiscale diffusion components.*


