:py:mod:`topo.tpgraph.diffusion`
================================

.. py:module:: topo.tpgraph.diffusion


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   topo.tpgraph.diffusion.Diffusor




Attributes
~~~~~~~~~~

.. autoapisummary::

   topo.tpgraph.diffusion._have_hnswlib
   topo.tpgraph.diffusion._have_nmslib


.. py:data:: _have_hnswlib
   :annotation: = True

   

.. py:data:: _have_nmslib
   :annotation: = True

   

.. py:class:: Diffusor(n_neighbors=10, n_eigs=50, use_eigs='max', metric='cosine', kernel_use='simple', eigen_expansion=True, plot_spectrum=False, verbose=False, cache=False, alpha=1, n_jobs=10, backend='nmslib', p=None, M=15, efC=50, efS=50, norm=False, transitions=True)

   Bases: :py:obj:`sklearn.base.TransformerMixin`

   Sklearn-compatible estimator for using fast anisotropic diffusion with an adaptive neighborhood search algorithm. The
   Diffusion Maps algorithm was initially proposed by Coifman et al in 2005, and was augmented by the work of many.
   This implementation aggregates recent advances in diffusion harmonics, and innovates only by implementing an
   adaptively decaying kernel (the rate of decay is dependent on neighborhood density)
   and an adaptive neighborhood estimation approach.

   :param n_eigs: Number of diffusion components to compute. This number can be iterated to get different views
                  from data at distinct spectral resolution.
   :type n_eigs: int (optional, default 50)
   :param use_eigs: Number of eigenvectors to use. If 'max', expands to the maximum number of positive eigenvalues
                    (reach of numerical precision), else to the maximum amount of computed components.
                    If 'knee', uses Kneedle to find an optimal cutoff point, and expands it by ``expansion``.
                    If 'comp_gap', tries to find a discrete eigengap from the computation process.
   :type use_eigs: int or str (optional, default 'knee')

   n_neighbors : int (optional, default 10)
       Number of k-nearest-neighbors to compute. The adaptive kernel will normalize distances by each cell
       distance of its median neighbor. Nonetheless, this hyperparameter remains as an user input regarding
       the minimal sample neighborhood resolution that drives the computation of the diffusion metrics. For
       practical purposes, the minimum amount of samples one would expect to constitute a neighborhood of its
       own. Increasing `k` can generate more globally-comprehensive metrics and maps, to a certain extend,
       however at the expense of fine-grained resolution. More generally, consider this a calculus
       discretization threshold.

   backend : str (optional, default 'hnwslib')
       Which backend to use to compute nearest-neighbors. Options for fast, approximate nearest-neighbors
       are 'hnwslib' (default) and 'nmslib'. For exact nearest-neighbors, use 'sklearn'.

   metric : str (optional, default 'cosine')
       Distance metric for building an approximate kNN graph. Defaults to
       'cosine'. Users are encouraged to explore different metrics, such as 'euclidean' and 'inner_product'.
       The 'hamming' and 'jaccard' distances are also available for string vectors.
        Accepted metrics include NMSLib*, HNSWlib** and sklearn metrics. Some examples are:

       -'sqeuclidean' (*, **)

       -'euclidean' (*, **)

       -'l1' (*)

       -'lp' - requires setting the parameter ``p`` (*) - similar to Minkowski

       -'cosine' (*, **)

       -'inner_product' (**)

       -'angular' (*)

       -'negdotprod' (*)

       -'levenshtein' (*)

       -'hamming' (*)

       -'jaccard' (*)

       -'jansen-shan' (*)

   p : int or float (optional, default 11/16 )
       P for the Lp metric, when ``metric='lp'``.  Can be fractional. The default 11/16 approximates
       an astroid norm with some computational efficiency (2^n bases are less painstakinly slow to compute).
       See https://en.wikipedia.org/wiki/Lp_space for some context.

   transitions : bool (optional, default False)
       Whether to estimate the diffusion transitions graph. If `True`, maps a basis encoding neighborhood
        transitions probability during eigendecomposition. If 'False' (default), maps the diffusion kernel.

   alpha : int or float (optional, default 1)
       Alpha in the diffusion maps literature. Controls how much the results are biased by data distribution.
           Defaults to 1, which is suitable for normalized data.

   kernel_use : str (optional, default 'decay_adaptive')
       Which type of kernel to use. There are four implemented, considering the adaptive decay and the
       neighborhood expansion, written as 'simple', 'decay', 'simple_adaptive' and 'decay_adaptive'. The first, 'simple'
       , is a locally-adaptive kernel similar to that proposed by Nadler et al.(https://doi.org/10.1016/j.acha.2005.07.004)
       and implemented in Setty et al. (https://doi.org/10.1038/s41587-019-0068-4). The 'decay' option applies an
       adaptive decay rate, but no neighborhood expansion. Those, followed by '_adaptive', apply the neighborhood expansion process.
        The default and recommended is 'decay_adaptive'.
       The neighborhood expansion can impact runtime, although this is not usually expressive for datasets under 10e6 samples.

   transitions : bool (optional, default False)
       Whether to decompose the transition graph when transforming.
   norm : bool (optional, default True)
       Whether to normalize the kernel transition probabilities to approximate the LPO.
   eigen_expansion : bool (optional, default False)
       Whether to expand the eigendecomposition and stop near a discrete eigengap (bit limit).
   n_jobs : int (optional, default 4)
       Number of threads to use in calculations. Defaults to 4 for safety, but performance
       scales dramatically when using more threads.
   plot_spectrum : bool (optional, default False)
       Whether to plot the spectrum decay analysis.
   verbose : bool (optional, default False)
       Controls verbosity.
   cache : bool (optional, default True)
       Whether to cache nearest-neighbors (before fit) and to store diffusion matrices after mapping (before transform).

   .. rubric:: Example

   import numpy as np
   from sklearn.datasets import load_digits
   from scipy.sparse import csr_matrix
   from topo.tpgraph.diffusion import Diffusor

   digits = load_digits()
   data = csr_matrix(digits)

   diff = Diffusor().fit(data)

   msdiffmap = diff.transform(data)

   .. py:method:: __repr__(self)

      Return repr(self).


   .. py:method:: fit(self, X)

      Fits an adaptive anisotropic diffusion kernel to the data.

      :param X: input data. Takes in numpy arrays and scipy csr sparse matrices.
      :param Use with sparse data for top performance. You can adjust a series of:
      :param parameters that can make the process faster and more informational depending:
      :param on your dataset.:

      :returns: *Diffusor object with kernel Diffusor.K and the transition potencial Diffusor.T .*


   .. py:method:: transform(self, X)

       Fits the renormalized Laplacian approximating the Laplace Beltrami-Operator
       in a discrete eigendecomposition. Then multiscales the resulting components.
       Parameters
       ----------
       X :
           input data. Takes in numpy arrays and scipy csr sparse matrices.
       Use with sparse data for top performance. You can adjust a series of
       parameters that can make the process faster and more informational depending
       on your dataset.

       Returns
       -------

      ``Diffusor.res['MultiscaleComponents']]``




   .. py:method:: ind_dist_grad(self, data)

      Utility function to get indices, distances and gradients from a multiscale diffusion map.

      :param data: Input data matrix (numpy array, pandas df, csr_matrix).

      :returns: *A tuple containing neighborhood indices, distances, gradient and a knn graph.*


   .. py:method:: res_dict(self)

      :returns: * *Dictionary containing normalized and multiscaled Diffusion Components*
                * *(Diffusor.res['StructureComponents']), their eigenvalues['EigenValues'] and*
                * *non - multiscaled components(['EigenVectors']).*


   .. py:method:: rescale(self, n_eigs=None)

      Re-scale the multiscale procedure to a new number of components.

      :param self:
      :type self: Diffusor object.
      :param n_eigs:
      :type n_eigs: int. Number of diffusion components to multiscale.

      :returns: *np.ndarray containing the new multiscaled basis.*


   .. py:method:: spectrum_plot(self, bla=None)

      Plot the decay spectra.

      :param self:
      :type self: Diffusor object.
      :param bla:
      :type bla: Here only for autodoc's sake.

      :returns: *A nice plot of the diffusion spectra.*



