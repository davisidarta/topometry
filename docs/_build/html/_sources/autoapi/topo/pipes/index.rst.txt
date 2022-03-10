:py:mod:`topo.pipes`
====================

.. py:module:: topo.pipes


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   topo.pipes.global_score
   topo.pipes.local_score
   topo.pipes.eval_models_layouts



.. py:function:: global_score(data, emb)


.. py:function:: local_score(data, emb, k=10, metric='cosine', n_jobs=12, data_is_graph=False, emb_is_graph=False)


.. py:function:: eval_models_layouts(TopOGraph, X, subsample=None, k=None, n_jobs=None, metric=None, bases=['diffusion', 'fuzzy', 'continuous'], graphs=['diff', 'cknn', 'fuzzy'], layouts=['tSNE', 'MAP', 'MDE', 'PaCMAP', 'TriMAP', 'NCVis'])

   Evaluates all orthogonal bases, topological graphs and layouts in the TopOGraph object.
   Compares results with PCA and PCA-derived layouts (i.e. t-SNE, UMAP etc).

   :param TopOGraph:
   :type TopOGraph: target TopOGraph object (can be empty).
   :param X:
   :type X: data matrix. Expects either numpy.ndarray or scipy.sparse.csr_matrix.
   :param subsample: If specified, subsamples the TopOGraph object and/or data matrix X to a number of samples
                     before computing results and scores. Useful if dealing with large datasets (>50,000 samples).
   :type subsample: optional (int, default None).
   :param k: Number of k-neighbors to use for evaluating results. Defaults to TopOGraph.base_knn.
   :type k: optional (int, default None).
   :param n_jobs: Number of threads to use in computations. Defaults to TopOGraph.n_jobs (default 1).
   :type n_jobs: optional (int, default None).
   :param metric: Distance metric to use. Defaults to TopOGraph.base_metric (default 'cosine').
   :type metric: optional (str, default None).

   bases : str (optional, default ['diffusion', 'continuous','fuzzy']).
       Which bases to compute. Defaults to all. To run only one or two bases, set it to
       ['fuzzy', 'diffusion'] or ['continuous'], for exemple.

   graphs : str (optional, default ['diff', 'cknn','fuzzy']).
       Which graphs to compute. Defaults to all. To run only one or two graphs, set it to
       ['fuzzy', 'diff'] or ['cknn'], for exemple.

   layouts : str (optional, default all ['tSNE', 'MAP', 'MDE', 'PaCMAP', 'TriMAP', 'NCVis']).
       Which layouts to compute. Defaults to all 6 options within TopOMetry: tSNE, MAP, MDE, PaCMAP,
       TriMAP and NCVis. To run only one or two layouts, set it to
       ['tSNE', 'MAP'] or ['PaCMAP'], for example.

   :returns: * *Populates the TopOGraph object and returns a list of lists (results)*
             * *- results[0] contains orthogonal bases scores.*
             * *- results[1] contains topological graph scores.*
             * *- results[2] contains layout scores.*


