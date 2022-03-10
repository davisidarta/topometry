:py:mod:`topo.plot`
===================

.. py:module:: topo.plot


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   topo.plot.scatter
   topo.plot.scatter3d
   topo.plot.hyperboloid
   topo.plot.two_to_3d_hyperboloid
   topo.plot.poincare
   topo.plot.sphere
   topo.plot.sphere_projection
   topo.plot.toroid
   topo.plot.draw_simple_ellipse
   topo.plot.gaussian_potential
   topo.plot.eval_gaussian
   topo.plot.eval_density_at_point
   topo.plot.get_cmap
   topo.plot.create_density_plot
   topo.plot.plot_bases_scores
   topo.plot.plot_graphs_scores
   topo.plot.plot_layouts_scores
   topo.plot.plot_all_layouts



.. py:function:: scatter(res, labels=None, pt_size=5, marker='o', opacity=1, cmap='Spectral')

   Basic scatter plot function.

   :param labels:
   :param pt_size:
   :param marker:
   :param opacity:
   :param cmap:


.. py:function:: scatter3d(res, labels=None, pt_size=5, marker='o', opacity=1, cmap='Spectral')


.. py:function:: hyperboloid(emb, labels=None, pt_size=5, marker='o', opacity=1, cmap='Spectral')


.. py:function:: two_to_3d_hyperboloid(emb)


.. py:function:: poincare(emb, labels=None, pt_size=5, marker='o', opacity=1, cmap='Spectral')


.. py:function:: sphere(emb, labels=None, pt_size=5, marker='o', opacity=1, cmap='Spectral')


.. py:function:: sphere_projection(emb, labels=None, pt_size=5, marker='o', opacity=1, cmap='Spectral')


.. py:function:: toroid(emb, R=3, r=1, labels=None, pt_size=5, marker='o', opacity=1, cmap='Spectral')


.. py:function:: draw_simple_ellipse(position, width, height, angle, ax=None, from_size=0.1, to_size=0.5, n_ellipses=3, alpha=0.1, color=None)


.. py:function:: gaussian_potential(emb, dims=[2, 3, 4], labels=None, pt_size=5, marker='o', opacity=1, cmap='Spectral')


.. py:function:: eval_gaussian(x, pos=np.array([0, 0]), cov=np.eye(2, dtype=np.float32))


.. py:function:: eval_density_at_point(x, embedding)


.. py:function:: get_cmap(n, name='hsv')


.. py:function:: create_density_plot(X, Y, embedding)


.. py:function:: plot_bases_scores(bases_scores, return_plot=True, figsize=(20, 8), fontsize=20)


.. py:function:: plot_graphs_scores(graphs_scores, return_plot=True, figsize=(20, 8), fontsize=20)


.. py:function:: plot_layouts_scores(layouts_scores, return_plot=True, figsize=(20, 8), fontsize=20)


.. py:function:: plot_all_layouts(TopOGraph, labels=None, pt_size=5, marker='o', opacity=1, cmap='Spectral')


