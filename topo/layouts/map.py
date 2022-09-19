# These are some adapted functions implemented in UMAP, added here with some changes
# Originally implemented by Leland McInnes at https://github.com/lmcinnes/umap
# License: BSD 3 clause
#
# For more information on the original UMAP implementation, please see: https://umap-learn.readthedocs.io/
#
# BSD 3-Clause License
#
# Copyright (c) 2017, Leland McInnes
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from scipy.optimize import curve_fit
from topo.base import dists as dist
from sklearn.neighbors import KDTree
from topo.spectral import _spectral, umap_layouts
from topo.tpgraph.fuzzy import fuzzy_simplicial_set
from topo.utils.umap_utils import ts, csr_unique, fast_knn_indices
from topo.spectral.umap_layouts import optimize_layout_euclidean, optimize_layout_generic, optimize_layout_inverse


try:
    import joblib
except ImportError:
    # sklearn.externals.joblib is deprecated in 0.21, will be removed in 0.23. Try installing joblib.
    from sklearn.externals import joblib

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf
INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


def make_epochs_per_sample(weights, n_epochs):
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.

    Originally implemented by Leland McInnes at https://github.com/lmcinnes/umap
    under the BSD 3-Clause License.

    Parameters
    ----------
    weights: array of shape (n_1_simplices)
        The weights ofhow much we wish to sample each 1-simplex.
    n_epochs: int
        The total number of epochs we want to train for.
    Returns
    -------
    An array of number of epochs per sample, one for each 1-simplex.
    """
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / n_samples[n_samples > 0]
    return result


def simplicial_set_embedding(
    graph,
    n_components,
    initial_alpha,
    a,
    b,
    gamma,
    negative_sample_rate,
    n_epochs,
    init,
    random_state,
    metric,
    metric_kwds,
    densmap,
    densmap_kwds,
    output_dens,
    output_metric=dist.named_distances_with_gradients["euclidean"],
    output_metric_kwds={},
    euclidean_output=True,
    parallel=True,
    verbose=False,
):
    """
    Perform a fuzzy simplicial set embedding, using a specified
    initialisation method and then minimizing the fuzzy set cross entropy
    between the 1-skeletons of the high and low dimensional fuzzy simplicial
    sets.

    Originally implemented by Leland McInnes at https://github.com/lmcinnes/umap
    under the BSD 3-Clause License.

    Parameters
    ----------
    graph: sparse matrix
        The 1-skeleton of the high dimensional fuzzy simplicial set as
        represented by a graph for which we require a sparse matrix for the
        (weighted) adjacency matrix.
    n_components: int
        The dimensionality of the euclidean space into which to embed the data.
    initial_alpha: float
        Initial learning rate for the SGD.
    a: float
        Parameter of differentiable approximation of right adjoint functor
    b: float
        Parameter of differentiable approximation of right adjoint functor
    gamma: float
        Weight to apply to negative samples.
    negative_sample_rate: int (optional, default 5)
        The number of negative samples to select per positive sample
        in the optimization process. Increasing this value will result
        in greater repulsive force being applied, greater optimization
        cost, but slightly more accuracy.
    n_epochs: int (optional, default 0)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If 0 is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).
    init: string
        How to initialize the low dimensional embedding. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * A numpy array of initial embedding positions.
    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.
    metric: string or callable
        The metric used to measure distance in high dimensional space; used if
        multiple connected components need to be layed out.
    metric_kwds: dict
        Key word arguments to be passed to the metric function; used if
        multiple connected components need to be layed out.
    densmap: bool
        Whether to use the density-augmented objective function to optimize
        the embedding according to the densMAP algorithm.
    densmap_kwds: dict
        Key word arguments to be used by the densMAP optimization.
    output_dens: bool
        Whether to output local radii in the original data and the embedding.
    output_metric: function
        Function returning the distance between two points in embedding space and
        the gradient of the distance wrt the first argument.
    output_metric_kwds: dict
        Key word arguments to be passed to the output_metric function.
    euclidean_output: bool
        Whether to use the faster code specialised for euclidean output metrics
    parallel: bool (optional, default False)
        Whether to run the computation using numba parallel.
        Running in parallel is non-deterministic, and is not used
        if a random seed has been set, to ensure reproducibility.
    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.
    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized of ``graph`` into an ``n_components`` dimensional
        euclidean space.
    aux_data: dict
        Auxiliary output returned with the embedding. When densMAP extension
        is turned on, this dictionary includes local radii in the original
        data (``rad_orig``) and in the embedding (``rad_emb``).
    """
    graph = graph.tocoo()
    graph.sum_duplicates()
    n_vertices = graph.shape[1]
    if (n_epochs <= 0) or (n_epochs is None):
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 1000
        else:
            n_epochs = 300

        # Use more epochs for densMAP
        if densmap:
            n_epochs += 200

    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()

    if isinstance(init, np.ndarray):
        initialisation = init
        embedding = init
    elif isinstance(init, str) and init == "random":
        embedding = random_state.uniform(
            low=-10.0, high=10.0, size=(graph.shape[0], n_components)
        ).astype(np.float32)
        initialisation = embedding
    elif isinstance(init, str) and init == "spectral":
        # We add a little noise to avoid local minima for optimization to come
        initialisation = _spectral.spectral_layout(
            graph,
            dim=n_components,
            random_state=random_state
        )
        expansion = 10.0 / np.abs(initialisation).max()
        embedding = (initialisation * expansion).astype(
            np.float32
        ) + random_state.normal(
            scale=0.0001, size=[graph.shape[0], n_components]
        ).astype(
            np.float32
        )
    else:
        init_data = np.array(init)
        if len(init_data.shape) == 2:
            if np.unique(init_data, axis=0).shape[0] < init_data.shape[0]:
                tree = KDTree(init_data)
                dist, ind = tree.query(init_data, k=2)
                nndist = np.mean(dist[:, 1])
                embedding = init_data + random_state.normal(
                    scale=0.001 * nndist, size=init_data.shape
                ).astype(np.float32)
            else:
                embedding = init_data

    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

    head = graph.row
    tail = graph.col
    weight = graph.data

    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

    aux_data = {}

    if densmap or output_dens:
        if verbose:
            print(ts() + " Computing original densities")

        dists = densmap_kwds["graph_dists"]

        mu_sum = np.zeros(n_vertices, dtype=np.float32)
        ro = np.zeros(n_vertices, dtype=np.float32)
        for i in range(len(head)):
            j = head[i]
            k = tail[i]

            D = dists[j, k] * dists[j, k]  # match sq-Euclidean used for embedding
            mu = graph.data[i]

            ro[j] += mu * D
            ro[k] += mu * D
            mu_sum[j] += mu
            mu_sum[k] += mu

        epsilon = 1e-8
        ro = np.log(epsilon + (ro / mu_sum))

        if densmap:
            R = (ro - np.mean(ro)) / np.std(ro)
            densmap_kwds["mu"] = graph.data
            densmap_kwds["mu_sum"] = mu_sum
            densmap_kwds["R"] = R

        if output_dens:
            aux_data["rad_orig"] = ro

    embedding = (
        10.0
        * (embedding - np.min(embedding, 0))
        / (np.max(embedding, 0) - np.min(embedding, 0))
    ).astype(np.float32, order="C")

    if euclidean_output:
        embedding = optimize_layout_euclidean(
            embedding,
            embedding,
            head,
            tail,
            n_epochs,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            initial_alpha,
            negative_sample_rate,
            parallel=parallel,
            verbose=verbose,
            densmap=densmap,
            densmap_kwds=densmap_kwds,
        )
    else:
        embedding = optimize_layout_generic(
            embedding,
            embedding,
            head,
            tail,
            n_epochs,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            initial_alpha,
            negative_sample_rate,
            output_metric,
            tuple(output_metric_kwds.values()),
            verbose=verbose,
        )

    if output_dens:
        if verbose:
            print(ts() + " Computing embedding densities")

        # Compute graph in embedding
        (knn_indices, knn_dists, rp_forest,) = fast_knn_indices(
            embedding,
            densmap_kwds["n_neighbors"],
            metric,
            {},
            False,
            random_state,
            verbose=verbose,
        )

        # TODO: USE SIGMAS, RHOS AND DISTS TO CREATE A 'METRIC' OF NON-LINEAR DISTORTION
        emb_graph, emb_sigmas, emb_rhos, emb_dists = fuzzy_simplicial_set(
            embedding,
            densmap_kwds["n_neighbors"],
            random_state,
            metric,
            {},
            knn_indices,
            knn_dists,
            verbose=verbose,
            return_dists=True,
        )

        emb_graph = emb_graph.tocoo()
        emb_graph.sum_duplicates()
        emb_graph.eliminate_zeros()

        n_vertices = emb_graph.shape[1]

        mu_sum = np.zeros(n_vertices, dtype=np.float32)
        re = np.zeros(n_vertices, dtype=np.float32)

        head = emb_graph.row
        tail = emb_graph.col
        for i in range(len(head)):
            j = head[i]
            k = tail[i]

            D = emb_dists[j, k]
            mu = emb_graph.data[i]

            re[j] += mu * D
            re[k] += mu * D
            mu_sum[j] += mu
            mu_sum[k] += mu

        epsilon = 1e-8
        re = np.log(epsilon + (re / mu_sum))

        aux_data["rad_emb"] = re
    aux_data["initiasation"] = initialisation
    return embedding, aux_data



def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    Originally implemented by Leland McInnes at https://github.com/lmcinnes/umap
    under the BSD 3-Clause License.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]

def fuzzy_embedding(graph,
                    n_components=2,
                    initial_alpha=1,
                    min_dist=0.3,
                    spread=1.2,
                    n_epochs=500,
                    metric='cosine',
                    metric_kwds={},
                    output_metric='euclidean',
                    output_metric_kwds={},
                    gamma=1.2,
                    negative_sample_rate=10,
                    init='spectral',
                    random_state=None,
                    euclidean_output=True,
                    parallel=True,
                    verbose=False,
                    a=None,
                    b=None,
                    densmap=False,
                    densmap_kwds={},
                    output_dens=False,
                    ):
    """\
    Perform a fuzzy simplicial set embedding, using a specified
    initialisation method and then minimizing the fuzzy set cross entropy
    between the 1-skeletons of the high and low dimensional fuzzy simplicial
    sets. 

    Originally implemented by Leland McInnes at https://github.com/lmcinnes/umap
    under the BSD 3-Clause License.

    Parameters
    ----------
    graph : sparse matrix
        The 1-skeleton of the high dimensional fuzzy simplicial set as
        represented by a graph for which we require a sparse matrix for the
        (weighted) adjacency matrix.
    n_components : int
        The dimensionality of the euclidean space into which to embed the data.
    initial_alpha : float
        Initial learning rate for the SGD.
    min_dist : float (optional, default 0.3)
        The effective minimum distance between embedded points. Smaller values will result in a more
        clustered/clumped embedding where nearby points on the manifold are drawn closer together,
        while larger values will result on a more even dispersal of points. The value should be set
        relative to the spread value, which determines the scale at which embedded points will be spread out.
    spread : float (optional, default 1.0)
        The effective scale of embedded points. In combination with min_dist this determines
        how clustered/clumped the embedded points are.
    gamma : float
        Weight to apply to negative samples.
    negative_sample_rate : int (optional, default 5)
        The number of negative samples to select per positive sample
        in the optimization process. Increasing this value will result
        in greater repulsive force being applied, greater optimization
        cost, but slightly more accuracy.
    n_epochs : int (optional, default 0)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If 0 is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).
    init : string
        How to initialize the low dimensional embedding. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * A numpy array of initial embedding positions.
    random_state : numpy RandomState or equivalent
        A state capable being used as a numpy random state.
    metric : string or callable
        The metric used to measure distance in high dimensional space; used if
        multiple connected components need to be layed out.
    metric_kwds : dict
        Key word arguments to be passed to the metric function; used if
        multiple connected components need to be layed out.
    densmap : bool
        Whether to use the density-augmented objective function to optimize
        the embedding according to the densMAP algorithm.
    densmap_kwds : dict
        Key word arguments to be used by the densMAP optimization.
    output_dens : bool
        Whether to output local radii in the original data and the embedding.
    output_metric : function
        Function returning the distance between two points in embedding space and
        the gradient of the distance wrt the first argument.
    output_metric_kwds : dict
        Key word arguments to be passed to the output_metric function.
    euclidean_output : bool
        Whether to use the faster code specialised for euclidean output metrics
    parallel : bool (optional, default False)
        Whether to run the computation using numba parallel.
        Running in parallel is non-deterministic, and is not used
        if a random seed has been set, to ensure reproducibility.
    verbose : bool (optional, default False)
        Whether to report information on the current progress of the algorithm.
    a : float
        Parameter of differentiable approximation of right adjoint functor
    b : float
        Parameter of differentiable approximation of right adjoint functor
    Returns
    -------
    embedding : array of shape (n_samples, n_components)
        The optimized of ``graph`` into an ``n_components`` dimensional
        euclidean space.
    aux_data : dict
        Auxiliary output returned with the embedding. When densMAP extension
        is turned on, this dictionary includes local radii in the original
        data (``rad_orig``) and in the embedding (``rad_emb``).
            Y_init : array of shape (n_samples, n_components)
                The spectral initialization of ``graph`` into an ``n_components`` dimensional
                euclidean space.
    """

    if random_state is None:
        random_state = np.random.RandomState()
    if metric_kwds is None:
        _metric_kwds = {}
    else:
        _metric_kwds = metric_kwds
    if output_metric_kwds is None:
        _output_metric_kwds = {}

    # Compat for umap 0.4 -> 0.5
    if a is None or b is None:
        a, b = find_ab_params(spread, min_dist)
    else:
        a = a
        b = b
        # the data matrix X is really only used for determining the number of connected components
        # for the init condition in the UMAP embedding (high-resolution spectral layout)

    Y, Y_aux = simplicial_set_embedding(graph,
                                        n_components,
                                        initial_alpha,
                                        a,
                                        b,
                                        gamma,
                                        negative_sample_rate,
                                        n_epochs,
                                        init,
                                        random_state,
                                        metric,
                                        metric_kwds,
                                        densmap,
                                        densmap_kwds,
                                        output_dens,
                                        output_metric,
                                        output_metric_kwds,
                                        euclidean_output,
                                        parallel,
                                        verbose)

    return Y, Y_aux
