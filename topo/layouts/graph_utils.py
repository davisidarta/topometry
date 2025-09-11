# These are some graph learning functions implemented in UMAP, added here with modifications
# for better speed and computational efficiency.
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
from scipy.sparse import coo_matrix
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
from topo.base import ann
from topo.base import dists as dist
from topo.spectral.umap_layouts import optimize_layout_euclidean, optimize_layout_generic, optimize_layout_inverse
from topo.spectral.eigen import spectral_layout
from topo.tpgraph.fuzzy import fuzzy_simplicial_set
from topo.utils.umap_utils import ts, fast_knn_indices

try:
    import joblib
except ImportError:
    # sklearn.externals.joblib is deprecated in 0.21, will be removed in 0.23. Try installing joblib.
    from sklearn.externals import joblib

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

def make_epochs_per_sample(weights, n_epochs):
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.
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
    save_every=None,                 # int or None. If int>0, store Y every `save_every` epochs
    save_limit=None,                 # optional cap on number of snapshots kept in-memory
    save_callback=None,              # optional callable(epoch:int, Y:np.ndarray) -> None
    include_init_snapshot=True,      # store epoch=0 (post-init) snapshot
):
    """Perform a fuzzy simplicial set embedding (UMAP/MAP), optionally saving
    intermediate embeddings every few epochs.

    Parameters
    ----------
    graph : sparse matrix (CSR/COO)
        Weighted adjacency of the high-dimensional fuzzy 1-skeleton.
    n_components : int
        Target embedding dimensionality.
    initial_alpha : float
        Initial learning rate for the SGD.
    a, b, gamma, negative_sample_rate : floats/ints
        Standard UMAP/MAP parameters.
    n_epochs : int
        Total optimization epochs. If <=0, a heuristic is used.
    init : {"spectral","random"} or ndarray
        Initialization strategy or explicit initial coordinates.
    random_state : numpy RandomState
        RNG.
    metric, metric_kwds : for densMAP internals
    densmap : bool
        Use density-augmented objective (densMAP).
    densmap_kwds : dict
        densMAP internals (expects "graph_dists" etc. if densMAP/output_dens).
    output_dens : bool
        If True, also compute embedding densities in aux_data.
    output_metric, output_metric_kwds, euclidean_output, parallel, verbose
        As in the original implementation.

    save_every : int or None, optional
        If provided and >0, store the embedding every `save_every` epochs into
        `aux_data["checkpoints"]` as a list of dicts:
            [{"epoch": e, "embedding": Y_e}, ...]
        WARNING: storing many snapshots can be memory intensive. Consider
        passing `save_callback` to stream snapshots to disk.

    save_limit : int or None, optional
        Maximum number of snapshots to keep in-memory in `aux_data`.
        If exceeded, the earliest snapshots are discarded (FIFO).

    save_callback : callable or None, optional
        If provided, called as `save_callback(epoch:int, Y:np.ndarray)` for
        each snapshot. Use this to persist to disk and avoid RAM growth.

    include_init_snapshot : bool, default True
        If True, also store a snapshot at epoch=0 (post initialisation/pre-SGD).

    Returns
    -------
    embedding : (n_samples, n_components) array
        Final optimized embedding.
    aux_data : dict
        Auxiliary outputs. New keys:
            - "checkpoints": list of {"epoch": int, "embedding": np.ndarray}
              (only if `save_every` is set or `include_init_snapshot` is True)
        Existing keys unchanged; when densMAP/output_dens are enabled, includes
        "rad_orig"/"rad_emb" radii etc.
    """
    graph = graph.tocoo()
    graph.sum_duplicates()
    n_vertices = graph.shape[1]

    # Heuristic epochs (kept from original)
    if (n_epochs is None) or (n_epochs <= 0):
        n_epochs = 1000 if graph.shape[0] <= 10000 else 300
        if densmap:
            n_epochs += 200

    # Prune tiny weights (uses total n_epochs as in the original)
    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()

    # ----- Initialisation (unchanged) -----
    if isinstance(init, np.ndarray):
        initialisation = init
        embedding = init
    elif isinstance(init, str) and init == "random":
        embedding = random_state.uniform(
            low=-10.0, high=10.0, size=(graph.shape[0], n_components)
        ).astype(np.float32)
        initialisation = embedding
    elif isinstance(init, str) and init == "spectral":
        initialisation = spectral_layout(
            graph, dim=n_components, random_state=random_state
        )
        expansion = 10.0 / np.abs(initialisation).max()
        embedding = (initialisation * expansion).astype(np.float32) + random_state.normal(
            scale=0.0001, size=[graph.shape[0], n_components]
        ).astype(np.float32)
    else:
        init_data = np.array(init)
        if len(init_data.shape) == 2:
            if np.unique(init_data, axis=0).shape[0] < init_data.shape[0]:
                tree = KDTree(init_data)
                dist_arr, ind = tree.query(init_data, k=2)
                nndist = np.mean(dist_arr[:, 1])
                embedding = init_data + random_state.normal(
                    scale=0.001 * nndist, size=init_data.shape
                ).astype(np.float32)
            else:
                embedding = init_data
        initialisation = embedding

    head = graph.row
    tail = graph.col
    weight = graph.data

    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

    aux_data = {}

    # ----- densMAP original densities (unchanged) -----
    if densmap or output_dens:
        if verbose:
            print(ts() + " Computing original densities")
        dists = densmap_kwds["graph_dists"]

        mu_sum = np.zeros(n_vertices, dtype=np.float32)
        ro = np.zeros(n_vertices, dtype=np.float32)
        for i in range(len(head)):
            j = head[i]
            k = tail[i]
            D = dists[j, k] * dists[j, k]
            mu = graph.data[i]
            ro[j] += mu * D; ro[k] += mu * D
            mu_sum[j] += mu; mu_sum[k] += mu

        epsilon = 1e-8
        ro = np.log(epsilon + (ro / mu_sum))

        if densmap:
            R = (ro - np.mean(ro)) / np.std(ro)
            densmap_kwds["mu"] = graph.data
            densmap_kwds["mu_sum"] = mu_sum
            densmap_kwds["R"] = R

        if output_dens:
            aux_data["rad_orig"] = ro

    # Normalize box (unchanged)
    embedding = (
        10.0 * (embedding - np.min(embedding, 0)) / (np.max(embedding, 0) - np.min(embedding, 0))
    ).astype(np.float32, order="C")

    # ----- NEW: checkpointing support -----
    checkpoints = []
    def _maybe_store(epoch, Y):
        """Store snapshot to memory and/or stream via callback."""
        if save_callback is not None:
            try:
                save_callback(int(epoch), Y)
            except Exception as _e:
                if verbose:
                    print(ts() + f" save_callback failed at epoch {epoch}: {_e}")
        if save_every is not None or include_init_snapshot:
            # Keep an in-memory copy (can be limited)
            snap = {"epoch": int(epoch), "embedding": Y.copy()}
            checkpoints.append(snap)
            if (save_limit is not None) and (len(checkpoints) > int(save_limit)):
                # FIFO drop earliest
                del checkpoints[0]

    # Store init snapshot if requested
    if include_init_snapshot:
        _maybe_store(epoch=0, Y=embedding)

    # Try to use optimizer-level callback if available; otherwise fall back to chunked loop
    def _run_optimizer_chunk(Y, chunk_epochs, epochs_per_sample):
        """Run one chunk of optimization and return updated Y."""
        if euclidean_output:
            return optimize_layout_euclidean(
                Y,
                Y,
                head,
                tail,
                chunk_epochs,
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
            return optimize_layout_generic(
                Y,
                Y,
                head,
                tail,
                chunk_epochs,
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

    # If no checkpointing requested, run once (original behavior)
    if not save_every or int(save_every) <= 0:
        epochs_per_sample = make_epochs_per_sample(weight, n_epochs)
        embedding = _run_optimizer_chunk(embedding, n_epochs, epochs_per_sample)

    else:
        # Chunked optimization loop, taking snapshots every `save_every` epochs.
        save_every = int(save_every)
        total_epochs = int(n_epochs)
        epochs_done = 0
        while epochs_done < total_epochs:
            chunk = min(save_every, total_epochs - epochs_done)
            # IMPORTANT: epochs_per_sample is defined relative to *this chunk*.
            # This approximate schedule yields good practical results and allows checkpointing
            # without modifying the low-level optimizer.
            epochs_per_sample = make_epochs_per_sample(weight, chunk)
            embedding = _run_optimizer_chunk(embedding, chunk, epochs_per_sample)
            epochs_done += chunk
            _maybe_store(epoch=epochs_done, Y=embedding)

    # ----- (unchanged) optional embedding densities -----
    if output_dens:
        if verbose:
            print(ts() + " Computing embedding densities")

        (knn_indices, knn_dists, rp_forest,) = fast_knn_indices(
            embedding,
            densmap_kwds["n_neighbors"],
            metric,
            {},
            False,
            random_state,
            verbose=verbose,
        )

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

        head_e = emb_graph.row
        tail_e = emb_graph.col
        for i in range(len(head_e)):
            j = head_e[i]
            k = tail_e[i]
            D = emb_dists[j, k]
            mu = emb_graph.data[i]
            re[j] += mu * D; re[k] += mu * D
            mu_sum[j] += mu; mu_sum[k] += mu

        epsilon = 1e-8
        re = np.log(epsilon + (re / mu_sum))
        aux_data["rad_emb"] = re

    aux_data["initiasation"] = initialisation  # (kept for BC; note misspelling preserved)
    if (save_every and int(save_every) > 0) or include_init_snapshot:
        aux_data["checkpoints"] = checkpoints

    return embedding, aux_data




def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]
