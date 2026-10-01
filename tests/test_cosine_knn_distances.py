"""
Regression tests for the cosine kNN inversion and the backend fixes that came with it.

`topo.base.ann.kNN` used to return `1 - d` for cosine, i.e. cosine *similarity*, while
everything downstream treated the graph as distances. Within each neighborhood the closest
points therefore got the lowest kernel weights - on TopOGraph's default path, since
`base_metric` defaults to 'cosine'.

Covers:
  - kNN returns true distances for every backend (cosine, and euclidean on nmslib's
    dense index, which reports squared L2).
  - Kernel weights decrease with distance, for Kernel(metric='cosine') and for the
    TopOGraph base kernel, and the two agree.
  - Fitting a cosine kernel leaves the caller's data alone.
  - A standalone Projector hands similarities to its affinity readers and distances to
    Isomap.
  - Euclidean graphs and kernels are untouched.
  - A dense nmslib index can be queried with the sparse rows it was fitted on.
  - Repeated queries on one transformer keep the same k.
  - Scaffold sizing works on the sklearn fallback, and with a cosine metric.
  - tp.sc.fit_adata skips a projection whose optional dependency is missing.
"""
import sys
import warnings

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.neighbors import NearestNeighbors

from topo.base.ann import kNN
from topo.topograph import TopOGraph
from topo.tpgraph.intrinsic_dim import automated_scaffold_sizing
from topo.tpgraph.kernels import Kernel


def _has(module):
    try:
        __import__(module)
        return True
    except ImportError:
        return False


requires_hnswlib = pytest.mark.skipif(not _has("hnswlib"), reason="hnswlib not installed")
requires_nmslib = pytest.mark.skipif(not _has("nmslib"), reason="nmslib not installed")

# (backend, sparse input) for every neighbor search TopoMetry can run
KNN_CASES = [
    pytest.param("sklearn", False, id="sklearn"),
    pytest.param("hnswlib", False, id="hnswlib", marks=requires_hnswlib),
    pytest.param("nmslib", False, id="nmslib-dense", marks=requires_nmslib),
    pytest.param("nmslib", True, id="nmslib-sparse", marks=requires_nmslib),
]
# TopOGraph resolves backend='nmslib' to hnswlib or sklearn, so only these two reach it
TOPOGRAPH_BACKENDS = [
    pytest.param("sklearn", id="sklearn"),
    pytest.param("hnswlib", id="hnswlib", marks=requires_hnswlib),
]

K = 15


def _blobs(n=400, p=40, n_clusters=5, seed=0):
    """Overlapping clusters: close enough that the kNN graph stays connected."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=0.5, size=(n_clusters, p))
    return centers[rng.integers(0, n_clusters, n)] + rng.normal(size=(n, p))


def _sphere(n=4000, dim=5, ambient=40, seed=0):
    """Points uniform on a `dim`-sphere, isometrically embedded in `ambient` dimensions."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, dim + 1))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    Q, _ = np.linalg.qr(rng.normal(size=(ambient, dim + 1)))
    return Z @ Q.T


def _edges(G):
    """Row index, column index and value of every stored entry, explicit zeros included."""
    G = G.tocsr()
    rows = np.repeat(np.arange(G.shape[0]), np.diff(G.indptr))
    return rows, G.indices, G.data


def _weight_vs_distance(W, D):
    """
    Per-row agreement between kernel weights and true distances, over each row's
    neighbors: the median Spearman correlation, and the fraction of rows whose closest
    neighbor holds the largest weight.
    """
    W = W.tocsr()
    rhos, closest_is_heaviest = [], []
    for i in range(W.shape[0]):
        idx = W.indices[W.indptr[i]:W.indptr[i + 1]]
        w = W.data[W.indptr[i]:W.indptr[i + 1]]
        keep = idx != i
        idx, w = idx[keep], w[keep]
        d = D[i, idx]
        rhos.append(spearmanr(d, w)[0])
        closest_is_heaviest.append(w[np.argmin(d)] == w.max())
    return np.median(rhos), np.mean(closest_is_heaviest)


def _topograph(backend, **kwargs):
    params = dict(base_knn=K, graph_knn=K, min_eigs=20, id_min_components=8,
                  id_max_components=20, n_jobs=1, backend=backend,
                  projection_methods=None, verbosity=0, random_state=0)
    params.update(kwargs)
    return TopOGraph(**params)


# ── kNN returns distances ─────────────────────────────────────────────────────
@pytest.mark.parametrize("backend, sparse_input", KNN_CASES)
def test_knn_cosine_returns_cosine_distances(backend, sparse_input):
    X = _blobs()
    G = kNN(csr_matrix(X) if sparse_input else X, n_neighbors=K, metric="cosine",
            backend=backend, n_jobs=1)
    rows, cols, vals = _edges(G)
    D = cosine_distances(X)

    np.testing.assert_allclose(vals, D[rows, cols], atol=1e-5)
    assert vals.min() >= 0.0 and vals.max() <= 2.0
    # a point is at distance exactly zero from itself, not at rounding-noise distance
    assert (rows == cols).sum() == X.shape[0]
    assert np.all(vals[rows == cols] == 0.0)

    # the stored neighbors are the nearest ones, not the farthest
    nearest = np.argsort(D, axis=1)[:, 1]
    G = G.tocsr()
    found = [nearest[i] in G.indices[G.indptr[i]:G.indptr[i + 1]] for i in range(X.shape[0])]
    assert np.mean(found) > 0.95


@pytest.mark.parametrize("backend, sparse_input", KNN_CASES)
def test_knn_euclidean_returns_euclidean_distances(backend, sparse_input):
    """nmslib's dense 'l2' HNSW index reports squared distances; kNN must not pass them on."""
    X = _blobs()
    G = kNN(csr_matrix(X) if sparse_input else X, n_neighbors=K, metric="euclidean",
            backend=backend, n_jobs=1)
    rows, cols, vals = _edges(G)
    np.testing.assert_allclose(vals, euclidean_distances(X)[rows, cols], rtol=1e-4, atol=1e-4)


@requires_nmslib
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_nmslib_sqeuclidean_returns_squared_distances(sparse_input):
    X = _blobs()
    G = kNN(csr_matrix(X) if sparse_input else X, n_neighbors=K, metric="sqeuclidean",
            backend="nmslib", n_jobs=1)
    rows, cols, vals = _edges(G)
    np.testing.assert_allclose(vals, euclidean_distances(X)[rows, cols] ** 2, rtol=1e-4, atol=1e-3)


# ── kernel weights follow distance ────────────────────────────────────────────
@pytest.mark.parametrize("backend, sparse_input", KNN_CASES)
def test_cosine_kernel_weights_decrease_with_distance(backend, sparse_input):
    X = _blobs()
    W = Kernel(metric="cosine", n_neighbors=K, backend=backend, n_jobs=1).fit(
        csr_matrix(X) if sparse_input else X).K
    rho, closest_is_heaviest = _weight_vs_distance(W, cosine_distances(X))
    # with the inversion these were about +0.6 and 0.0; symmetrization keeps the fixed
    # values short of -1 and 1
    assert rho < -0.6, f"weights do not decrease with distance: Spearman {rho:+.2f}"
    assert closest_is_heaviest > 0.5


@pytest.mark.parametrize("backend", TOPOGRAPH_BACKENDS)
def test_topograph_base_kernel_weights_decrease_with_distance(backend):
    """The default path: base_metric='cosine', kernel built from the kNN graph as 'precomputed'."""
    X = _blobs()
    tg = _topograph(backend).fit(X)
    assert tg.base_metric == "cosine"

    rows, cols, vals = _edges(tg.base_knn_graph)
    np.testing.assert_allclose(vals, cosine_distances(X)[rows, cols], atol=1e-5)

    rho, closest_is_heaviest = _weight_vs_distance(tg.base_kernel.K, cosine_distances(X))
    assert rho < -0.6, f"weights do not decrease with distance: Spearman {rho:+.2f}"
    assert closest_is_heaviest > 0.5


@pytest.mark.parametrize("backend", TOPOGRAPH_BACKENDS)
def test_topograph_base_kernel_matches_cosine_kernel(backend):
    """TopOGraph's precomputed route and Kernel(metric='cosine') build the same kernel."""
    X = _blobs()
    W_topograph = _topograph(backend).fit(X).base_kernel.K
    W_kernel = Kernel(metric="cosine", n_neighbors=K, backend=backend, n_jobs=1).fit(X).K
    # not bit-equal on hnswlib: Kernel hands it unit-norm rows, which shifts its float32
    # distances in the last digits
    np.testing.assert_allclose(W_topograph.toarray(), W_kernel.toarray(), rtol=1e-4, atol=1e-6)
    assert not W_topograph.diagonal().any(), "cosine kernel has self-loops"


@pytest.mark.parametrize("backend, sparse_input", KNN_CASES)
def test_cosine_kernel_does_not_modify_its_input(backend, sparse_input):
    """Rows were L2-normalized in place for backends that need unit vectors."""
    X = csr_matrix(_blobs()) if sparse_input else _blobs()
    before = X.copy()
    Kernel(metric="cosine", n_neighbors=K, backend=backend, n_jobs=1).fit(X)
    if sparse_input:
        np.testing.assert_array_equal(X.data, before.data)
    else:
        np.testing.assert_array_equal(X, before)


def test_precomputed_graph_is_used_as_given():
    """A graph the user supplies is not assumed to hold cosine distances."""
    X = _blobs()
    G = NearestNeighbors(n_neighbors=K, metric="cosine").fit(X).kneighbors_graph(X, mode="distance")
    tg = _topograph("sklearn", base_metric="precomputed").fit(G)
    W_kernel = Kernel(metric="precomputed", n_neighbors=K, backend="sklearn", n_jobs=1).fit(G).K
    np.testing.assert_array_equal(tg.base_kernel.K.toarray(), W_kernel.toarray())


# ── standalone Projector ──────────────────────────────────────────────────────
def test_projector_readers_get_the_graph_they_expect(monkeypatch):
    """
    Projector reads its kNN graph as affinities for the spectral initialization and MAP, and
    as distances for Isomap. All three used to receive cosine similarities.
    """
    import topo.layouts.projector as projector
    X = _blobs()
    D = cosine_distances(X)
    seen = {}

    def capture(name, result):
        def reader(graph, *args, **kwargs):
            seen[name] = graph
            return result
        return reader

    Y = np.zeros((X.shape[0], 2))
    monkeypatch.setattr(projector, "spectral_layout", capture("init", Y))
    monkeypatch.setattr(projector, "fuzzy_embedding", capture("MAP", (Y, {})))
    monkeypatch.setattr(projector, "Isomap", capture("Isomap", Y))
    for method in ("MAP", "Isomap"):
        projector.Projector(metric="cosine", projection_method=method, n_neighbors=K, n_jobs=1,
                            nbrs_backend="sklearn", random_state=0).fit(X)

    for name, expected in (("init", 1 - D), ("MAP", 1 - D), ("Isomap", D)):
        rows, cols, vals = _edges(seen[name])
        np.testing.assert_allclose(vals, expected[rows, cols], atol=1e-5, err_msg=name)


# ── euclidean is untouched ────────────────────────────────────────────────────
def test_euclidean_knn_is_the_sklearn_graph():
    """Unchanged apart from the self-distance, which sklearn returns as rounding noise."""
    X = _blobs()
    G = kNN(X, n_neighbors=K, metric="euclidean", backend="sklearn", n_jobs=1)
    # K neighbors besides the point itself, as on the other backends
    ref = NearestNeighbors(n_neighbors=K + 1, metric="euclidean", n_jobs=1).fit(X).kneighbors_graph(
        X, mode="distance")
    rows, cols, vals = _edges(G)
    np.testing.assert_array_equal(cols, ref.indices)
    np.testing.assert_array_equal(vals[rows != cols], ref.data[rows != cols])
    assert np.all(vals[rows == cols] == 0.0)


def test_euclidean_topograph_kernel_is_built_from_the_graph_unchanged():
    X = _blobs()
    tg = _topograph("sklearn", base_metric="euclidean").fit(X)
    W_kernel = Kernel(metric="precomputed", n_neighbors=K, backend="sklearn", n_jobs=1).fit(
        tg.base_knn_graph).K
    np.testing.assert_array_equal(tg.base_kernel.K.toarray(), W_kernel.toarray())


# ── nmslib: dense index, sparse rows ──────────────────────────────────────────
@requires_nmslib
def test_dense_nmslib_index_accepts_the_sparse_rows_it_was_fitted_on():
    from topo.base.ann import NMSlibTransformer
    X = _blobs(n=200)
    X_sparse = csr_matrix(X)

    from_dense = NMSlibTransformer(n_neighbors=K, metric="cosine", dense=True, n_jobs=1).fit(X).transform(X)
    t = NMSlibTransformer(n_neighbors=K, metric="cosine", dense=True, n_jobs=1).fit(X_sparse)
    assert "sparse" not in t.space

    from_sparse = t.transform(X_sparse)
    np.testing.assert_array_equal(from_sparse.indices, from_dense.indices)
    np.testing.assert_array_equal(from_sparse.data, from_dense.data)

    indices, distances = t.ind_dist_grad(X_sparse, return_grad=False, return_graph=False)
    assert indices.shape == distances.shape == (X.shape[0], K + 1)

    refit = NMSlibTransformer(n_neighbors=K, metric="cosine", dense=True, n_jobs=1).fit_transform(X_sparse)
    np.testing.assert_array_equal(refit.indices, from_dense.indices)


@requires_nmslib
def test_sparse_nmslib_index_accepts_dense_rows():
    from topo.base.ann import NMSlibTransformer
    X = _blobs(n=200)
    t = NMSlibTransformer(n_neighbors=K, metric="cosine", n_jobs=1).fit(csr_matrix(X))
    assert "sparse" in t.space
    np.testing.assert_array_equal(t.transform(X).indices, t.transform(csr_matrix(X)).indices)


# ── repeated queries keep k ───────────────────────────────────────────────────
@pytest.mark.parametrize("transformer", [
    pytest.param("HNSWlibTransformer", marks=requires_hnswlib),
    pytest.param("NMSlibTransformer", marks=requires_nmslib),
])
def test_repeated_queries_keep_k_fixed(transformer):
    import topo.base.ann as ann
    X = _blobs(n=200)
    t = getattr(ann, transformer)(n_neighbors=K, metric="euclidean", n_jobs=1).fit(X)

    for _ in range(3):
        G = t.transform(X)
        assert np.all(np.diff(G.indptr) == K + 1)
        indices, distances = t.ind_dist_grad(X, return_grad=False, return_graph=False)
        assert indices.shape == distances.shape == (X.shape[0], K + 1)
    assert t.n_neighbors == K


# ── scaffold sizing ───────────────────────────────────────────────────────────
def test_scaffold_sizing_runs_on_the_sklearn_fallback():
    """`random_state` used to be forwarded to sklearn's NearestNeighbors, which rejects it."""
    n = automated_scaffold_sizing(_blobs(), ks=(10, 20), backend="sklearn", n_jobs=1,
                                  min_components=2, max_components=30, random_state=0)
    assert 2 <= n <= 30


@pytest.mark.parametrize("method", ["fsa", "mle"])
def test_intrinsic_dimension_with_cosine_metric(method):
    """Read off raw cosine distances, both estimators return about half the dimension."""
    _, details = automated_scaffold_sizing(_sphere(dim=5), method=method, ks=30, metric="cosine",
                                           backend="sklearn", n_jobs=1, min_components=2,
                                           max_components=30, return_details=True)
    estimate = np.median(details["local_id"])
    assert 3.8 < estimate < 6.2, f"{method} estimate {estimate:.2f} for dimension 5"


# ── single-cell wrapper ───────────────────────────────────────────────────────
def test_fit_adata_skips_projection_with_missing_dependency(monkeypatch):
    """PaCMAP is in the default projections but is an optional dependency."""
    anndata = pytest.importorskip("anndata")
    from topo.single_cell import fit_adata

    monkeypatch.setitem(sys.modules, "pacmap", None)  # makes `import pacmap` raise ImportError
    adata = anndata.AnnData(_blobs(n=300).astype(np.float32))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_adata(adata, projections=("MAP", "PaCMAP"), do_leiden=False, base_knn=K, graph_knn=K,
                  min_eigs=20, id_min_components=8, id_max_components=20, n_jobs=1,
                  backend="sklearn", verbosity=0, random_state=0)

    assert "X_TopoMAP" in adata.obsm and "X_msTopoMAP" in adata.obsm
    assert "X_TopoPaCMAP" not in adata.obsm
    assert any("PaCMAP" in str(w.message) and issubclass(w.category, RuntimeWarning) for w in caught)
