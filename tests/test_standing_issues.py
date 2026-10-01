"""
Regression tests for the standing issues fixed in 1.1.1, other than the cosine kNN inversion
(see test_cosine_knn_distances.py).
"""
import sys
import warnings

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

from topo.base import ann
from topo.base.ann import kNN, resolve_backend


def _has(module):
    try:
        __import__(module)
        return True
    except ImportError:
        return False


requires_hnswlib = pytest.mark.skipif(not _has("hnswlib"), reason="hnswlib not installed")
requires_nmslib = pytest.mark.skipif(not _has("nmslib"), reason="nmslib not installed")

BACKENDS = [
    pytest.param("sklearn", id="sklearn"),
    pytest.param("hnswlib", id="hnswlib", marks=requires_hnswlib),
    pytest.param("nmslib", id="nmslib", marks=requires_nmslib),
]

K = 15


def _blobs(n=400, p=40, n_clusters=5, seed=0):
    """Overlapping clusters: close enough that the kNN graph stays connected."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=0.5, size=(n_clusters, p))
    return centers[rng.integers(0, n_clusters, n)] + rng.normal(size=(n, p))


def _without(monkeypatch, *modules):
    """Make `modules` look uninstalled to the backend resolver."""
    real = ann._is_installed
    monkeypatch.setattr(ann, "_is_installed", lambda m: False if m in modules else real(m))


# ── neighbor-search backends ──────────────────────────────────────────────────
@pytest.mark.parametrize("backend", BACKENDS)
def test_every_backend_returns_k_neighbors_besides_the_point_itself(backend):
    """sklearn counted the point itself among its k neighbors; the ANN backends did not."""
    X = _blobs()
    G = kNN(X, n_neighbors=K, metric="euclidean", backend=backend, n_jobs=1).tocsr()
    assert np.all(np.diff(G.indptr) == K + 1)


def test_resolve_backend_keeps_an_installed_backend():
    assert resolve_backend("sklearn") == "sklearn"
    for backend in ("hnswlib", "nmslib"):
        if _has(backend):
            assert resolve_backend(backend) == backend


def test_resolve_backend_falls_back_with_a_warning(monkeypatch):
    _without(monkeypatch, "hnswlib")
    expected = "nmslib" if _has("nmslib") else "sklearn"
    with pytest.warns(UserWarning):
        assert resolve_backend("hnswlib") == expected

    _without(monkeypatch, "hnswlib", "nmslib")
    for backend in ("hnswlib", "nmslib"):
        with pytest.warns(UserWarning, match="scikit-learn"):
            assert resolve_backend(backend) == "sklearn"
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # asking for sklearn is not a fallback
        assert resolve_backend("sklearn") == "sklearn"


def test_knn_default_backend_works_without_ann_libraries(monkeypatch):
    """kNN(), geodesic_correlation() and IntrinsicDim default to hnswlib and had no fallback."""
    from topo.eval.local_scores import geodesic_correlation
    from topo.tpgraph.intrinsic_dim import IntrinsicDim
    _without(monkeypatch, "hnswlib", "nmslib")
    X = _blobs(n=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G = kNN(X, n_neighbors=K, metric="euclidean")
        rows = np.repeat(np.arange(X.shape[0]), np.diff(G.indptr))
        np.testing.assert_allclose(G.data, euclidean_distances(X)[rows, G.indices], atol=1e-8)

        assert np.isfinite(geodesic_correlation(X, X[:, :10], n_neighbors=K, n_jobs=1))

        est = IntrinsicDim(k=[10, 20], plot=False, n_jobs=1)
        est.fit(X)
        assert set(est.local_id["fsa"]) == {"10", "20"}


@requires_nmslib
def test_topograph_honours_the_nmslib_backend():
    """backend='nmslib' was replaced by hnswlib when installed, and by sklearn otherwise."""
    from topo.topograph import TopOGraph
    tg = TopOGraph(backend="nmslib")
    tg._parse_backend()
    assert tg.backend == "nmslib"


@pytest.mark.parametrize("transformer, module", [("HNSWlibTransformer", "hnswlib"), ("NMSlibTransformer", "nmslib")])
def test_transformer_raises_when_its_library_is_missing(monkeypatch, transformer, module):
    """fit() used to print a message and return None, which failed later as an AttributeError."""
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ImportError, match="pip install"):
        getattr(ann, transformer)(n_neighbors=K).fit(_blobs(n=100))


@requires_hnswlib
def test_hnswlib_transformer_accepts_sparse_input():
    """fit() returned None for CSR input."""
    X = _blobs(n=200)
    dense = ann.HNSWlibTransformer(n_neighbors=K, metric="cosine", n_jobs=1).fit(X).transform(X)
    t = ann.HNSWlibTransformer(n_neighbors=K, metric="cosine", n_jobs=1).fit(csr_matrix(X))
    assert t is not None
    np.testing.assert_array_equal(t.transform(csr_matrix(X)).indices, dense.indices)


@pytest.mark.parametrize("transformer", [
    pytest.param("HNSWlibTransformer", marks=requires_hnswlib),
    pytest.param("NMSlibTransformer", marks=requires_nmslib),
])
def test_transformer_recall_check_and_gradients(transformer, capsys):
    X = _blobs(n=200)
    t = getattr(ann, transformer)(n_neighbors=K, metric="euclidean", n_jobs=1).fit(X)
    recall = t.test_efficiency(X)
    assert 0.95 <= recall <= 1.0
    assert "recall" in capsys.readouterr().out
    assert t.n_neighbors == K

    indices, distances, graph = t.ind_dist_grad(X)
    assert graph.shape == (X.shape[0], X.shape[0])
    with pytest.raises(NotImplementedError):   # they were computed from the indices, not the data
        t.ind_dist_grad(X, return_grad=True)


# ── kernels ───────────────────────────────────────────────────────────────────
from scipy.sparse import find  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from topo.topograph import TopOGraph  # noqa: E402
from topo.tpgraph.cknn import cknn_graph  # noqa: E402
from topo.tpgraph.kernels import Kernel, compute_kernel  # noqa: E402


def _topograph(backend="sklearn", **kwargs):
    params = dict(base_knn=K, graph_knn=K, min_eigs=20, id_min_components=8,
                  id_max_components=20, n_jobs=1, backend=backend,
                  projection_methods=None, verbosity=0, random_state=0)
    params.update(kwargs)
    return TopOGraph(**params)


def _reference_kernel(G, k):
    """exp(-(d / sigma_i)^2) over each row's neighbors, symmetrized: the adaptive kernel."""
    G = G.tocsr()
    sigma = np.array([np.sort(G.data[G.indptr[i]:G.indptr[i + 1]])[k // 2 - 1] for i in range(G.shape[0])])
    x, y, d = find(G)
    W = csr_matrix((np.exp(-(d / (sigma[x] + 1e-10)) ** 2), (x, y)), shape=G.shape)
    return ((W + W.T) / 2).toarray()


def test_use_angular_false_is_honoured():
    """It was forced to True for cosine whatever the caller passed."""
    X = _blobs()
    angular = Kernel(metric="cosine", n_neighbors=K, backend="sklearn", n_jobs=1).fit(X)
    raw = Kernel(metric="cosine", n_neighbors=K, backend="sklearn", n_jobs=1, use_angular=False).fit(X)
    assert abs(angular.K - raw.K).max() > 1e-3
    np.testing.assert_allclose(raw.K.toarray(), _reference_kernel(raw.knn_, K), rtol=1e-10)
    # the default is unchanged: angles
    default = compute_kernel(X, metric="cosine", n_neighbors=K, backend="sklearn", n_jobs=1)
    np.testing.assert_array_equal(default.toarray(), angular.K.toarray())


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_kernel_has_no_self_loops_on_any_backend(metric):
    """scikit-learn's noisy euclidean self-distances became weight-1 self-loops."""
    X = _blobs().astype(np.float32)
    for n_jobs in (1, 4):       # sklearn only returns the noise when it works in chunks
        W = compute_kernel(X, metric=metric, n_neighbors=K, backend="sklearn", n_jobs=n_jobs)
        assert not W.diagonal().any()
    G = NearestNeighbors(n_neighbors=K + 1, metric=metric).fit(X).kneighbors_graph(X, mode="distance").tocsr()
    rows = np.repeat(np.arange(G.shape[0]), np.diff(G.indptr))
    G.data[G.indices == rows] = 1e-7     # a graph handed over with noisy self-distances
    assert not compute_kernel(G, metric="precomputed", n_neighbors=K).diagonal().any()


def test_pairwise_kernel():
    """`pairwise=True` passed the metric name where the second data matrix goes, and raised."""
    X = _blobs(n=120)
    W = compute_kernel(X, metric="euclidean", n_neighbors=K, pairwise=True, n_jobs=1)
    assert W.nnz == X.shape[0] * (X.shape[0] - 1)
    D = csr_matrix(euclidean_distances(X))
    D.setdiag(0.0)      # stored zeros: the self-distance takes part in the bandwidth, as in kNN graphs
    np.testing.assert_allclose(W.toarray(), _reference_kernel(D, K), rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("alpha_decaying", [False, True])
def test_neighborhood_expansion_widens_the_graph(alpha_decaying):
    """`new_k` always came out equal to `k`, so nothing was ever expanded."""
    X = _blobs()
    kwargs = dict(metric="euclidean", n_neighbors=K, backend="sklearn", n_jobs=1,
                  alpha_decaying=alpha_decaying, return_densities=True)
    W, dens = compute_kernel(X, **kwargs)
    W_wide, dens_wide = compute_kernel(X, expand_nbr_search=True, **kwargs)

    new_k = dens_wide["expanded_k_neighbor"]
    assert K < new_k <= 2 * K
    assert np.all(np.diff(dens_wide["knn_expanded"].indptr) == new_k + 1)
    assert W_wide.nnz > W.nnz
    assert dens_wide["adaptive_bw_nbr_expanded"].mean() > dens_wide["adaptive_bw"].mean()
    np.testing.assert_array_equal(dens_wide["adaptive_bw"], dens["adaptive_bw"])
    assert np.isfinite(W_wide.data).all() and not W_wide.diagonal().any()

    kernel = Kernel(metric="euclidean", n_neighbors=K, backend="sklearn", n_jobs=1,
                    expand_nbr_search=True, alpha_decaying=alpha_decaying).fit(X)
    assert kernel.expanded_k_neighbor_ == new_k
    # nothing to widen in a graph that is handed over
    Kernel(metric="precomputed", n_neighbors=K, expand_nbr_search=True).fit(dens["knn"])


@pytest.mark.parametrize("version", ["bw_adaptive_nbr_expansion", "bw_adaptive_alpha_decaying_nbr_expansion"])
def test_topograph_expansion_kernels_search_the_data(version):
    """They fitted the kernel on the kNN graph, as if its rows were the data."""
    X = _blobs()
    tg = _topograph(base_kernel_version=version).fit(X)
    expected = Kernel(metric="cosine", n_neighbors=K, backend="sklearn", n_jobs=1, expand_nbr_search=True,
                      alpha_decaying="alpha" in version).fit(X)
    np.testing.assert_array_equal(tg.base_kernel.K.toarray(), expected.K.toarray())
    assert tg.base_kernel.expanded_k_neighbor_ > K

    G = NearestNeighbors(n_neighbors=K + 1).fit(X).kneighbors_graph(X, mode="distance")
    with pytest.raises(ValueError, match="precomputed"):
        _topograph(base_metric="precomputed", base_kernel_version=version).fit(G)


def test_cknn_rule_is_applied_per_pair():
    """The normalization was one number for the whole graph, and the weights were distances."""
    X = _blobs(p=5)
    delta = 1.2
    A, W, d_k = cknn_graph(X, n_neighbors=K, delta=delta, metric="euclidean", weighted=None,
                           return_densities=True, backend="sklearn", n_jobs=1)
    D = euclidean_distances(X)
    G = NearestNeighbors(n_neighbors=K + 1).fit(X).kneighbors_graph(X, mode="distance").tocsr()
    np.testing.assert_allclose(d_k, [np.sort(G.data[G.indptr[i]:G.indptr[i + 1]])[K // 2 - 1]
                                     for i in range(X.shape[0])])

    ratio = D / np.sqrt(np.outer(d_k, d_k))
    candidates = ((G + G.T) > 0).toarray()
    np.fill_diagonal(candidates, False)
    expected = candidates & (ratio < delta)
    np.testing.assert_array_equal(A.toarray().astype(bool), expected)
    np.testing.assert_allclose(W.toarray(), np.where(expected, np.exp(-ratio ** 2), 0.0), rtol=1e-5, atol=1e-7)
    assert abs(A - A.T).nnz == 0

    # a larger delta can only add edges
    A_wide = cknn_graph(X, n_neighbors=K, delta=2.0, metric="euclidean", backend="sklearn", n_jobs=1)
    assert A_wide.nnz > A.nnz and (A_wide - A).min() >= 0


def test_cknn_warns_about_isolated_points_and_delta_reaches_the_kernel():
    X = _blobs()
    with pytest.warns(UserWarning, match="delta"):
        cknn_graph(X, n_neighbors=K, delta=0.5, metric="euclidean", backend="sklearn", n_jobs=1)

    # TopOGraph accepted `delta` but never passed it on
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tight = _topograph(base_kernel_version="cknn", delta=1.2).fit(X)
        loose = _topograph(base_kernel_version="cknn", delta=3.0).fit(X)
    assert loose.base_kernel.K.nnz > tight.base_kernel.K.nnz
    rho = _per_row_spearman(loose.base_kernel.K, cosine_distances(X))
    assert rho < -0.6, f"CkNN weights do not decrease with distance: Spearman {rho:+.2f}"


def _per_row_spearman(W, D):
    from scipy.stats import spearmanr
    W = W.tocsr()
    rhos = []
    for i in range(W.shape[0]):
        idx = W.indices[W.indptr[i]:W.indptr[i + 1]]
        if len(idx) > 3:
            rhos.append(spearmanr(D[i, idx], W.data[W.indptr[i]:W.indptr[i + 1]])[0])
    return np.median(rhos)


def test_package_compiles_without_syntax_warnings():
    """A docstring in kernels.py had an invalid escape sequence."""
    import pathlib
    import topo
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for path in pathlib.Path(topo.__file__).parent.rglob("*.py"):
            compile(path.read_text(), str(path), "exec")


# ── geodesics ─────────────────────────────────────────────────────────────────
def _arc(n=120, radians=2.5, ambient=6, seed=0):
    """Points along an arc of a great circle, in order; the angle between the ends is `radians`."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, radians, n)
    Q, _ = np.linalg.qr(rng.normal(size=(ambient, 2)))
    return np.c_[np.cos(t), np.sin(t)] @ Q.T, t


def test_cosine_geodesics_are_angles():
    """Summed along a path, 1 - cos underestimates: it is not a metric. Angles add up."""
    from topo.eval.local_scores import geodesic_distance
    X, t = _arc()
    kernel = Kernel(metric="cosine", n_neighbors=4, backend="sklearn", n_jobs=1).fit(X)
    np.testing.assert_allclose(kernel.SP[0], t, atol=1e-6)

    raw = geodesic_distance(kernel.knn_, n_jobs=1)[0]       # what summing 1 - cos gives
    assert raw[-1] < 0.05 * t[-1]

    with pytest.raises(ValueError, match="precomputed"):     # no distances kept to walk on
        Kernel(metric="precomputed", n_neighbors=4).fit(kernel.knn_).SP
    cached = Kernel(metric="precomputed", n_neighbors=4, cache_input=True).fit(kernel.knn_)
    np.testing.assert_allclose(cached.SP[0], raw, atol=1e-12)


def test_isomap_unrolls_an_arc_measured_by_cosine():
    from topo.layouts.isomap import Isomap
    X, t = _arc()
    Y = Isomap(X, n_components=1, n_neighbors=4, metric="cosine", backend="sklearn", n_jobs=1)
    # an isometric unrolling: embedding distances equal the angles
    np.testing.assert_allclose(np.abs(Y[:, 0] - Y[0, 0]), t, atol=1e-3)


def test_geodesic_correlation_with_cosine_metric():
    from topo.eval.local_scores import geodesic_correlation
    X, t = _arc()
    assert geodesic_correlation(X, X, metric="cosine", n_neighbors=4, n_jobs=1, backend="sklearn") > 0.999


def test_evaluation_geodesics_use_angles_for_a_cosine_base_graph(monkeypatch):
    import topo.eval.local_scores as local_scores
    from topo.tpgraph.kernels import _cosine_distance_to_angle
    X = _blobs(n=200)
    tg = _topograph().fit(X)
    seen = []

    def capture(graph, *args, **kwargs):
        seen.append(graph)
        raise RuntimeError("captured")

    monkeypatch.setattr(local_scores, "geodesic_distance", capture)
    with pytest.raises(RuntimeError, match="captured"):
        tg.eval_models_layouts(X, landmarks=None, kernels=[], eigenmap_methods=[], projections=[])
    np.testing.assert_allclose(seen[0].data, _cosine_distance_to_angle(tg.base_knn_graph.data))


# ── projections ───────────────────────────────────────────────────────────────
from topo.layouts.projector import Projector  # noqa: E402


def test_topograph_gives_isomap_distances_not_affinities(monkeypatch):
    """Isomap was run on the diffusion operator, whose entries are larger for closer points."""
    import topo.layouts.projector as projector
    X = _blobs(n=200)
    tg = _topograph().fit(X)
    seen = {}

    def isomap(graph, *args, **kwargs):
        seen["graph"] = graph
        return np.zeros((X.shape[0], 2))

    monkeypatch.setattr(projector, "Isomap", isomap)
    tg.project(projection_method="Isomap", multiscale=True)
    msZ = tg.spectral_scaffold(multiscale=True)[:, :tg._scaffold_components_ms]
    G = seen["graph"].tocsr()
    rows = np.repeat(np.arange(G.shape[0]), np.diff(G.indptr))
    np.testing.assert_allclose(G.data, euclidean_distances(msZ)[rows, G.indices], rtol=1e-4, atol=1e-6)


def test_topograph_isomap_and_y_aliases():
    X = _blobs(n=200)
    tg = _topograph().fit(X)
    for method in ("Isomap", "MAP"):
        tg.project(projection_method=method, multiscale=True, num_iters=50)
        tg.project(projection_method=method, multiscale=False, num_iters=50)
    assert all(np.isfinite(Y).all() for Y in tg.ProjectionDict.values())
    np.testing.assert_array_equal(tg.Y("msTopoMAP"), tg.msTopoMAP)
    np.testing.assert_array_equal(tg.Y("TopoMAP"), tg.TopoMAP)


@pytest.mark.parametrize("method", ["Isomap", "MAP"])
def test_projector_landmarks(method):
    """Any use of landmarks raised AttributeError (a misspelled attribute)."""
    X = _blobs(n=200)
    landmarks = np.arange(0, 200, 2)
    Y = Projector(metric="euclidean", projection_method=method, n_neighbors=K, n_jobs=1,
                  nbrs_backend="sklearn", num_iters=50, landmarks=landmarks, random_state=0).fit_transform(X)
    Y = Y[0] if isinstance(Y, tuple) else Y
    assert Y.shape == (100, 2) and np.isfinite(Y).all()


def test_geodesic_distance_among_a_subset_of_vertices():
    from topo.eval.local_scores import geodesic_distance
    X = _blobs(n=150)
    G = kNN(X, n_neighbors=K, metric="euclidean", backend="sklearn", n_jobs=1)
    full = geodesic_distance(G, n_jobs=1)
    np.testing.assert_array_equal(geodesic_distance(G, n_jobs=2), full)
    subset = np.arange(0, 150, 3)
    for n_jobs in (1, 2):
        np.testing.assert_array_equal(geodesic_distance(G, indices=subset, n_jobs=n_jobs),
                                      full[np.ix_(subset, subset)])
    assert geodesic_distance(G, indices=7, n_jobs=1).shape == (150,)


def test_umap_projection():
    """UMAP was handed a graph matrix where it takes (indices, distances)."""
    pytest.importorskip("umap")
    X = _blobs(n=200)
    Y = Projector(metric="euclidean", projection_method="UMAP", n_neighbors=K, n_jobs=1,
                  nbrs_backend="sklearn", num_iters=50, random_state=0).fit_transform(X)
    assert Y.shape == (200, 2) and np.isfinite(Y).all()
    G = NearestNeighbors(n_neighbors=K + 1).fit(X).kneighbors_graph(X, mode="distance")
    Y = Projector(metric="precomputed", projection_method="UMAP", n_neighbors=K, n_jobs=1,
                  num_iters=50, random_state=0).fit_transform(G)
    assert Y.shape == (200, 2) and np.isfinite(Y).all()
    tg = _topograph().fit(X)
    assert tg.project(projection_method="UMAP", multiscale=True, num_iters=50).shape == (200, 2)
