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


# ── reproducibility ───────────────────────────────────────────────────────────
def test_eigendecomposition_is_reproducible():
    """Eigenvector signs flipped between runs: SciPy no longer seeds ARPACK's start vector."""
    from topo.spectral.eigen import EigenDecomposition, eigendecompose
    X = _blobs(n=200)
    kernel = Kernel(metric="euclidean", n_neighbors=K, backend="sklearn", n_jobs=1).fit(X)
    for seed in (None, 7):
        runs = [eigendecompose(kernel.P, n_components=10, random_state=seed) for _ in range(3)]
        for evals, evecs in runs[1:]:
            np.testing.assert_array_equal(evals, runs[0][0])
            np.testing.assert_array_equal(evecs, runs[0][1])
    # the largest entry of every eigenvector is positive
    evecs = runs[0][1]
    assert np.all(evecs[np.argmax(np.abs(evecs), axis=0), np.arange(evecs.shape[1])] > 0)

    a = EigenDecomposition(n_components=10, method="msDM", random_state=3).fit(kernel).transform(X=None)
    b = EigenDecomposition(n_components=10, method="msDM", random_state=3).fit(kernel).transform(X=None)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("backend", ["sklearn", pytest.param("hnswlib", marks=requires_hnswlib)])
def test_topograph_is_reproducible_with_a_seed(backend):
    """With one job: index construction and layout optimization are not deterministic in parallel."""
    X = _blobs(n=200)
    first = _topograph(backend, projection_methods=["MAP"]).fit(X)
    second = _topograph(backend, projection_methods=["MAP"]).fit(X)
    for multiscale in (True, False):
        np.testing.assert_array_equal(first.spectral_scaffold(multiscale), second.spectral_scaffold(multiscale))
    np.testing.assert_array_equal(first.P_of_msZ.toarray(), second.P_of_msZ.toarray())
    np.testing.assert_array_equal(first.msTopoMAP, second.msTopoMAP)


def test_spectral_layout_of_a_disconnected_graph():
    """Each component was embedded with the Laplacian of the whole graph, which cannot fit."""
    from scipy.sparse import block_diag
    from topo.spectral.eigen import spectral_layout
    blocks = [Kernel(metric="euclidean", n_neighbors=10, backend="sklearn", n_jobs=1).fit(_blobs(n=n, seed=s)).K
              for n, s in ((60, 0), (80, 1), (50, 2))]
    graph = block_diag(blocks).tocsr()
    Y = spectral_layout(graph, 2, np.random.RandomState(0))
    assert Y.shape == (190, 2) and np.isfinite(Y).all()
    # the components are laid out apart from each other
    centers = np.array([Y[:60].mean(0), Y[60:140].mean(0), Y[140:].mean(0)])
    spread = max(Y[:60].std(), Y[60:140].std(), Y[140:].std())
    assert np.linalg.norm(centers[0] - centers[1]) > spread


# ── intrinsic dimension ───────────────────────────────────────────────────────
def _sphere(n=3000, dim=5, ambient=40, seed=0):
    """Points uniform on a `dim`-sphere, isometrically embedded in `ambient` dimensions."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, dim + 1))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    Q, _ = np.linalg.qr(rng.normal(size=(ambient, dim + 1)))
    return Z @ Q.T


@pytest.mark.parametrize("k", [[20], 20, (20, 40), range(20, 60, 20), np.int64(20)])
def test_intrinsic_dim_accepts_every_form_of_k(k):
    """A one-element list and a tuple both raised."""
    from topo.tpgraph.intrinsic_dim import IntrinsicDim
    est = IntrinsicDim(k=k, plot=False, backend="sklearn", n_jobs=1)
    est.fit(_blobs(n=150))
    expected = {str(int(v)) for v in np.atleast_1d(np.asarray(list(k) if not np.isscalar(k) else [k]))}
    assert set(est.local_id["fsa"]) == set(est.global_id["mle"]) == expected


def test_global_intrinsic_dimension_of_a_sphere():
    """The global FSA estimate divided the median of the local ones by log(2) a second time."""
    from topo.tpgraph.intrinsic_dim import IntrinsicDim, mle_global
    X = _sphere(dim=5)
    est = IntrinsicDim(k=[30], plot=False, backend="sklearn", n_jobs=1)
    est.fit(X)
    assert 3.8 < est.global_id["fsa"]["30"] < 6.2
    assert 3.8 < est.global_id["mle"]["30"] < 6.2
    G = kNN(X, n_neighbors=30, backend="sklearn", n_jobs=1)
    assert 3.8 < mle_global(G, n_neighbors=30) < 6.2        # raised when no local estimates were passed


@pytest.mark.parametrize("id_method", ["fsa", "mle"])
def test_topograph_intrinsic_dimension_accessors(id_method):
    """local_ids() and global_id_mle()/fsa() returned None; global_id was the scaffold size."""
    X = _blobs()
    tg = _topograph(id_method=id_method, id_ks=20).fit(X)
    local = tg.local_ids()
    assert set(local) == {id_method} and local[id_method].shape == (X.shape[0],)
    estimate = tg.global_id_fsa() if id_method == "fsa" else tg.global_id_mle()
    assert estimate is not None and estimate > 0
    assert tg.global_id == estimate
    assert (tg.global_id_mle() if id_method == "fsa" else tg.global_id_fsa()) is None
    assert tg.n_scaffold_components == tg._scaffold_components_ms == 20     # capped by id_max_components
    assert tg.global_id != tg.n_scaffold_components
    assert _topograph().global_id is None                                   # before fit


# ── TopOGraph accessors and analyses ──────────────────────────────────────────
@pytest.fixture(scope="module")
def fitted():
    X = _blobs()
    return X, _topograph(projection_methods=["MAP"]).fit(X)


def test_y_aliases_match_projection_keys(fitted):
    """Y('TopoPaCMAP') looked the layout up under a key that names a graph kernel."""
    _, tg = fitted
    np.testing.assert_array_equal(tg.Y("TopoMAP"), tg.TopoMAP)
    np.testing.assert_array_equal(tg.Y("msTopoMAP"), tg.msTopoMAP)
    fake = np.zeros((2, 2))
    tg.ProjectionDict["PaCMAP of DM with bw_adaptive"] = fake
    tg.ProjectionDict["PaCMAP of msDM with bw_adaptive"] = fake + 1
    try:
        np.testing.assert_array_equal(tg.Y("TopoPaCMAP"), tg.TopoPaCMAP)
        np.testing.assert_array_equal(tg.Y("msTopoPaCMAP"), tg.msTopoPaCMAP)
    finally:
        del tg.ProjectionDict["PaCMAP of DM with bw_adaptive"], tg.ProjectionDict["PaCMAP of msDM with bw_adaptive"]


def test_pseudotime_uses_each_eigenvector_with_its_own_eigenvalue(fitted):
    """Eigenvalue j+1 was paired with column j, and the weights were applied to the scaffold,
    whose columns already carry them."""
    _, tg = fitted
    eig = tg.EigenbasisDict["msDM with bw_adaptive"]
    k = 10
    out = tg.pseudotime(root=3, k=k, multiscale=True)
    psi = eig.eigenvectors[:, :k] * (eig.eigenvalues[:k] / (1 - eig.eigenvalues[:k]))
    d2 = ((psi - psi[3]) ** 2).sum(1)
    np.testing.assert_allclose(out["pseudotime"], (d2 - d2.min()) / (d2.max() - d2.min() + 1e-12))
    # with these weights, the coordinates are the multiscale scaffold itself
    np.testing.assert_allclose(psi, tg.spectral_scaffold(multiscale=True)[:, :k])
    # every available eigenvector can be used
    assert np.isfinite(tg.pseudotime(root=3, k=10_000)["pseudotime"]).all()


def test_spectral_selectivity_uses_the_selected_components(fitted, monkeypatch):
    """It never trimmed to the scaffold size, and its eigenvalues were shifted by one."""
    _, tg = fitted
    n_all = tg.spectral_scaffold(multiscale=True).shape[1]
    monkeypatch.setattr(tg, "_scaffold_components_ms", 6)
    assert n_all > 6
    trimmed = tg.spectral_selectivity(k_neighbors=10, weight_mode="none")
    full = tg.spectral_selectivity(k_neighbors=10, weight_mode="none", use_scaffold_components=False)
    assert trimmed["axis"].max() < 6 <= full["axis"].max() + 1 or full["axis"].max() >= trimmed["axis"].max()
    assert not np.allclose(trimmed["EAS"], full["EAS"])

    # eigenvalue weights line up with the columns: weighting by them equals passing them in
    evals = tg.EigenbasisDict["msDM with bw_adaptive"].eigenvalues[:6]
    np.testing.assert_allclose(tg.spectral_selectivity(k_neighbors=10)["EAS"],
                               tg.spectral_selectivity(k_neighbors=10, evals=evals)["EAS"])


def test_riemann_diagnostics_defaults(fitted):
    """Documented default: the multiscale layout; and the Laplacian the tp.sc wrappers use."""
    from topo.eval.rmetric import RiemannMetric
    _, tg = fitted
    out = tg.riemann_diagnostics()
    np.testing.assert_allclose(out["G"], RiemannMetric(tg.msTopoMAP, tg.graph_kernel.L).get_rmetric())


def test_saving_does_not_strip_the_fitted_object(fitted, tmp_path):
    import topo as tp
    _, tg = fitted
    index = tg.base_nbrs_class
    assert index is not None
    tp.save_topograph(tg, str(tmp_path / "tg.pkl"))
    assert tg.base_nbrs_class is index
    tg.write_pkl(str(tmp_path / "tg2.pkl"))
    assert tg.base_nbrs_class is index
    loaded = tp.load_topograph(str(tmp_path / "tg.pkl"))
    assert loaded.base_nbrs_class is None
    np.testing.assert_array_equal(loaded.msTopoMAP, tg.msTopoMAP)


def test_visualize_optimization_writes_a_gif(fitted, tmp_path, monkeypatch):
    """It needed imageio, which it never used, and a canvas method matplotlib removed."""
    import matplotlib
    matplotlib.use("Agg")
    from PIL import Image
    _, tg = fitted
    monkeypatch.setitem(sys.modules, "imageio", None)
    monkeypatch.setitem(sys.modules, "imageio.v2", None)
    path = tg.visualize_optimization(num_iters=30, save_every=10, filename=str(tmp_path / "map.gif"))
    with Image.open(path) as gif:
        assert gif.n_frames >= 3


def test_run_models_computes_every_kernel_combination():
    """Graph kernels other than the first were never computed, so the legacy workflow hit KeyErrors."""
    X = _blobs(n=200)
    tg = _topograph(delta=2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tg.run_models(X, kernels=["bw_adaptive", "cknn"], projections=["MAP"])
    for base in ("bw_adaptive", "cknn"):
        for tag in ("msDM", "DM"):
            assert f"{tag} with {base}" in tg.EigenbasisDict
            for graph in ("bw_adaptive", "cknn"):
                assert f"{graph} from {tag} with {base}" in tg.GraphKernelDict
                assert f"MAP of {graph} from {tag} with {base}" in tg.ProjectionDict
    assert tg.projection_methods is None      # restored


def test_set_refined_from_precomputed_installs_the_graph_with_coords(fitted):
    """Passing `coords` skipped the graph altogether."""
    X, _ = fitted
    tg = _topograph().fit(X)
    Z = tg.spectral_scaffold(multiscale=True)[:, :5]
    G = kNN(Z, n_neighbors=K, backend="sklearn", n_jobs=1)
    tg.set_refined_from_precomputed(G, multiscale=True, coords=Z)
    assert tg.knn_msZ is G
    np.testing.assert_array_equal(tg.spectral_scaffold(multiscale=True), Z.astype(np.float32))


# ── Riemann metric ────────────────────────────────────────────────────────────
def test_riemann_metric_keeps_the_laplacian_sparse(fitted):
    """Every call densified the Laplacian: n^2 * 8 bytes per copy."""
    from scipy.sparse import issparse
    from topo.eval import rmetric
    _, tg = fitted
    L, Y = tg.graph_kernel.L, tg.msTopoMAP
    assert issparse(L) and issparse(rmetric._symmetrize(L))
    assert issparse(rmetric.RiemannMetric(Y, L).L)

    np.testing.assert_allclose(rmetric.RiemannMetric(Y, L).get_rmetric(),
                               rmetric.RiemannMetric(Y, L.toarray()).get_rmetric(), rtol=1e-8, atol=1e-10)
    for t in (0, 3):
        sparse_vals, sparse_lims = rmetric.calculate_deformation(Y, L, diffusion_t=t)
        dense_vals, dense_lims = rmetric.calculate_deformation(Y, L.toarray(), diffusion_t=t)
        np.testing.assert_allclose(sparse_vals, dense_vals, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(sparse_lims, dense_lims, rtol=1e-6, atol=1e-8)


def test_riemann_plots_run_on_current_matplotlib(fitted):
    """They called matplotlib.cm.get_cmap, removed in matplotlib 3.9."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from topo.eval import rmetric
    _, tg = fitted
    L, Y = tg.graph_kernel.L, tg.msTopoMAP
    fig, axes = plt.subplots(1, 3)
    rmetric.plot_riemann_metric_localized(Y, L, n_plot=20, ax=axes[0], seed=0, colors=np.arange(Y.shape[0]))
    rmetric.plot_riemann_metric_global(Y, L, grid_res=4, k_avg=10, ax=axes[1])
    rmetric.plot_metric_contraction_expansion(Y, L, ax=axes[2])
    plt.close(fig)


# ── single-cell wrappers ──────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def fitted_adata():
    anndata = pytest.importorskip("anndata")
    pytest.importorskip("scanpy")
    import matplotlib
    matplotlib.use("Agg")
    import topo as tp
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=0.5, size=(5, 40))
    labels = rng.integers(0, 5, 300)
    adata = anndata.AnnData((centers[labels] + rng.normal(size=(300, 40))).astype(np.float32))
    adata.obs["group"] = np.array(list("abcde"))[labels]
    tg = tp.sc.fit_adata(adata, projections=("MAP",), do_leiden=False, projection_methods=["MAP"],
                         base_knn=K, graph_knn=K, min_eigs=20, id_min_components=8, id_max_components=20,
                         n_jobs=1, backend="sklearn", verbosity=0, random_state=0)
    return adata, tg


def test_preprocess_does_not_modify_its_input():
    """Its docstring promises a copy; it normalized, logged and annotated the object passed in."""
    anndata = pytest.importorskip("anndata")
    pytest.importorskip("scanpy")
    import topo as tp
    counts = np.random.default_rng(1).poisson(1.0, size=(200, 300)).astype(np.float32)
    adata = anndata.AnnData(counts.copy())
    out = tp.sc.preprocess(adata, n_top_genes=100, flavor="seurat")
    assert out is not adata and out.shape == (200, 100)
    np.testing.assert_array_equal(adata.X, counts)
    assert adata.raw is None and "counts" not in adata.layers and "highly_variable" not in adata.var
    assert "counts" in out.layers and out.raw is not None


def test_sc_intrinsic_dim_stores_the_estimates(fitted_adata):
    """It read the scaffold size as the global estimate and got None for the local ones."""
    import topo as tp
    adata, tg = fitted_adata
    tp.sc.intrinsic_dim(adata, tg, id_k_values=[10, 20])
    assert adata.uns["topometry_id_global_fsa"] == tg.global_id != tg.n_scaffold_components
    np.testing.assert_array_equal(adata.obs["local_id_fsa"].to_numpy(), tg.local_ids()["fsa"])
    assert {"id_fsa_k10", "id_fsa_k20", "id_mle_k10", "id_mle_k20"} <= set(adata.obs.columns)
    tp.sc.intrinsic_dim(adata, tg, id_k_values=[10])        # a single k used to be skipped with an error
    assert "10" in adata.uns["intrinsic_dim_estimator"]["local_id"]["fsa"]


def test_sc_pseudotime_matches_topograph(fitted_adata):
    import topo as tp
    adata, tg = fitted_adata
    out = tp.sc.pseudotime_analysis(adata, tg, starting_cluster="a", groupby="group", verbose=False)
    expected = tg.pseudotime(root=out["root"], k=out["k_use"], multiscale=True)["pseudotime"]
    np.testing.assert_allclose(adata.obs["topo_pseudotime"].to_numpy(), expected, atol=1e-12)


def test_plot_riemann_diagnostics_honours_its_arguments(fitted_adata):
    """`diffusion_t` was ignored, the palette had to exist in .uns, and one panel was always
    colored by 'topo_clusters'."""
    import matplotlib.pyplot as plt
    import topo as tp
    adata, tg = fitted_adata
    assert "group_colors" not in adata.uns and "topo_clusters" not in adata.obs
    deformation = {}
    for t in (0, 4):
        fig = tp.sc.plot_riemann_diagnostics(adata, tg, proj_key="X_TopoMAP", groupby="group",
                                             diffusion_t=t, show=False, verbose=False)
        plt.close(fig)
        deformation[t] = adata.obs["deformation_TopoMAP"].to_numpy().copy()
    assert not np.allclose(deformation[0], deformation[4])
    for groupby in (None, "not a column"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.close(tp.sc.plot_riemann_diagnostics(adata, tg, groupby=groupby, show=False, verbose=False))


def test_sc_riemann_diagnostics_stores_the_laplacian_it_uses(fitted_adata):
    import topo as tp
    adata, tg = fitted_adata
    tp.sc.riemann_diagnostics(adata, tg, diffusion_t=0, diffusion_op=None)
    assert abs(adata.obsp["topometry_laplacian"] - tg.graph_kernel.L).max() == 0
    expected = tg.riemann_diagnostics(Y=adata.obsm["X_TopoMAP"], L=tg.graph_kernel.L, compute_metric=False)
    np.testing.assert_allclose(adata.obs["metric_deformation__X_TopoMAP"].to_numpy(), expected["deformation"])


def test_topological_workflow_runs_with_its_defaults():
    """Its default 'LE' eigenmap and its graph-kernel keys no longer existed in TopOGraph."""
    anndata = pytest.importorskip("anndata")
    pytest.importorskip("scanpy")
    import topo as tp
    adata = anndata.AnnData(_blobs(n=200).astype(np.float32))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = tp.sc.topological_workflow(adata, _topograph(delta=2.0), kernels=["bw_adaptive", "cknn"])
    assert "X_MAP of cknn from msDM with bw_adaptive" in out.obsm
    assert "X_Isomap of bw_adaptive from DM with cknn" in out.obsm
    assert "bw_adaptive from msDM with cknn_leiden" in out.obs
    assert out.obsp["msDM with cknn_distances"].shape == (200, 200)
    with pytest.warns(UserWarning, match="LE"):
        tp.sc.topological_workflow(anndata.AnnData(_blobs(n=200).astype(np.float32)), _topograph(),
                                   kernels=["bw_adaptive"], eigenmap_methods=["DM", "LE"], projections=["MAP"])


def _fake_bbknn(adata, batch_key=None, use_rep="X_pca", neighbors_within_batch=3, **kwargs):
    """Stands in for sc.external.pp.bbknn: writes neighbor distances and UMAP-style connectivities."""
    Z = np.asarray(adata.obsm[use_rep])
    G = NearestNeighbors(n_neighbors=K + 1).fit(Z).kneighbors_graph(Z, mode="distance").tocsr()
    G.setdiag(0.0)
    G.eliminate_zeros()
    C = G.copy()
    C.data = np.exp(-C.data / C.data.mean())
    adata.obsp["distances"], adata.obsp["connectivities"] = G, C


def test_integrate_bbknn_builds_kernels_from_distances(monkeypatch):
    """It handed BBKNN's connectivities - affinities - to TopOGraph as distances."""
    import types
    anndata = pytest.importorskip("anndata")
    sc = pytest.importorskip("scanpy")
    import topo as tp
    monkeypatch.setitem(sys.modules, "bbknn", types.ModuleType("bbknn"))
    monkeypatch.setattr(sc.external.pp, "bbknn", _fake_bbknn)
    X = _blobs(n=200).astype(np.float32)
    adata = anndata.AnnData(X.copy())
    adata.obs["batch"] = np.where(np.arange(200) % 2 == 0, "a", "b")
    tg = _topograph(projection_methods=["MAP"], id_ks=10)
    tp.sc.integrate_bbknn(adata, batch_key="batch", tg=tg)

    rows = np.repeat(np.arange(200), np.diff(tg.base_knn_graph.indptr))
    np.testing.assert_allclose(tg.base_knn_graph.data, euclidean_distances(X)[rows, tg.base_knn_graph.indices],
                               rtol=1e-4, atol=1e-4)
    assert _per_row_spearman(tg.base_kernel.K, euclidean_distances(X)) < -0.6
    msZ = tg.spectral_scaffold(multiscale=True)
    assert _per_row_spearman(tg._kernel_msZ.K, euclidean_distances(msZ)) < -0.6


def test_blend_distance_graphs_keeps_distances():
    pytest.importorskip("scanpy")
    from topo.single_cell import _blend_distance_graphs
    ref = csr_matrix(np.array([[0, 2.0, 0], [2.0, 0, 4.0], [0, 4.0, 0]]))
    other = csr_matrix(np.array([[0, 6.0, 1.0], [6.0, 0, 0], [1.0, 0, 0]]))
    out = _blend_distance_graphs(ref, other, alpha=0.25).toarray()
    np.testing.assert_allclose(out, [[0, 3.0, 1.0], [3.0, 0, 4.0], [1.0, 4.0, 0]])


# ── display in notebooks ──────────────────────────────────────────────────────
def test_listing_and_displaying_an_estimator_evaluates_nothing(fitted):
    """
    scikit-learn's dir() and notebook display read every attribute of an estimator. On a
    Kernel that raised ValueError ('knn' of a precomputed kernel) or started an all-pairs
    shortest-path computation ('SP'), so displaying a TopOGraph in Jupyter failed.
    """
    X, tg = fitted
    kernel = Kernel(metric="euclidean", n_neighbors=K, backend="sklearn", n_jobs=1).fit(X)
    for obj in (tg, tg.base_kernel, kernel, tg.eigenbasis, Kernel(), _topograph()):
        assert "fit" in dir(obj)
        bundle = obj._repr_mimebundle_(include=None, exclude=None)     # as IPython calls it
        assert set(bundle) == {"text/plain"} and bundle["text/plain"] == repr(obj)
        assert not hasattr(obj, "_repr_html_")      # IPython asks for this one separately
    assert kernel._SP is None, "listing the attributes computed the shortest paths"
    assert not hasattr(tg.base_kernel, "SP")        # no distance graph is kept for it

    # an unfitted kernel has no kernel matrix: hasattr says so instead of raising
    assert not hasattr(Kernel(), "K") and not hasattr(tg.base_kernel, "knn")
    with pytest.raises(ValueError):     # still a ValueError for code that catches it
        Kernel().K


def test_ipython_shows_the_text_summary(fitted, capsys):
    formatters = pytest.importorskip("IPython.core.formatters")
    _, tg = fitted
    for obj in (tg, tg.base_kernel, tg.eigenbasis):
        data, _ = formatters.DisplayFormatter().format(obj)
        assert set(data) == {"text/plain"} and data["text/plain"] == repr(obj)
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err and "Traceback" not in captured.out
