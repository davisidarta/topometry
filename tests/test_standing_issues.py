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
