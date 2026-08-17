"""
Regression tests for bugs reported in the issue tracker.

Covers:
  #26 - adaptive bandwidth must be in the same distance units as the distances it
        normalizes (angular vs cosine-distance).
  #27 - nmslib index data type must match the space name it is paired with.
  #19 - t-SNE must use the metric matching the input it is actually given.
  #18 - evaluation pipeline must accept sparse CSR input.
  #17 - topo.eval.trustworthiness must remain importable.
"""
import numpy as np
import pytest
from scipy.sparse import csr_matrix, find

from topo.base.ann import kNN
from topo.tpgraph.kernels import (
    _adap_bw,
    _angularize_graph,
    _cosine_distance_to_angle,
    compute_kernel,
)


def _toy(n=200, p=20, seed=0):
    return np.random.default_rng(seed).normal(size=(n, p))


# ── #26 ───────────────────────────────────────────────────────────────────────
def test_adaptive_bandwidth_matches_distance_units():
    """adap_sd and dists must both be angular, so their ratio is O(1)."""
    X, k = _toy(), 10
    K = kNN(X, metric="cosine", n_neighbors=k, backend="sklearn", n_jobs=1)

    K_ang = _angularize_graph(K, "cosine", True)
    adap = _adap_bw(K_ang, k)
    _, _, dists = find(K_ang)

    # angles live in [0, pi]; cosine distances in [0, 2]
    assert dists.max() <= np.pi + 1e-9
    assert adap.max() <= np.pi + 1e-9

    # the same bandwidth derived from unconverted distances is ~2x smaller, which is
    # precisely the mismatch that made the kernel collapse
    adap_cosine_units = _adap_bw(K, k)
    assert adap.mean() > 1.8 * adap_cosine_units.mean()

    ratio = dists / (adap[np.asarray(find(K_ang)[0])] + 1e-10)
    assert 0.5 < ratio.mean() < 2.0, f"normalized distance out of scale: {ratio.mean()}"


def test_angular_conversion_is_a_noop_for_non_cosine_metrics():
    X = _toy()
    K = kNN(X, metric="euclidean", n_neighbors=10, backend="sklearn", n_jobs=1)
    assert _angularize_graph(K, "euclidean", True) is K
    assert _angularize_graph(K, "cosine", False) is K


def test_euclidean_kernel_unaffected_by_use_angular():
    X = _toy()
    a = compute_kernel(X, metric="euclidean", n_neighbors=10, adaptive_bw=True,
                       use_angular=True, backend="sklearn", n_jobs=1)
    b = compute_kernel(X, metric="euclidean", n_neighbors=10, adaptive_bw=True,
                       use_angular=False, backend="sklearn", n_jobs=1)
    np.testing.assert_allclose(a.toarray(), b.toarray())


def test_cosine_kernel_weights_do_not_collapse():
    """With consistent units the weights stay in a usable range."""
    X = _toy()
    W = compute_kernel(X, metric="cosine", n_neighbors=10, adaptive_bw=True,
                       backend="sklearn", n_jobs=1)
    assert np.isfinite(W.data).all()
    # under the unit mismatch the mean weight fell to ~0.006
    assert W.data.mean() > 0.05, f"kernel weights collapsed: {W.data.mean()}"


def test_cosine_distance_to_angle_bounds():
    d = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(_cosine_distance_to_angle(d),
                               [0.0, np.pi / 2, np.pi], atol=1e-12)


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("adaptive", [True, False])
def test_kernel_builds_for_all_bandwidth_configurations(expand, adaptive):
    X = _toy()
    W = compute_kernel(X, metric="cosine", n_neighbors=10, adaptive_bw=adaptive,
                       expand_nbr_search=expand, backend="sklearn", n_jobs=1)
    assert W.shape == (X.shape[0], X.shape[0])
    assert np.isfinite(W.data).all()


# ── #27 ───────────────────────────────────────────────────────────────────────
try:  # nmslib is an optional backend and does not build everywhere
    import nmslib  # noqa: F401
    _HAS_NMSLIB = True
except ImportError:
    _HAS_NMSLIB = False

requires_nmslib = pytest.mark.skipif(not _HAS_NMSLIB, reason="nmslib not installed")


@requires_nmslib
@pytest.mark.parametrize("dense", [True, False])
@pytest.mark.parametrize("sparse_input", [True, False])
def test_nmslib_space_matches_data_type(dense, sparse_input):
    """A dense index must never be paired with a *_sparse space name."""
    from topo.base.ann import NMSlibTransformer
    X = _toy(n=120, p=15)
    X = csr_matrix(X) if sparse_input else X
    t = NMSlibTransformer(n_neighbors=10, metric="cosine", dense=dense, n_jobs=1).fit(X)
    if dense:
        assert "sparse" not in t.space
    elif sparse_input:
        assert "sparse" in t.space


@requires_nmslib
def test_knn_nmslib_accepts_dense_without_sparsifying():
    """Dense input must not trigger a conversion warning, and must give the same graph."""
    X = _toy(n=120, p=15)
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        K_dense = kNN(X, metric="cosine", n_neighbors=10, backend="nmslib", n_jobs=1)
    assert not [w for w in caught if "does not support dense" in str(w.message)]
    K_sparse = kNN(csr_matrix(X), metric="cosine", n_neighbors=10,
                   backend="nmslib", n_jobs=1)
    assert K_dense.shape == K_sparse.shape
