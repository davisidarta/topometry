import numpy as np
import scipy.sparse as sp
import pytest
import anndata
from anndata import AnnData
from pathlib import Path
from topo.single_cell import (
    run_cca_integration,
    save_cca_reference,
    load_cca_reference,
    map_to_cca_reference,
)


def _make_adata(n, p, label, seed, n_types=4):
    """Same helper as in test_cca_integration.py."""
    rng = np.random.default_rng(seed)
    ct = rng.integers(0, n_types, n)
    centers = rng.normal(0, 3, (n_types, p))
    X = np.exp(centers[ct] + rng.normal(0, 0.8, (n, p)))
    X[rng.random((n, p)) < 0.8] = 0.0
    X = np.round(X).astype(np.float32)
    a = AnnData(X=sp.csr_matrix(X))
    a.obs_names = [f"{label}_c{i}" for i in range(n)]
    a.var_names = [f"gene{j:04d}" for j in range(p)]
    a.obs["batch"] = label
    return a


@pytest.fixture
def reference(tmp_path):
    """A completed integration result saved to disk."""
    a = _make_adata(200, 400, "A", seed=0)
    b = _make_adata(200, 400, "B", seed=1)
    adata_int = run_cca_integration([a, b], scale_output=False)
    ref_path = tmp_path / "reference.h5ad"
    save_cca_reference(adata_int, ref_path)
    return adata_int, ref_path


# ── Test 1: run_cca_integration stores reference fields ──────────────────
def test_integration_stores_reference_fields():
    a = _make_adata(200, 400, "A", seed=0)
    b = _make_adata(200, 400, "B", seed=1)
    result = run_cca_integration([a, b], scale_output=False)

    assert "cca_loadings" in result.varm, \
        "varm['cca_loadings'] must be stored by run_cca_integration"
    assert "X_cca" in result.obsm, \
        "obsm['X_cca'] must be stored by run_cca_integration"
    assert "ref_mu"    in result.uns["cca_integration"]
    assert "ref_sigma" in result.uns["cca_integration"]

    U_k      = result.varm["cca_loadings"]
    cc_ref   = result.obsm["X_cca"]
    n_comp   = result.uns["cca_integration"]["n_components"]

    assert U_k.shape    == (result.n_vars, n_comp), \
        f"Expected ({result.n_vars}, {n_comp}), got {U_k.shape}"
    assert cc_ref.shape == (result.n_obs, n_comp), \
        f"Expected ({result.n_obs}, {n_comp}), got {cc_ref.shape}"


# ── Test 2: save and load round-trip ─────────────────────────────────────
def test_save_load_roundtrip(tmp_path, reference):
    adata_int, ref_path = reference
    loaded = load_cca_reference(ref_path)

    assert loaded.n_obs  == adata_int.n_obs
    assert loaded.n_vars == adata_int.n_vars
    assert "cca_loadings" in loaded.varm
    assert "X_cca"        in loaded.obsm
    np.testing.assert_allclose(
        loaded.varm["cca_loadings"],
        adata_int.varm["cca_loadings"], atol=1e-5)


# ── Test 3: load_cca_reference raises on missing fields ──────────────────
def test_load_raises_on_missing_fields(tmp_path):
    bad = AnnData(X=sp.csr_matrix(np.ones((10, 5), dtype=np.float32)))
    bad.obs_names = [f"c{i}" for i in range(10)]
    bad.var_names = [f"g{j}" for j in range(5)]
    bad_path = tmp_path / "bad.h5ad"
    bad.write_h5ad(bad_path)
    with pytest.raises(ValueError, match="missing required fields"):
        load_cca_reference(bad_path)


# ── Test 4: query_only — basic structure ─────────────────────────────────
def test_query_only_basic(reference):
    adata_int, _ = reference
    query = _make_adata(80, 400, "Q", seed=5)
    result = map_to_cca_reference(query, adata_int, mode="query_only")

    assert result.n_obs == 80
    assert result.n_vars == adata_int.n_vars
    assert sp.issparse(result.X)
    assert result.X.dtype == np.float32
    assert "original" in result.layers
    assert "cca_mapping" in result.uns
    assert result.uns["cca_mapping"]["mode"] == "query_only"


# ── Test 5: query_only — corrected values non-negative ───────────────────
def test_query_only_nonnegative(reference):
    adata_int, _ = reference
    query = _make_adata(80, 400, "Q", seed=6)
    result = map_to_cca_reference(query, adata_int, mode="query_only")
    X = result.X.toarray()
    assert X.min() >= 0.0, f"Corrected values must be >= 0, got min={X.min()}"


# ── Test 6: query_only — reference stays frozen ───────────────────────────
def test_query_only_reference_frozen(reference):
    adata_int, _ = reference
    X_ref_before = adata_int.X.toarray().copy()
    query = _make_adata(80, 400, "Q", seed=7)
    _ = map_to_cca_reference(query, adata_int, mode="query_only")
    X_ref_after = adata_int.X.toarray()
    np.testing.assert_array_equal(X_ref_before, X_ref_after,
        err_msg="Reference expression must not be modified in query_only mode")


# ── Test 7: query_only — query cells shift ───────────────────────────────
def test_query_only_cells_shift(reference):
    adata_int, _ = reference
    query = _make_adata(150, 400, "Q", seed=8)
    result = map_to_cca_reference(query, adata_int, mode="query_only")
    X_corr = result.X.toarray()
    X_orig = result.layers["original"].toarray()
    assert np.abs(X_corr - X_orig).max() > 0.0, \
        "Corrected and original query expression should differ"


# ── Test 8: query_only from disk ─────────────────────────────────────────
def test_query_only_from_path(tmp_path, reference):
    adata_int, ref_path = reference
    query = _make_adata(80, 400, "Q", seed=9)
    result = map_to_cca_reference(query, ref_path, mode="query_only")
    assert result.n_obs == 80


# ── Test 9: partial gene coverage — warn and impute ──────────────────────
def test_partial_gene_coverage_warns(reference):
    import warnings
    adata_int, _ = reference
    features = list(adata_int.var_names)
    query_full = _make_adata(80, 400, "Q", seed=10)
    query_partial = query_full[:, query_full.var_names[:300]].copy()
    coverage = len([g for g in features if g in set(query_partial.var_names)])
    frac = coverage / len(features)
    if frac >= 0.8:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = map_to_cca_reference(
                query_partial, adata_int, mode="query_only")
        assert result.n_obs == 80
    else:
        with pytest.raises(ValueError, match="covers only"):
            map_to_cca_reference(
                query_partial, adata_int,
                mode="query_only", min_shared_features=0.8)


# ── Test 10: full_reintegration — output contains all cells ──────────────
def test_full_reintegration_all_cells(reference):
    adata_int, _ = reference
    n_ref = adata_int.n_obs
    query = _make_adata(80, 400, "Q", seed=11)
    result = map_to_cca_reference(query, adata_int, mode="full_reintegration")
    assert result.n_obs == n_ref + 80
    assert result.n_vars == adata_int.n_vars


# ── Test 11: full_reintegration — feature set preserved ──────────────────
def test_full_reintegration_features_fixed(reference):
    adata_int, _ = reference
    query = _make_adata(80, 400, "Q", seed=12)
    result = map_to_cca_reference(query, adata_int, mode="full_reintegration")
    assert list(result.var_names) == list(adata_int.var_names), \
        "full_reintegration must preserve the reference feature set"


# ── Test 12: invalid mode raises ─────────────────────────────────────────
def test_invalid_mode_raises(reference):
    adata_int, _ = reference
    query = _make_adata(80, 400, "Q", seed=13)
    with pytest.raises(ValueError, match="mode must be"):
        map_to_cca_reference(query, adata_int, mode="invalid")


# ── Test 13: query_only — run log populated ──────────────────────────────
def test_query_only_run_log(reference):
    adata_int, _ = reference
    query = _make_adata(150, 400, "Q", seed=14)
    result = map_to_cca_reference(query, adata_int, mode="query_only")
    meta = result.uns["cca_mapping"]
    assert isinstance(meta["n_anchors_raw"],  int)
    assert isinstance(meta["n_anchors_filt"], int)
    assert meta["n_anchors_raw"] >= meta["n_anchors_filt"] >= 0
    assert meta["projection_method"] == "back_projected_U_k"
    assert meta["n_ref_cells"] == adata_int.n_obs
