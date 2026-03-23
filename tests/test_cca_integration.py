import numpy as np
import scipy.sparse as sp
import pytest
import anndata
from anndata import AnnData
from topo.single_cell import run_cca_integration


def _make_adata(n: int, p: int, label: str, seed: int,
                n_types: int = 4) -> AnnData:
    """
    Structured toy data: n_types Gaussian clusters, integer counts,
    ~80% sparsity. Passes _is_raw_counts() heuristic.
    """
    rng = np.random.default_rng(seed)
    ct = rng.integers(0, n_types, n)
    centers = rng.normal(0, 3, (n_types, p))
    X = np.exp(centers[ct] + rng.normal(0, 0.8, (n, p)))
    X[rng.random((n, p)) < 0.8] = 0.0
    X = np.round(X).astype(np.float32)   # integer-valued floats → raw counts
    a = AnnData(X=sp.csr_matrix(X))
    a.obs_names = [f"{label}_c{i}" for i in range(n)]
    a.var_names = [f"gene{j:04d}" for j in range(p)]
    a.obs["batch"] = label
    return a


# ── Test 1: Basic two-batch, all adaptive params ──────────────────────
def test_two_batch_adaptive():
    a = _make_adata(300, 500, "A", seed=0)
    b = _make_adata(280, 500, "B", seed=1)
    result = run_cca_integration([a, b], scale_output=False)

    assert result.n_obs == 580
    # n_vars must equal the recorded n_features value
    assert result.n_vars == result.uns["cca_integration"]["n_features"]
    assert result.n_vars >= 100, "Expected at least 100 integration features"
    assert sp.issparse(result.X)
    assert result.X.dtype == np.float32
    assert set(result.obs["batch"].unique()) == {"A", "B"}
    assert "original" in result.layers
    assert "cca_integration" in result.uns

    meta = result.uns["cca_integration"]
    assert meta["n_datasets"] == 2
    assert meta["correction_mode"] == "symmetric"
    assert meta["n_components"] == 30
    assert meta["n_anchors_filt_per_merge"][0] > 0, \
        "Structured data should yield anchors"
    # All per-merge lists must have exactly N-1 = 1 entry for 2 datasets
    for key in ["k_anchor_per_merge", "k_filter_per_merge",
                "k_score_per_merge", "k_weight_per_merge",
                "sd_bandwidth_per_merge", "n_anchors_raw_per_merge",
                "n_anchors_filt_per_merge"]:
        assert len(meta[key]) == 1, f"{key} should have 1 entry"


# ── Test 2: Output is in log-normalised space (not z-scored) ──────────
def test_output_is_lognorm_space():
    a = _make_adata(300, 500, "A", seed=2)
    b = _make_adata(300, 500, "B", seed=3)
    result = run_cca_integration([a, b], scale_output=False)

    X = result.X.toarray()
    assert X.mean() > 0.0,  "Mean must be > 0 in lognorm space"
    assert X.max()  > 1.0,  "Max must be > 1 in lognorm space"
    assert X.min()  >= 0.0, "Values must be >= 0 (clamp enforced)"

    X_orig = result.layers["original"].toarray()
    assert X_orig.mean() > 0.0
    assert X_orig.max()  > 1.0
    assert X_orig.min()  >= 0.0


# ── Test 3: Symmetric correction — both batches shift ─────────────────
def test_symmetric_correction_both_batches_shift():
    a = _make_adata(300, 500, "A", seed=4)
    b = _make_adata(300, 500, "B", seed=5)
    result = run_cca_integration([a, b], scale_output=False)

    X_orig = result.layers["original"].toarray()
    X_corr = result.X.toarray()
    batch  = result.obs["batch"].values

    idx_a = np.where(batch == "A")[0]
    idx_b = np.where(batch == "B")[0]

    shift_a = np.abs(X_corr[idx_a].mean() - X_orig[idx_a].mean())
    shift_b = np.abs(X_corr[idx_b].mean() - X_orig[idx_b].mean())
    assert shift_a > 0.0, "Batch A must shift (symmetric correction)"
    assert shift_b > 0.0, "Batch B must shift (symmetric correction)"


# ── Test 4: Single AnnData + batch_key ────────────────────────────────
def test_single_adata_batch_key():
    a = _make_adata(200, 400, "A", seed=6)
    b = _make_adata(200, 400, "B", seed=7)
    combined = anndata.concat([a, b])
    result = run_cca_integration(combined, batch_key="batch", scale_output=False)
    assert result.n_obs == 400
    assert set(result.obs["batch"].unique()) == {"A", "B"}


# ── Test 5: Three-batch guide tree ────────────────────────────────────
def test_three_batches_guide_tree():
    batches = [_make_adata(150, 400, f"B{i}", seed=i) for i in range(3)]
    result  = run_cca_integration(batches, scale_output=False)
    assert result.n_obs == 450
    assert set(result.obs["batch"].unique()) == {"B0", "B1", "B2"}
    meta = result.uns["cca_integration"]
    assert len(meta["merge_order"])           == 2
    assert len(meta["n_anchors_filt_per_merge"]) == 2
    assert len(meta["n_anchors_raw_per_merge"])  == 2
    assert len(meta["k_anchor_per_merge"])    == 2
    assert len(meta["sd_bandwidth_per_merge"]) == 2


# ── Test 6: Manual hyperparameters respected ──────────────────────────
def test_manual_hyperparameters():
    a = _make_adata(200, 400, "A", seed=8)
    b = _make_adata(200, 400, "B", seed=9)
    result = run_cca_integration(
        [a, b],
        n_features=200, n_components=15,
        k_anchor=5, k_filter=60, k_score=20, k_weight=50,
        sd_bandwidth=1.0, scale_output=False)
    assert result.n_vars == 200
    meta = result.uns["cca_integration"]
    assert meta["n_components"]               == 15
    assert meta["k_anchor_per_merge"][0]      == 5
    assert meta["k_score_per_merge"][0]       == 20
    assert meta["k_filter_per_merge"][0]      == 60
    assert meta["sd_bandwidth_per_merge"][0]  == 1.0
    assert meta["k_weight_per_merge"][0]      == 50


# ── Test 7: Zero-anchor fallback — no crash ───────────────────────────
def test_zero_anchor_fallback():
    import warnings
    X_a = sp.random(40, 100, density=0.1, format="csr", random_state=0)
    X_b = sp.random(40, 100, density=0.1, format="csr", random_state=1)
    a = AnnData(X=X_a.astype(np.float32))
    b = AnnData(X=X_b.astype(np.float32))
    a.obs_names = [f"A{i}" for i in range(40)]
    b.obs_names = [f"B{i}" for i in range(40)]
    a.var_names = b.var_names = [f"g{j}" for j in range(100)]
    a.obs["batch"] = "A"
    b.obs["batch"] = "B"
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = run_cca_integration(
            [a, b], n_features=80, n_components=5,
            k_anchor=2, k_filter=0, scale_output=False)
    assert result.n_obs == 80


# ── Test 8: raw >= filt anchor counts ─────────────────────────────────
def test_anchor_counts_raw_ge_filtered():
    a = _make_adata(200, 400, "A", seed=10)
    b = _make_adata(200, 400, "B", seed=11)
    result = run_cca_integration([a, b], scale_output=False)
    meta = result.uns["cca_integration"]
    assert meta["n_anchors_raw_per_merge"][0]  >= \
           meta["n_anchors_filt_per_merge"][0] >= 0


# ── Test 9: Corrected values non-negative ─────────────────────────────
def test_corrected_values_nonnegative():
    a = _make_adata(300, 500, "A", seed=14)
    b = _make_adata(300, 500, "B", seed=15)
    result = run_cca_integration([a, b], scale_output=False)
    X = result.X.toarray()
    assert X.min() >= 0.0, f"Min corrected value is {X.min():.4f} < 0"


# ── Test 10: Multithreaded — no gross divergence ───────────────────────
def test_multithreaded_no_gross_divergence():
    a = _make_adata(200, 400, "A", seed=12)
    b = _make_adata(200, 400, "B", seed=13)
    r1 = run_cca_integration([a.copy(), b.copy()], n_threads=1, seed=0, scale_output=False)
    r2 = run_cca_integration([a.copy(), b.copy()], n_threads=2, seed=0, scale_output=False)
    diff = np.abs((r1.X - r2.X).toarray())
    assert diff.max() < 0.5, \
        f"Multithreaded result diverges too much: max diff = {diff.max():.4f}"


# ── Test 11: Real data smoke test ─────────────────────────────────────
def test_pbmc3k_smoke():
    import scanpy as sc_
    adata = sc_.datasets.pbmc3k()
    rng = np.random.default_rng(42)
    adata.obs["batch"] = rng.choice(["b1", "b2"], size=adata.n_obs)

    result = run_cca_integration(adata, batch_key="batch", n_threads=2, scale_output=False)

    assert result.n_obs == adata.n_obs
    assert sp.issparse(result.X)
    assert result.X.dtype == np.float32

    meta = result.uns["cca_integration"]
    assert meta["n_anchors_filt_per_merge"][0] > 0, \
        "Expected anchors on structured PBMC data"
    assert meta["correction_mode"] == "symmetric"
    assert result.n_vars == meta["n_features"]

    X = result.X.toarray()
    assert X.mean()  > 0.0,  "Output must be in lognorm space"
    assert X.min()  >= 0.0,  "Output must be >= 0"

    print(
        f"\nPBMC smoke: {result.n_obs} cells, {result.n_vars} features\n"
        f"  k_anchor:    {meta['k_anchor_per_merge']}\n"
        f"  n_anchors:   raw={meta['n_anchors_raw_per_merge']}  "
        f"filt={meta['n_anchors_filt_per_merge']}\n"
        f"  sd_bandwidth:{meta['sd_bandwidth_per_merge']}"
    )


# ── Test 12: CCA gene loadings from raw U ─────────────────────────────
def test_cca_gene_loadings_from_raw_u():
    """
    U_k must be computed from raw singular vectors, not row-normalised ones.
    Verify that column norms of U_k are 1.0 and cc_a rows are unit-norm.
    """
    from topo.single_cell import _compute_cca
    rng = np.random.default_rng(0)
    n_a, n_b, p = 80, 60, 200
    X_a = rng.standard_normal((n_a, p)).astype(np.float32)
    X_b = rng.standard_normal((n_b, p)).astype(np.float32)
    cc_a, cc_b, U_k = _compute_cca(X_a, X_b, n_components=10, seed=0)

    # U_k columns must be unit-norm (column-normalised)
    col_norms = np.linalg.norm(U_k, axis=0)
    np.testing.assert_allclose(col_norms, 1.0, atol=1e-5,
        err_msg="U_k column norms must equal 1.0 (column-normalised)")

    # cc_a rows must be unit-norm (row-normalised)
    row_norms = np.linalg.norm(cc_a, axis=1)
    np.testing.assert_allclose(row_norms, 1.0, atol=1e-5,
        err_msg="cc_a row norms must equal 1.0 (row-normalised)")

    # Shape checks
    assert U_k.shape == (p, min(10, min(n_a, n_b) - 1))
    assert cc_a.shape[0] == n_a
    assert cc_b.shape[0] == n_b
