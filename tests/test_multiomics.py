"""
Tests for multi-omic integration functions:
  - atac_lsi
  - wnn_integration
  - compute_gene_activity_scores (requires pyranges)
  - ModalityBridge (fit_modality_bridge, apply_modality_bridge)
  - fit_adata WNN pathway
"""
import numpy as np
import scipy.sparse as sp
import pytest
import anndata as ad
from anndata import AnnData

import topo as tp
from topo.topograph import TopOGraph


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def atac_adata():
    """Sparse binary ATAC peak matrix with depth artifact."""
    np.random.seed(42)
    n_cells, n_peaks = 200, 5000
    X = sp.random(n_cells, n_peaks, density=0.05, format="csr")
    X.data = np.ones_like(X.data)
    X = X.astype(float)
    # Depth artifact: cells 0-99 have 2x fragments
    depth = np.ones(n_cells)
    depth[:100] = 2.0
    X = sp.diags(depth) @ X
    adata = AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"peak_{i}" for i in range(n_peaks)]
    return adata


@pytest.fixture
def paired_multiome():
    """Paired RNA+ATAC data with 3 cell types (clear block signal)."""
    rng = np.random.default_rng(42)
    n_cells = 300
    n_genes = 1000
    n_peaks = 4000
    n_ct = 3
    ct = np.repeat(np.arange(n_ct), n_cells // n_ct)

    # RNA
    centers_rna = np.zeros((n_ct, n_genes))
    for c in range(n_ct):
        centers_rna[c, c * 300:(c + 1) * 300] = 5.0
    X_rna = centers_rna[ct] + rng.normal(0, 1.0, (n_cells, n_genes))
    X_rna = np.maximum(X_rna, 0)

    # ATAC
    centers_atac = np.zeros((n_ct, n_peaks))
    for c in range(n_ct):
        centers_atac[c, c * 500:(c + 1) * 500] = 1.0
    X_atac = ((centers_atac[ct] + rng.normal(0, 0.3, (n_cells, n_peaks))) > 0.5).astype(float)

    adata_rna = AnnData(X=sp.csr_matrix(X_rna))
    adata_rna.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata_rna.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata_rna.var["highly_variable"] = True

    adata_atac = AnnData(X=sp.csr_matrix(X_atac))
    adata_atac.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata_atac.var_names = [f"peak_{i}" for i in range(n_peaks)]

    return adata_rna, adata_atac, ct


# ── TestAtacLsi ───────────────────────────────────────────────────────

class TestAtacLsi:
    def test_output_shape(self, atac_adata):
        tp.sc.atac_lsi(atac_adata, n_components=30, inplace=True)
        lsi = atac_adata.obsm["X_lsi"]
        assert lsi.shape == (200, 29)  # 30 - 1 for depth component

    def test_l2_normalization(self, atac_adata):
        tp.sc.atac_lsi(atac_adata, n_components=30, inplace=True)
        norms = np.linalg.norm(atac_adata.obsm["X_lsi"], axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_component1_discarded(self, atac_adata):
        """Output has n_components-1 columns (depth component removed)."""
        tp.sc.atac_lsi(atac_adata, n_components=50, inplace=True)
        assert atac_adata.obsm["X_lsi"].shape[1] == 49

    def test_inplace_false_no_modification(self, atac_adata):
        result = tp.sc.atac_lsi(atac_adata, n_components=20, inplace=False)
        assert result is not None
        assert result.shape == (200, 19)
        assert "X_lsi" not in atac_adata.obsm

    def test_zero_count_cell_warning(self):
        """Cell with all-zero peaks should emit a warning."""
        X = sp.csr_matrix((50, 1000), dtype=float)
        adata = AnnData(X=X)
        adata.var_names = [f"p{i}" for i in range(1000)]
        adata.obs_names = [f"c{i}" for i in range(50)]
        with pytest.warns(UserWarning, match="zero total counts"):
            tp.sc.atac_lsi(adata, n_components=10, inplace=True)

    def test_uns_metadata(self, atac_adata):
        tp.sc.atac_lsi(atac_adata, n_components=30, inplace=True)
        assert "lsi" in atac_adata.uns
        assert atac_adata.uns["lsi"]["n_components"] == 30
        assert 1 not in atac_adata.uns["lsi"]["components_used"]

    def test_precomputed_bypass(self, atac_adata):
        """wnn_integration skips atac_lsi when pre-existing LSI is present."""
        tp.sc.atac_lsi(atac_adata, n_components=20, inplace=True)
        lsi_before = atac_adata.obsm["X_lsi"].copy()
        # Build a minimal RNA adata for WNN
        rng = np.random.default_rng(99)
        n = atac_adata.n_obs
        adata_rna = AnnData(X=sp.csr_matrix(rng.normal(0, 1, (n, 200))))
        adata_rna.obs_names = atac_adata.obs_names.copy()
        adata_rna.var_names = [f"g{i}" for i in range(200)]
        adata_rna.var["highly_variable"] = True
        # atac_use_precomputed_lsi=True should not recompute LSI
        wnn = tp.sc.wnn_integration(
            {"rna": adata_rna, "atac": atac_adata},
            atac_use_precomputed_lsi=True, n_neighbors=10,
        )
        assert np.array_equal(atac_adata.obsm["X_lsi"], lsi_before)


# ── TestWnnIntegration ────────────────────────────────────────────────

class TestWnnIntegration:
    def test_output_graph_shape(self, paired_multiome):
        adata_rna, adata_atac, _ = paired_multiome
        tp.sc.atac_lsi(adata_atac, n_components=15, inplace=True)
        wnn = tp.sc.wnn_integration(
            {"rna": adata_rna, "atac": adata_atac},
            atac_use_precomputed_lsi=True, n_neighbors=15,
        )
        assert wnn.obsp["WNN"].shape == (300, 300)

    def test_row_stochastic(self, paired_multiome):
        adata_rna, adata_atac, _ = paired_multiome
        tp.sc.atac_lsi(adata_atac, n_components=15, inplace=True)
        wnn = tp.sc.wnn_integration(
            {"rna": adata_rna, "atac": adata_atac},
            atac_use_precomputed_lsi=True, n_neighbors=15,
        )
        row_sums = np.array(wnn.obsp["WNN"].sum(axis=1)).flatten()
        assert np.allclose(row_sums, 1.0, atol=1e-3)

    def test_values_in_range(self, paired_multiome):
        adata_rna, adata_atac, _ = paired_multiome
        tp.sc.atac_lsi(adata_atac, n_components=15, inplace=True)
        wnn = tp.sc.wnn_integration(
            {"rna": adata_rna, "atac": adata_atac},
            atac_use_precomputed_lsi=True, n_neighbors=15,
        )
        W = wnn.obsp["WNN"]
        assert W.min() >= -1e-6
        assert W.max() <= 1 + 1e-6

    def test_modality_weights_shape(self, paired_multiome):
        adata_rna, adata_atac, _ = paired_multiome
        tp.sc.atac_lsi(adata_atac, n_components=15, inplace=True)
        wnn = tp.sc.wnn_integration(
            {"rna": adata_rna, "atac": adata_atac},
            atac_use_precomputed_lsi=True, n_neighbors=15,
        )
        assert wnn.obsm["modality_weights"].shape == (300, 2)

    def test_modality_weights_sum_to_one(self, paired_multiome):
        adata_rna, adata_atac, _ = paired_multiome
        tp.sc.atac_lsi(adata_atac, n_components=15, inplace=True)
        wnn = tp.sc.wnn_integration(
            {"rna": adata_rna, "atac": adata_atac},
            atac_use_precomputed_lsi=True, n_neighbors=15,
        )
        sums = wnn.obsm["modality_weights"].sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-6)

    def test_csr_format(self, paired_multiome):
        adata_rna, adata_atac, _ = paired_multiome
        tp.sc.atac_lsi(adata_atac, n_components=15, inplace=True)
        wnn = tp.sc.wnn_integration(
            {"rna": adata_rna, "atac": adata_atac},
            atac_use_precomputed_lsi=True, n_neighbors=15,
        )
        assert sp.issparse(wnn.obsp["WNN"])
        assert wnn.obsp["WNN"].format == "csr"

    def test_wnn_clustering_quality(self, paired_multiome):
        """WNN ARI should be >= 0.50 on synthetic data with clear structure."""
        from sklearn.metrics import adjusted_rand_score
        from sklearn.cluster import AgglomerativeClustering

        adata_rna, adata_atac, ct = paired_multiome
        tp.sc.atac_lsi(adata_atac, n_components=15, inplace=True)
        wnn = tp.sc.wnn_integration(
            {"rna": adata_rna, "atac": adata_atac},
            atac_use_precomputed_lsi=True, n_neighbors=15,
        )
        W = wnn.obsp["WNN"]
        dense = 1.0 - W.toarray()
        np.fill_diagonal(dense, 0)
        dense = np.clip(dense, 0, None)
        cl = AgglomerativeClustering(n_clusters=3, metric="precomputed", linkage="average")
        labels = cl.fit_predict(dense)
        ari = adjusted_rand_score(ct, labels)
        assert ari >= 0.50, f"WNN ARI {ari:.4f} below 0.50 floor"

    def test_mudata_input(self, paired_multiome):
        """If mudata installed, dict and MuData inputs give identical WNN."""
        try:
            import mudata
        except ImportError:
            pytest.skip("mudata not installed")

        adata_rna, adata_atac, _ = paired_multiome
        tp.sc.atac_lsi(adata_atac, n_components=15, inplace=True)
        # Dict input
        wnn_dict = tp.sc.wnn_integration(
            {"rna": adata_rna, "atac": adata_atac},
            atac_use_precomputed_lsi=True, n_neighbors=15,
        )
        # MuData input
        mdata = mudata.MuData({"rna": adata_rna.copy(), "atac": adata_atac.copy()})
        wnn_mudata = tp.sc.wnn_integration(
            mdata, atac_use_precomputed_lsi=True, n_neighbors=15,
        )
        # Both pathways should produce structurally equivalent graphs.
        # Tolerance is loose because hnswlib kNN is nondeterministic across
        # separate runs (thread scheduling) and MuData may return views.
        diff = np.abs(wnn_dict.obsp["WNN"] - wnn_mudata.obsp["WNN"]).max()
        assert diff < 0.1, f"Dict/MuData WNN differ by {diff}"

    def test_protein_prenormalized_vs_internal(self):
        """CLR applied externally vs internally gives same WNN."""
        rng = np.random.default_rng(42)
        n = 100
        obs_names = [f"c{i}" for i in range(n)]

        # RNA
        rna = AnnData(X=sp.csr_matrix(rng.normal(3, 1, (n, 200)).clip(0)))
        rna.obs_names = obs_names
        rna.var_names = [f"g{i}" for i in range(200)]
        rna.var["highly_variable"] = True

        # Protein (raw counts)
        X_prot = rng.poisson(5, (n, 30)).astype(float)
        prot1 = AnnData(X=sp.csr_matrix(X_prot))
        prot1.obs_names = obs_names
        prot1.var_names = [f"prot{i}" for i in range(30)]

        # Pre-normalized protein (CLR)
        X_clr = X_prot + 1.0
        log_X = np.log(X_clr)
        geo = np.exp(log_X.mean(axis=1, keepdims=True))
        X_clr = np.log(X_clr / geo)
        prot2 = AnnData(X=sp.csr_matrix(X_clr))
        prot2.obs_names = obs_names
        prot2.var_names = [f"prot{i}" for i in range(30)]

        wnn1 = tp.sc.wnn_integration(
            {"rna": rna, "protein": prot1},
            protein_normalized=False, n_neighbors=10,
        )
        wnn2 = tp.sc.wnn_integration(
            {"rna": rna.copy(), "protein": prot2},
            protein_normalized=True, n_neighbors=10,
        )
        diff = np.abs(wnn1.obsp["WNN"] - wnn2.obsp["WNN"]).max()
        assert diff < 1e-4, f"CLR internal/external WNN differ by {diff}"


# ── TestModalityBridge ────────────────────────────────────────────────

class TestModalityBridge:
    def _make_bridge_data(self):
        rng = np.random.default_rng(42)
        n_ct = 3
        n_paired, n_query = 300, 201
        n_genes, n_peaks = 1000, 4000

        ct_p = np.repeat(np.arange(n_ct), n_paired // n_ct)
        ct_q = np.array([i % n_ct for i in range(n_query)])

        def _rna(n, ct):
            c = np.zeros((n_ct, n_genes))
            for i in range(n_ct):
                c[i, i*300:(i+1)*300] = 5.0
            return sp.csr_matrix(np.maximum(c[ct] + rng.normal(0, 0.5, (n, n_genes)), 0))

        def _atac(n, ct):
            c = np.zeros((n_ct, n_peaks))
            for i in range(n_ct):
                c[i, i*500:(i+1)*500] = 1.0
            return sp.csr_matrix(((c[ct] + rng.normal(0, 0.3, (n, n_peaks))) > 0.5).astype(float))

        # Paired reference
        adata_p = AnnData(X=_rna(n_paired, ct_p))
        adata_p.var_names = [f"g{i}" for i in range(n_genes)]
        adata_p.obs_names = [f"p{i}" for i in range(n_paired)]
        adata_p.var["highly_variable"] = True

        atac_p = AnnData(X=_atac(n_paired, ct_p))
        atac_p.var_names = [f"pk{i}" for i in range(n_peaks)]
        atac_p.obs_names = [f"p{i}" for i in range(n_paired)]
        tp.sc.atac_lsi(atac_p, n_components=20, inplace=True)
        adata_p.obsm["X_lsi"] = atac_p.obsm["X_lsi"]

        # Query ATAC
        atac_q = AnnData(X=_atac(n_query, ct_q))
        atac_q.var_names = [f"pk{i}" for i in range(n_peaks)]
        atac_q.obs_names = [f"q{i}" for i in range(n_query)]
        tp.sc.atac_lsi(atac_q, n_components=20, inplace=True)

        return adata_p, atac_q, ct_p, ct_q

    def test_fit_returns_bridge_object(self):
        adata_p, _, _, _ = self._make_bridge_data()
        bridge = tp.sc.fit_modality_bridge(adata_p, k=10)
        assert bridge.is_fitted
        from topo.bridge import ModalityBridge
        assert isinstance(bridge, ModalityBridge)

    def test_bridge_hvg_names_stored(self):
        adata_p, _, _, _ = self._make_bridge_data()
        bridge = tp.sc.fit_modality_bridge(adata_p, k=10)
        assert len(bridge.hvg_names) == 1000

    def test_apply_output_shape(self):
        adata_p, atac_q, _, _ = self._make_bridge_data()
        bridge = tp.sc.fit_modality_bridge(adata_p, k=10)
        imputed = tp.sc.apply_modality_bridge(atac_q, bridge, inplace=False)
        assert imputed.shape == (201, 1000)

    def test_imputed_values_nonnegative(self):
        adata_p, atac_q, _, _ = self._make_bridge_data()
        bridge = tp.sc.fit_modality_bridge(adata_p, k=10)
        imputed = tp.sc.apply_modality_bridge(atac_q, bridge, inplace=False)
        assert (imputed.X >= 0).all()

    def test_serialization_roundtrip(self, tmp_path):
        adata_p, atac_q, _, _ = self._make_bridge_data()
        bridge = tp.sc.fit_modality_bridge(adata_p, k=10)
        imputed1 = tp.sc.apply_modality_bridge(atac_q, bridge, inplace=False)

        path = str(tmp_path / "bridge.joblib")
        bridge.save(path)
        bridge2 = tp.sc.ModalityBridge.load(path)
        imputed2 = tp.sc.apply_modality_bridge(atac_q, bridge2, inplace=False)
        assert np.allclose(imputed1.X, imputed2.X, atol=1e-5)

    def test_lsi_dim_mismatch_raises(self):
        import copy
        adata_p, atac_q, _, _ = self._make_bridge_data()
        bridge = tp.sc.fit_modality_bridge(adata_p, k=10)
        bad = copy.deepcopy(bridge)
        bad.lsi_coords = bad.lsi_coords[:, :5]
        bad._rebuild_index()
        with pytest.raises(ValueError, match="LSI component count mismatch"):
            tp.sc.apply_modality_bridge(atac_q, bad, inplace=False)

    def test_label_transfer_ari(self):
        """Imputed RNA preserves cell-type structure (ARI >= 0.70)."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score

        adata_p, atac_q, ct_p, ct_q = self._make_bridge_data()
        bridge = tp.sc.fit_modality_bridge(adata_p, k=10)
        imputed = tp.sc.apply_modality_bridge(atac_q, bridge, inplace=False)
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = km.fit_predict(imputed.X)
        ari = adjusted_rand_score(ct_q, labels)
        assert ari >= 0.70, f"Label transfer ARI {ari:.4f} below 0.70"


# ── TestFitAdataWnn ───────────────────────────────────────────────────

class TestFitAdataWnn:
    def test_precomputed_graph_key_accepted(self, paired_multiome):
        adata_rna, adata_atac, _ = paired_multiome
        tp.sc.atac_lsi(adata_atac, n_components=15, inplace=True)
        wnn = tp.sc.wnn_integration(
            {"rna": adata_rna, "atac": adata_atac},
            atac_use_precomputed_lsi=True, n_neighbors=15,
        )
        adata_rna.obsp["WNN"] = wnn.obsp["WNN"]
        tg = tp.sc.fit_adata(
            adata_rna,
            precomputed_graph_key="WNN",
            graph_input_type="affinity",
            do_leiden=False,
        )
        assert "X_ms_spectral_scaffold" in adata_rna.obsm
        assert "X_spectral_scaffold" in adata_rna.obsm

    def test_graph_input_type_affinity_accepted(self):
        n = 50
        np.random.seed(42)
        from sklearn.neighbors import kneighbors_graph
        X = np.random.randn(n, 10)
        knn = kneighbors_graph(X, n_neighbors=10, mode="connectivity").tocsr().astype(float)
        knn_norm = sp.diags(1.0 / np.array(knn.sum(axis=1)).flatten()) @ knn
        tg = TopOGraph(graph_input_type="affinity", min_eigs=10, verbosity=0)
        tg.fit(knn_norm)
        assert tg.n == 50

    def test_default_behavior_unchanged(self):
        """Default fit_adata (no precomputed graph) still works."""
        rng = np.random.default_rng(99)
        n, p = 200, 500
        X = rng.normal(3, 1, (n, p)).astype(np.float32)
        X = np.maximum(X, 0)
        adata = AnnData(X=sp.csr_matrix(X))
        adata.obs_names = [f"c{i}" for i in range(n)]
        adata.var_names = [f"g{i}" for i in range(p)]
        tg = tp.sc.fit_adata(adata, do_leiden=False, projections=None)
        assert "X_ms_spectral_scaffold" in adata.obsm
        assert "X_spectral_scaffold" in adata.obsm

    def test_end_to_end_wnn_to_scaffold(self, paired_multiome):
        """Full pipeline: wnn_integration -> fit_adata -> scaffold coordinates."""
        adata_rna, adata_atac, _ = paired_multiome
        tp.sc.atac_lsi(adata_atac, n_components=15, inplace=True)
        wnn = tp.sc.wnn_integration(
            {"rna": adata_rna, "atac": adata_atac},
            atac_use_precomputed_lsi=True, n_neighbors=15,
        )
        adata_rna.obsp["WNN"] = wnn.obsp["WNN"]
        tg = tp.sc.fit_adata(
            adata_rna,
            precomputed_graph_key="WNN",
            graph_input_type="affinity",
            do_leiden=False,
        )
        scaffold = adata_rna.obsm["X_ms_spectral_scaffold"]
        assert scaffold.shape[0] == 300
        assert scaffold.shape[1] > 0
