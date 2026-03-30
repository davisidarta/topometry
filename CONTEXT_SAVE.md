# topometry Multi-Omic Implementation — Context Save
Generated: 2026-03-30T16:00:00Z
Session status: COMPLETE — All milestones done, all criteria met

---

## A. Repository structure
The package lives in `topo/` with `topo/__init__.py` as entry point. The main class `TopOGraph` is in `topo/topograph.py`. Single-cell integration is in `topo/single_cell.py` (exposed as `tp.sc` when scanpy is installed). kNN construction is in `topo/base/ann.py` via the `kNN()` function. Kernel/affinity construction is in `topo/tpgraph/kernels.py` via `Kernel` class and `compute_kernel()`. Tests are in `tests/` (2 files: `test_cca_integration.py`, `test_cca_mapping.py`). Notebooks are in `notebooks/` (7 pre-existing notebooks). Package is configured via `setup.cfg` (version 1.1.0). Imported as `import topometry as tp`.

## B. TopOGraph class — full digest
File: `topo/topograph.py`
Class: TopOGraph (inherits BaseEstimator, TransformerMixin)

Constructor parameters relevant to this implementation:
- `base_knn` (int, 30): k for base kNN graph
- `base_metric` (str, 'cosine'): distance metric for base kNN
- `base_kernel` (Kernel or None): pre-fitted kernel to reuse
- `base_kernel_version` (str, 'bw_adaptive'): kernel type
- `graph_knn` (int, 30): k for scaffold-space kNN
- `graph_kernel_version` (str, 'bw_adaptive'): kernel for scaffold graphs
- `graph_metric` (str, 'euclidean'): distance for scaffold kNN
- `backend` (str, 'hnswlib'): ANN backend
- `n_jobs` (int, -1): threads
- `uom` (bool, False): union-of-manifolds mode

How a user-supplied graph is currently handled:
In `fit()` at line 738: `if self.base_metric == 'precomputed': self.base_knn_graph = X.copy()`
The precomputed distance matrix is passed as X directly. Then at line 743: `if self.base_knn_graph is None:` — if it's not None (i.e., precomputed was set), kNN construction is skipped. Then at line 766: `self._compute_kernel_from_version_knn(self.base_knn_graph, ...)` — kernel weighting is ALWAYS applied to the base_knn_graph.

Where kernel weighting occurs:
Function: `_compute_kernel_from_version_knn` at line 2381 in topograph.py.
This function creates a `Kernel(metric="precomputed", ...)` and calls `.fit(knn)` on the kNN distance graph. The Kernel class's `fit()` method (kernels.py:584) calls `compute_kernel()` which applies the actual bandwidth-adaptive/fuzzy/cknn weighting.
This is a DISCRETE function call that can be bypassed with a conditional.

Existing graph_input_type parameter: NO — does not exist yet. Must be added.

## C. kNN and affinity construction — full digest

Function: kNN
File: topo/base/ann.py, Line: 18
Signature: `kNN(X, Y=None, n_neighbors=5, metric='euclidean', n_jobs=-1, backend='hnswlib', low_memory=True, M=60, p=11/16, efC=200, efS=200, n_trees=50, return_instance=False, verbose=False, **kwargs)`
What it does: Builds a k-nearest-neighbors graph using hnswlib, nmslib, or sklearn. Returns sparse CSR distance matrix (symmetrized). If `return_instance=True`, also returns the fitted ANN object.
Inputs: Dense or sparse matrix X, k, metric, backend choice
Output: scipy.sparse.csr_matrix of kNN distances (NOT affinities — raw distances)
Called from: TopOGraph.fit() (lines 747, 875, 959-963, 1099, 1118)
Relevance: Blueprint Sections 2 (ATAC kNN), 4 (per-modality kNN in WNN), 6 (bridge index)

Function: compute_kernel
File: topo/tpgraph/kernels.py, Line: 78
Signature: `compute_kernel(X, metric='cosine', n_neighbors=10, fuzzy=False, cknn=False, delta=1.0, pairwise=False, sigma=None, adaptive_bw=True, expand_nbr_search=False, alpha_decaying=False, return_densities=False, symmetrize=True, backend='hnswlib', n_jobs=-1, verbose=False, use_angular=False, square_distances=True, **kwargs)`
What it does: Converts a kNN distance graph (or raw data) into a kernel/affinity matrix using adaptive bandwidth Gaussian, fuzzy simplicial sets, or cKNN.
Inputs: kNN distance matrix (sparse) when metric='precomputed'
Output: Kernel matrix K (sparse, values in [0,1]-ish), plus optional density dict
Called from: Kernel.fit() (line 615)
Relevance: Blueprint Section 4 — WNN kernel must match this kernel formulation

Function: Kernel class
File: topo/tpgraph/kernels.py, Line: 312
Key properties: `.K` (kernel/affinity matrix), `.P` (row-normalized diffusion operator = `.diff_op()`), `.knn` (kNN graph)
The `.P` property (line 802) calls `diff_op()` which row-normalizes K to make a row-stochastic diffusion operator.

Function: _compute_kernel_from_version_knn (TopOGraph method)
File: topo/topograph.py, Line: 2381
What it does: Creates a Kernel with appropriate params based on kernel_version string and fits it on a kNN graph. This is the injection point for the `graph_input_type="affinity"` bypass.

## D. fit_adata — full digest
File: topo/single_cell.py
Signature: `fit_adata(adata, tg=None, *, projections=("MAP","PaCMAP"), do_leiden=True, leiden_key_base="topo_clusters", leiden_resolutions=(0.2,0.8), leiden_primary_index=1, **topograph_kwargs)`

How TopOGraph is instantiated inside fit_adata:
Line 682: `tg = TopOGraph(**topograph_kwargs)`
Line 700: `tg.fit(adata.X)`

Existing precomputed graph parameter: NONE — there is no parameter for passing a precomputed graph. Must be added (`precomputed_graph_key` and `graph_input_type`).

What is stored in adata after fit_adata completes:
- `adata.obsm["X_ms_spectral_scaffold"]` — msDM scaffold coordinates
- `adata.obsm["X_spectral_scaffold"]` — DM scaffold coordinates
- `adata.obsm["X_msTopoMAP"]`, `adata.obsm["X_TopoMAP"]` — MAP 2D projections
- `adata.obsm["X_msTopoPaCMAP"]`, `adata.obsm["X_TopoPaCMAP"]` — PaCMAP 2D projections
- `adata.obsp["topometry_connectivities"]` — refined DM operator (P_of_Z)
- `adata.obsp["topometry_distances"]` — same as connectivities (for scanpy compat)
- `adata.obsp["topometry_connectivities_ms"]` / `"topometry_distances_ms"` — msDM operator
- `adata.obs["topo_clusters"]` and resolution-specific columns (Leiden)

Changes needed per blueprint Section 5:
Add `precomputed_graph_key: Optional[str] = None` and `graph_input_type: str = "knn"` parameters. When `precomputed_graph_key` is set, retrieve `adata.obsp[precomputed_graph_key]` and pass to TopOGraph with the given `graph_input_type`, skipping internal kNN construction. When None, preserve existing behavior exactly.

## E. Public API surface (topo/__init__.py)
Currently exported relevant symbols:
- `TopOGraph`, `load_topograph`, `save_topograph` from `topo.topograph`
- `sc` (single_cell module) — when scanpy installed
- `ann` from `topo.base`
- `spt` (spectral), `tpg` (tpgraph), `eval`, `utils`, `pipes`, `pl` (plot), `lt` (layouts)

What must be added after implementation:
- `tp.sc.atac_lsi` — new function in single_cell.py
- `tp.sc.wnn_integration` — new function in single_cell.py
- `tp.sc.compute_gene_activity_scores` — new function in single_cell.py
- `tp.sc.fit_modality_bridge` — new function in single_cell.py
- `tp.sc.apply_modality_bridge` — new function in single_cell.py
- `tp.sc.ModalityBridge` — class from topo/bridge.py, exposed via single_cell.py

## F. Existing test conventions
Tests are in `tests/test_cca_integration.py` and `tests/test_cca_mapping.py`. They use plain functions (not classes) with pytest. Helper `_make_adata()` creates structured toy data with Gaussian clusters. Tests import from `topo.single_cell` directly. There is no `conftest.py`. Fixtures use pytest's `tmp_path`.

## G. Pre-existing notebooks
Located in `notebooks/`:
- `integration_tests.ipynb` — CCA integration testing
- `R1_integration_test.ipynb` — Review round 1 integration test
- `test_cca_integration.ipynb` — CCA integration tests
- `test_cca_integration_realdata.ipynb` — CCA with real data
- `test_integration_utilities.ipynb` — Integration utility tests
- `test_testpypi_install.ipynb` — TestPyPI install test
- `test_visualize_optimization.ipynb` — MAP optimization visualization

Stale docs/ references found: `cca_implementation_report.md` line 9 mentions `docs/test_cca_integration.ipynb`. This is a report document, not executable code. Will update if needed.

## H. Dependencies
Already declared in setup.cfg: numpy, scipy, scikit-learn, matplotlib, pandas, numba, setuptools
Missing for blueprint:
- `pyranges` — needed for `compute_gene_activity_scores` (optional dep)
- `mudata` — soft dependency for WNN (optional)
- `joblib` — used but not declared (likely transitive via scikit-learn)
- `hnswlib` — used but not declared as dependency
- `scanpy` — soft dependency (already handled at import)

Need to add `[options.extras_require]` section with `multiomics = pyranges; mudata` in setup.cfg.

## I. Audit summary

=== AUDIT SUMMARY ===
Audit A — Kernel weighting injection point:
  File: topo/topograph.py
  Function: _compute_kernel_from_version_knn (line 2381)
  Called from fit() at lines 766 (base kernel) and 1134/1149 (scaffold kernels)
  Bypass strategy: When graph_input_type="affinity", skip _compute_kernel_from_version_knn for the base kernel step. Instead, validate the input matrix and create a proxy Kernel object with .P set to the supplied affinity matrix directly.

Audit B — Kernel function to reuse in WNN:
  Function: Kernel class (topo/tpgraph/kernels.py line 312)
  Signature: `Kernel(metric="precomputed", n_neighbors=k, adaptive_bw=True, ...).fit(knn_distance_graph)`
  The `.P` property returns the row-stochastic diffusion operator.
  For WNN, use: `Kernel(metric="precomputed", n_neighbors=n_neighbors, adaptive_bw=True, ...).fit(knn_sparse)` then `.P` for the affinity.

Audit C — fit_adata TopOGraph instantiation:
  Pattern: `tg = TopOGraph(**topograph_kwargs)` then `tg.fit(adata.X)`
  Existing precomputed graph parameter: NONE

Audit D — hnswlib wrapper function:
  Function: kNN (topo/base/ann.py line 18)
  Signature: `kNN(X, n_neighbors=5, metric='euclidean', n_jobs=-1, backend='hnswlib', return_instance=False, verbose=False, **kwargs)`
  Called as: `knn_graph = kNN(X_dense, n_neighbors=30, metric='euclidean', n_jobs=-1, backend='hnswlib')`

Audit E — Pre-existing notebooks:
  Notebooks found: integration_tests.ipynb, R1_integration_test.ipynb, test_cca_integration.ipynb, test_cca_integration_realdata.ipynb, test_integration_utilities.ipynb, test_testpypi_install.ipynb, test_visualize_optimization.ipynb
  Stale docs/ references found: cca_implementation_report.md:9 (report file, not code)
  All pre-existing notebooks: NOT VALIDATED YET (will test as baseline)
=== END AUDIT SUMMARY ===

## J. Implementation plan
Milestone 1 (Audit): COMPLETE
Milestone 2 (TopOGraph graph_input_type): COMPLETE
Milestone 3 (atac_lsi): COMPLETE
Milestone 4 (wnn_integration): COMPLETE
Milestone 5 (compute_gene_activity_scores): COMPLETE
Milestone 6 (ModalityBridge + bridge functions): COMPLETE
Milestone 7 (fit_adata adjustments): COMPLETE
Milestone 8 (Final validation sweep): COMPLETE

Post-milestone work:
- Fixed pre-existing CCA test failures (seurat_v3 overflow fallback)
- Added setup.cfg [options.extras_require] multiomics
- Created tests/test_multiomics.py (26 tests, 4 classes)
- Created tests/fixtures/generate_multiomics_fixtures.py
- Updated .gitignore for fixture files

Pre-existing failures (not caused by this implementation):
- pytest: 11 failed, 3 passed, 11 errors — ALL failures are in CCA integration tests due to `ValueError: cannot specify integer bins when input data contains infinity` (numeric overflow in scanpy HVG seurat_v3 method with synthetic data)
- These failures exist before any changes and are unrelated to multi-omic integration

## K. Open questions

1. **Kernel consistency for WNN (Section 4)**: The blueprint says the WNN kernel must match TopOGraph's internal kernel. TopOGraph uses `Kernel(metric="precomputed", adaptive_bw=True, ...).fit(knn_distances)` which produces `.P` (row-stochastic). For WNN, we should reuse the same `Kernel` class to convert per-modality kNN distances to affinities. The WNN weighted sum then gets row-normalized. This should be consistent. No blocker.

2. **TopOGraph.fit() precomputed path**: Currently `base_metric='precomputed'` causes `X` to be treated as a precomputed distance matrix (stored as `base_knn_graph`). For `graph_input_type='affinity'`, we need a different path where X is treated as an already-weighted affinity matrix. The cleanest approach: add `graph_input_type` to __init__, and in fit(), when `graph_input_type='affinity'`, skip both kNN construction AND kernel weighting, instead creating a proxy Kernel with `.P = X` and `.K = X`. No blocker.

3. **EigenDecomposition expects a Kernel object**: The `EigenDecomposition.fit(kernel)` at line 1065 expects a Kernel with `.P` property. When bypassing kernel construction for affinity input, we need a proxy object that exposes `.P` — the `_ProxyKernel` class already used in UoM code (line 794-796) serves this purpose. No blocker.

4. **Automated sizing with affinity input**: `_automated_sizing(X)` at line 782 runs on raw X data. When `graph_input_type='affinity'`, X is a square affinity matrix, not features. Need to skip `_automated_sizing` and use `min_eigs` as the scaffold size, or require the user to set `min_eigs` explicitly. Minor issue — will set a reasonable default.
