#!/usr/bin/env python
"""
Generate PBMC 10k Multiome test fixtures for topometry multi-omic tests.

Downloads the 10x Genomics PBMC 10k Multiome dataset, subsets to 500 cells,
runs basic preprocessing, and saves as h5ad fixtures.

Usage:
    python tests/fixtures/generate_multiomics_fixtures.py

Output:
    tests/fixtures/pbmc500_rna.h5ad
    tests/fixtures/pbmc500_atac.h5ad

These files are NOT committed to git (listed in .gitignore).
"""
import os
import sys
import numpy as np
import warnings

warnings.filterwarnings("ignore")

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))
RNA_OUT = os.path.join(FIXTURE_DIR, "pbmc500_rna.h5ad")
ATAC_OUT = os.path.join(FIXTURE_DIR, "pbmc500_atac.h5ad")
SEED = 42
N_CELLS = 500


def download_pbmc_10k():
    """Download PBMC 10k Multiome filtered feature barcode matrix."""
    import scanpy as sc

    url = (
        "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/"
        "pbmc_granulocyte_sorted_10k/"
        "pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5"
    )
    cache_dir = os.path.join(FIXTURE_DIR, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    h5_path = os.path.join(cache_dir, "pbmc_10k_multiome_filtered.h5")

    if not os.path.exists(h5_path):
        print(f"Downloading PBMC 10k Multiome to {h5_path} ...")
        import urllib.request

        def _progress(count, block_size, total_size):
            pct = count * block_size * 100.0 / total_size
            sys.stdout.write(f"\r  {pct:.1f}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, h5_path, reporthook=_progress)
        print("\n  Done.")
    else:
        print(f"Using cached {h5_path}")

    # Read with scanpy (handles the 10x multiome h5 format)
    adata = sc.read_10x_h5(h5_path, gex_only=False)
    adata.var_names_make_unique()
    return adata


def split_modalities(adata):
    """Split combined 10x Multiome AnnData into RNA and ATAC."""
    if "feature_types" not in adata.var.columns:
        raise ValueError(
            "Cannot split modalities: 'feature_types' column not in adata.var. "
            "Ensure the h5 file is from a 10x Multiome experiment."
        )
    rna_mask = adata.var["feature_types"] == "Gene Expression"
    atac_mask = adata.var["feature_types"] == "Peaks"

    adata_rna = adata[:, rna_mask].copy()
    adata_atac = adata[:, atac_mask].copy()
    return adata_rna, adata_atac


def subset_cells(adata_rna, adata_atac, n_cells=N_CELLS, seed=SEED):
    """Randomly subset to n_cells (fixed seed for reproducibility)."""
    rng = np.random.default_rng(seed)
    n_total = adata_rna.n_obs
    if n_total <= n_cells:
        print(f"  Dataset has {n_total} cells, no subsetting needed.")
        return adata_rna, adata_atac

    idx = rng.choice(n_total, size=n_cells, replace=False)
    idx.sort()
    return adata_rna[idx].copy(), adata_atac[idx].copy()


def preprocess_rna(adata_rna):
    """Basic RNA preprocessing: normalize, log1p, HVG."""
    import scanpy as sc

    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.highly_variable_genes(adata_rna, n_top_genes=3000, flavor="seurat_v3",
                                 layer=None, subset=False)
    adata_rna.layers["counts"] = adata_rna.X.copy()
    return adata_rna


def preprocess_atac(adata_atac):
    """Basic ATAC preprocessing: LSI."""
    import topo as tp

    tp.sc.atac_lsi(adata_atac, n_components=50, inplace=True)
    return adata_atac


def main():
    print("=== Generating PBMC 10k Multiome fixtures ===")

    # Step 1: Download
    adata = download_pbmc_10k()
    print(f"Loaded: {adata.shape}")

    # Step 2: Split modalities
    adata_rna, adata_atac = split_modalities(adata)
    print(f"RNA: {adata_rna.shape}, ATAC: {adata_atac.shape}")

    # Step 3: Subset
    adata_rna, adata_atac = subset_cells(adata_rna, adata_atac)
    print(f"Subsetted: RNA {adata_rna.shape}, ATAC {adata_atac.shape}")

    # Step 4: Preprocess
    print("Preprocessing RNA...")
    adata_rna = preprocess_rna(adata_rna)
    print("Preprocessing ATAC (LSI)...")
    adata_atac = preprocess_atac(adata_atac)

    # Step 5: Save
    adata_rna.write_h5ad(RNA_OUT)
    adata_atac.write_h5ad(ATAC_OUT)
    print(f"\nSaved:")
    print(f"  RNA:  {RNA_OUT}")
    print(f"  ATAC: {ATAC_OUT}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
