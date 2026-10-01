# Quick-start cheat-sheet

## Fitting a TopOGraph

TopoMetry is built around the `TopOGraph` class. From a data matrix `data` (`np.ndarray` or
`scipy.sparse.csr_matrix`, samples in rows), one call learns everything: the base kernel, the
spectral scaffolds, the refined graphs and the 2-D layouts.

```python
import topo as tp

tg = tp.TopOGraph(n_jobs=-1, random_state=42)
tg.fit(data)
```

The results are available as attributes and methods:

```python
msZ = tg.spectral_scaffold(multiscale=True)   # multiscale diffusion maps (msDM) scaffold
Z = tg.spectral_scaffold(multiscale=False)    # single-time diffusion maps (DM) scaffold

tg.knn_X       # neighbor distances in the input space
tg.P_of_X      # diffusion operator on the input space
tg.P_of_msZ    # refined diffusion operator, built on the msDM scaffold
tg.P_of_Z      # refined diffusion operator, built on the DM scaffold

tg.msTopoMAP   # 2-D MAP layout of the msDM refined graph
tg.TopoMAP     # 2-D MAP layout of the DM refined graph

tg.global_id               # intrinsic dimensionality estimated on the input
tg.n_scaffold_components   # number of scaffold components selected from it
```

`TopoPaCMAP` and `msTopoPaCMAP` are also computed when [PaCMAP](https://github.com/YingfanWang/PaCMAP)
is installed (`pip install pacmap`).

To plot a layout:

```python
tp.pl.scatter(tg.msTopoMAP, labels=labels)
```

## Other projections

`project` computes one more layout and stores it in `tg.ProjectionDict`. `multiscale` chooses
between the msDM and the DM scaffold.

```python
isomap = tg.project(projection_method='Isomap', multiscale=True)
tsne = tg.project(projection_method='t-SNE', multiscale=True)
```

Available methods are `'MAP'`, `'Isomap'`, `'t-SNE'`, `'UMAP'`, `'PaCMAP'`, `'TriMAP'`,
`'IsomorphicMDE'`, `'IsometricMDE'` and `'NCVis'`. All but the first three need their own
package installed.

## Single-cell data

With an `AnnData` object, `tp.sc.fit_adata` fits a `TopOGraph` on `adata.X` and writes the
scaffolds, layouts and clusters back to `adata`:

```python
tg = tp.sc.fit_adata(adata, n_jobs=-1, random_state=42)

adata.obsm['X_ms_spectral_scaffold']   # msDM scaffold
adata.obsm['X_msTopoMAP']              # layout
adata.obs['topo_clusters']             # Leiden clusters on the refined graph
```

`tp.sc.run_and_report(adata, filename='report.pdf')` runs the whole analysis and writes a PDF
report. See the tutorials for a guided tour.

## Choosing kernels and metrics

```python
tg = tp.TopOGraph(
    base_knn=30,                         # neighbors in the input space
    graph_knn=30,                        # neighbors in the scaffold
    base_metric='cosine',                # metric in the input space
    graph_metric='euclidean',            # metric in the scaffold
    base_kernel_version='bw_adaptive',   # or 'fuzzy', 'cknn', 'gaussian', ...
    graph_kernel_version='bw_adaptive',
)
```

## Backends and reproducibility

- Neighbors are searched with [hnswlib](https://github.com/nmslib/hnswlib) when it is installed
  (`pip install hnswlib`). Otherwise TopoMetry falls back to nmslib, and then to scikit-learn's
  exact search, with a warning. Pass `backend='sklearn'` to ask for exact search.
- With `n_jobs=1` and a `random_state`, two fits give identical results. With more threads,
  approximate index construction and layout optimization are not deterministic.

## Computing several models at once

`run_models` fits every combination of the given base and graph kernels and computes the given
projections on both scaffolds. This is useful for comparing and scoring models rather than
choosing one _a priori_.

```python
tg.run_models(data,
              kernels=['bw_adaptive', 'fuzzy'],
              projections=['MAP', 'Isomap'])

tg.BaseKernelDict.keys()    # 'bw_adaptive', 'fuzzy'
tg.EigenbasisDict.keys()    # 'msDM with bw_adaptive', 'DM with bw_adaptive', ...
tg.GraphKernelDict.keys()   # 'fuzzy from msDM with bw_adaptive', ...
tg.ProjectionDict.keys()    # 'MAP of fuzzy from msDM with bw_adaptive', ...
```
