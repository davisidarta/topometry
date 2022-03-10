# Quick-start

## Fitting a TopOGraph
Now, let's go through a quick start!

TopOMetry functions around 

From a  data matrix ``data`` (np.ndarray, pd.DataFrame or sp.csr_matrix), you can set up a ``TopoGraph`` 
with default parameters: 

```
import topo as tp
   
# Learn topological metrics and basis from data. The default is to use diffusion harmonics.
tg = tp.TopOGraph()
tg.fit(data)
```

After learning a topological basis, we can access topological metrics and basis in the ``TopOGraph`` object, and build different
topological graphs. 

```
# Learn a topological graph. Again, the default is to use diffusion harmonics.
tgraph = tg.transform(data) 
```

Then, it is possible to optimize the topological graph layout. TopOMetry has 5 different layout options: tSNE, MAP, 
TriMAP, PaCMAP and MDE.

```
# Graph layout optimization
map_emb = tg.MAP()
mde_emb = tg.MDE()
pacmap_emb = tg.PaCMAP()
trimap_emb = tg.TriMAP()
tsne_emb = tg.tSNE()
```

We can also plot the embeddings:

```
tp.plot.scatter(map_emb)
```

## Computing several models at once

The `tp.pipes.`

If no parameters are passed to the `ct.topological_workflow` function, by default it will perform the following steps:

Similarity learning and building a topological orthogonal basis with:
* Multiscale diffusion maps (`'diffusion'`)
* Fuzzy simplicial sets Laplacian Eigenmaps ('fuzzy');

Learn the topological graphs with:
* Diffusion harmonics (`'diff'`)
* Fuzzy simplicial sets

Next, it performs layout optimization with:
* MAP - a lighter [UMAP](https://umap-learn.readthedocs.io/en/latest/index.html) with looser assumptions;
    - MAP and MDE use information both from the orthogonal basis and the topological graph
* [MDE](https://web.stanford.edu/~boyd/papers/min_dist_emb.html) - a general framework for graph layout optimization, with the pyMDE implementation. 
* [t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf?fbclid=IwA) - using [MulticoreTSNE](https://github.com/DmitryUlyanov/Multicore-TSNE)
* [PaCMAP](https://arxiv.org/abs/2012.04456) 
* [TriMAP](https://arxiv.org/abs/1910.00204)
* [NCVis](https://dl.acm.org/doi/abs/10.1145/3366423.3380061)

And perform layout optimization with



