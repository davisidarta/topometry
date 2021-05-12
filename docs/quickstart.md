
From a large data matrix ``data`` (np.ndarray, pd.DataFrame or sp.csr_matrix), you can set up a ``TopoGraph`` with default parameters: 

```
import topo.models as tp
   
# Learn topological metrics and basis from data. The default is to use diffusion harmonics.
tg = tp.TopoGraph()
tg = tg.fit(data)

```

After learning a topological basis, we can access topological metrics and basis in the ``tg`` object, and build different
topological graphs.
```
# Learn a topological graph. Again, the default is to use diffusion harmonics.
tgraph = tg.transform(data) # tgraph.K is the affinity graph, and tgraph.T is the transition graph
   
# Graph layout optimization with UMAP (not so much 'uniform' in this case)
topo_umap, aux = tp.MAP(tg.MSDiffMaps, tgraph.T)

emb_t, aux = tp.MAP(tg.MSDiffMaps, traph)
```
