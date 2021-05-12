
From a large data matrix ``data`` (np.ndarray, pd.DataFrame or sp.csr_matrix), you can set up a ``TopoGraph`` with default parameters: 

```
import topo.models as tp
   
# Learn topological metrics and basis from data. The default is to use diffusion harmonics.
tg = tp.TopOGraph()
tg = tg.fit(data)
```

Note: `topo.ml` is the high-level model module which contains the `TopOGraph` object.

After learning a topological basis, we can access topological metrics and basis in the ``TopOGraph`` object, and build different
topological graphs. 

```
# Learn a topological graph. Again, the default is to use diffusion harmonics.
tgraph = tg.transform(data) 
```

Then, it is possible to optimize the topological graph layout. The first option is to do so with
our adaptation of UMAP (MAP), which will minimize the cross-entropy between the topological basis
and its graph:

```
# Graph layout optimization with MAP
map_emb, aux = tp.MAP(tg.MSDiffMaps, tgraph)
```

The second, albeit most interesting option is to use pyMDE to find a Minimum Distortion Embedding. TopOMetry implements some
custom MDE problems within the TopOGraph model :

```
# Set up MDE problem
mde = tg.MDE(tgraph)
mde_emb = mde.embed()
```
