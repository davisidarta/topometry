# Quick-start

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



