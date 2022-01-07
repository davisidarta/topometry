# Quick-start with the MNIST digits dataset

    This tutorial covers a quick-start with TopOMetry using the MNIST handwritten digits dataset. This is composed of ~1,800 handwritten digits images composed of 64 (8 x 8) pixels each. Our task will be to represent this high-dimensional space (of 64 dimensions) into latent orthogonal bases and to visualize comprehensive layouts of this data.
    
    First, we'll load some libraries:


```python
# Load some libraries:
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix
import topo as tp
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
%matplotlib inline
```

    Next, we'll load the MNIST handwritten digits dataset from scikit-learn:


```python
# Load the digits data
digits = load_digits()

# Convert to sparse for even more speed
data = csr_matrix(digits.data)
```

    Then, we'll create an empty TopOGraph object to perform our analyses. The TopOGraph class is the main class used in TopOMetry to coordinate analyses between the multiple other classes available in the library. The TopOGraph can learn similarities, new orthogonal bases and affinity graphs with any pairwise combination of three algorithms: Diffusion Maps, Continuous-k-Nearest Neighbors and fuzzy simplicial sets, rendering 9 model options (3 bases x 3 graphs). By default, the TopOGraph runs the 'diffusion' basis and the 'diff' graph.  


```python
# Set up a TopOGraph object:
tg = tp.TopOGraph(n_jobs=12, n_eigs=20)

# Fit a topological orthogonal basis:
tg.fit(data)

# Fit a topological graph:
db_diff_graph = tg.transform()
```

    Computing neighborhood graph...
     Base kNN graph computed in 0.192627 (sec)
    Building topological basis...using diffusion model.
     Topological basis fitted with multiscale self-adaptive diffusion maps in 0.555791 (sec)
        Building topological graph...
         Topological `diff` graph extracted in = 0.152692 (sec)


    /home/davi/.local/lib/python3.9/site-packages/kneed/knee_locator.py:304: UserWarning: No knee/elbow found
      warnings.warn("No knee/elbow found")


(http://jmlr.org/papers/v22/20-1061.html) , one of the 6 graph layout optimization methods included in TopOMetry. The other methods are:

* MAP (Manifold Approximation and Projection)[] - a lighter (UMAP)[] with looser assumptions
* tSNE (t-Stochasthic Neighborhood Embedding)[] - a classic of visualization
* MDE (Minimum Distortion Embedding)[] - the ultimate swiss-army knife for graph layout optimization
* TriMAP[] - dimensionality reduction using triplets
* NCVis (Noise Contrastive Visualization)[] - for blazing fast performance
* PaCMAP (Pairwise-Controlled Manifold Approximation and Projection)[] - for global/local balanced embeddings

For this tutorial, we'll first visualize the graph layout with PaCMAP:


```python
db_PaCMAP = tg.PaCMAP(num_iters=1000, distance='angular')
```

             Spectral layout not stored at TopOGraph.SpecLayout. Trying to compute...


    /home/davi/.local/lib/python3.9/site-packages/pacmap/pacmap.py:383: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      if Yinit is None or Yinit == "pca":
    /home/davi/.local/lib/python3.9/site-packages/pacmap/pacmap.py:389: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      elif Yinit == "random":


             Obtained PaCMAP embedding in = 106.948818 (sec)



```python
plt.scatter(db_PaCMAP[:, 0], db_PaCMAP[:, 1], c=digits.target.astype('int32'), cmap='Spectral', s=0.5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('db_PaCMAP projection of the Digits dataset', fontsize=12);
```


    
![png](MNIST_TopOMetry_Tutorial_files/MNIST_TopOMetry_Tutorial_8_0.png)
    


Now let's check how other layouts look:


```python
# TriMAP, tSNE and NCVis use only the basis, not the graph
emb_db_trimap = tg.TriMAP()
emb_db_tsne = tg.tSNE()
emb_db_ncvis = tg.NCVis()

# MAP and MDE also use graph data
emb_db_diff_map = tg.MAP()
emb_db_diff_mde = tg.MDE()
```

             Obtained TriMAP embedding in = 47.268870 (sec)
             Obtained tSNE embedding in = 12.291106 (sec)
             Optimized MAP embedding in = 14.215852 (sec)
             Obtained MDE embedding in = 3.011959 (sec)



```python
plt.scatter(emb_db_trimap[:, 0], emb_db_trimap[:, 1], c=digits.target.astype('int32'), cmap='Spectral', s=0.5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('db_TriMAP projection of the Digits dataset', fontsize=12);
```


    
![png](MNIST_TopOMetry_Tutorial_files/MNIST_TopOMetry_Tutorial_11_0.png)
    



```python
plt.scatter(emb_db_tsne[:, 0], emb_db_tsne[:, 1], c=digits.target.astype('int32'), cmap='Spectral', s=0.5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('db_tSNE projection of the Digits dataset', fontsize=12)
```




    Text(0.5, 1.0, 'db_tSNE projection of the Digits dataset')




    
![png](MNIST_TopOMetry_Tutorial_files/MNIST_TopOMetry_Tutorial_12_1.png)
    



```python
plt.scatter(emb_db_ncvis[:, 0], emb_db_ncvis[:, 1], c=digits.target.astype('int32'), cmap='Spectral', s=0.5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('db_NCVis projection of the Digits dataset', fontsize=12)
```




    Text(0.5, 1.0, 'db_NCVis projection of the Digits dataset')




    
![png](MNIST_TopOMetry_Tutorial_files/MNIST_TopOMetry_Tutorial_13_1.png)
    



```python
plt.scatter(emb_db_diff_map[:, 0], emb_db_diff_map[:, 1], c=digits.target.astype('int32'), cmap='Spectral', s=0.5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('db_diff_MAP projection of the Digits dataset', fontsize=12)
```




    Text(0.5, 1.0, 'db_diff_MAP projection of the Digits dataset')




    
![png](MNIST_TopOMetry_Tutorial_files/MNIST_TopOMetry_Tutorial_14_1.png)
    



```python
plt.scatter(emb_db_diff_mde[:, 0], emb_db_diff_mde[:, 1], c=digits.target.astype('int32'), cmap='Spectral', s=0.5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('db_diff_MDE projection of the Digits dataset', fontsize=12)
```




    Text(0.5, 1.0, 'db_diff_MDE projection of the Digits dataset')




    
![png](MNIST_TopOMetry_Tutorial_files/MNIST_TopOMetry_Tutorial_15_1.png)
    


We can also quantify how much each method preserves global and local structure:


```python
results = tp.pipes.eval_models_layouts(tg, data, bases=['diffusion'], graphs=['diff'],
                                       layouts=['MAP', 'tSNE', 'PaCMAP', 'TriMAP', 'MDE', 'NCVis'])
```

    Computing scores...
    Computing PCA for comparison...
    Computing UMAP...
    Computing default tSNE...
    Computing default PaCMAP...


    /home/davi/.local/lib/python3.9/site-packages/pacmap/pacmap.py:383: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      if Yinit is None or Yinit == "pca":
    /home/davi/.local/lib/python3.9/site-packages/pacmap/pacmap.py:389: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      elif Yinit == "random":


    Computing default TriMAP...
    Computing default MDE...
    Computing default NCVis...



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    /tmp/ipykernel_71762/2342656512.py in <module>
    ----> 1 results = tp.pipes.eval_models_layouts(tg, data, bases=['diffusion'], graphs=['diff'],
          2                                        layouts=['MAP', 'tSNE', 'PaCMAP', 'TriMAP', 'MDE', 'NCVis'])


    ~/.local/lib/python3.9/site-packages/topo/pipes.py in eval_models_layouts(TopOGraph, X, bases, graphs, layouts)
        610 
        611         import ncvis
    --> 612         ncvis_emb = ncvis.NCVis(distance=TopOGraph.graph_metric,
        613                                 n_neighbors=TopOGraph.graph_knn, n_jobs=TopOGraph.n_jobs)
        614         ncvis_pca, ncvis_lap = global_scores(X, ncvis_emb, n_dim=TopOGraph.n_eigs)


    wrapper/ncvis.pyx in ncvis.NCVis.__init__()


    TypeError: __init__() got an unexpected keyword argument 'n_jobs'



```python
plot_layouts_scores = results[2]
```

That's it for this tutorial! I hope TopOMetry can be useful for your work!

Of course, you're now wondering: 
