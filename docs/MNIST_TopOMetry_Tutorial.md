# Quick-start with the MNIST digits dataset

This tutorial covers a quick-start with `TopOMetry` using the MNIST handwritten digits dataset. This dataset is composed of ~1,800 handwritten digits images composed of 64 (8 x 8) pixels each. Our task will be to represent this high-dimensional space (of 64 dimensions) into a latent orthonormal eigenbasis. From this eigenbasis we will learn a new topological graph and visualize it with graph-layout algorithms. Although there are extensive options within `TopOMetry`, most people are interested in learning similarities, orthonormal eigenbases and graphs and graph-layouts associated with such eigenbases. In this tutorial, we'll see how to do so with the most robust `TopOMetry` algorithm: multiscale diffusion maps, which covers nearly all use-cases.

    
First, we'll load some libraries:


```python
# Load some libraries:
import numpy as np
import topo as tp

# Import the MNIST data set:
from sklearn.datasets import load_digits

# Matplotlib for plotting:
from matplotlib import pyplot as plt
%matplotlib inline
```

Load the MNIST handwritten digits dataset from scikit-learn:


```python
# Load the digits data and the labels:
X, labels = load_digits(return_X_y=True)
```

### Set and fit a TopOGraph object

Then, we'll create an empty TopOGraph object to perform our analyses. The TopOGraph class is the main class used in TopOMetry to coordinate analyses between the multiple other classes available in the library. 
 

The TopOGraph can learn similarities, latent orthonormal eigenbases and new topological graphs from these eigenbases using various kernels and eigenmap strategies. By default, it uses an adaptive bandwidth kernel to learn a reweighted diffusion operator and and multiscale diffusion maps - this strategy is guaranteed to approximate the Laplace-Beltrami Operator (LBO) regardless of data geometry and sampling distribution. 

Because we already know we have 10 classes, we'll use a slightly larger number of components (`n_eigs`) to account for possible digits images which may be odd looking (e.g. 8's that look like 3's or 1's). If we hadn't known it beforehand, we could use larger numbers and try to select a number of components to keep by visualizing an eigengap. We could also try to use Fischer Separability Analysis (FSA) to estimate the global dimensionality of the data, which is also included in TopOMetry.


```python
# Set up a TopOGraph object:
tg = tp.TopOGraph(n_eigs=15, n_jobs=-1, base_metric='euclidean')
tg
```




    TopOGraph object without any fitted data.
     . Base Kernels:
     . Eigenbases:
     . Graph Kernels:
     . Projections: 
     Active base kernel  -  .base_kernel 
     Active eigenbasis  -  .eigenbasis 
     Active graph kernel  -  .graph_kernel



The `fit()` method in the `TopOGraph` class learns a similarity kernel (referred to as 'base kernel') which will be used to compute an eigenbasis. A new kernel (referred to as 'graph kernel') will be used to learn topological affinities from this eigenbasis, which can be used for clustering and graph-layout optimization.


```python
tg.fit(X)
```

    Computing neighborhood graph...
     Base kNN graph computed in 0.100219 (sec)
     Fitted the bw_adaptive kernel in 0.044877 (sec)
    Computing eigenbasis...
     Fitted eigenbasis with Diffusion Maps from the bw_adaptive kernel in 0.166896 (sec)





    TopOGraph object with 1797 samples and 64 observations and:
     . Base Kernels: 
        bw_adaptive - .BaseKernelDict['bw_adaptive']
     . Eigenbases: 
        DM with bw_adaptive - .EigenbasisDict['DM with bw_adaptive']
     . Graph Kernels:
     . Projections: 
     Active base kernel  -  .base_kernel 
     Active eigenbasis  -  .eigenbasis 
     Active graph kernel  -  .graph_kernel



Let's visualize the eigenspectrum of our multiscale diffusion maps:


```python
tg.eigenspectrum()
```


    
![png](MNIST_TopOMetry_Tutorial_files/MNIST_TopOMetry_Tutorial_9_0.png)
    


Let's inspect our eigenbasis class:


```python
tg.eigenbasis
```




    EigenDecomposition() estimator fitted with 1797 samples using Diffusion Maps.



The `EigenDecomposition` class returns its results with the `transform()` method. All computations are performed during `fit()`.


```python
diffusion_maps = tg.eigenbasis.transform()
diffusion_maps.shape
```




    (1797, 15)



Let's visualize the multiscale diffusion maps results:


```python
plt.scatter(diffusion_maps[:, 0], diffusion_maps[:, 1], c=labels, cmap='Spectral', s=0.5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('Multiscale Diffusion Maps projection of the Digits dataset', fontsize=12)
```




    Text(0.5, 1.0, 'Multiscale Diffusion Maps projection of the Digits dataset')




    
![png](MNIST_TopOMetry_Tutorial_files/MNIST_TopOMetry_Tutorial_15_1.png)
    


As we see, the diffusion maps do a decent job in separating major classes. However, this is not so great for visualization because each diffusion component will carry information regarding different classes:


```python
plt.figure(figsize=(9 * 2 + 5, 2))
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
)
plot_num = 1
for i in range(0, diffusion_maps.shape[1]):
    plt.subplot(1, diffusion_maps.shape[1], plot_num)
    plt.title('Multiscale DC_' + str(plot_num))
    plt.scatter(range(0, diffusion_maps.shape[0]), diffusion_maps[:,i], s=0.1, c=labels, cmap='tab20')
    plot_num += 1
    plt.xticks(())
    plt.yticks(())
```


    
![png](MNIST_TopOMetry_Tutorial_files/MNIST_TopOMetry_Tutorial_17_0.png)
    


Therefore, ideally we should try to encode the information retrieved by several components into a single visualization to get a better understanding of this data. 

That's why we'll learn a new affinity graph and use it for visualization and clustering! This is done with the `transform()` method.

NOTE: You may have noticed the oscilatory pattern here. This is no coincidence. The eigenvectors of the Laplace-Beltrami Operator, like the eigenvectors of the graph Laplacian, form a basis for the space of functions defined on the manifold. This basis is similar to a Fourier basis in that it decomposes functions into a sum of sinusoidal functions with different frequencies. This makes them particularly useful for tasks such as spectral clustering or dimensionality reduction, where the structure of the data (i.e., the functions to be decomposed) is not known a priori.


```python
tg.transform()
tg
```

        Building topological graph from eigenbasis...
            Computing neighborhood graph...
     Computed in 0.133223 (sec)
     Fitted the bw_adaptive graph kernel in 0.064460 (sec)





    TopOGraph object with 1797 samples and 64 observations and:
     . Base Kernels: 
        bw_adaptive - .BaseKernelDict['bw_adaptive']
     . Eigenbases: 
        DM with bw_adaptive - .EigenbasisDict['DM with bw_adaptive']
     . Graph Kernels: 
        bw_adaptive from DM with bw_adaptive - .GraphKernelDict['bw_adaptive from DM with bw_adaptive']
     . Projections: 
     Active base kernel  -  .base_kernel 
     Active eigenbasis  -  .eigenbasis 
     Active graph kernel  -  .graph_kernel



The graph kernel object is acessible at `TopOGraph.graph_kernel`. By default, TopOMetry uses its diffusion operator for visualization:


```python
# The graph kernel diffusion operator
tg.graph_kernel.P
```




    <1797x1797 sparse matrix of type '<class 'numpy.float64'>'
    	with 67754 stored elements in Compressed Sparse Row format>



We can see this diffusion operator encodes the main 10 digits classes present in the data, with some uncertaintiny between similar classes:


```python
plt.spy(tg.graph_kernel.P, markersize=0.1)
```




    <matplotlib.lines.Line2D at 0x7f9f5526f7c0>




    
![png](MNIST_TopOMetry_Tutorial_files/MNIST_TopOMetry_Tutorial_23_1.png)
    


### Project the results with a graph-layout method

Now let's visualize this graph! FOr visualization, we''ll use the `project()` method in `TopOGraph`. It takes as arguments the number of components, the method to be use for graph projection, an optional initialization and other keyword arguments.

For visualization, we'll use two methods that come with TopOMetry without need to install further libraries:
* ['Isomap'](https://doi.org/10.1126/science.290.5500.2319) - one of the very first manifold learning methods (it can be quite slow)
        
        Isomap preserves geodesics distances, being called a 'global' method
* 'MAP'- a lighter [UMAP](https://umap-learn.readthedocs.io/en/latest/index.html) with looser assumptions
    
        MAP / UMAP preserve neighborhood relationships with a focus on local neighborhoods, being called a 'local' method 


```python
isomap = tg.project(projection_method='Isomap')

plt.scatter(isomap[:, 0], isomap[:, 1], c=labels, cmap='Spectral', s=0.5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('Isomap projection of diffusion operator on Multiscale DM of the Digits dataset', fontsize=12)
```

     Computed Isomap in 1.964845 (sec)





    Text(0.5, 1.0, 'Isomap projection of diffusion operator on Multiscale DM of the Digits dataset')




    
![png](MNIST_TopOMetry_Tutorial_files/MNIST_TopOMetry_Tutorial_26_2.png)
    



```python
map = tg.project(projection_method='MAP')

plt.scatter(map[:, 0], map[:, 1], c=labels, cmap='Spectral', s=0.5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('MAP projection of diffusion operator on Multiscale DM of the Digits dataset', fontsize=12)
```

     Computed MAP in 38.834775 (sec)





    Text(0.5, 1.0, 'MAP projection of diffusion operator on Multiscale DM of the Digits dataset')




    
![png](MNIST_TopOMetry_Tutorial_files/MNIST_TopOMetry_Tutorial_27_2.png)
    


### Use additional graph-layout methods

Other layout options are also available, but require installing other packages that are not listed as hard-dependencies:
* ['UMAP'](https://umap-learn.readthedocs.io/en/latest/index.html) - arguably the state-of-the-art for graph layout optimization (requires installing `umap-learn`)
* ['PaCMAP'](http://jmlr.org/papers/v22/20-1061.html) (Pairwise-controlled Manifold Approximation and Projection) - for balanced visualizations (requires installing `pacmap`)
* ['TriMAP'](https://github.com/eamid/trimap) - dimensionality reduction using triplets (requires installing `trimap`)
* 'IsomorphicMDE' - [MDE](https://github.com/cvxgrp/pymde) with preservation of nearest neighbors (requires installing `pymde`)
* 'IsometricMDE' - [MDE](https://github.com/cvxgrp/pymde) with preservation of pairwise distances (requires installing `pymde`)
* ['NCVis'](https://github.com/stat-ml/ncvis) (Noise Contrastive Visualization) - a UMAP-like method with blazing fast performance (requires installing `ncvis`)

Let's use PaCMAP for this example:


```python
%pip install pacmap
```

    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: pacmap in /home/davi/.local/lib/python3.10/site-packages (0.7.0)
    Requirement already satisfied: scikit-learn>=0.20 in /home/davi/.local/lib/python3.10/site-packages (from pacmap) (0.24.0)
    Requirement already satisfied: annoy>=1.11 in /home/davi/.local/lib/python3.10/site-packages (from pacmap) (1.17.1)
    Requirement already satisfied: numba>=0.50 in /home/davi/.local/lib/python3.10/site-packages (from pacmap) (0.56.4)
    Requirement already satisfied: numpy>=1.20 in /home/davi/.local/lib/python3.10/site-packages (from pacmap) (1.23.5)
    Requirement already satisfied: setuptools in /home/davi/.local/lib/python3.10/site-packages (from numba>=0.50->pacmap) (67.4.0)
    Requirement already satisfied: llvmlite<0.40,>=0.39.0dev0 in /home/davi/.local/lib/python3.10/site-packages (from numba>=0.50->pacmap) (0.39.1)
    Requirement already satisfied: joblib>=0.11 in /home/davi/.local/lib/python3.10/site-packages (from scikit-learn>=0.20->pacmap) (1.2.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /home/davi/.local/lib/python3.10/site-packages (from scikit-learn>=0.20->pacmap) (3.1.0)
    Requirement already satisfied: scipy>=0.19.1 in /usr/lib/python3/dist-packages (from scikit-learn>=0.20->pacmap) (1.8.0)
    Note: you may need to restart the kernel to use updated packages.


PaCMAP has its own graph-learning algorithm, so instead of using the diffusion operator we learned from the multiscale diffusion maps,  TopOMetry automatically feeds it the diffusion maps itself.


```python
pacmap = tg.project(projection_method='PaCMAP')

plt.scatter(pacmap[:, 0], pacmap[:, 1], c=labels, cmap='Spectral', s=0.5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('PaCMAP projection of Multiscale DM of the Digits dataset', fontsize=12)
```

     Computed PaCMAP in 106.888411 (sec)





    Text(0.5, 1.0, 'PaCMAP projection of Multiscale DM of the Digits dataset')




    
![png](MNIST_TopOMetry_Tutorial_files/MNIST_TopOMetry_Tutorial_31_2.png)
    


### Condensing analysis into a single line of code

The `TopOGraph` class has the utility function `TopOGraph.run_models()` to run any given combinations of kernels, eigendecomposition strategies and projection methods. With it, you can perform all the analysis we have shown so far with a single line of code that populates TopOMetry slots. It takes as input a array-like dataset and the specified options:

HINT: You may want to change the verbosity parameter of your TopOGraph object before running all models - it can get quite verborrhagic.


```python
# These are the default options:
tg.run_models(X, kernels=['cknn', 'bw_adaptive'],
                   eigenmap_methods=['DM', 'LE'],
                   projections=['MAP', 'Isomap'])
```

     Fitted the cknn kernel in 0.114642 (sec)
    Computing eigenbasis...
     Fitted eigenbasis with Diffusion Maps from the cknn kernel in 1.097669 (sec)
        Building topological graph from eigenbasis...
            Computing neighborhood graph...
     Computed in 0.165898 (sec)
     Fitted the cknn graph kernel in 0.129180 (sec)
     Computed MAP in 21.914634 (sec)
     Computed Isomap in 2.730034 (sec)
        Building topological graph from eigenbasis...
            Computing neighborhood graph...
     Computed in 0.137594 (sec)
     Fitted the bw_adaptive graph kernel in 0.045746 (sec)
     Computed MAP in 26.702895 (sec)
     Computed Isomap in 2.473890 (sec)
    Computing eigenbasis...
     Fitted eigenbasis with Laplacian Eigenmaps from the cknn in 0.861823 (sec)
        Building topological graph from eigenbasis...
            Computing neighborhood graph...
     Computed in 0.141452 (sec)
     Fitted the cknn graph kernel in 0.071322 (sec)
     Computed MAP in 26.111806 (sec)
     Computed Isomap in 2.897260 (sec)
        Building topological graph from eigenbasis...
            Computing neighborhood graph...
     Computed in 0.136629 (sec)
     Fitted the bw_adaptive graph kernel in 0.038628 (sec)
     Computed MAP in 19.988005 (sec)
     Computed Isomap in 1.734149 (sec)
    Computing eigenbasis...
        Building topological graph from eigenbasis...
            Computing neighborhood graph...
     Computed in 0.134680 (sec)
     Fitted the cknn graph kernel in 0.056481 (sec)
     Computed MAP in 16.370777 (sec)
     Computed Isomap in 1.631021 (sec)
        Building topological graph from eigenbasis...
            Computing neighborhood graph...
     Computed in 0.142196 (sec)
     Fitted the bw_adaptive graph kernel in 0.000022 (sec)
     Computed MAP in 5.608308 (sec)
     Computed Isomap in 1.642030 (sec)
    Computing eigenbasis...
     Fitted eigenbasis with Laplacian Eigenmaps from the bw_adaptive in 0.074959 (sec)
        Building topological graph from eigenbasis...
            Computing neighborhood graph...
     Computed in 0.113214 (sec)
     Fitted the cknn graph kernel in 0.048047 (sec)
     Computed MAP in 4.825907 (sec)
     Computed Isomap in 1.534137 (sec)
        Building topological graph from eigenbasis...
            Computing neighborhood graph...
     Computed in 0.142320 (sec)
     Fitted the bw_adaptive graph kernel in 0.042134 (sec)
     Computed MAP in 4.795768 (sec)
     Computed Isomap in 2.079404 (sec)


As we can see, our `TopOGraph` is now populated with all the desired results:


```python
tg
```




    TopOGraph object with 1797 samples and 64 observations and:
     . Base Kernels: 
        bw_adaptive - .BaseKernelDict['bw_adaptive'] 
        cknn - .BaseKernelDict['cknn']
     . Eigenbases: 
        DM with bw_adaptive - .EigenbasisDict['DM with bw_adaptive'] 
        DM with cknn - .EigenbasisDict['DM with cknn'] 
        LE with cknn - .EigenbasisDict['LE with cknn'] 
        LE with bw_adaptive - .EigenbasisDict['LE with bw_adaptive']
     . Graph Kernels: 
        bw_adaptive from DM with bw_adaptive - .GraphKernelDict['bw_adaptive from DM with bw_adaptive'] 
        cknn from DM with cknn - .GraphKernelDict['cknn from DM with cknn'] 
        bw_adaptive from DM with cknn - .GraphKernelDict['bw_adaptive from DM with cknn'] 
        cknn from LE with cknn - .GraphKernelDict['cknn from LE with cknn'] 
        bw_adaptive from LE with cknn - .GraphKernelDict['bw_adaptive from LE with cknn'] 
        cknn from DM with bw_adaptive - .GraphKernelDict['cknn from DM with bw_adaptive'] 
        cknn from LE with bw_adaptive - .GraphKernelDict['cknn from LE with bw_adaptive'] 
        bw_adaptive from LE with bw_adaptive - .GraphKernelDict['bw_adaptive from LE with bw_adaptive']
     . Projections: 
        Isomap of bw_adaptive from DM with bw_adaptive - .ProjectionDict['Isomap of bw_adaptive from DM with bw_adaptive'] 
        MAP of bw_adaptive from DM with bw_adaptive - .ProjectionDict['MAP of bw_adaptive from DM with bw_adaptive'] 
        PaCMAP of DM with bw_adaptive - .ProjectionDict['PaCMAP of DM with bw_adaptive'] 
        MAP of cknn from DM with cknn - .ProjectionDict['MAP of cknn from DM with cknn'] 
        Isomap of cknn from DM with cknn - .ProjectionDict['Isomap of cknn from DM with cknn'] 
        MAP of bw_adaptive from DM with cknn - .ProjectionDict['MAP of bw_adaptive from DM with cknn'] 
        Isomap of bw_adaptive from DM with cknn - .ProjectionDict['Isomap of bw_adaptive from DM with cknn'] 
        MAP of cknn from LE with cknn - .ProjectionDict['MAP of cknn from LE with cknn'] 
        Isomap of cknn from LE with cknn - .ProjectionDict['Isomap of cknn from LE with cknn'] 
        MAP of bw_adaptive from LE with cknn - .ProjectionDict['MAP of bw_adaptive from LE with cknn'] 
        Isomap of bw_adaptive from LE with cknn - .ProjectionDict['Isomap of bw_adaptive from LE with cknn'] 
        MAP of cknn from DM with bw_adaptive - .ProjectionDict['MAP of cknn from DM with bw_adaptive'] 
        Isomap of cknn from DM with bw_adaptive - .ProjectionDict['Isomap of cknn from DM with bw_adaptive'] 
        MAP of cknn from LE with bw_adaptive - .ProjectionDict['MAP of cknn from LE with bw_adaptive'] 
        Isomap of cknn from LE with bw_adaptive - .ProjectionDict['Isomap of cknn from LE with bw_adaptive'] 
        MAP of bw_adaptive from LE with bw_adaptive - .ProjectionDict['MAP of bw_adaptive from LE with bw_adaptive'] 
        Isomap of bw_adaptive from LE with bw_adaptive - .ProjectionDict['Isomap of bw_adaptive from LE with bw_adaptive'] 
     Active base kernel  -  .base_kernel 
     Active eigenbasis  -  .eigenbasis 
     Active graph kernel  -  .graph_kernel



We may want to plot one of the learned projections:


```python
map_cknn_le = tg.ProjectionDict['MAP of cknn from LE with bw_adaptive']

plt.scatter(map_cknn_le[:, 0], map_cknn_le[:, 1], c=labels, cmap='Spectral', s=0.5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('A projection of the Digits dataset using TopOMetry', fontsize=12)
```




    Text(0.5, 1.0, 'A projection of the Digits dataset using TopOMetry')




    
![png](MNIST_TopOMetry_Tutorial_files/MNIST_TopOMetry_Tutorial_37_1.png)
    


This is it for this first tutorial! To check how to use the `Kernel`, `EigenDecomposition` and `Projector` classes that are used to build TopOMetry to create your own analysis pipeline, check the next tutorial. Feel free to open an issue at GitHub if you have any questions.
