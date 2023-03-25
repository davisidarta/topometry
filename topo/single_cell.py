# Wrapper functions for single-cell analysis with Scanpy and TopOMetry
#
# All of these functions call scanpy and thus require it for working
# However, I opted not to include it as a hard-dependency as not all users are interested in single-cell analysis
#
try:
    import scanpy as sc
    _HAVE_SCANPY = True
except ImportError:
    _HAVE_SCANPY = False

# Functions will be defined only if user has scanpy installed.
if _HAVE_SCANPY:
    import numpy as np
    import scanpy.external as sce
    from scipy.sparse import csr_matrix
    from topo.topograph import TopOGraph

    def preprocess(AnnData, normalize=True, log=True, target_sum=1e4, min_mean=0.0125, max_mean=8, min_disp=0.3, max_value=10, save_to_raw=True, plot_hvg=False, scale=True, **kwargs):
        """
        A wrapper around Scanpy's preprocessing functions. Normalizes RNA library by size, logarithmizes it and
        selects highly variable genes for subsetting the AnnData object. Automatically subsets the Anndata
        object and saves the full expression matrix to AnnData.raw.


        Parameters
        ----------
        AnnData : the target AnnData object.

        normalize : bool (optional, default True).
            Whether to size-normalize each cell.
        
        log : bool (optional, default True).
            Whether to log-transform for variance stabilization.

        target_sum : int (optional, default 1e4).
            constant for library size normalization.

        min_mean : float (optional, default 0.0125).
            Minimum gene expression level for inclusion as highly-variable gene.

        max_mean : float (optional, default 8.0).
            Maximum gene expression level for inclusion as highly-variable gene.

        min_disp : float (optional, default 0.3).
            Minimum expression dispersion for inclusion as highly-variable gene.

        save_to_raw : bool (optional, default True).
            Whether to save the full expression matrix to AnnData.raw.

        plot_hvg : bool (optional, default False).
            Whether to plot the high-variable genes plot.

        scale : bool (optional, default True).
            Whether to zero-center and scale the data to unit variance.

        max_value : float (optional, default 10.0).
            Maximum value for clipping the data after scaling.

        **kwargs : dict (optional, default {})
            Additional keyword arguments for `sc.pp.highly_variable_genes()`.

        Returns
        -------

        Updated AnnData object.

        """
        if normalize:
            sc.pp.normalize_total(AnnData, target_sum=target_sum)
        if log:
            sc.pp.log1p(AnnData)
        sc.pp.highly_variable_genes(
            AnnData, min_mean=min_mean, max_mean=max_mean, min_disp=min_disp, **kwargs)
        if plot_hvg:
            sc.pl.highly_variable_genes(AnnData)
        if save_to_raw:
            AnnData.raw = AnnData
        AnnData = AnnData[:, AnnData.var.highly_variable]
        if scale:
            sc.pp.scale(AnnData, max_value=max_value)
        return AnnData.copy()

    def default_workflow(AnnData, n_neighbors=15, n_pcs=50, metric='euclidean',
                         resolution=0.8, min_dist=0.5, spread=1, maxiter=600):
        """

        A wrapper around scanpy's default workflow: mean-center and scale the data, perform PCA and use its
        output to construct k-Nearest-Neighbors graphs. These graphs are used for louvain clustering and
        laid out with UMAP. For simplicity, only the main arguments are included; we recommend you tailor
        a simple personalized workflow if you want to optimize hyperparameters.

        Parameters
        ----------

        AnnData: the target AnnData object.

        n_neighbors: int (optional, default 15).
            Number of neighbors for kNN graph construction.

        n_pcs: int (optional, default 50).
            Number of principal components to retain for downstream analysis.

        metric: str (optional, default 'euclidean').
            Metric used for neighborhood graph construction. Common values are 'euclidean' and 'cosine'.

        resolution: float (optional, default 0.8).
            Resolution parameter for the leiden graph community clustering algorithm.

        min_dist: float (optional, default 0.5).
            Key hyperparameter for UMAP embedding. Smaller values lead to more 'concentrated' graphs, however can
            lead to loss of global structure. Recommended values are between 0.3 and 0.8.

        spread: float (optional, default 1.0).
            Key hyperparameter for UMAP embedding. Controls the global spreading of data in the embedidng during
            optimization. Larger values lead to more spread out layouts, but can lead to loss of local structure.
            Ideally, this parameter should vary with `min_dist`.

        maxiter: int (optional, 600).
            Number of maximum iterations for the UMAP embedding optimization.


        Returns
        -------

        Updated AnnData object.

        """
        sc.tl.pca(AnnData, n_comps=n_pcs)
        sc.pp.neighbors(AnnData, n_neighbors=n_neighbors, metric=metric, n_pcs=n_pcs)
        sc.tl.leiden(AnnData, resolution=resolution)
        sc.tl.umap(AnnData, min_dist=min_dist, spread=spread, maxiter=maxiter)
        AnnData.obsm['X_pca_umap'] = AnnData.obsm['X_umap']
        AnnData.obs['pca_leiden'] = AnnData.obs['leiden']
        return AnnData.copy()

    def default_integration_workflow(AnnData,
                                     integration_method=[
                                         'harmony', 'scanorama', 'bbknn'],
                                     batch_key='batch',
                                     n_neighbors=15,
                                     n_pcs=50,
                                     metric='euclidean',
                                     resolution=0.8,
                                     min_dist=0.5,
                                     spread=1,
                                     maxiter=600,
                                     **kwargs):
        """

        A wrapper around scanpy's default integration workflows: harmony, scanorama and bbknn.

        Parameters
        ----------

        AnnData: the target AnnData object.

        integration_method: str( optional, default ['harmony', 'scanorama', 'bbknn', 'scvi']).
            Which integration methods to run. Defaults to all.

        n_neighbors: int (optional, default 15).
            Number of neighbors for kNN graph construction.

        n_pcs: int (optional, default 50).
            Number of principal components to retain for downstream analysis in scanorama.

        metric: str (optional, default 'euclidean').
            Metric used for neighborhood graph construction. Common values are 'euclidean' and 'cosine'.

        resolution: float (optional, default 0.8).
            Resolution parameter for the leiden graph community clustering algorithm.

        min_dist: float (optional, default 0.5).
            Key hyperparameter for UMAP embedding. Smaller values lead to more 'concentrated' graphs, however can
            lead to loss of global structure. Recommended values are between 0.3 and 0.8.

        spread: float (optional, default 1.0).
            Key hyperparameter for UMAP embedding. Controls the global spreading of data in the embedidng during
            optimization. Larger values lead to more spread out layouts, but can lead to loss of local structure.
            Ideally, this parameter should vary with `min_dist`.

        maxiter: int (optional, 600).
            Number of maximum iterations for the UMAP embedding optimization.

        kwargs: additional parameters to be passed for the integration method. To use this option,
            select only one integration method at a time - otherwise, it'll raise several errors.


        Returns
        -------

        Batch-corrected and updated AnnData object.

        """
        # Batch-correct latent representations
        # With harmony

        if 'harmony' in integration_method:
            sce.pp.harmony_integrate(AnnData, key=batch_key, basis='X_pca',
                                     adjusted_basis='X_pca_harmony', **kwargs)
            sc.pp.neighbors(AnnData, use_rep='X_pca_harmony',
                            n_neighbors=n_neighbors, metric=metric)
            sc.tl.leiden(AnnData, key_added='pca_harmony_leiden',
                         resolution=resolution)
            sc.tl.umap(AnnData, min_dist=min_dist,
                       maxiter=maxiter, spread=spread)
            AnnData.obsm['X_pca_harmony_umap'] = AnnData.obsm['X_umap']

        if 'scanorama' in integration_method:
            try:
                import scanorama
            except ImportError:
                return ((print("scanorama is required for using scanorama as batch-correction method."
                               " Please install it with `pip install scanorama`. ")))

            # subset the individual dataset to the same variable genes as in MNN-correct.
            # split per batch into new objects.
            batches = AnnData.obs[batch_key].cat.categories.tolist()
            alldata = {}
            for batch in batches:
                alldata[batch] = AnnData[AnnData.obs[batch_key] == batch, ]

            # convert to list of AnnData objects
            adatas = list(alldata.values())
            # run scanorama.integrate
            scanorama.integrate_scanpy(adatas, dimred=n_pcs, **kwargs)
            # Get all the integrated matrices.
            scanorama_int = [ad.obsm['X_scanorama'] for ad in adatas]
            # make into one matrix.
            all_s = np.concatenate(scanorama_int)
            print(all_s.shape)
            # add to the AnnData object
            AnnData.obsm["X_pca_scanorama"] = all_s
            sc.pp.neighbors(AnnData, use_rep='X_pca_scanorama',
                            n_neighbors=n_neighbors, metric=metric)
            sc.tl.leiden(AnnData, key_added='pca_scanorama_leiden',
                         resolution=resolution)
            sc.tl.umap(AnnData, min_dist=min_dist,
                       maxiter=maxiter, spread=spread)
            AnnData.obsm['X_pca_scanorama_umap'] = AnnData.obsm['X_umap']

        if 'bbknn' in integration_method:
            try:
                import bbknn
            except ImportError:
                return ((print("bbknn is required for using BBKNN as batch-correction method."
                               " Please install it with `pip install bbknn`. ")))
            if 'pca_leiden' not in AnnData.obs.keys():
                sc.pp.neighbors(
                    AnnData, n_neighbors=n_neighbors, metric=metric)
                sc.tl.leiden(AnnData, resolution=resolution,
                             key_added='pca_leiden')
            bbknn.ridge_regression(AnnData, batch_key=batch_key,
                                   confounder_key=['pca_leiden'])
            bbknn.bbknn(AnnData, batch_key=batch_key, use_rep='X_pca',
                        n_pcs=None, **kwargs)
            bbknn_graph = csr_matrix(AnnData.obsp['connectivities'])
            sc.tl.leiden(AnnData, key_added='pca_BBKNN_leiden',
                         adjacency=bbknn_graph, resolution=resolution)
            sc.tl.umap(AnnData, min_dist=min_dist,
                       maxiter=maxiter, spread=spread)
            AnnData.obsm['X_pca_bbknn_umap'] = AnnData.obsm['X_umap']

        # if 'scvi' in integration_method:
        #     try:
        #         import scvi
        #     except ImportError:
        #         return((print("scvi is required for using scvi as batch-correction method."
        #                       " Please install it with `pip install scvi-tools`. ")))
        #     scvi.data.setup_anndata(AnnData, batch_key=batch_key)
        #     vae = scvi.model.SCVI(AnnData, n_layers=5, n_latent=n_pcs, gene_likelihood="nb")
        #     vae.train()
        #     AnnData.obsm["X_scVI"] = vae.get_latent_representation()
        #     sc.pp.neighbors(AnnData, use_rep='X_scvi', n_neighbors=n_neighbors, metric=metric)
        #     sc.tl.leiden(AnnData, key_added='scvi_leiden', resolution=resolution)
        #     sc.tl.umap(AnnData, min_dist=min_dist, maxiter=maxiter, spread=spread)
        #     AnnData.obsm['X_scvi_umap'] = AnnData.obsm['X_umap']

        return AnnData

    def topological_workflow(AnnData, topograph=None,
                             kernels=['fuzzy', 'cknn',
                                      'bw_adaptive'],
                             eigenmap_methods=['DM', 'LE'],
                             projections=['Isomap', 'MAP'],
                             resolution=0.8,
                             X_to_csr=False, **kwargs):
        """

        A wrapper around TopOMetry's topological workflow. Clustering is performed
        with the leiden algorithm on TopOMetry's topological graphs. This wrapper takes an AnnData object containing a
        cell per feature matrix (np.ndarray or scipy.sparse.csr_matrix) and a TopOGraph object. If no TopOGraph object is
        provided, it will generate a new one, *which will not be stored*. The TopOGraph object keeps all analysis results,
        but similarity matrices, dimensional reductions and clustering results are also added to the AnnData object.

        All parameters for the topological analysis must have been added to the TopOGraph object beforehand; otherwise,
        default parameters will be used. Within this wrapper, users only select which kernel and eigendecomposition models to use
        and what graph-layout layout options to use them with. For hyperparameter tuning, the embeddings must be obtained separetely.


        Parameters
        ----------

        AnnData: the target AnnData object.

        topograph: celltometry.TopOGraph (optional).
            The TopOGraph object containing parameters for the topological analysis.

        kernels : list of str (optional, default ['fuzzy', 'cknn', 'bw_adaptive']).
            List of kernel versions to run. These will be used to learn an eigenbasis and to learn a new graph kernel from it.
            Options are:
            * 'fuzzy'
            * 'cknn'
            * 'bw_adaptive'
            * 'bw_adaptive_alpha_decaying'
            * 'bw_adaptive_nbr_expansion'
            * 'bw_adaptive_alpha_decaying_nbr_expansion'
            * 'gaussian'
            Will not run all by default to avoid long waiting times in reckless calls.

        eigenmap_methods : list of str (optional, default ['DM', 'LE', 'top']).
            List of eigenmap methods to run. Options are:
            * 'DM'
            * 'LE'
            * 'top'
            * 'bottom'
        
        projections : list of str (optional, default ['Isomap', 'MAP']).
            List of projection methods to run. Options are the same of the `topo.layouts.Projector()` object:
            * '(L)Isomap'
            * 't-SNE'
            * 'MAP'
            * 'UMAP'
            * 'PaCMAP'
            * 'TriMAP'
            * 'IsomorphicMDE' - MDE with preservation of nearest neighbors
            * 'IsometricMDE' - MDE with preservation of pairwise distances
            * 'NCVis'

        resolution : float (optional, default 0.8).
            Resolution parameter for the leiden graph community clustering algorithm.

        X_to_csr : bool (optional, default False).
            Whether to convert the data matrix in AnnData.X to a csr_matrix format prior to analysis. This is quite
            useful if the data matrix is rather sparse and may significantly speed up computations.

        kwargs : dict (optional)
            Additional parameters to be passed to the sc.tl.leiden() function for clustering.

        Returns
        -------

        Updated AnnData object.

        """

        if topograph is None:
            topograph = TopOGraph()
        if X_to_csr:
            from scipy.sparse import csr_matrix
            data = csr_matrix(AnnData.X)
        else:
            data = AnnData.X
        topograph.run_models(data,
                   kernels=kernels,
                   eigenmap_methods=eigenmap_methods,
                   projections=projections)
        
        # Get results to AnnData
        for kernel in kernels:  
            for eigenmap_method in eigenmap_methods:
                if eigenmap_method in ['DM', 'LE']:
                    basis_key = eigenmap_method + ' with ' + str(kernel)
                elif eigenmap_method == 'top':
                    basis_key = 'Top eigenpairs with ' + str(kernel)
                elif eigenmap_method == 'bottom':
                    basis_key = 'Bottom eigenpairs with ' + str(kernel)
                else:
                    raise ValueError('Unknown eigenmap method.')
                AnnData.obsm['X_' + basis_key] = topograph.EigenbasisDict[basis_key].transform(data) # returns the scaled eigenvectors

                for kernel in kernels:
                    graph_key = kernel + ' from ' + basis_key
                    AnnData.obsp[basis_key + '_distances'] = topograph.eigenbasis_knn_graph
                    AnnData.obsp[graph_key + '_connectivities'] = topograph.GraphKernelDict[graph_key].P
                    sc.tl.leiden(AnnData, adjacency = topograph.GraphKernelDict[graph_key].P, resolution=resolution, key_added = graph_key + '_leiden', **kwargs)
                for projection in projections:
                    if projection in ['MAP', 'UMAP', 'MDE', 'Isomap']: 
                        suffix_key = graph_key
                    else:
                        suffix_key = basis_key
                    projection_key = projection + ' of ' + suffix_key
                    AnnData.obsm['X_' + projection_key] = topograph.ProjectionDict[projection_key]
        return AnnData

