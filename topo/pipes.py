import numpy as np
from scipy.sparse import issparse, csr_matrix
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances
from sklearn.manifold import trustworthiness
from topo.utils._utils import get_landmark_indices
from topo.base.ann import kNN
from topo.topograph import TopOGraph
from topo.eval.global_scores import global_score_pca
from topo.eval.local_scores import geodesic_distance, geodesic_correlation



def global_score(data, emb, Y_pca=False):
    global_scores_pca = global_score_pca(data, emb, Y_pca=Y_pca)
    return global_scores_pca

def eval_models_layouts(TopOGraph, X,
                        methods=['tw', 'gc', 'gs'],
                        kernels=['cknn', 'bw_adaptive'],
                        eigenmap_methods=['DM', 'LE'],
                        projections=['MAP'],
                        additional_eigenbases=None,
                        additional_projections=None,
                        n_neighbors=5, n_jobs=-1,
                        landmark_method='kmeans',
                        metric='euclidean',
                        n_pcs=30,
                        landmarks=None,
                        run_uncomputed_models=True,
                          **kwargs):
    """
    Evaluates all orthogonal bases, topological graphs and layouts in the TopOGraph object.
    
    Currently uses three different quality metrics: trustworthiness (https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html),
    geodesic correlation (defined in the TopOMetry manuscript as the Spearman R correlation between high- and low-dimensional geodesic distances),
    and global score (defined in the TriMAP paper as the MRE normalized by PCA's MRE).

    
    Parameters
    --------------

    TopOGraph : target TopOGraph object (can be empty).

    X : data matrix. Expects either numpy.ndarray or scipy.sparse.csr_matrix.

    methods : list of str (optional, default ['tw', 'gc', 'gs']).
        Methods to use in the evaluation. Options are `'tw'` (trustworthiness), `'gc'` (geodesic correlation), 
        and `'gs'` (global score). `gc` is computationally expensive, so it is recommended to use a small number of landmarks (`landmarks`)
        or not use it at all. Take in mind `gc` is intrinsically related to ISOMAP and distance-preservation methods.

    kernels : list of str (optional, default ['bw_adaptive']).
        List of kernel versions to run and evaluate. These will be used to learn an eigenbasis and to learn a new graph kernel from it.
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
        List of eigenmap methods to run and evaluate. Options are:
        * 'DM'
        * 'LE'
        * 'top'
        * 'bottom'

    projections : list of str (optional, default ['MAP']).
        List of projection methods to run and evaluate. Options are the same of the `topo.layouts.Projector()` object:
        * '(L)Isomap'
        * 't-SNE'
        * 'MAP'
        * 'UMAP'
        * 'PaCMAP'
        * 'TriMAP'
        * 'IsomorphicMDE' - MDE with preservation of nearest neighbors
        * 'IsometricMDE' - MDE with preservation of pairwise distances
        * 'NCVis'

    additional_eigenbases : dict (optional, default None).
        Dictionary containing named additional eigenbases (e.g. factor analysis, AE's latent layer, ICA, etc) to be evaluated.

    additional_projections : dict (optional, default None).
        Dictionary containing named additional projections (e.g. t-SNE, UMAP, etc) to be evaluated.

    n_neighbors : int (optional, default 5).
        Number of nearest neighbors to use for the kNN graph.

    n_jobs : int (optional, default -1).
        Number of jobs to use for parallelization. If -1, uses all available cores.

    landmarks : optional (int, default None).
        If specified, subsamples the TopOGraph object and/or data matrix X to a number of landmark samples
        before computing results and scores. Useful if dealing with large datasets (>30,000 samples).

    landmark_method : str (optional, default 'random').
        Method to use for landmark selection. Options are 'random' and 'kmeans'.

    kwargs : dict (optional, default {}).
        Additional keyword arguments to pass to the `topo.base.ann.kNN()` function.

        
    Returns
    -------

    Populates the TopOGraph object and returns a dictionary of dictionaries with the results

    """

    import gc
    # Run models
    if TopOGraph.verbosity > 0:
        print('Running specified models...')
    if run_uncomputed_models:
        TopOGraph.run_models(X, kernels, eigenmap_methods, projections)
    # Define landmarks if applicable
    if landmarks is not None:
        if landmark_method == 'random':
            if isinstance(landmarks, int):
                landmark_indices = get_landmark_indices(
                    TopOGraph.base_knn_graph, n_landmarks=landmarks, method=landmark_method, random_state=TopOGraph.random_state)
                if landmark_indices.shape[0] == TopOGraph.base_knn_graph.shape[0]:
                    landmark_indices = None
            elif isinstance(landmarks, np.ndarray):
                landmark_indices = landmarks
            else:
                raise ValueError(
                    '\'landmarks\' must be either an integer or a numpy array.')
            base_graph = TopOGraph.base_knn_graph[landmark_indices, :][:, landmark_indices]
        elif landmark_method == 'kmeans':
            if isinstance(landmarks, int):
                landmarks = get_landmark_indices(
                    TopOGraph.base_knn_graph, n_landmarks=landmarks, method=landmark_method, random_state=TopOGraph.random_state)
            else:
                raise ValueError(
                    '\'landmarks\' must be either an integer or a numpy array.')
            base_graph = kNN(landmarks, n_neighbors=n_neighbors,
                            metric=TopOGraph.base_metric,
                            n_jobs=n_jobs,
                            backend=TopOGraph.backend,
                            return_instance=False,
                            verbose=False, **kwargs)
            gc.collect()
        else:
            raise ValueError(
                    '\'landmark_method\' must be either `random` or `kmeans`.')
    else:
        base_graph = TopOGraph.base_knn_graph

    gc.collect()
    # Run PCA
    from sklearn.decomposition import PCA
    if TopOGraph.verbosity >= 1:
        print('Computing PCA for comparison...')
    import numpy as np
    if issparse(X) == True:
        if isinstance(X, csr_matrix):
            data = X.todense()
            gc.collect()
    if issparse(X) == False:
        if not isinstance(X, np.ndarray):
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                data = np.asarray(X.values.T)
            else:
                return print('Uknown data format.')
        else:
            data = X
    gc.collect()
    # Now with PCA
    pca_emb = PCA(n_components=n_pcs).fit_transform(data)
    # Create empty dicts for the results
    EigenbasisTWResults = {}
    EigenbasisGCResults = {}
    EigenbasisGSResults = {}

    if TopOGraph.verbosity > 0:
        print('Assessing base graph...')
    if 'gc' in methods:
        base_geodesics = squareform(geodesic_distance(base_graph, directed=False, n_jobs=n_jobs))
    if 'tw' in methods:
        if isinstance(X, csr_matrix):
            data_pdist = pairwise_distances(X.toarray(), metric=metric, n_jobs=n_jobs)
        else:
            data_pdist = pairwise_distances(X, metric=metric, n_jobs=n_jobs)
    gc.collect()
    for key in TopOGraph.EigenbasisDict.keys():
        if 'gc' in methods:
            if TopOGraph.verbosity > 0:
                print('Computing geodesics for eigenbasis \'{}...\''.format(key))
            emb_graph = kNN(TopOGraph.EigenbasisDict[key].results(), n_neighbors=n_neighbors,
                            metric=metric,
                            n_jobs=n_jobs,
                            backend=TopOGraph.backend,
                            return_instance=False,
                            verbose=False, **kwargs)
            if landmarks is not None:
                emb_graph = emb_graph[landmark_indices, :][:, landmark_indices]
                gc.collect()
            embedding_geodesics = squareform(geodesic_distance(emb_graph, directed=False, n_jobs=n_jobs)
                                            )
            gc.collect()
            if TopOGraph.verbosity > 0:
                print('Computing geodesic correlation for eigenbasis \'{}...\''.format(key))
            EigenbasisGCResults[key], _ = spearmanr(
                base_geodesics, embedding_geodesics)
            gc.collect()
        if 'gs' in methods:
            EigenbasisGSResults[key] = global_score(
                X, TopOGraph.EigenbasisDict[key].results(), Y_pca=pca_emb)
            if TopOGraph.verbosity > 0:
                print('Computed global score for eigenbasis {}'.format(key))
            gc.collect()
        if 'tw' in methods:
            EigenbasisTWResults[key] = trustworthiness(data_pdist,
                                                        TopOGraph.EigenbasisDict[key].results(), metric='precomputed')
            if TopOGraph.verbosity > 0:
                print('Computed trustworthiness for projection {}'.format(key))
            gc.collect()

    ProjectionTWResults = {}
    ProjectionGCResults = {}
    ProjectionGSResults = {}
    for key in TopOGraph.ProjectionDict.keys():
        if 'gc' in methods:
            if TopOGraph.verbosity > 0:
                print('Computing geodesics for projection \' {}...\''.format(key))
            emb_graph = kNN(TopOGraph.ProjectionDict[key],
                            n_neighbors=n_neighbors,
                            metric=metric,
                            n_jobs=n_jobs,
                            backend=TopOGraph.backend,
                            return_instance=False,
                            verbose=False, **kwargs)
            gc.collect()
            if landmarks is not None:
                emb_graph = emb_graph[landmark_indices, :][:, landmark_indices]
                gc.collect()
            embedding_geodesics = squareform(geodesic_distance(emb_graph, directed=False, n_jobs=n_jobs)
                )
            gc.collect()
            if TopOGraph.verbosity > 0:
                print('Computing Sgeodesic correlation for projection \'{}...\''.format(key))
            ProjectionGCResults[key], _ = spearmanr(
                base_geodesics, embedding_geodesics)
            gc.collect()
        if 'gs' in methods:
            ProjectionGSResults[key] = global_score(
                X, TopOGraph.ProjectionDict[key], Y_pca=pca_emb)
            if TopOGraph.verbosity > 0:
                print('Computed global score for projection {}'.format(key))
            gc.collect()
        if 'tw' in methods:
            ProjectionTWResults[key] = trustworthiness(data_pdist,
                                                        TopOGraph.ProjectionDict[key], metric='precomputed')
            if TopOGraph.verbosity > 0:
                print('Computed trustworthiness for projection {}'.format(key))
            gc.collect()

    gc.collect()
    if 'gc' in methods:
        if TopOGraph.verbosity > 0:
            print('Computing Spearman R for PCA...')
        emb_graph = kNN(pca_emb,
                        n_neighbors=n_neighbors,
                        metric=TopOGraph.base_metric,
                        n_jobs=n_jobs,
                        backend=TopOGraph.backend,
                        return_instance=False,
                        verbose=False, **kwargs)
        if landmarks is not None:
            emb_graph = emb_graph[landmark_indices, :][:, landmark_indices]
            gc.collect()
        embedding_geodesics = squareform(geodesic_distance(emb_graph, directed=False, n_jobs=n_jobs)
            )
        gc.collect()
        if TopOGraph.verbosity > 0:
            print('Computing Spearman R for PCA...')
        EigenbasisGCResults['PCA'], _ = spearmanr(
            base_geodesics, embedding_geodesics)
        gc.collect()
    if 'tw' in methods:
        EigenbasisTWResults['PCA'] = trustworthiness(data_pdist,
                                                pca_emb, metric='precomputed')
        if TopOGraph.verbosity > 0:
            print('Computed trustworthiness for projection {}'.format(key))
        gc.collect()
    if additional_eigenbases is not None:
        for key in additional_eigenbases.keys():
            if 'gc' in methods:
                if TopOGraph.verbosity > 0:
                    print('Computing geodesics for eigenbasis \'{}...\''.format(key))
                emb_graph = kNN(additional_eigenbases[key], n_neighbors=n_neighbors,
                                metric=metric,
                                n_jobs=n_jobs,
                                backend=TopOGraph.backend,
                                return_instance=False,
                                verbose=False, **kwargs)
                gc.collect()
                if landmarks is not None:
                    emb_graph = emb_graph[landmark_indices, :][:, landmark_indices]
                    gc.collect()
                embedding_geodesics = squareform(geodesic_distance(emb_graph, directed=False, n_jobs=n_jobs)
                                                )
                gc.collect()
                if TopOGraph.verbosity > 0:
                    print('Computing Spearman R for eigenbasis \'{}...\''.format(key))
                EigenbasisGCResults[key], _ = spearmanr(
                    base_geodesics, embedding_geodesics)
                gc.collect()
            if 'gs' in methods:
                EigenbasisGSResults[key] = global_score(
                    X, additional_eigenbases[key], Y_pca=pca_emb)
                if TopOGraph.verbosity > 0:
                    print('Computed global score for eigenbasis {}'.format(key))
                gc.collect()
            if 'tw' in methods:
                EigenbasisTWResults[key] = trustworthiness(data_pdist,
                                                            additional_eigenbases[key], metric='precomputed')
                if TopOGraph.verbosity > 0:
                    print('Computed trustworthiness for eigenbasis {}'.format(key))
                gc.collect()
    if additional_projections is not None:
        for key in additional_projections.keys():
            if 'gc' in methods:
                if TopOGraph.verbosity > 0:
                    print('Computing geodesics for eigenbasis \'{}...\''.format(key))
                emb_graph = kNN(additional_projections[key], n_neighbors=n_neighbors,
                                metric=metric,
                                n_jobs=n_jobs,
                                backend=TopOGraph.backend,
                                return_instance=False,
                                verbose=False, **kwargs)
                gc.collect()
                if landmarks is not None:
                    emb_graph = emb_graph[landmark_indices, :][:, landmark_indices]
                    gc.collect()
                embedding_geodesics = squareform(geodesic_distance(emb_graph, directed=False, n_jobs=n_jobs)
                                                )
                gc.collect()
                if TopOGraph.verbosity > 0:
                    print('Computing Spearman R for projection \'{}...\''.format(key))
                ProjectionGCResults[key], _ = spearmanr(
                    base_geodesics, embedding_geodesics)
                gc.collect()
            if 'gs' in methods:
                ProjectionGSResults[key] = global_score(
                    X, additional_projections[key], Y_pca=pca_emb)
                if TopOGraph.verbosity > 0:
                    print('Computed global score for projection {}'.format(key))
                gc.collect()
            if 'tw' in methods:
                ProjectionTWResults[key] = trustworthiness(data_pdist,
                                                            additional_projections[key], metric='precomputed')
                if TopOGraph.verbosity > 0:
                    print('Computed trustworthiness for projection {}'.format(key))
                gc.collect()

    res_dict = {'Eigenbasis - Trustworthiness' : EigenbasisTWResults,
                'Eigenbasis - Geodesic correlation' : EigenbasisGCResults,
                'Eigenbasis - Global score' : EigenbasisGSResults,
                'Projection - Trustworthiness' : ProjectionTWResults,
                'Projection - Geodesic correlation' : ProjectionGCResults,
                'Projection - Global score' : ProjectionGSResults,   
    }

    return res_dict



def explained_variance(X, title='some data', n_pcs=200, figsize=(12,6), sup_title_fontsize=20, title_fontsize=16, return_dicts=False):
    """
    Plots the explained variance by PCA with varying number of highly variable genes.

    Parameters
    ----------
    X: np.ndarray (2D) of observations per sample.

    title: str (optional, default 'some data').

    n_pcs: int (optional, default 200).
        Number of principal components to use.

    figsize: tuple of int (optional, default (12,6)).

    sup_title_fontsize: int (optional, default 20).

    title_fontsize: int (optional, default 16).

    return_dicts: bool (optional, default False).
        Whether to return explained covariance ratio and singular values dictionaries.

    Returns
    -------
    A plot. If `return_dicts=True`, also returns a tuple of dictionaries (explained_cov_ratio, singular_values) with the keys
    being strings with the number of genes and the values being the explained covariance ratio
    and the singular values for PCA.
    
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    explained_cov_ratio = {}
    singular_values = {}
    pca = PCA(n_components=n_pcs)
    pca.fit(X)
    explained_cov_ratio = pca.explained_variance_ratio_
    singular_values = pca.singular_values_
    plt.figure(figsize=figsize)
    plt.subplots_adjust(left=0.2, right=0.98, bottom=0.001,
                        top=0.9, wspace=0.15, hspace=0.01)
    plt.suptitle(title, fontsize=sup_title_fontsize)
    plt.subplot(1, 2, 1)
    plt.plot(singular_values)
    plt.title('Eigenspectrum', fontsize=title_fontsize)
    plt.xlabel('Principal component', fontsize=title_fontsize-6)
    plt.ylabel('Singular values', fontsize=title_fontsize-6)
    plt.legend(fontsize=11)
    plt.subplot(1, 2, 2)
    plt.plot(explained_cov_ratio.cumsum())
    plt.title('Total explained variance', fontsize=title_fontsize)
    plt.xlabel('Principal component', fontsize=13)
    plt.ylabel('Cumulative explained variance', fontsize=title_fontsize-6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
    if return_dicts:
        return explained_cov_ratio, singular_values