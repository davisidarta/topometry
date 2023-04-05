import numpy as np
from sklearn.utils import check_random_state
from scipy.sparse import issparse, csr_matrix
from scipy.stats import spearmanr, kendalltau
from scipy.spatial.distance import squareform
from topo.utils._utils import get_landmark_indices
from topo.base.ann import kNN
from topo.topograph import TopOGraph
from topo.eval.global_scores import global_score_pca
from topo.eval.local_scores import geodesic_distance


def local_score(data, emb, landmarks=None,
                        landmark_method='random',
                        metric='euclidean',
                        n_neighbors=3, n_jobs=-1,
                        cor_method='spearman', random_state=None, **kwargs):
    
    if random_state is None:
            random_state = np.random.RandomState()
    elif isinstance(random_state, np.random.RandomState):
        pass
    elif isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)
    else:
        print('RandomState error! No random state was defined!')

    if isinstance(data, csr_matrix):
        if data.shape[0] == data.shape[1]:
            # graph
            DATA_IS_GRAPH = True
        else:
            # data
            DATA_IS_GRAPH = False
    else:
        if np.shape(data)[0] == np.shape(data)[1]:
            # graph
            DATA_IS_GRAPH = True
        else:
            # data
            DATA_IS_GRAPH = False
    if isinstance(emb, csr_matrix):
        if emb.shape[0] == emb.shape[1]:
            # graph
            EMB_IS_GRAPH = True
        else:
            # data
            EMB_IS_GRAPH = False
    else:
        if np.shape(emb)[0] == np.shape(emb)[1]:
            # graph
            EMB_IS_GRAPH = True
        else:
            # data
            EMB_IS_GRAPH = False

    
    if not DATA_IS_GRAPH:
        data_graph = kNN(data, n_neighbors=n_neighbors,
                        metric=metric,
                        n_jobs=n_jobs,
                        return_instance=False,
                        verbose=False, **kwargs)
    else:
        data_graph = data.copy()

    if not EMB_IS_GRAPH:
        emb_graph = kNN(emb, n_neighbors=n_neighbors,
                        metric=metric,
                        n_jobs=n_jobs,
                        return_instance=False,
                        verbose=False, **kwargs)
    else:
        emb_graph = emb.copy()

    # Define landmarks if applicable
    if landmarks is not None:
        if isinstance(landmarks, int):
            landmark_indices = get_landmark_indices(
                data_graph, n_landmarks=landmarks, method=landmark_method, random_state=TopOGraph.random_state)
            if landmark_indices.shape[0] == TopOGraph.base_knn_graph.shape[0]:
                landmark_indices = None
        elif isinstance(landmarks, np.ndarray):
            landmark_indices = landmarks
        else:
            raise ValueError(
                '\'landmarks\' must be either an integer or a numpy array.')
        
    if landmarks is not None:
        data_graph = data_graph[landmark_indices, :][:, landmark_indices]
        emb_graph = emb_graph[landmark_indices, :][:, landmark_indices]
    
    base_geodesics = squareform(geodesic_distance(
        data_graph, directed=False, n_jobs=n_jobs))
    embedding_geodesics = squareform(geodesic_distance(
        emb_graph, directed=False, n_jobs=n_jobs))
    
    if cor_method == 'spearman':
        print('Computing Spearman R...')
        results = spearmanr(
            base_geodesics, embedding_geodesics).correlation
    else:
        print('Computing Kendall Tau for eigenbasis...')
        results = kendalltau(
            base_geodesics, embedding_geodesics)
    return results.correlation

def global_score(data, emb):
    global_scores_pca = global_score_pca(data, emb)
    return global_scores_pca

def eval_models_layouts(TopOGraph, X,
                        landmarks=None,
                        kernels=['cknn', 'bw_adaptive'],
                        eigenmap_methods=['DM', 'LE'],
                        projections=['MAP'],
                        additional_eigenbases=None,
                        additional_projections=None,
                        landmark_method='random',
                        n_neighbors=5, n_jobs=-1,
                        cor_method='spearman', **kwargs):
    """
    Evaluates all orthogonal bases, topological graphs and layouts in the TopOGraph object.
    Compares results with PCA and PCA-derived layouts (i.e. t-SNE, UMAP etc).

    Parameters
    --------------

    TopOGraph : target TopOGraph object (can be empty).

    X : data matrix. Expects either numpy.ndarray or scipy.sparse.csr_matrix.

    landmarks : optional (int, default None).
        If specified, subsamples the TopOGraph object and/or data matrix X to a number of landmark samples
        before computing results and scores. Useful if dealing with large datasets (>30,000 samples).

    kernels : list of str (optional, default ['fuzzy', 'cknn', 'bw_adaptive_alpha_decaying']).
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

    projections : list of str (optional, default ['Isomap', 'MAP']).
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
        Dictionary containing named additional eigenbases (e.g. factor analysis, VAEs, ICA, etc) to be evaluated.

    additional_projections : dict (optional, default None).
        Dictionary containing named additional projections (e.g. t-SNE, UMAP, etc) to be evaluated.

    n_neighbors : int (optional, default 5).
        Number of nearest neighbors to use for the kNN graph.

    n_jobs : int (optional, default -1).
        Number of jobs to use for parallelization. If -1, uses all available cores.

    cor_method : str (optional, default 'spearman').
        Correlation method to use for local scores. Options are 'spearman' and 'kendall'.

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
    TopOGraph.run_models(X, kernels, eigenmap_methods, projections)
    # Define landmarks if applicable
    if landmarks is not None:
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

    # Compute geodesics
    gc.collect()
    EigenbasisLocalResults = {}
    EigenbasisGlobalResults = {}
    if TopOGraph.verbosity > 0:
        print('Computing base geodesics...')
    if landmarks is not None:
        base_graph = TopOGraph.base_knn_graph[landmark_indices, :][:, landmark_indices]
    else:
        base_graph = TopOGraph.base_knn_graph
    base_geodesics = squareform(geodesic_distance(
        base_graph, directed=False, n_jobs=n_jobs))
    gc.collect()
    for key in TopOGraph.EigenbasisDict.keys():
        if TopOGraph.verbosity > 0:
            print('Computing geodesics for eigenbasis \'{}...\''.format(key))
        emb_graph = kNN(TopOGraph.EigenbasisDict[key].results(), n_neighbors=n_neighbors,
                        metric=TopOGraph.base_metric,
                        n_jobs=n_jobs,
                        backend=TopOGraph.backend,
                        return_instance=False,
                        verbose=False, **kwargs)
        if landmarks is not None:
            emb_graph = emb_graph[landmark_indices, :][:, landmark_indices]
            gc.collect()
        embedding_geodesics = squareform(geodesic_distance(
            emb_graph, directed=False, n_jobs=n_jobs))
        gc.collect()
        if cor_method == 'spearman':
            if TopOGraph.verbosity > 0:
                print('Computing Spearman R for eigenbasis \'{}...\''.format(key))
            EigenbasisLocalResults[key], _ = spearmanr(
                base_geodesics, embedding_geodesics)
        else:
            if TopOGraph.verbosity > 0:
                print('Computing Kendall Tau for eigenbasis \'{}...\''.format(key))
            EigenbasisLocalResults[key], _ = kendalltau(
                base_geodesics, embedding_geodesics)
        gc.collect()
        EigenbasisGlobalResults[key] = global_score(
            X, TopOGraph.EigenbasisDict[key].results())
        if TopOGraph.verbosity > 0:
            print('Finished for eigenbasis {}'.format(key))
        gc.collect()
    ProjectionLocalResults = {}
    ProjectionGlobalResults = {}
    for key in TopOGraph.ProjectionDict.keys():
        if TopOGraph.verbosity > 0:
            print('Computing geodesics for projection \' {}...\''.format(key))
        emb_graph = kNN(TopOGraph.ProjectionDict[key],
                        n_neighbors=n_neighbors,
                        metric=TopOGraph.graph_metric,
                        n_jobs=n_jobs,
                        backend=TopOGraph.backend,
                        return_instance=False,
                        verbose=False, **kwargs)
        if landmarks is not None:
            emb_graph = emb_graph[landmark_indices, :][:, landmark_indices]
            gc.collect()
        embedding_geodesics = squareform(geodesic_distance(
            emb_graph, directed=False, n_jobs=n_jobs))
        gc.collect()
        if cor_method == 'spearman':
            if TopOGraph.verbosity > 0:
                print('Computing Spearman R for projection \'{}...\''.format(key))
            ProjectionLocalResults[key], _ = spearmanr(
                base_geodesics, embedding_geodesics)
        else:
            if TopOGraph.verbosity > 0:
                print('Computing Kendall Tau for projection \'{}...\''.format(key))
            ProjectionLocalResults[key], _ = kendalltau(
                base_geodesics, embedding_geodesics)
        gc.collect()
        ProjectionGlobalResults[key] = global_score(
            X, TopOGraph.ProjectionDict[key])
        gc.collect()
    from sklearn.decomposition import PCA
    if TopOGraph.verbosity >= 1:
        print('Computing PCA for comparison...')
    import numpy as np
    if issparse(X) == True:
        if isinstance(X, csr_matrix):
            data = X.todense()
    if issparse(X) == False:
        if not isinstance(X, np.ndarray):
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                data = np.asarray(X.values.T)
            else:
                return print('Uknown data format.')
        else:
            data = X
    pca_emb = PCA(n_components=TopOGraph.n_eigs).fit_transform(data)
    emb_graph = kNN(pca_emb,
                    n_neighbors=n_neighbors,
                    metric=TopOGraph.graph_metric,
                    n_jobs=n_jobs,
                    backend=TopOGraph.backend,
                    return_instance=False,
                    verbose=False, **kwargs)
    if landmarks is not None:
        emb_graph = emb_graph[landmark_indices, :][:, landmark_indices]
        gc.collect()
    embedding_geodesics = squareform(geodesic_distance(
        emb_graph, directed=False, n_jobs=n_jobs))
    gc.collect()
    if TopOGraph.verbosity > 0:
        print('Computing Spearman R for PCA...')
    EigenbasisLocalResults['PCA'], _ = spearmanr(
        base_geodesics, embedding_geodesics)
    gc.collect()
    ProjectionGlobalResults['PCA'] = global_score(X, pca_emb)
    gc.collect()
    if additional_eigenbases is not None:
        for key in additional_eigenbases.keys():
            if TopOGraph.verbosity > 0:
                print('Computing geodesics for additional eigenbasis \'{}...\''.format(key))
            emb_graph = kNN(additional_eigenbases[key],
                            n_neighbors=n_neighbors,
                            metric=TopOGraph.base_metric,
                            n_jobs=n_jobs,
                            backend=TopOGraph.backend,
                            return_instance=False,
                            verbose=False, **kwargs)
            if landmarks is not None:
                emb_graph = emb_graph[landmark_indices, :][:, landmark_indices]
                gc.collect()
            embedding_geodesics = squareform(geodesic_distance(
                emb_graph, directed=False, n_jobs=n_jobs))
            gc.collect()
            if cor_method == 'spearman':
                if TopOGraph.verbosity > 0:
                    print('Computing Spearman R for additional eigenbasis \'{}...\''.format(key))
                EigenbasisLocalResults[key], _ = spearmanr(
                    base_geodesics, embedding_geodesics)
            else:
                if TopOGraph.verbosity > 0:
                    print('Computing Kendall Tau for additional eigenbasis \'{}...\''.format(key))
                EigenbasisLocalResults[key], _ = kendalltau(
                    base_geodesics, embedding_geodesics)
            gc.collect()
            EigenbasisGlobalResults[key] = global_score(
                X, additional_eigenbases[key])
            if TopOGraph.verbosity > 0:
                print('Finished for eigenbasis {}'.format(key))
            gc.collect()
    if additional_projections is not None:
        for key in additional_projections.keys():
            if TopOGraph.verbosity > 0:
                print('Computing geodesics for additional projection \' {}...\''.format(key))
            emb_graph = kNN(additional_projections[key],
                            n_neighbors=n_neighbors,
                            metric=TopOGraph.graph_metric,
                            n_jobs=n_jobs,
                            backend=TopOGraph.backend,
                            return_instance=False,
                            verbose=False, **kwargs)
            if landmarks is not None:
                emb_graph = emb_graph[landmark_indices, :][:, landmark_indices]
            embedding_geodesics = squareform(geodesic_distance(
                emb_graph, directed=False, n_jobs=n_jobs))
            gc.collect()
            if cor_method == 'spearman':
                if TopOGraph.verbosity > 0:
                    print('Computing Spearman R for additional projection \'{}...\''.format(key))
                ProjectionLocalResults[key], _ = spearmanr(
                    base_geodesics, embedding_geodesics)
            else:
                if TopOGraph.verbosity > 0:
                    print('Computing Kendall Tau for additional projection \'{}...\''.format(key))
                ProjectionLocalResults[key], _ = kendalltau(
                    base_geodesics, embedding_geodesics)
            gc.collect()
            ProjectionGlobalResults[key] = global_score(
                X, additional_projections[key])
            gc.collect()
    res_dict = {'EigenbasisLocal': EigenbasisLocalResults,
                'EigenbasisGlobal': EigenbasisGlobalResults,
                'ProjectionLocal': ProjectionLocalResults,
                'ProjectionGlobal': ProjectionGlobalResults}

    return res_dict
