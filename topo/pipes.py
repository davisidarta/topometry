import matplotlib
from scipy.sparse import issparse, csr_matrix
from sklearn.utils import resample
from topo.tpgraph.fuzzy import fuzzy_simplicial_set
from topo.layouts.map import fuzzy_embedding
from topo.base.ann import kNN

matplotlib.use('Agg')  # plotting backend compatible with screen
from topo.topograph import TopOGraph
from topo.eval.global_scores import global_score_pca
from topo.eval.local_scores import knn_spearman_r


def global_score(data, emb):
    global_scores_pca = global_score_pca(data, emb)
    return global_scores_pca


def local_score(data, emb, k=10, metric='cosine', n_jobs=12, data_is_graph=False, emb_is_graph=False):
    try:
        import hnswlib
        _have_hnswlib = True
    except ImportError:
        _have_hnswlib = False
    try:
        import nmslib
        _have_nmslib = True
    except ImportError:
        _have_nmslib = False

    if issparse(data):
        if _have_nmslib:
            from topo.base.ann import NMSlibTransformer
            if not data_is_graph:
                data_graph = NMSlibTransformer(n_neighbors=k, n_jobs=n_jobs, metric=metric).fit_transform(data)
            if not emb_is_graph:
                emb = csr_matrix(emb)
                emb_graph = NMSlibTransformer(n_neighbors=k, n_jobs=n_jobs, metric=metric).fit_transform(emb)
        elif _have_hnswlib:
            from topo.base.ann import HNSWlibTransformer
            if not data_is_graph:
                data = data.toarray()
                data_graph = HNSWlibTransformer(n_neighbors=k, n_jobs=n_jobs, metric=metric).fit_transform(data)
            if not emb_is_graph:
                emb_graph = HNSWlibTransformer(n_neighbors=k, n_jobs=n_jobs, metric=metric).fit_transform(emb)
        else:
            from sklearn.neighbors import NearestNeighbors
            if not data_is_graph:
                data_nbrs = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs,
                                             metric=metric).fit(data)
                data_graph = data_nbrs.kneighbors(data)
            if not emb_is_graph:
                emb_nbrs = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs,
                                            metric=metric).fit(emb)
                emb_graph = emb_nbrs.kneighbors(emb)
    else:
        if _have_hnswlib:
            from topo.base.ann import HNSWlibTransformer
            if not data_is_graph:
                if issparse(data):
                    data = data.toarray()
                data_graph = HNSWlibTransformer(n_neighbors=k, n_jobs=n_jobs, metric=metric).fit_transform(data)
            if not emb_is_graph:
                emb_graph = HNSWlibTransformer(n_neighbors=k, n_jobs=n_jobs, metric=metric).fit_transform(emb)
        elif _have_nmslib:
            from topo.base.ann import NMSlibTransformer
            if not data_is_graph:
                data = csr_matrix(data)
                data_graph = NMSlibTransformer(n_neighbors=k, n_jobs=n_jobs, metric=metric).fit_transform(data)
            if not emb_is_graph:
                emb = csr_matrix(emb)
                emb_graph = NMSlibTransformer(n_neighbors=k, n_jobs=n_jobs, metric=metric).fit_transform(emb)
        else:
            from sklearn.neighbors import NearestNeighbors
            if not data_is_graph:
                data_nbrs = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs,
                                             metric=metric).fit(data)
                data_graph = data_nbrs.kneighbors(data)
            if not emb_is_graph:
                emb_nbrs = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs,
                                            metric=metric).fit(emb)
                emb_graph = emb_nbrs.kneighbors(emb)

    if emb_is_graph:
        emb_graph = emb
    if data_is_graph:
        data_graph = data

    local_scores_r = knn_spearman_r(data_graph, emb_graph)
    return local_scores_r


def eval_models_layouts(TopOGraph, X, subsample=None, k=None, n_jobs=None, metric=None,
                        bases=['diffusion', 'fuzzy', 'continuous'],
                        graphs=['diff', 'cknn', 'fuzzy'],
                        layouts=['tSNE', 'MAP', 'MDE', 'PaCMAP', 'TriMAP', 'NCVis']):
    """
    Evaluates all orthogonal bases, topological graphs and layouts in the TopOGraph object.
    Compares results with PCA and PCA-derived layouts (i.e. t-SNE, UMAP etc).

    Parameters
    --------------

    TopOGraph : target TopOGraph object (can be empty).

    X : data matrix. Expects either numpy.ndarray or scipy.sparse.csr_matrix.

    subsample : optional (int, default None).
        If specified, subsamples the TopOGraph object and/or data matrix X to a number of samples
        before computing results and scores. Useful if dealing with large datasets (>50,000 samples).

    k : optional (int, default None).
        Number of k-neighbors to use for evaluating results. Defaults to TopOGraph.base_knn.

    n_jobs : optional (int, default None).
        Number of threads to use in computations. Defaults to TopOGraph.n_jobs (default 1).

    metric : optional (str, default None).
        Distance metric to use. Defaults to TopOGraph.base_metric (default 'cosine').


    bases : str (optional, default ['diffusion', 'continuous','fuzzy']).
        Which bases to compute. Defaults to all. To run only one or two bases, set it to
        ['fuzzy', 'diffusion'] or ['continuous'], for exemple.

    graphs : str (optional, default ['diff', 'cknn','fuzzy']).
        Which graphs to compute. Defaults to all. To run only one or two graphs, set it to
        ['fuzzy', 'diff'] or ['cknn'], for exemple.

    layouts : str (optional, default all ['tSNE', 'MAP', 'MDE', 'PaCMAP', 'TriMAP', 'NCVis']).
        Which layouts to compute. Defaults to all 6 options within TopOMetry: tSNE, MAP, MDE, PaCMAP,
        TriMAP and NCVis. To run only one or two layouts, set it to
        ['tSNE', 'MAP'] or ['PaCMAP'], for example.

    Returns
    -------

    Populates the TopOGraph object and returns a list of lists (results):

    - results[0] contains orthogonal bases scores.
    - results[1] contains topological graph scores.
    - results[2] contains layout scores.

    """
    if k is None:
        k = TopOGraph.base_knn
    if n_jobs is None:
        n_jobs = TopOGraph.n_jobs
    if metric is None:
        metric = TopOGraph.base_metric
    if str('diffusion') in bases:
        eval_db = True
    else:
        eval_db = False
    if str('continuous') in bases:
        eval_cb = True
    else:
        eval_cb = False
    if str('fuzzy') in bases:
        eval_fb = True
    else:
        eval_fb = False
    if str('diff') in graphs:
        eval_diff = True
    else:
        eval_diff = False
    if str('cknn') in graphs:
        eval_cknn = True
    else:
        eval_cknn = False
    if str('fuzzy') in graphs:
        eval_fuzzy = True
    else:
        eval_fuzzy = False
    if str('tSNE') in layouts:
        eval_tSNE = True
    else:
        eval_tSNE = False
    if str('MAP') in layouts:
        eval_MAP = True
    else:
        eval_MAP = False
    if str('MDE') in layouts:
        eval_MDE = True
    else:
        eval_MDE = False
    if str('PaCMAP') in layouts:
        eval_PaCMAP = True
    else:
        eval_PaCMAP = False
    if str('TriMAP') in layouts:
        eval_TriMAP = True
    else:
        eval_TriMAP = False
    if str('NCVis') in layouts:
        eval_NCVis = True
    else:
        eval_NCVis = False
    db_pca = None
    db_r = None
    db_diff_r = None
    db_cknn_r = None
    db_fuzzy_r = None
    cb_pca = None
    cb_r = None
    cb_diff_r = None
    cb_cknn_r = None
    cb_fuzzy_r = None
    fb_pca = None
    fb_r = None
    fb_diff_r = None
    fb_cknn_r = None
    fb_fuzzy_r = None
    db_diff_scores = None
    db_cknn_scores = None
    db_fuzzy_scores = None
    cb_diff_scores = None
    cb_cknn_scores = None
    cb_fuzzy_scores = None
    fb_diff_scores = None
    fb_cknn_scores = None
    fb_fuzzy_scores = None



    if subsample is not None:
        if TopOGraph.verbosity >= 1:
            print('Subsampling...will recompute topological graphs...')

        X = resample(X, n_samples=subsample)
        TopOGraph.base_knn_graph = kNN(X, n_neighbors=k,
                           metric=metric,
                           n_jobs=n_jobs,
                           backend=TopOGraph.backend,
                           M=TopOGraph.M,
                           p=TopOGraph.p,
                           efC=TopOGraph.efC,
                           efS=TopOGraph.efS,
                           return_instance=False,
                           verbose=False)

        if TopOGraph.MSDiffMap is not None:
            TopOGraph.MSDiffMap = resample(TopOGraph.MSDiffMap, n_samples=subsample)
            if TopOGraph.db_tSNE is not None:
                TopOGraph.db_tSNE = resample(TopOGraph.db_tSNE, n_samples=subsample)
            if TopOGraph.db_NCVis is not None:
                TopOGraph.db_NCVis = resample(TopOGraph.db_NCVis, n_samples=subsample)
            if TopOGraph.db_PaCMAP is not None:
                TopOGraph.db_PaCMAP = resample(TopOGraph.db_PaCMAP, n_samples=subsample)
            if TopOGraph.db_TriMAP is not None:
                TopOGraph.db_TriMAP = resample(TopOGraph.db_TriMAP, n_samples=subsample)
            if TopOGraph.db_diff_graph is not None:
                TopOGraph.db_diff_graph = None      # There's no way to resample a square distance matrix
                if TopOGraph.db_diff_MAP is not None:
                    TopOGraph.db_diff_MAP = resample(TopOGraph.db_diff_MAP, n_samples=subsample)
                if TopOGraph.db_diff_MDE is not None:
                    TopOGraph.db_diff_MDE = resample(TopOGraph.db_diff_MDE, n_samples=subsample)
            if TopOGraph.db_fuzzy_graph is not None:
                TopOGraph.db_fuzzy_graph = None     # There's no way to resample a square distance matrix
                if TopOGraph.db_fuzzy_MAP is not None:
                    TopOGraph.db_fuzzy_MAP = resample(TopOGraph.db_fuzzy_MAP, n_samples=subsample)
                if TopOGraph.db_fuzzy_MDE is not None:
                    TopOGraph.db_fuzzy_MDE = resample(TopOGraph.db_fuzzy_MDE, n_samples=subsample)
            if TopOGraph.db_cknn_graph is not None:
                TopOGraph.db_cknn_graph = None      # There's no way to resample a square distance matrix
                if TopOGraph.db_cknn_MAP is not None:
                    TopOGraph.db_cknn_MAP = resample(TopOGraph.db_cknn_MAP, n_samples=subsample)
                if TopOGraph.db_cknn_MDE is not None:
                    TopOGraph.db_cknn_MDE = resample(TopOGraph.db_cknn_MDE, n_samples=subsample)

        if TopOGraph.CLapMap is not None:
            TopOGraph.CLapMap = resample(TopOGraph.CLapMap, n_samples=subsample)
            if TopOGraph.cb_tSNE is not None:
                TopOGraph.cb_tSNE = resample(TopOGraph.cb_tSNE, n_samples=subsample)
            if TopOGraph.cb_NCVis is not None:
                TopOGraph.cb_NCVis = resample(TopOGraph.cb_NCVis, n_samples=subsample)
            if TopOGraph.cb_PaCMAP is not None:
                TopOGraph.cb_PaCMAP = resample(TopOGraph.cb_PaCMAP, n_samples=subsample)
            if TopOGraph.cb_TriMAP is not None:
                TopOGraph.cb_TriMAP = resample(TopOGraph.cb_TriMAP, n_samples=subsample)
            if TopOGraph.cb_diff_graph is not None:
                TopOGraph.cb_diff_graph = None      # There's no way to resample a square distance matrix
                if TopOGraph.cb_diff_MAP is not None:
                    TopOGraph.cb_diff_MAP = resample(TopOGraph.cb_diff_MAP, n_samples=subsample)
                if TopOGraph.cb_diff_MDE is not None:
                    TopOGraph.cb_diff_MDE = resample(TopOGraph.cb_diff_MDE, n_samples=subsample)
            if TopOGraph.cb_fuzzy_graph is not None:
                TopOGraph.cb_fuzzy_graph = None     # There's no way to resample a square distance matrix
                if TopOGraph.cb_fuzzy_MAP is not None:
                    TopOGraph.cb_fuzzy_MAP = resample(TopOGraph.cb_fuzzy_MAP, n_samples=subsample)
                if TopOGraph.cb_fuzzy_MDE is not None:
                    TopOGraph.cb_fuzzy_MDE = resample(TopOGraph.cb_fuzzy_MDE, n_samples=subsample)
            if TopOGraph.cb_cknn_graph is not None:
                TopOGraph.cb_cknn_graph = None      # There's no way to resample a square distance matrix
                if TopOGraph.cb_cknn_MAP is not None:
                    TopOGraph.cb_cknn_MAP = resample(TopOGraph.cb_cknn_MAP, n_samples=subsample)
                if TopOGraph.cb_cknn_MDE is not None:
                    TopOGraph.cb_cknn_MDE = resample(TopOGraph.cb_cknn_MDE, n_samples=subsample)

        if TopOGraph.FuzzyLapMap is not None:
            TopOGraph.FuzzyLapMap = resample(TopOGraph.FuzzyLapMap, n_samples=subsample)
            if TopOGraph.fb_tSNE is not None:
                TopOGraph.fb_tSNE = resample(TopOGraph.fb_tSNE, n_samples=subsample)
            if TopOGraph.fb_NCVis is not None:
                TopOGraph.fb_NCVis = resample(TopOGraph.fb_NCVis, n_samples=subsample)
            if TopOGraph.fb_PaCMAP is not None:
                TopOGraph.fb_PaCMAP = resample(TopOGraph.fb_PaCMAP, n_samples=subsample)
            if TopOGraph.fb_TriMAP is not None:
                TopOGraph.fb_TriMAP = resample(TopOGraph.fb_TriMAP, n_samples=subsample)
            if TopOGraph.fb_diff_graph is not None:
                TopOGraph.fb_diff_graph = None      # There's no way to resample a square distance matrix
                if TopOGraph.fb_diff_MAP is not None:
                    TopOGraph.fb_diff_MAP = resample(TopOGraph.fb_diff_MAP, n_samples=subsample)
                if TopOGraph.fb_diff_MDE is not None:
                    TopOGraph.fb_diff_MDE = resample(TopOGraph.fb_diff_MDE, n_samples=subsample)
            if TopOGraph.fb_fuzzy_graph is not None:
                TopOGraph.fb_fuzzy_graph = None     # There's no way to resample a square distance matrix
                if TopOGraph.fb_fuzzy_MAP is not None:
                    TopOGraph.fb_fuzzy_MAP = resample(TopOGraph.fb_fuzzy_MAP, n_samples=subsample)
                if TopOGraph.fb_fuzzy_MDE is not None:
                    TopOGraph.fb_fuzzy_MDE = resample(TopOGraph.fb_fuzzy_MDE, n_samples=subsample)
            if TopOGraph.fb_cknn_graph is not None:
                TopOGraph.fb_cknn_graph = None # There's no way to resample a square distance matrix
                if TopOGraph.fb_cknn_MAP is not None:
                    TopOGraph.fb_cknn_MAP = resample(TopOGraph.fb_cknn_MAP, n_samples=subsample)
                if TopOGraph.fb_cknn_MDE is not None:
                    TopOGraph.fb_cknn_MDE = resample(TopOGraph.fb_cknn_MDE, n_samples=subsample)

    else:
        if TopOGraph.base_knn_graph is None:
            TopOGraph.base_knn_graph = kNN(X, n_neighbors=k,
                                           metric=metric,
                                           n_jobs=n_jobs,
                                           backend=TopOGraph.backend,
                                           M=TopOGraph.M,
                                           p=TopOGraph.p,
                                           efC=TopOGraph.efC,
                                           efS=TopOGraph.efS,
                                           return_instance=False,
                                           verbose=False)

    TopOGraph.run_layouts(X=X,
                          bases=bases,
                          graphs=graphs,
                          layouts=layouts)



    if TopOGraph.verbosity > 0:
        print('Computing scores...')
    if eval_db:
        if TopOGraph.verbosity > 0:
            print('Computing diffusion-related scores...')
        DB_knn_graph = kNN(TopOGraph.MSDiffMap, n_neighbors=k,
                           metric=metric,
                           n_jobs=n_jobs,
                           backend=TopOGraph.backend,
                           M=TopOGraph.M,
                           p=TopOGraph.p,
                           efC=TopOGraph.efC,
                           efS=TopOGraph.efS,
                           return_instance=False,
                           verbose=False)

        db_pca = global_score(X, TopOGraph.MSDiffMap)
        db_r = local_score(TopOGraph.base_knn_graph, DB_knn_graph,
                           k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True, emb_is_graph=True)
        db_scores = db_pca, db_r



        if eval_TriMAP:
            data_db_TriMAP_pca = global_score(X, TopOGraph.db_TriMAP)
            db_TriMAP_r = local_score(DB_knn_graph, TopOGraph.db_TriMAP,
                                      k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            data_db_TriMAP_scores = data_db_TriMAP_pca, db_TriMAP_r
        if eval_PaCMAP:
            data_db_PaCMAP_pca = global_score(X, TopOGraph.db_PaCMAP)
            db_PaCMAP_r = local_score(DB_knn_graph, TopOGraph.db_PaCMAP,
                                      k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            data_db_PaCMAP_scores = data_db_PaCMAP_pca, db_PaCMAP_r
        if eval_NCVis:
            data_db_NCVis_pca = global_score(X, TopOGraph.db_NCVis)
            db_NCVis_r = local_score(DB_knn_graph, TopOGraph.db_NCVis,
                                     k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            data_db_NCVis_scores = data_db_NCVis_pca, db_NCVis_r
        if eval_tSNE:
            data_db_tSNE_pca = global_score(X, TopOGraph.db_tSNE)
            db_tSNE_r = local_score(DB_knn_graph, TopOGraph.db_tSNE,
                                    k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            data_db_tSNE_scores = data_db_tSNE_pca, db_tSNE_r

        if eval_diff:
            db_diff_r = local_score(DB_knn_graph, TopOGraph.db_diff_graph, emb_is_graph=True,
                                    k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            db_diff_scores = db_diff_r
            if eval_MAP:
                data_db_diff_MAP_pca = global_score(X, TopOGraph.db_diff_MAP)
                db_diff_MAP_r = local_score(TopOGraph.db_diff_graph, TopOGraph.db_diff_MAP,
                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_db_diff_MAP_scores = data_db_diff_MAP_pca, db_diff_MAP_r
            if eval_MDE:
                data_db_diff_MDE_pca = global_score(X, TopOGraph.db_diff_MDE)
                db_diff_MDE_r = local_score(TopOGraph.db_diff_graph, TopOGraph.db_diff_MDE,
                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_db_diff_MDE_scores = data_db_diff_MDE_pca, db_diff_MDE_r

        if eval_cknn:
            db_cknn_r = local_score(DB_knn_graph, TopOGraph.db_cknn_graph, emb_is_graph=True,
                                    k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            db_cknn_scores = db_cknn_r

            if eval_MAP:
                data_db_cknn_MAP_pca = global_score(X, TopOGraph.db_cknn_MAP)
                db_cknn_MAP_r = local_score(TopOGraph.db_cknn_graph, TopOGraph.db_cknn_MAP,
                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_db_cknn_MAP_scores = data_db_cknn_MAP_pca, db_cknn_MAP_r
            if eval_MDE:
                data_db_cknn_MDE_pca = global_score(X, TopOGraph.db_cknn_MDE)
                db_cknn_MDE_r = local_score(TopOGraph.db_cknn_graph, TopOGraph.db_cknn_MDE,
                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_db_cknn_MDE_scores = data_db_cknn_MDE_pca, db_cknn_MDE_r

        if eval_fuzzy:
            db_fuzzy_r = local_score(DB_knn_graph, TopOGraph.db_fuzzy_graph, emb_is_graph=True,
                                     k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            db_fuzzy_scores = db_fuzzy_r
            if eval_MAP:
                data_db_fuzzy_MAP_pca = global_score(X, TopOGraph.db_fuzzy_MAP)
                db_fuzzy_MAP_r = local_score(TopOGraph.db_fuzzy_graph, TopOGraph.db_fuzzy_MAP,
                                             k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_db_fuzzy_MAP_scores = data_db_fuzzy_MAP_pca, db_fuzzy_MAP_r
            if eval_MDE:
                data_db_fuzzy_MDE_pca = global_score(X, TopOGraph.db_fuzzy_MDE)
                db_fuzzy_MDE_r = local_score(TopOGraph.db_fuzzy_graph, TopOGraph.db_fuzzy_MDE,
                                             k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_db_fuzzy_MDE_scores = data_db_fuzzy_MDE_pca, db_fuzzy_MDE_r

    if eval_cb:
        if TopOGraph.verbosity > 0:
            print('Computing continuous-related scores...')

        CB_knn_graph = kNN(TopOGraph.CLapMap, n_neighbors=k,
                           metric=metric,
                           n_jobs=n_jobs,
                           backend=TopOGraph.backend,
                           M=TopOGraph.M,
                           p=TopOGraph.p,
                           efC=TopOGraph.efC,
                           efS=TopOGraph.efS,
                           return_instance=False,
                           verbose=False)
        cb_pca = global_score(X, TopOGraph.CLapMap)
        cb_r = local_score(TopOGraph.base_knn_graph, CB_knn_graph,
                           k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True, emb_is_graph=True)

        cb_scores = cb_pca, cb_r



        if eval_TriMAP:
            data_cb_TriMAP_pca = global_score(X, TopOGraph.cb_TriMAP)
            cb_TriMAP_r = local_score(CB_knn_graph, TopOGraph.cb_TriMAP,
                                      k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            data_cb_TriMAP_scores = data_cb_TriMAP_pca, cb_TriMAP_r

        if eval_PaCMAP:
            data_cb_PaCMAP_pca = global_score(X, TopOGraph.cb_PaCMAP)
            cb_PaCMAP_r = local_score(CB_knn_graph, TopOGraph.cb_PaCMAP,
                                      k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            data_cb_PaCMAP_scores = data_cb_PaCMAP_pca, cb_PaCMAP_r

        if eval_NCVis:
            data_cb_NCVis_pca = global_score(X, TopOGraph.cb_NCVis)
            cb_NCVis_r = local_score(CB_knn_graph, TopOGraph.cb_NCVis,
                                     k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            data_cb_NCVis_scores = data_cb_NCVis_pca, cb_NCVis_r

        if eval_tSNE:
            data_cb_tSNE_pca = global_score(X, TopOGraph.cb_tSNE)
            cb_tSNE_r = local_score(CB_knn_graph, TopOGraph.cb_tSNE,
                                    k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            data_cb_tSNE_scores = data_cb_tSNE_pca, cb_tSNE_r

        if eval_diff:
            cb_diff_r = local_score(CB_knn_graph, TopOGraph.cb_diff_graph,
                                    emb_is_graph=True, k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)

            cb_diff_scores = cb_diff_r

            if eval_MAP:
                data_cb_diff_MAP_pca = global_score(X, TopOGraph.cb_diff_MAP)
                cb_diff_MAP_r = local_score(TopOGraph.cb_diff_graph, TopOGraph.cb_diff_MAP,
                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_cb_diff_MAP_scores = data_cb_diff_MAP_pca, cb_diff_MAP_r
            if eval_MDE:
                data_cb_diff_MDE_pca = global_score(X, TopOGraph.cb_diff_MDE)
                cb_diff_MDE_r = local_score(TopOGraph.cb_diff_graph, TopOGraph.cb_diff_MDE,
                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_cb_diff_MDE_scores = data_cb_diff_MDE_pca, cb_diff_MDE_r

        if eval_cknn:
            cb_cknn_r = local_score(CB_knn_graph, TopOGraph.cb_cknn_graph,
                                    emb_is_graph=True, k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            cb_cknn_scores = cb_cknn_r

            if eval_MAP:
                data_cb_cknn_MAP_pca = global_score(X, TopOGraph.cb_cknn_MAP)
                cb_cknn_MAP_r = local_score(TopOGraph.cb_cknn_graph, TopOGraph.cb_cknn_MAP,
                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_cb_cknn_MAP_scores = data_cb_cknn_MAP_pca, cb_cknn_MAP_r
            if eval_MDE:
                data_cb_cknn_MDE_pca = global_score(X, TopOGraph.cb_cknn_MDE)
                cb_cknn_MDE_r = local_score(TopOGraph.cb_cknn_graph, TopOGraph.cb_cknn_MDE,
                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_cb_cknn_MDE_scores = data_cb_cknn_MDE_pca, cb_cknn_MDE_r

        if eval_fuzzy:
            cb_fuzzy_r = local_score(CB_knn_graph, TopOGraph.cb_fuzzy_graph,
                                     emb_is_graph=True, k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            cb_fuzzy_scores = cb_fuzzy_r

            if eval_MAP:
                data_cb_fuzzy_MAP_pca = global_score(X, TopOGraph.cb_fuzzy_MAP)
                cb_fuzzy_MAP_r = local_score(TopOGraph.cb_fuzzy_graph, TopOGraph.cb_fuzzy_MAP,
                                             k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_cb_fuzzy_MAP_scores = data_cb_fuzzy_MAP_pca, cb_fuzzy_MAP_r
            if eval_MDE:
                data_cb_fuzzy_MDE_pca = global_score(X, TopOGraph.cb_fuzzy_MDE)
                cb_fuzzy_MDE_r = local_score(TopOGraph.cb_fuzzy_graph, TopOGraph.cb_fuzzy_MDE,
                                             k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_cb_fuzzy_MDE_scores = data_cb_fuzzy_MDE_pca, cb_fuzzy_MDE_r

    if eval_fb:
        if TopOGraph.verbosity > 0:
            print('Computing fuzzy-related scores...')
        FB_knn_graph = kNN(TopOGraph.FuzzyLapMap, n_neighbors=k,
                           metric=metric,
                           n_jobs=n_jobs,
                           backend=TopOGraph.backend,
                           M=TopOGraph.M,
                           p=TopOGraph.p,
                           efC=TopOGraph.efC,
                           efS=TopOGraph.efS,
                           return_instance=False,
                           verbose=False)
        fb_pca = global_score(X, TopOGraph.FuzzyLapMap)
        fb_r = local_score(TopOGraph.base_knn_graph, FB_knn_graph,
                           k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True, emb_is_graph=True)

        fb_scores = fb_pca, fb_r



        if eval_TriMAP:
            data_fb_TriMAP_pca = global_score(X, TopOGraph.fb_TriMAP)
            fb_TriMAP_r = local_score(FB_knn_graph, TopOGraph.fb_TriMAP,
                                      k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            data_fb_TriMAP_scores = data_fb_TriMAP_pca, fb_TriMAP_r

        if eval_PaCMAP:
            data_fb_PaCMAP_pca = global_score(X, TopOGraph.fb_PaCMAP)
            fb_PaCMAP_r = local_score(FB_knn_graph, TopOGraph.fb_PaCMAP,
                                      k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            data_fb_PaCMAP_scores = data_fb_PaCMAP_pca, fb_PaCMAP_r

        if eval_NCVis:
            data_fb_NCVis_pca = global_score(X, TopOGraph.fb_NCVis)
            fb_NCVis_r = local_score(FB_knn_graph, TopOGraph.fb_NCVis,
                                     k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            data_fb_NCVis_scores = data_fb_NCVis_pca, fb_NCVis_r

        if eval_tSNE:
            data_fb_tSNE_pca = global_score(X, TopOGraph.fb_tSNE)
            fb_tSNE_r = local_score(FB_knn_graph, TopOGraph.fb_tSNE,
                                    k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            data_fb_tSNE_scores = data_fb_tSNE_pca, fb_tSNE_r

        if eval_diff:
            fb_diff_r = local_score(FB_knn_graph, TopOGraph.fb_diff_graph,
                                    emb_is_graph=True, k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            fb_diff_scores = fb_diff_r

            if eval_MAP:
                data_fb_diff_MAP_pca = global_score(X, TopOGraph.fb_diff_MAP)
                fb_diff_MAP_r = local_score(TopOGraph.fb_diff_graph, TopOGraph.fb_diff_MAP,
                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_fb_diff_MAP_scores = data_fb_diff_MAP_pca, fb_diff_MAP_r
            if eval_MDE:
                data_fb_diff_MDE_pca = global_score(X, TopOGraph.fb_diff_MDE)
                fb_diff_MDE_r = local_score(TopOGraph.fb_diff_graph, TopOGraph.fb_diff_MDE,
                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_fb_diff_MDE_scores = data_fb_diff_MDE_pca, fb_diff_MDE_r

        if eval_cknn:
            fb_cknn_r = local_score(FB_knn_graph, TopOGraph.fb_cknn_graph,
                                    emb_is_graph=True, k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            fb_cknn_scores = fb_cknn_r

            if eval_MAP:
                data_fb_cknn_MAP_pca = global_score(X, TopOGraph.fb_cknn_MAP)
                fb_cknn_MAP_r = local_score(TopOGraph.fb_cknn_graph, TopOGraph.fb_cknn_MAP,
                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_fb_cknn_MAP_scores = data_fb_cknn_MAP_pca, fb_cknn_MAP_r
            if eval_MDE:
                data_fb_cknn_MDE_pca = global_score(X, TopOGraph.fb_cknn_MDE)
                fb_cknn_MDE_r = local_score(TopOGraph.fb_cknn_graph, TopOGraph.fb_cknn_MDE,
                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_fb_cknn_MDE_scores = data_fb_cknn_MDE_pca, fb_cknn_MDE_r

        if eval_fuzzy:
            fb_fuzzy_r = local_score(FB_knn_graph, TopOGraph.fb_fuzzy_graph,
                                     emb_is_graph=True, k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            fb_fuzzy_scores = fb_fuzzy_r

            if eval_MAP:
                data_fb_fuzzy_MAP_pca = global_score(X, TopOGraph.fb_fuzzy_MAP)
                fb_fuzzy_MAP_r = local_score(TopOGraph.fb_fuzzy_graph, TopOGraph.fb_fuzzy_MAP,
                                             k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_fb_fuzzy_MAP_scores = data_fb_fuzzy_MAP_pca, fb_fuzzy_MAP_r
            if eval_MDE:
                data_fb_fuzzy_MDE_pca = global_score(X, TopOGraph.fb_fuzzy_MDE)
                fb_fuzzy_MDE_r = local_score(TopOGraph.fb_fuzzy_graph, TopOGraph.fb_fuzzy_MDE,
                                             k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_fb_fuzzy_MDE_scores = data_fb_fuzzy_MDE_pca, fb_fuzzy_MDE_r

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
    pca_pca = global_score(X, pca_emb)
    pca_r = local_score(X, pca_emb, k=k, metric=metric, n_jobs=n_jobs)

    embedding_scores = {'PCA': (pca_pca, pca_r)}
    graph_scores = {}
    layout_scores = {}

    if eval_MAP:
        if TopOGraph.verbosity >= 1:
            print('Computing default UMAP...')
        # Compare to PCA + (U)MAP and just (U)MAP
        try:
            import umap
            umap_operator = umap.UMAP(metric=metric, n_neighbors=k,
                                      n_jobs=n_jobs, min_dist=0.4)
            umap_emb = umap_operator.fit_transform(data)
        except:
            print("UMAP is not installed. Will use MAP as comparison.")
            fuzzy_results = fuzzy_simplicial_set(X, n_neighbors=k,
                                                         backend=TopOGraph.backend, metric=metric,
                                                         n_jobs=n_jobs, efC=TopOGraph.efC, M=TopOGraph.M,
                                                         verbose=TopOGraph.bases_graph_verbose)

            umap_emb, aux = fuzzy_embedding(graph=fuzzy_results[0],
                                                   verbose=TopOGraph.layout_verbose, min_dist=0.4)

        umap_on_pca_pca = global_score(X, umap_emb)
        umap_on_pca_r = local_score(X, umap_emb, k=k, metric=metric, n_jobs=n_jobs)
        umap_on_pca_scores = umap_on_pca_pca, umap_on_pca_r

        layout_scores['UMAP'] = np.absolute(umap_on_pca_scores)

    if eval_tSNE:
        if TopOGraph.verbosity >= 1:
            print('Computing default tSNE...')
        try:
            from MulticoreTSNE import MulticoreTSNE as TSNE
            tsne_emb = TSNE(verbose=TopOGraph.layout_verbose, n_jobs=n_jobs,
                            init=pca_emb[:, 0:2], metric=metric).fit_transform(pca_emb)

        except:
            print("MulticoreTSNE is not installed. Will use scikit-learn TSNE implementation.")
            from sklearn.manifold import TSNE
            tsne_emb = TSNE(verbose=TopOGraph.verbosity, n_jobs=n_jobs,
                            init=pca_emb[:, 0:2], metric=metric).fit_transform(pca_emb)

        tsne_on_pca_pca = global_score(X, tsne_emb)
        tsne_on_pca_r = local_score(X, tsne_emb, k=k, metric=metric, n_jobs=n_jobs)
        tsne_on_pca_scores = tsne_on_pca_pca, tsne_on_pca_r
        layout_scores['tSNE'] = np.absolute(tsne_on_pca_scores)

    if eval_PaCMAP:
        if TopOGraph.verbosity >= 1:
            print('Computing default PaCMAP...')
        from topo.layouts.pairwise import PaCMAP
        pca_pacmap_emb = PaCMAP(data=pca_emb, init=pca_emb[:, 0:2], n_neighbors=k, verbose=TopOGraph.layout_verbose)
        pacmap_on_pca_pca = global_score(X, pca_pacmap_emb)
        pacmap_on_pca_r = local_score(X, pca_pacmap_emb,
                                      k=k, metric=metric, n_jobs=n_jobs)
        pacmap_on_pca_scores = pacmap_on_pca_pca, pacmap_on_pca_r
        layout_scores['PaCMAP'] = np.absolute(pacmap_on_pca_scores)

    if eval_TriMAP:
        if TopOGraph.verbosity >= 1:
            print('Computing default TriMAP...')
        from topo.layouts.trimap import TriMAP


        pca_trimap_emb = TriMAP(X=pca_emb, verbose=TopOGraph.layout_verbose, n_outliers=k)
        trimap_on_pca_pca = global_score(X, pca_trimap_emb)
        trimap_on_pca_r = local_score(X, pca_trimap_emb,
                                      k=k, metric=metric, n_jobs=n_jobs)
        trimap_on_pca_scores = trimap_on_pca_pca, trimap_on_pca_r
        layout_scores['TriMAP'] = np.absolute(trimap_on_pca_scores)

    if eval_MDE:
        if TopOGraph.verbosity >= 1:
            print('Computing default MDE...')
        import torch
        import pymde
        pca_mde_emb = pymde.preserve_neighbors(torch.tensor(pca_emb),
                                               n_neighbors=TopOGraph.base_knn, verbose=False).embed()
        pca_mde_emb = pca_mde_emb.numpy()
        mde_on_pca_pca = global_score(X, pca_mde_emb)
        mde_on_pca_r = local_score(X, pca_mde_emb, k=k, metric=metric, n_jobs=n_jobs)
        mde_on_pca_scores = mde_on_pca_pca, mde_on_pca_r
        layout_scores['MDE'] = np.absolute(mde_on_pca_scores)

    # if eval_NCVis:
    #     if TopOGraph.verbosity >= 1:
    #         print('Computing default NCVis...')
    #
    #     import ncvis
    #     ncvis_emb = ncvis.NCVis(n_neighbors=TopOGraph.graph_knn)
    #     ncvis_pca, ncvis_lap = global_scores(X, ncvis_emb, n_dim=TopOGraph.n_eigs)
    #     ncvis_r, ncvis_t = local_scores(TopOGraph.base_knn_graph, ncvis_emb, data_is_graph=True)
    #     ncvis_scores = ncvis_pca, ncvis_lap, ncvis_r, ncvis_t
    #     layout_scores['NCVis'] = np.absolute(ncvis_scores)

    if eval_db:
        embedding_scores['DB'] = np.absolute(db_scores)
        if eval_PaCMAP:
            layout_scores['db_PaCMAP'] = np.absolute(data_db_PaCMAP_scores)
        if eval_TriMAP:
            layout_scores['db_TriMAP'] = np.absolute(data_db_TriMAP_scores)
        if eval_tSNE:
            layout_scores['db_tSNE'] = np.absolute(data_db_tSNE_scores)
        if eval_NCVis:
            layout_scores['db_NCVis'] = np.absolute(data_db_NCVis_scores)
        if eval_diff:
            graph_scores['db_diff'] = np.absolute(db_diff_scores)
            if eval_MAP:
                layout_scores['db_diff_MAP'] = np.absolute(data_db_diff_MAP_scores)
            if eval_MDE:
                layout_scores['db_diff_MDE'] = np.absolute(data_db_diff_MDE_scores)
        if eval_cknn:
            graph_scores['db_cknn'] = np.absolute(db_cknn_scores)
            if eval_MAP:
                layout_scores['db_cknn_MAP'] = np.absolute(data_db_cknn_MAP_scores)
            if eval_MDE:
                layout_scores['db_cknn_MDE'] = np.absolute(data_db_cknn_MDE_scores)
        if eval_fuzzy:
            graph_scores['db_fuzzy'] = np.absolute(db_fuzzy_scores)
            if eval_MAP:
                layout_scores['db_fuzzy_MAP'] = np.absolute(data_db_fuzzy_MAP_scores)
            if eval_MDE:
                layout_scores['db_fuzzy_MDE'] = np.absolute(data_db_fuzzy_MDE_scores)

    if eval_cb:
        embedding_scores['CB'] = np.absolute(cb_scores)
        if eval_PaCMAP:
            layout_scores['cb_PaCMAP'] = np.absolute(data_cb_PaCMAP_scores)
        if eval_TriMAP:
            layout_scores['cb_TriMAP'] = np.absolute(data_cb_TriMAP_scores)
        if eval_tSNE:
            layout_scores['cb_tSNE'] = np.absolute(data_cb_tSNE_scores)
        if eval_NCVis:
            layout_scores['cb_NCVis'] = np.absolute(data_cb_NCVis_scores)
        if eval_diff:
            graph_scores['cb_diff'] = np.absolute(cb_diff_scores)
            if eval_MAP:
                layout_scores['cb_diff_MAP'] = np.absolute(data_cb_diff_MAP_scores)
            if eval_MDE:
                layout_scores['cb_diff_MDE'] = np.absolute(data_cb_diff_MDE_scores)
        if eval_cknn:
            graph_scores['cb_cknn'] = np.absolute(cb_cknn_scores)
            if eval_MAP:
                layout_scores['cb_cknn_MAP'] = np.absolute(data_cb_cknn_MAP_scores)
            if eval_MDE:
                layout_scores['cb_cknn_MDE'] = np.absolute(data_cb_cknn_MDE_scores)
        if eval_fuzzy:
            graph_scores['cb_fuzzy'] = np.absolute(cb_fuzzy_scores)
            if eval_MAP:
                layout_scores['cb_fuzzy_MAP'] = np.absolute(data_cb_fuzzy_MAP_scores)
            if eval_MDE:
                layout_scores['cb_fuzzy_MDE'] = np.absolute(data_cb_fuzzy_MDE_scores)

    if eval_fb:
        embedding_scores['FB'] = np.absolute(fb_scores)
        if eval_PaCMAP:
            layout_scores['fb_PaCMAP'] = np.absolute(data_fb_PaCMAP_scores)
        if eval_TriMAP:
            layout_scores['fb_TriMAP'] = np.absolute(data_fb_TriMAP_scores)
        if eval_tSNE:
            layout_scores['fb_tSNE'] = np.absolute(data_fb_tSNE_scores)
        if eval_NCVis:
            layout_scores['fb_NCVis'] = np.absolute(data_fb_NCVis_scores)
        if eval_diff:
            graph_scores['fb_diff'] = np.absolute(fb_diff_scores)
            if eval_MAP:
                layout_scores['fb_diff_MAP'] = np.absolute(data_fb_diff_MAP_scores)
            if eval_MDE:
                layout_scores['fb_diff_MDE'] = np.absolute(data_fb_diff_MDE_scores)
        if eval_cknn:
            graph_scores['fb_cknn'] = np.absolute(fb_cknn_scores)
            if eval_MAP:
                layout_scores['fb_cknn_MAP'] = np.absolute(data_fb_cknn_MAP_scores)
            if eval_MDE:
                layout_scores['fb_cknn_MDE'] = np.absolute(data_fb_cknn_MDE_scores)
        if eval_fuzzy:
            graph_scores['fb_fuzzy'] = np.absolute(fb_fuzzy_scores)
            if eval_MAP:
                layout_scores['fb_fuzzy_MAP'] = np.absolute(data_fb_fuzzy_MAP_scores)
            if eval_MDE:
                layout_scores['fb_fuzzy_MDE'] = np.absolute(data_fb_fuzzy_MDE_scores)

    return embedding_scores, graph_scores, layout_scores
