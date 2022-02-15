import matplotlib
from scipy.sparse import issparse, csr_matrix

from topo.layouts.graph_utils import fuzzy_simplicial_set_ann
from topo.layouts.map import fuzzy_embedding

matplotlib.use('Agg')  # plotting backend compatible with screen
from topo.topograph import TopOGraph
from topo.eval.global_scores import global_score_pca, global_score_laplacian
from topo.eval.local_scores import knn_spearman_r, knn_kendall_tau

def global_scores(data, emb, k=10, data_is_graph=False, metric='cosine', n_jobs=12):
    global_scores_pca = global_score_pca(data, emb)
    global_scores_lap = global_score_laplacian(data, emb, k=k, n_jobs=n_jobs, data_is_graph=data_is_graph)
    return global_scores_pca, global_scores_lap

def local_scores(data, emb, k=10, metric='cosine', n_jobs=12, data_is_graph=False, emb_is_graph=False):
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
    local_scores_tau = knn_kendall_tau(data_graph, emb_graph)
    return local_scores_r, local_scores_tau

def eval_models_layouts(TopOGraph, X, k=None, n_jobs=None, metric=None,
                bases=['diffusion', 'fuzzy', 'continuous'],
                graphs=['diff', 'cknn', 'fuzzy'],
                layouts=['tSNE', 'MAP','MDE','PaCMAP','TriMAP', 'NCVis']):
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
    db_lap = None
    db_r = None
    db_t = None
    db_diff_r = None
    db_diff_t = None
    db_cknn_r = None
    db_cknn_t = None
    db_fuzzy_r = None
    db_fuzzy_t = None
    cb_pca = None
    cb_lap = None
    cb_r = None
    cb_t = None
    cb_diff_r = None
    cb_diff_t = None
    cb_cknn_r = None
    cb_cknn_t = None
    cb_fuzzy_r = None
    cb_fuzzy_t = None
    fb_pca = None
    fb_lap = None
    fb_r = None
    fb_t = None
    fb_diff_r = None
    fb_diff_t = None
    fb_cknn_r = None
    fb_cknn_t = None
    fb_fuzzy_r = None
    fb_fuzzy_t = None
    db_diff_scores = None
    db_cknn_scores = None
    db_fuzzy_scores = None
    cb_diff_scores = None
    cb_cknn_scores = None
    cb_fuzzy_scores = None
    fb_diff_scores = None
    fb_cknn_scores = None
    fb_fuzzy_scores = None


    TopOGraph.run_layouts(X=X,
                bases=bases,
                graphs=graphs,
                layouts=layouts)

    if TopOGraph.verbosity > 0:
        print('Computing scores...')
    if eval_db:
        db_pca, db_lap = global_scores(X, TopOGraph.MSDiffMap, k=k, metric=metric, n_jobs=n_jobs)
        db_r, db_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.MSDiffMap,
                                  k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
        db_scores = db_pca, db_lap, db_r, db_t
        if eval_TriMAP:
            data_db_TriMAP_pca, data_db_TriMAP_lap = global_scores(X, TopOGraph.db_TriMAP, k=k, metric=metric, n_jobs=n_jobs)
            db_TriMAP_r, db_TriMAP_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.db_TriMAP,
                                  k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            data_db_TriMAP_scores = data_db_TriMAP_pca, data_db_TriMAP_lap, db_TriMAP_r, db_TriMAP_t
        if eval_PaCMAP:
            data_db_PaCMAP_pca, data_db_PaCMAP_lap = global_scores(X, TopOGraph.db_PaCMAP, k=k, metric=metric, n_jobs=n_jobs)
            db_PaCMAP_r, db_PaCMAP_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.db_PaCMAP,
                                  k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            data_db_PaCMAP_scores = data_db_PaCMAP_pca, data_db_PaCMAP_lap, db_PaCMAP_r, db_PaCMAP_t
        if eval_NCVis:
            data_db_NCVis_pca, data_db_NCVis_lap = global_scores(X, TopOGraph.db_NCVis, k=k, metric=metric, n_jobs=n_jobs)
            db_NCVis_r, db_NCVis_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.db_NCVis,
                                  k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            data_db_NCVis_scores = data_db_NCVis_pca, data_db_NCVis_lap, db_NCVis_r, db_NCVis_t
        if eval_tSNE:
            data_db_tSNE_pca, data_db_tSNE_pca = global_scores(X, TopOGraph.db_tSNE, k=k, metric=metric, n_jobs=n_jobs)
            db_tSNE_r, db_tSNE_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.db_tSNE,
                                  k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            data_db_tSNE_scores = data_db_tSNE_pca, data_db_tSNE_pca, db_tSNE_r, db_tSNE_t

        if eval_diff:
            db_diff_r, db_diff_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.db_diff_graph, emb_is_graph=True,
                                  k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            db_diff_scores = db_diff_r, db_diff_t
            if eval_MAP:
                data_db_diff_MAP_pca, data_db_diff_MAP_pca = global_scores(X, TopOGraph.db_diff_MAP,
                                                                           k=k, metric=metric, n_jobs=n_jobs)
                db_diff_MAP_r, db_diff_MAP_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.db_diff_MAP,
                                  k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_db_diff_MAP_scores = data_db_diff_MAP_pca, data_db_diff_MAP_pca, db_diff_MAP_r, db_diff_MAP_t
            if eval_MDE:
                data_db_diff_MDE_pca, data_db_diff_MDE_lap = global_scores(X, TopOGraph.db_diff_MDE,
                                                                           k=k, metric=metric, n_jobs=n_jobs)
                db_diff_MDE_r, db_diff_MDE_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.db_diff_MDE,
                                  k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_db_diff_MDE_scores = data_db_diff_MDE_pca, data_db_diff_MDE_lap, db_diff_MDE_r, db_diff_MDE_t

        if eval_cknn:
            db_cknn_r, db_cknn_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.db_cknn_graph, emb_is_graph=True,
                                  k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            db_cknn_scores = db_cknn_r, db_cknn_t

            if eval_MAP:
                data_db_cknn_MAP_pca, data_db_cknn_MAP_lap = global_scores(X, TopOGraph.db_cknn_MAP,
                                                                           k=k, metric=metric, n_jobs=n_jobs)
                db_cknn_MAP_r, db__MAP_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.db_cknn_MAP,
                                  k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_db_cknn_MAP_scores = data_db_cknn_MAP_pca, data_db_cknn_MAP_lap, db_cknn_MAP_r, db__MAP_t
            if eval_MDE:
                data_db_cknn_MDE_pca, data_db_cknn_MDE_lap = global_scores(X, TopOGraph.db_cknn_MDE,
                                                                           k=k, metric=metric, n_jobs=n_jobs)
                db_cknn_MDE_r, db_cknn_MDE_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.db_cknn_MDE,
                                  k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_db_cknn_MDE_scores = data_db_cknn_MDE_pca, data_db_cknn_MDE_lap, db_cknn_MDE_r, db_cknn_MDE_t

        if eval_fuzzy:
            db_fuzzy_r, db_fuzzy_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.db_fuzzy_graph, emb_is_graph=True,
                                  k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            db_fuzzy_scores = db_fuzzy_r, db_fuzzy_t
            if eval_MAP:
                data_db_fuzzy_MAP_pca, data_db_fuzzy_MAP_lap = global_scores(X, TopOGraph.db_fuzzy_MAP,
                                                                           k=k, metric=metric, n_jobs=n_jobs)
                db_fuzzy_MAP_r, db__MAP_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.db_fuzzy_MAP,
                                  k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_db_fuzzy_MAP_scores = data_db_fuzzy_MAP_pca, data_db_fuzzy_MAP_lap, db_fuzzy_MAP_r, db__MAP_t
            if eval_MDE:
                data_db_fuzzy_MDE_pca, data_db_fuzzy_MDE_lap = global_scores(X, TopOGraph.db_fuzzy_MDE,
                                                                           k=k, metric=metric, n_jobs=n_jobs)
                db_fuzzy_MDE_r, db_fuzzy_MDE_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.db_fuzzy_MDE,
                                  k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_db_fuzzy_MDE_scores = data_db_fuzzy_MDE_pca, data_db_fuzzy_MDE_lap, db_fuzzy_MDE_r, db_fuzzy_MDE_t

    if eval_cb:
        cb_pca, cb_lap = global_scores(X, TopOGraph.CLapMap, k=k, metric=metric, n_jobs=n_jobs)
        cb_r, cb_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.CLapMap,
                                  k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)

        cb_scores = cb_pca, cb_lap, cb_r, cb_t
        if eval_TriMAP:
            data_cb_TriMAP_pca, data_cb_TriMAP_lap = global_scores(X, TopOGraph.cb_TriMAP,
                                                                           k=k, metric=metric, n_jobs=n_jobs)
            cb_TriMAP_r, cb_TriMAP_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.cb_TriMAP,
                                  k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            data_cb_TriMAP_scores = data_cb_TriMAP_pca, data_cb_TriMAP_lap, cb_TriMAP_r, cb_TriMAP_t

        if eval_PaCMAP:
            data_cb_PaCMAP_pca, data_cb_PaCMAP_lap = global_scores(X, TopOGraph.cb_PaCMAP,
                                                                           k=k, metric=metric, n_jobs=n_jobs)
            cb_PaCMAP_r, cb_PaCMAP_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.cb_PaCMAP,
                                  k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            data_cb_PaCMAP_scores = data_cb_PaCMAP_pca, data_cb_PaCMAP_lap, cb_PaCMAP_r, cb_PaCMAP_t

        if eval_NCVis:
            data_cb_NCVis_pca, data_cb_NCVis_lap = global_scores(X, TopOGraph.cb_NCVis,
                                                                           k=k, metric=metric, n_jobs=n_jobs)
            cb_NCVis_r, cb_NCVis_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.cb_NCVis,
                                  k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            data_cb_NCVis_scores = data_cb_NCVis_pca, data_cb_NCVis_lap, cb_NCVis_r, cb_NCVis_t

        if eval_tSNE:
            data_cb_tSNE_pca, data_db_tSNE_pca = global_scores(X, TopOGraph.cb_tSNE,
                                                                           k=k, metric=metric, n_jobs=n_jobs)
            cb_tSNE_r, cb_tSNE_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.cb_tSNE,
                                  k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            data_cb_tSNE_scores = data_cb_tSNE_pca, data_cb_tSNE_pca, cb_tSNE_r, cb_tSNE_t

        if eval_diff:
            cb_diff_r, cb_diff_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.cb_diff_graph,
                                emb_is_graph=True, k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)

            cb_diff_scores = cb_diff_r, cb_diff_t

            if eval_MAP:
                data_cb_diff_MAP_pca, data_cb_diff_MAP_pca = global_scores(X, TopOGraph.cb_diff_MAP,
                                                                           k=k, metric=metric, n_jobs=n_jobs)
                cb_diff_MAP_r, cb_diff_MAP_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.cb_diff_MAP,
                                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_cb_diff_MAP_scores = data_cb_diff_MAP_pca, data_cb_diff_MAP_pca, cb_diff_MAP_r, cb_diff_MAP_t
            if eval_MDE:
                data_cb_diff_MDE_pca, data_cb_diff_MDE_lap = global_scores(X, TopOGraph.cb_diff_MDE,
                                                                           k=k, metric=metric, n_jobs=n_jobs)
                cb_diff_MDE_r, cb_diff_MDE_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.cb_diff_MDE,
                                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_cb_diff_MDE_scores = data_cb_diff_MDE_pca, data_cb_diff_MDE_lap, cb_diff_MDE_r, cb_diff_MDE_t

        if eval_cknn:
            cb_cknn_r, cb_cknn_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.cb_cknn_graph,
                                emb_is_graph=True, k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            cb_cknn_scores = cb_cknn_r, cb_cknn_t

            if eval_MAP:
                data_cb_cknn_MAP_pca, data_cb_cknn_MAP_lap = global_scores(X, TopOGraph.cb_cknn_MAP,
                                                                           k=k, metric=metric, n_jobs=n_jobs)
                cb_cknn_MAP_r, cb__MAP_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.cb_cknn_MAP,
                                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_cb_cknn_MAP_scores = data_cb_cknn_MAP_pca, data_cb_cknn_MAP_lap, cb_cknn_MAP_r, cb__MAP_t
            if eval_MDE:
                data_cb_cknn_MDE_pca, data_cb_cknn_MDE_lap = global_scores(X, TopOGraph.cb_cknn_MDE,
                                                                           k=k, metric=metric, n_jobs=n_jobs)
                cb_cknn_MDE_r, cb_cknn_MDE_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.cb_cknn_MDE,
                                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_cb_cknn_MDE_scores = data_cb_cknn_MDE_pca, data_cb_cknn_MDE_lap, cb_cknn_MDE_r, cb_cknn_MDE_t

        if eval_fuzzy:
            cb_fuzzy_r, cb_fuzzy_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.cb_fuzzy_graph,
                                emb_is_graph=True, k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            cb_fuzzy_scores = cb_fuzzy_r, cb_fuzzy_t

            if eval_MAP:
                data_cb_fuzzy_MAP_pca, data_cb_fuzzy_MAP_lap = global_scores(X, TopOGraph.cb_fuzzy_MAP,
                                                                           k=k, metric=metric, n_jobs=n_jobs)
                cb_fuzzy_MAP_r, cb__MAP_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.cb_fuzzy_MAP,
                                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_cb_fuzzy_MAP_scores = data_cb_fuzzy_MAP_pca, data_cb_fuzzy_MAP_lap, cb_fuzzy_MAP_r, cb__MAP_t
            if eval_MDE:
                data_cb_fuzzy_MDE_pca, data_cb_fuzzy_MDE_lap = global_scores(X, TopOGraph.cb_fuzzy_MDE,
                                                                           k=k, metric=metric, n_jobs=n_jobs)
                cb_fuzzy_MDE_r, cb_fuzzy_MDE_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.cb_fuzzy_MDE,
                                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_cb_fuzzy_MDE_scores = data_cb_fuzzy_MDE_pca, data_cb_fuzzy_MDE_lap, cb_fuzzy_MDE_r, cb_fuzzy_MDE_t

    if eval_fb:
        fb_pca, fb_lap = global_scores(X, TopOGraph.FuzzyLapMap, k=k, metric=metric, n_jobs=n_jobs)
        fb_r, fb_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.FuzzyLapMap,
                                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)

        fb_scores = fb_pca, fb_lap, fb_r, fb_t

        if eval_TriMAP:
            data_fb_TriMAP_pca, data_fb_TriMAP_lap = global_scores(X, TopOGraph.fb_TriMAP,
                                                                           k=k, metric=metric, n_jobs=n_jobs)
            fb_TriMAP_r, fb_TriMAP_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.fb_TriMAP,
                                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            data_fb_TriMAP_scores = data_fb_TriMAP_pca, data_fb_TriMAP_lap, fb_TriMAP_r, fb_TriMAP_t

        if eval_PaCMAP:
            data_fb_PaCMAP_pca, data_fb_PaCMAP_lap = global_scores(X, TopOGraph.fb_PaCMAP,
                                                                           k=k, metric=metric, n_jobs=n_jobs)
            fb_PaCMAP_r, fb_PaCMAP_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.fb_PaCMAP,
                                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            data_fb_PaCMAP_scores = data_fb_PaCMAP_pca, data_fb_PaCMAP_lap, fb_PaCMAP_r, fb_PaCMAP_t

        if eval_NCVis:
            data_fb_NCVis_pca, data_fb_NCVis_lap = global_scores(X, TopOGraph.fb_NCVis,
                                                                           k=k, metric=metric, n_jobs=n_jobs)
            fb_NCVis_r, fb_NCVis_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.fb_NCVis,
                                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            data_fb_NCVis_scores = data_fb_NCVis_pca, data_fb_NCVis_lap, fb_NCVis_r, fb_NCVis_t

        if eval_tSNE:
            data_fb_tSNE_pca, data_db_tSNE_pca = global_scores(X, TopOGraph.fb_tSNE,
                                                                           k=k, metric=metric, n_jobs=n_jobs)
            fb_tSNE_r, fb_tSNE_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.fb_tSNE,
                                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            data_fb_tSNE_scores = data_fb_tSNE_pca, data_fb_tSNE_pca, fb_tSNE_r, fb_tSNE_t

        if eval_diff:
            fb_diff_r, fb_diff_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.fb_diff_graph,
                                emb_is_graph=True, k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            fb_diff_scores = fb_diff_r, fb_diff_t

            if eval_MAP:
                data_fb_diff_MAP_pca, data_fb_diff_MAP_pca = global_scores(X, TopOGraph.fb_diff_MAP,
                                                                           k=k, metric=metric, n_jobs=n_jobs)
                fb_diff_MAP_r, fb_diff_MAP_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.fb_diff_MAP,
                                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_fb_diff_MAP_scores = data_fb_diff_MAP_pca, data_fb_diff_MAP_pca, fb_diff_MAP_r, fb_diff_MAP_t
            if eval_MDE:
                data_fb_diff_MDE_pca, data_fb_diff_MDE_lap = global_scores(X, TopOGraph.fb_diff_MDE,
                                                                           k=k, metric=metric, n_jobs=n_jobs)
                fb_diff_MDE_r, fb_diff_MDE_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.fb_diff_MDE,
                                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_fb_diff_MDE_scores = data_fb_diff_MDE_pca, data_fb_diff_MDE_lap, fb_diff_MDE_r, fb_diff_MDE_t

        if eval_cknn:
            fb_cknn_r, fb_cknn_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.fb_cknn_graph,
                                emb_is_graph=True, k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            fb_cknn_scores = fb_cknn_r, fb_cknn_t

            if eval_MAP:
                data_fb_cknn_MAP_pca, data_fb_cknn_MAP_lap = global_scores(X, TopOGraph.fb_cknn_MAP,
                                                                           k=k, metric=metric, n_jobs=n_jobs)
                fb_cknn_MAP_r, fb__MAP_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.fb_cknn_MAP,
                                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_fb_cknn_MAP_scores = data_fb_cknn_MAP_pca, data_fb_cknn_MAP_lap, fb_cknn_MAP_r, fb__MAP_t
            if eval_MDE:
                data_fb_cknn_MDE_pca, data_fb_cknn_MDE_lap = global_scores(X, TopOGraph.fb_cknn_MDE,
                                                                           k=k, metric=metric, n_jobs=n_jobs)
                fb_cknn_MDE_r, fb_cknn_MDE_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.fb_cknn_MDE,
                                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_fb_cknn_MDE_scores = data_fb_cknn_MDE_pca, data_fb_cknn_MDE_lap, fb_cknn_MDE_r, fb_cknn_MDE_t

        if eval_fuzzy:
            fb_fuzzy_r, fb_fuzzy_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.fb_fuzzy_graph,
                                emb_is_graph=True, k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
            fb_fuzzy_scores = fb_fuzzy_r, fb_fuzzy_t

            if eval_MAP:
                data_fb_fuzzy_MAP_pca, data_fb_fuzzy_MAP_lap = global_scores(X, TopOGraph.fb_fuzzy_MAP,
                                                                           k=k, metric=metric, n_jobs=n_jobs)
                fb_fuzzy_MAP_r, fb__MAP_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.fb_fuzzy_MAP,
                                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_fb_fuzzy_MAP_scores = data_fb_fuzzy_MAP_pca, data_fb_fuzzy_MAP_lap, fb_fuzzy_MAP_r, fb__MAP_t
            if eval_MDE:
                data_fb_fuzzy_MDE_pca, data_fb_fuzzy_MDE_lap = global_scores(X, TopOGraph.fb_fuzzy_MDE,
                                                                           k=k, metric=metric, n_jobs=n_jobs)
                fb_fuzzy_MDE_r, fb_fuzzy_MDE_t = local_scores(TopOGraph.base_knn_graph, TopOGraph.fb_fuzzy_MDE,
                                                            k=k, metric=metric, n_jobs=n_jobs, data_is_graph=True)
                data_fb_fuzzy_MDE_scores = data_fb_fuzzy_MDE_pca, data_fb_fuzzy_MDE_lap, fb_fuzzy_MDE_r, fb_fuzzy_MDE_t

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
                data = np.array(X.values.T)
            else:
                return print('Uknown data format.')

    pca_emb = PCA(n_components=TopOGraph.n_eigs).fit_transform(data)
    pca_pca, pca_lap = global_scores(X, pca_emb, k=k, metric=metric, n_jobs=n_jobs)
    pca_r, pca_t = local_scores(X, pca_emb)

    embedding_scores = {'PCA' : (pca_pca, pca_lap, pca_r, pca_t)}
    graph_scores = {}
    layout_scores = {}

    if eval_MAP:
        if TopOGraph.verbosity >= 1:
            print('Computing UMAP...')
        # Compare to PCA + (U)MAP and just (U)MAP
        pca_fuzzy_results = fuzzy_simplicial_set_ann(pca_emb, n_neighbors=TopOGraph.base_knn,
                                                     backend=TopOGraph.backend, metric=TopOGraph.base_metric,
                                                     n_jobs=TopOGraph.n_jobs, efC=TopOGraph.efC, M=TopOGraph.M,
                                                     verbose=TopOGraph.bases_graph_verbose)

        umap_on_pca_emb, aux = fuzzy_embedding(graph=pca_fuzzy_results[0],
                                          verbose=TopOGraph.layout_verbose)

        umap_on_pca_pca, umap_on_pca_lap = global_scores(X, umap_on_pca_emb, k=k, metric=metric, n_jobs=n_jobs)
        umap_on_pca_r, umap_on_pca_t = local_scores(TopOGraph.base_knn_graph, umap_on_pca_emb, data_is_graph=True, k=k)
        umap_on_pca_scores = umap_on_pca_pca, umap_on_pca_lap, umap_on_pca_r, umap_on_pca_t

        layout_scores['UMAP'] = np.absolute(umap_on_pca_scores)


    if eval_tSNE:
        if TopOGraph.verbosity >= 1:
            print('Computing default tSNE...')
        from MulticoreTSNE import MulticoreTSNE as TSNE
        tsne_emb = TSNE(verbose=TopOGraph.layout_verbose, n_jobs=TopOGraph.n_jobs,
                        init=pca_emb[:, 0:2], metric=TopOGraph.graph_metric).fit_transform(pca_emb)
        tsne_on_pca_pca, tsne_on_pca_lap = global_scores(X, tsne_emb, k=k, metric=metric, n_jobs=n_jobs)
        tsne_on_pca_r, tsne_on_pca_t = local_scores(TopOGraph.base_knn_graph, tsne_emb, data_is_graph=True, k=k)
        tsne_on_pca_scores = tsne_on_pca_pca, tsne_on_pca_lap, tsne_on_pca_r, tsne_on_pca_t
        layout_scores['tSNE'] = np.absolute(tsne_on_pca_scores)

    if eval_PaCMAP:
        if TopOGraph.verbosity >= 1:
            print('Computing default PaCMAP...')
        from topo.layouts.pairwise import PaCMAP
        pca_pacmap_emb = PaCMAP(data=pca_emb, init=pca_emb[:, 0:2], verbose=TopOGraph.layout_verbose)
        pacmap_on_pca_pca, pacmap_on_pca_lap = global_scores(X, pca_pacmap_emb, k=k, metric=metric, n_jobs=n_jobs)
        pacmap_on_pca_r, pacmap_on_pca_t = local_scores(TopOGraph.base_knn_graph, pca_pacmap_emb, data_is_graph=True,
                                                        k=k)
        pacmap_on_pca_scores = pacmap_on_pca_pca, pacmap_on_pca_lap, pacmap_on_pca_r, pacmap_on_pca_t
        layout_scores['PaCMAP'] = np.absolute(pacmap_on_pca_scores)

    if eval_TriMAP:
        if TopOGraph.verbosity >= 1:
            print('Computing default TriMAP...')
        from topo.layouts.trimap import TriMAP
        pca_trimap_emb = TriMAP(X=pca_emb, init=pca_emb[:, 0:2], verbose=TopOGraph.layout_verbose)
        trimap_on_pca_pca, trimap_on_pca_lap = global_scores(X, pca_trimap_emb, k=k, metric=metric, n_jobs=n_jobs)
        trimap_on_pca_r, trimap_on_pca_t = local_scores(TopOGraph.base_knn_graph, pca_trimap_emb, data_is_graph=True,
                                                        k=k)
        trimap_on_pca_scores = trimap_on_pca_pca, trimap_on_pca_lap, trimap_on_pca_r, trimap_on_pca_t
        layout_scores['TriMAP'] = np.absolute(trimap_on_pca_scores)

    if eval_MDE:
        if TopOGraph.verbosity >= 1:
            print('Computing default MDE...')
        import torch
        import pymde
        pca_mde_emb = pymde.preserve_neighbors(torch.tensor(pca_emb),
                                               n_neighbors=TopOGraph.base_knn, verbose=False).embed()
        pca_mde_emb = pca_mde_emb.numpy()
        mde_on_pca_pca, mde_on_pca_lap = global_scores(X, pca_mde_emb, k=k, metric=metric, n_jobs=n_jobs)
        mde_on_pca_r, mde_on_pca_t = local_scores(TopOGraph.base_knn_graph, pca_mde_emb, data_is_graph=True, k=k)
        mde_on_pca_scores = mde_on_pca_pca, mde_on_pca_lap, mde_on_pca_r, mde_on_pca_t
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


