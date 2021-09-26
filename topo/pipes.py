import matplotlib
matplotlib.use('Agg')  # plotting backend compatible with screen
import sys
from topo.models import TopOGraph
from topo.eval.global_scores import global_score_pca, global_score_laplacian
from topo.eval.local_scores import geodesic_distance, knn_spearman_r, knn_kendall_tau

filename = sys.argv[1]  # read filename from command line

def TopoMAP(data, *tg_kwargs, **map_kwargs):
    """""""""
    Easy, direct application of topological graphs (``TopoGraph``) and Manifold Approximation and Projection (``MAP``) for layout optimization with triple
    approximation of the Laplace-Beltrami Operator.

    Parameters
    ----------
    data: np.ndarray, csr_matrix, pd.DataFrame
        Input data. Will be converted to csr_matrix by default

    *tg_kwargs: dict
        keyword arguments for the ``TopoGraph`` instance.

    **map_kwargs: dict
        keyword arguments for the ``MAP`` function

    Returns
    -------
    TopoGraph:
        object containing topological graph analysis
    embedding: np.ndarray
        lower dimensional projection for optimal visualization

    """""""""

    tg = TopOGraph(*tg_kwargs).fit(data)
    graph = tg.transform()
    basis = tg.MSDiffMap
    emb = tg.MAP(basis, graph, **map_kwargs)

    return tg, emb



def TopoMDE(data, *tg_kwargs, **mde_kwargs):
    """""""""
    Easy, direct application of topological graphs (``TopoGraph``) and Manifold Approximation and Projection (``MAP``) for layout optimization with triple
    approximation of the Laplace-Beltrami Operator.

    Parameters
    ----------
    data: np.ndarray, csr_matrix, pd.DataFrame
        Input data. Will be converted to csr_matrix by default

    *tg_kwargs: dict
        keyword arguments for the ``TopoGraph`` instance.

    **mde_kwargs: dict
        keyword arguments for the ``MDE`` function

    Returns
    -------
    TopoGraph:
        object containing topological graph analysis
    embedding: np.ndarray
        lower dimensional projection for optimal visualization

    """""""""

    tg = TopOGraph(*tg_kwargs).fit(data)
    graph = tg.transform()
    topo = tg.MSDiffMap

    emb = tg.MDE(graph, **mde_kwargs)

    return tg, emb


def global_scores(data, emb):
    global_scores_pca = global_score_pca(data, emb)
    global_scores_lap = global_score_laplacian(data, emb)
    return global_scores_pca, global_scores_lap

def local_scores(data, emb, k=10, metric='cosine', n_jobs=12, emb_is_graph=False):
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
    if _have_hnswlib:
        if _have_hnswlib:
            from topo.base.ann import HNSWlibTransformer
            data_graph = HNSWlibTransformer(n_neighbors=k, n_jobs=n_jobs, metric=metric).fit_transform(data)
            if not emb_is_graph:
                emb_graph = HNSWlibTransformer(n_neighbors=k, n_jobs=n_jobs, metric=metric).fit_transform(emb)

    elif _have_nmslib:
        from topo.base.ann import NMSlibTransformer
        data_graph = NMSlibTransformer(n_neighbors=k, n_jobs=n_jobs, metric=metric).fit_transform(data)
        if not emb_is_graph:
            emb_graph = NMSlibTransformer(n_neighbors=k, n_jobs=n_jobs, metric=metric).fit_transform(emb)

    else:
        from sklearn.neighbors import NearestNeighbors
        data_nbrs = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs,
                                     metric=metric).fit(data)
        data_graph = data_nbrs.kneighbors(data)
        if not emb_is_graph:
            emb_nbrs = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs,
                                        metric=metric).fit(emb)
            emb_graph = emb_nbrs.kneighbors(emb)

    if emb_is_graph:
        emb_graph = emb

    local_scores_r = knn_spearman_r(data_graph, emb_graph)
    local_scores_tau = knn_kendall_tau(data_graph, emb_graph)
    return local_scores_r, local_scores_tau

def eval_models_layouts(TopOGraph, X,
                basis=['diffusion', 'fuzzy', 'continuous'],
                graphs=['diff', 'cknn', 'fuzzy'],
                layouts=['tSNE', 'MAP','MDE','PaCMAP','TriMAP']):
    if str('diffusion') in basis:
        eval_db = True
    if str('continuous') in basis:
        eval_cb = True
    if str('fuzzy') in basis:
        eval_fb = True
    if str('diff') in graphs:
        eval_diff = True
    if str('cknn') in graphs:
        eval_cknn = True
    if str('fuzzy') in graphs:
        eval_fuzzy = True
    if str('tSNE') in layouts:
        eval_tSNE = True
    if str('MAP') in layouts:
        eval_MAP = True
    if str('MDE') in layouts:
        eval_MDE = True
    if str('PaCMAP') in layouts:
        eval_PaCMAP = True
    if str('TriMAP') in layouts:
        eval_TriMAP = True

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

    TopOGraph.run_layouts(X=X,
                basis=basis,
                graphs=graphs,
                layouts=layouts)

    if eval_db:
        db_pca, db_lap = global_scores(X, TopOGraph.MSDiffMap)
        db_r, db_t = local_scores(X, TopOGraph.DiffBasis.K, emb_is_graph=True)

        db_scores = db_pca, db_lap, db_r, db_t

        if eval_diff:
            db_diff_r, db_diff_t = local_scores(X, TopOGraph.Diff_Diff_Graph, emb_is_graph=True)

            db_diff_scores = db_diff_r, db_diff_t

            if eval_tSNE:
                data_db_diff_tSNE_pca, data_db_diff_tSNE_pca = global_scores(X, TopOGraph.db_diff_tSNE)
                db_diff_tSNE_r, db_diff_tSNE_t = local_scores(X, TopOGraph.db_diff_tSNE)
                data_db_diff_tSNE_scores = data_db_diff_tSNE_pca, data_db_diff_tSNE_pca, db_diff_tSNE_r, db_diff_tSNE_t
            if eval_MAP:
                data_db_diff_MAP_pca, data_db_diff_MAP_pca = global_scores(X, TopOGraph.db_diff_MAP)
                db_diff_MAP_r, db_diff_MAP_t = local_scores(X, TopOGraph.db_diff_MAP)
                data_db_diff_MAP_scores = data_db_diff_MAP_pca, data_db_diff_MAP_pca, db_diff_MAP_r, db_diff_MAP_t
            if eval_MDE:
                mde = TopOGraph.MDE()
                data_db_diff_MDE_pca, data_db_diff_MDE_lap = global_scores(X, TopOGraph.db_diff_MDE)
                db_diff_MDE_r, db_diff_MDE_t = local_scores(X, TopOGraph.db_diff_MDE)
                data_db_diff_MDE_scores = data_db_diff_MDE_pca, data_db_diff_MDE_lap, db_diff_MDE_r, db_diff_MDE_t
            if eval_TriMAP:
                data_db_diff_TriMAP_pca, data_db_diff_TriMAP_lap = global_scores(X, TopOGraph.db_diff_TriMAP)
                db_diff_TriMAP_r, db_diff_TriMAP_t = local_scores(X, TopOGraph.db_diff_TriMAP)
                data_db_diff_TriMAP_scores = data_db_diff_TriMAP_pca, data_db_diff_TriMAP_lap, db_diff_TriMAP_r, db_diff_TriMAP_t
            if eval_PaCMAP:
                data_db_diff_PaCMAP_pca, data_db_diff_PaCMAPP_lap = global_scores(X, TopOGraph.db_diff_PaCMAP)
                db_diff_PaCMAP_r, db_diff_PaCMAP_t = local_scores(X, TopOGraph.db_diff_PaCMAP)
                data_db_diff_PaCMAP_scores = data_db_diff_PaCMAP_pca, data_db_diff_PaCMAPP_lap, db_diff_PaCMAP_r, db_diff_PaCMAP_t

        if eval_cknn:
            db_cknn_r, db_cknn_t = local_scores(X, TopOGraph.Diff_Cknn_Graph, emb_is_graph=True)
            db_cknn_scores = db_cknn_r, db_cknn_t
            if eval_tSNE:
                data_db_cknn_tSNE_pca, data_db_cknn_tSNE_lap = global_scores(X, TopOGraph.db_cknn_tSNE)
                db_cknn_tSNE_r, db_cknn_tSNE_t = local_scores(X, TopOGraph.db_cknn_tSNE)
                data_db_cknn_tSNE_scores = data_db_cknn_tSNE_pca, data_db_cknn_tSNE_lap, db_cknn_tSNE_r, db_cknn_tSNE_t
            if eval_MAP:
                data_db_cknn_MAP_pca, data_db_cknn_MAP_lap = global_scores(X, TopOGraph.db_cknn_MAP)
                db_cknn_MAP_r, db__MAP_t = local_scores(X, TopOGraph.db_cknn_MAP)
                data_db_cknn_MAP_scores = data_db_cknn_MAP_pca, data_db_cknn_MAP_lap, db_cknn_MAP_r, db__MAP_t
            if eval_MDE:
                mde = TopOGraph.MDE()
                data_db_cknn_MDE_pca, data_db_cknn_MDE_lap = global_scores(X, TopOGraph.db_cknn_MDE)
                db_cknn_MDE_r, db_cknn_MDE_t = local_scores(X, TopOGraph.db_cknn_MDE)
                data_db_cknn_MDE_scores = data_db_cknn_MDE_pca, data_db_cknn_MDE_lap, db_cknn_MDE_r, db_cknn_MDE_t
            if eval_TriMAP:
                data_db_cknn_TriMAP_pca, data_db_cknn_TriMAP_lap = global_scores(X, TopOGraph.db_cknn_TriMAP)
                db_cknn_TriMAP_r, db_cknn_TriMAP_t = local_scores(X, TopOGraph.db_cknn_TriMAP)
                data_db_cknn_TriMAP_scores = data_db_cknn_TriMAP_pca, data_db_cknn_TriMAP_lap, db_cknn_TriMAP_r, db_cknn_TriMAP_t
            if eval_PaCMAP:
                data_db_cknn_PaCMAP_pca, data_db_cknn_PaCMAP_lap = global_scores(X, TopOGraph.db_cknn_PaCMAP)
                db_cknn_PaCMAP_r, db_cknn_PaCMAP_t = local_scores(X, TopOGraph.db_cknn_PaCMAP)
                data_db_cknn_PaCMAP_scores = data_db_cknn_PaCMAP_pca, data_db_cknn_PaCMAP_lap, db_cknn_PaCMAP_r, db_cknn_PaCMAP_t

        if eval_fuzzy:
            db_fuzzy_r, db_fuzzy_t = local_scores(X, TopOGraph.Diff_Fuzzy_Graph, emb_is_graph=True)
            db_fuzzy_scores = db_fuzzy_r, db_fuzzy_t
            if eval_tSNE:
                data_db_fuzzy_tSNE_pca, data_db_fuzzy_tSNE_lap = global_scores(X, TopOGraph.db_fuzzy_tSNE)
                db_fuzzy_tSNE_r, db_fuzzy_tSNE_t = local_scores(X, TopOGraph.db_fuzzy_tSNE)
                data_db_fuzzy_tSNE_scores = data_db_fuzzy_tSNE_pca, data_db_fuzzy_tSNE_lap, db_fuzzy_tSNE_r, db_fuzzy_tSNE_t
            if eval_MAP:
                data_db_fuzzy_MAP_pca, data_db_fuzzy_MAP_lap = global_scores(X, TopOGraph.db_fuzzy_MAP)
                db_fuzzy_MAP_r, db__MAP_t = local_scores(X, TopOGraph.db_fuzzy_MAP)
                data_db_fuzzy_MAP_scores = data_db_fuzzy_MAP_pca, data_db_fuzzy_MAP_lap, db_fuzzy_MAP_r, db__MAP_t
            if eval_MDE:
                mde = TopOGraph.MDE()
                data_db_fuzzy_MDE_pca, data_db_fuzzy_MDE_lap = global_scores(X, TopOGraph.db_fuzzy_MDE)
                db_fuzzy_MDE_r, db_fuzzy_MDE_t = local_scores(X, TopOGraph.db_fuzzy_MDE)
                data_db_fuzzy_MDE_scores = data_db_fuzzy_MDE_pca, data_db_fuzzy_MDE_lap, db_fuzzy_MDE_r, db_fuzzy_MDE_t
            if eval_TriMAP:
                data_db_fuzzy_TriMAP_pca, data_db_fuzzy_TriMAP_lap = global_scores(X, TopOGraph.db_fuzzy_TriMAP)
                db_fuzzy_TriMAP_r, db_fuzzy_TriMAP_t = local_scores(X, TopOGraph.db_fuzzy_TriMAP)
                data_db_fuzzy_TriMAP_scores = data_db_fuzzy_TriMAP_pca, data_db_fuzzy_TriMAP_lap, db_fuzzy_TriMAP_r, db_fuzzy_TriMAP_t
            if eval_PaCMAP:
                data_db_fuzzy_PaCMAP_pca, data_db_fuzzy_PaCMAP_lap = global_scores(X, TopOGraph.db_fuzzy_PaCMAP)
                db_fuzzy_PaCMAP_r, db_fuzzy_PaCMAP_t = local_scores(X, TopOGraph.db_fuzzy_PaCMAP)
                data_db_fuzzy_PaCMAP_scores = data_db_fuzzy_PaCMAP_pca, data_db_fuzzy_PaCMAP_lap, db_fuzzy_PaCMAP_r, db_fuzzy_PaCMAP_t

    if eval_cb:
        cb_pca, cb_lap = global_scores(X, TopOGraph.MSDiffMap)
        cb_r, cb_t = local_scores(X, TopOGraph.DiffBasis.K, emb_is_graph=True)

        cb_scores = cb_pca, cb_lap, cb_r, cb_t

        if eval_diff:
            cb_diff_r, cb_diff_t = local_scores(X, TopOGraph.Diff_Diff_Graph, emb_is_graph=True)

            cb_diff_scores = cb_diff_r, cb_diff_t

            if eval_tSNE:
                data_cb_diff_tSNE_pca, data_cb_diff_tSNE_pca = global_scores(X, TopOGraph.cb_diff_tSNE)
                cb_diff_tSNE_r, cb_diff_tSNE_t = local_scores(X, TopOGraph.cb_diff_tSNE)
                data_cb_diff_tSNE_scores = data_cb_diff_tSNE_pca, data_cb_diff_tSNE_pca, cb_diff_tSNE_r, cb_diff_tSNE_t
            if eval_MAP:
                data_cb_diff_MAP_pca, data_cb_diff_MAP_pca = global_scores(X, TopOGraph.cb_diff_MAP)
                cb_diff_MAP_r, cb_diff_MAP_t = local_scores(X, TopOGraph.cb_diff_MAP)
                data_cb_diff_MAP_scores = data_cb_diff_MAP_pca, data_cb_diff_MAP_pca, cb_diff_MAP_r, cb_diff_MAP_t
            if eval_MDE:
                mde = TopOGraph.MDE()
                data_cb_diff_MDE_pca, data_cb_diff_MDE_lap = global_scores(X, TopOGraph.cb_diff_MDE)
                cb_diff_MDE_r, cb_diff_MDE_t = local_scores(X, TopOGraph.cb_diff_MDE)
                data_cb_diff_MDE_scores = data_cb_diff_MDE_pca, data_cb_diff_MDE_lap, cb_diff_MDE_r, cb_diff_MDE_t
            if eval_TriMAP:
                data_cb_diff_TriMAP_pca, data_cb_diff_TriMAP_lap = global_scores(X, TopOGraph.cb_diff_TriMAP)
                cb_diff_TriMAP_r, cb_diff_TriMAP_t = local_scores(X, TopOGraph.cb_diff_TriMAP)
                data_cb_diff_TriMAP_scores = data_cb_diff_TriMAP_pca, data_cb_diff_TriMAP_lap, cb_diff_TriMAP_r, cb_diff_TriMAP_t
            if eval_PaCMAP:
                data_cb_diff_PaCMAP_pca, data_cb_diff_PaCMAP_lap = global_scores(X, TopOGraph.cb_diff_PaCMAP)
                cb_diff_PaCMAP_r, cb_diff_PaCMAP_t = local_scores(X, TopOGraph.cb_diff_PaCMAP)
                data_cb_diff_PaCMAP_scores = data_cb_diff_PaCMAP_pca, data_cb_diff_PaCMAP_lap, cb_diff_PaCMAP_r, cb_diff_PaCMAP_t

        if eval_cknn:
            cb_cknn_r, cb_cknn_t = local_scores(X, TopOGraph.Diff_Cknn_Graph, emb_is_graph=True)
            cb_cknn_scores = cb_cknn_r, cb_cknn_t
            if eval_tSNE:
                data_cb_cknn_tSNE_pca, data_cb_cknn_tSNE_lap = global_scores(X, TopOGraph.cb_cknn_tSNE)
                cb_cknn_tSNE_r, cb_cknn_tSNE_t = local_scores(X, TopOGraph.cb_cknn_tSNE)
                data_cb_cknn_tSNE_scores = data_cb_cknn_tSNE_pca, data_cb_cknn_tSNE_lap, cb_cknn_tSNE_r, cb_cknn_tSNE_t
            if eval_MAP:
                data_cb_cknn_MAP_pca, data_cb_cknn_MAP_lap = global_scores(X, TopOGraph.cb_cknn_MAP)
                cb_cknn_MAP_r, cb__MAP_t = local_scores(X, TopOGraph.cb_cknn_MAP)
                data_cb_cknn_MAP_scores = data_cb_cknn_MAP_pca, data_cb_cknn_MAP_lap, cb_cknn_MAP_r, cb__MAP_t
            if eval_MDE:
                mde = TopOGraph.MDE()
                data_cb_cknn_MDE_pca, data_cb_cknn_MDE_lap = global_scores(X, TopOGraph.cb_cknn_MDE)
                cb_cknn_MDE_r, cb_cknn_MDE_t = local_scores(X, TopOGraph.cb_cknn_MDE)
                data_cb_cknn_MDE_scores = data_cb_cknn_MDE_pca, data_cb_cknn_MDE_lap, cb_cknn_MDE_r, cb_cknn_MDE_t
            if eval_TriMAP:
                data_cb_cknn_TriMAP_pca, data_cb_cknn_TriMAP_lap = global_scores(X, TopOGraph.cb_cknn_TriMAP)
                cb_cknn_TriMAP_r, cb_cknn_TriMAP_t = local_scores(X, TopOGraph.cb_cknn_TriMAP)
                data_cb_cknn_TriMAP_scores = data_cb_cknn_TriMAP_pca, data_cb_cknn_TriMAP_lap, cb_cknn_TriMAP_r, cb_cknn_TriMAP_t
            if eval_PaCMAP:
                data_cb_cknn_PaCMAP_pca, data_cb_cknn_PaCMAP_lap = global_scores(X, TopOGraph.cb_cknn_PaCMAP)
                cb_cknn_PaCMAP_r, cb_cknn_PaCMAP_t = local_scores(X, TopOGraph.cb_cknn_PaCMAP)
                data_cb_cknn_PaCMAP_scores = data_cb_cknn_PaCMAP_pca, data_cb_cknn_PaCMAP_lap, cb_cknn_PaCMAP_r, cb_cknn_PaCMAP_t

        if eval_fuzzy:
            cb_fuzzy_r, cb_fuzzy_t = local_scores(X, TopOGraph.Diff_Fuzzy_Graph, emb_is_graph=True)
            cb_fuzzy_scores = cb_fuzzy_r, cb_fuzzy_t
            if eval_tSNE:
                data_cb_fuzzy_tSNE_pca, data_cb_fuzzy_tSNE_lap = global_scores(X, TopOGraph.cb_fuzzy_tSNE)
                cb_fuzzy_tSNE_r, cb_fuzzy_tSNE_t = local_scores(X, TopOGraph.cb_fuzzy_tSNE)
                data_cb_fuzzy_tSNE_scores = data_cb_fuzzy_tSNE_pca, data_cb_fuzzy_tSNE_lap, cb_fuzzy_tSNE_r, cb_fuzzy_tSNE_t
            if eval_MAP:
                data_cb_fuzzy_MAP_pca, data_cb_fuzzy_MAP_lap = global_scores(X, TopOGraph.cb_fuzzy_MAP)
                cb_fuzzy_MAP_r, cb__MAP_t = local_scores(X, TopOGraph.cb_fuzzy_MAP)
                data_cb_fuzzy_MAP_scores = data_cb_fuzzy_MAP_pca, data_cb_fuzzy_MAP_lap, cb_fuzzy_MAP_r, cb__MAP_t
            if eval_MDE:
                mde = TopOGraph.MDE()
                data_cb_fuzzy_MDE_pca, data_cb_fuzzy_MDE_lap = global_scores(X, TopOGraph.cb_fuzzy_MDE)
                cb_fuzzy_MDE_r, cb_fuzzy_MDE_t = local_scores(X, TopOGraph.cb_fuzzy_MDE)
                data_cb_fuzzy_MDE_scores = data_cb_fuzzy_MDE_pca, data_cb_fuzzy_MDE_lap, cb_fuzzy_MDE_r, cb_fuzzy_MDE_t
            if eval_TriMAP:
                data_cb_fuzzy_TriMAP_pca, data_cb_fuzzy_TriMAP_lap = global_scores(X, TopOGraph.cb_fuzzy_TriMAP)
                cb_fuzzy_TriMAP_r, cb_fuzzy_TriMAP_t = local_scores(X, TopOGraph.cb_fuzzy_TriMAP)
                data_cb_fuzzy_TriMAP_scores = data_cb_fuzzy_TriMAP_pca, data_cb_fuzzy_TriMAP_lap, cb_fuzzy_TriMAP_r, cb_fuzzy_TriMAP_t
            if eval_PaCMAP:
                data_cb_fuzzy_PaCMAP_pca, data_cb_fuzzy_PaCMAP_lap = global_scores(X, TopOGraph.cb_fuzzy_PaCMAP)
                cb_fuzzy_PaCMAP_r, cb_fuzzy_PaCMAP_t = local_scores(X, TopOGraph.cb_fuzzy_PaCMAP)
                data_cb_fuzzy_PaCMAP_scores = data_cb_fuzzy_PaCMAP_pca, data_cb_fuzzy_PaCMAP_lap, cb_fuzzy_PaCMAP_r, cb_fuzzy_PaCMAP_t


    if eval_fb:
        fb_pca, fb_lap = global_scores(X, TopOGraph.MSDiffMap)
        fb_r, fb_t = local_scores(X, TopOGraph.DiffBasis.K, emb_is_graph=True)

        fb_scores = fb_pca, fb_lap, fb_r, fb_t

        if eval_diff:
            fb_diff_r, fb_diff_t = local_scores(X, TopOGraph.Diff_Diff_Graph, emb_is_graph=True)

            fb_diff_scores = fb_diff_r, fb_diff_t

            if eval_tSNE:
                data_fb_diff_tSNE_pca, data_fb_diff_tSNE_pca = global_scores(X, TopOGraph.fb_diff_tSNE)
                fb_diff_tSNE_r, fb_diff_tSNE_t = local_scores(X, TopOGraph.fb_diff_tSNE)
                data_fb_diff_tSNE_scores = data_fb_diff_tSNE_pca, data_fb_diff_tSNE_pca, fb_diff_tSNE_r, fb_diff_tSNE_t
            if eval_MAP:
                data_fb_diff_MAP_pca, data_fb_diff_MAP_pca = global_scores(X, TopOGraph.fb_diff_MAP)
                fb_diff_MAP_r, fb_diff_MAP_t = local_scores(X, TopOGraph.fb_diff_MAP)
                data_fb_diff_MAP_scores = data_fb_diff_MAP_pca, data_fb_diff_MAP_pca, fb_diff_MAP_r, fb_diff_MAP_t
            if eval_MDE:
                mde = TopOGraph.MDE()
                data_fb_diff_MDE_pca, data_fb_diff_MDE_lap = global_scores(X, TopOGraph.fb_diff_MDE)
                fb_diff_MDE_r, fb_diff_MDE_t = local_scores(X, TopOGraph.fb_diff_MDE)
                data_fb_diff_MDE_scores = data_fb_diff_MDE_pca, data_fb_diff_MDE_lap, fb_diff_MDE_r, fb_diff_MDE_t
            if eval_TriMAP:
                data_fb_diff_TriMAP_pca, data_fb_diff_TriMAP_lap = global_scores(X, TopOGraph.fb_diff_TriMAP)
                fb_diff_TriMAP_r, fb_diff_TriMAP_t = local_scores(X, TopOGraph.fb_diff_TriMAP)
                data_fb_diff_TriMAP_scores = data_fb_diff_TriMAP_pca, data_fb_diff_TriMAP_lap, fb_diff_TriMAP_r, fb_diff_TriMAP_t
            if eval_PaCMAP:
                data_fb_diff_PaCMAP_pca, data_fb_diff_PaCMAP_lap = global_scores(X, TopOGraph.fb_diff_PaCMAP)
                fb_diff_PaCMAP_r, fb_diff_PaCMAP_t = local_scores(X, TopOGraph.fb_diff_PaCMAP)
                data_fb_diff_PaCMAP_scores = data_fb_diff_PaCMAP_pca, data_fb_diff_PaCMAP_lap, fb_diff_PaCMAP_r, fb_diff_PaCMAP_t

        if eval_cknn:
            fb_cknn_r, fb_cknn_t = local_scores(X, TopOGraph.Diff_Cknn_Graph, emb_is_graph=True)
            fb_cknn_scores = fb_cknn_r, fb_cknn_t
            if eval_tSNE:
                data_fb_cknn_tSNE_pca, data_fb_cknn_tSNE_lap = global_scores(X, TopOGraph.fb_cknn_tSNE)
                fb_cknn_tSNE_r, fb_cknn_tSNE_t = local_scores(X, TopOGraph.fb_cknn_tSNE)
                data_fb_cknn_tSNE_scores = data_fb_cknn_tSNE_pca, data_fb_cknn_tSNE_lap, fb_cknn_tSNE_r, fb_cknn_tSNE_t
            if eval_MAP:
                data_fb_cknn_MAP_pca, data_fb_cknn_MAP_lap = global_scores(X, TopOGraph.fb_cknn_MAP)
                fb_cknn_MAP_r, fb__MAP_t = local_scores(X, TopOGraph.fb_cknn_MAP)
                data_fb_cknn_MAP_scores = data_fb_cknn_MAP_pca, data_fb_cknn_MAP_lap, fb_cknn_MAP_r, fb__MAP_t
            if eval_MDE:
                mde = TopOGraph.MDE()
                data_fb_cknn_MDE_pca, data_fb_cknn_MDE_lap = global_scores(X, TopOGraph.fb_cknn_MDE)
                fb_cknn_MDE_r, fb_cknn_MDE_t = local_scores(X, TopOGraph.fb_cknn_MDE)
                data_fb_cknn_MDE_scores = data_fb_cknn_MDE_pca, data_fb_cknn_MDE_lap, fb_cknn_MDE_r, fb_cknn_MDE_t
            if eval_TriMAP:
                data_fb_cknn_TriMAP_pca, data_fb_cknn_TriMAP_lap = global_scores(X, TopOGraph.fb_cknn_TriMAP)
                fb_cknn_TriMAP_r, fb_cknn_TriMAP_t = local_scores(X, TopOGraph.fb_cknn_TriMAP)
                data_fb_cknn_TriMAP_scores = data_fb_cknn_TriMAP_pca, data_fb_cknn_TriMAP_lap, fb_cknn_TriMAP_r, fb_cknn_TriMAP_t
            if eval_PaCMAP:
                data_fb_cknn_PaCMAP_pca, data_fb_cknn_PaCMAP_lap = global_scores(X, TopOGraph.fb_cknn_PaCMAP)
                fb_cknn_PaCMAP_r, fb_cknn_PaCMAP_t = local_scores(X, TopOGraph.fb_cknn_PaCMAP)
                data_fb_cknn_PaCMAP_scores = data_fb_cknn_PaCMAP_pca, data_fb_cknn_PaCMAP_lap, fb_cknn_PaCMAP_r, fb_cknn_PaCMAP_t

        if eval_fuzzy:
            fb_fuzzy_r, fb_fuzzy_t = local_scores(X, TopOGraph.Diff_Fuzzy_Graph, emb_is_graph=True)
            fb_fuzzy_scores = fb_fuzzy_r, fb_fuzzy_t
            if eval_tSNE:
                data_fb_fuzzy_tSNE_pca, data_fb_fuzzy_tSNE_lap = global_scores(X, TopOGraph.fb_fuzzy_tSNE)
                fb_fuzzy_tSNE_r, fb_fuzzy_tSNE_t = local_scores(X, TopOGraph.fb_fuzzy_tSNE)
                data_fb_fuzzy_tSNE_scores = data_fb_fuzzy_tSNE_pca, data_fb_fuzzy_tSNE_lap, fb_fuzzy_tSNE_r, fb_fuzzy_tSNE_t
            if eval_MAP:
                data_fb_fuzzy_MAP_pca, data_fb_fuzzy_MAP_lap = global_scores(X, TopOGraph.fb_fuzzy_MAP)
                fb_fuzzy_MAP_r, fb__MAP_t = local_scores(X, TopOGraph.fb_fuzzy_MAP)
                data_fb_fuzzy_MAP_scores = data_fb_fuzzy_MAP_pca, data_fb_fuzzy_MAP_lap, fb_fuzzy_MAP_r, fb__MAP_t
            if eval_MDE:
                mde = TopOGraph.MDE()
                data_fb_fuzzy_MDE_pca, data_fb_fuzzy_MDE_lap = global_scores(X, TopOGraph.fb_fuzzy_MDE)
                fb_fuzzy_MDE_r, fb_fuzzy_MDE_t = local_scores(X, TopOGraph.fb_fuzzy_MDE)
                data_fb_fuzzy_MDE_scores = data_fb_fuzzy_MDE_pca, data_fb_fuzzy_MDE_lap, fb_fuzzy_MDE_r, fb_fuzzy_MDE_t
            if eval_TriMAP:
                data_fb_fuzzy_TriMAP_pca, data_fb_fuzzy_TriMAP_lap = global_scores(X, TopOGraph.fb_fuzzy_TriMAP)
                fb_fuzzy_TriMAP_r, fb_fuzzy_TriMAP_t = local_scores(X, TopOGraph.fb_fuzzy_TriMAP)
                data_fb_fuzzy_TriMAP_scores = data_fb_fuzzy_TriMAP_pca, data_fb_fuzzy_TriMAP_lap, fb_fuzzy_TriMAP_r, fb_fuzzy_TriMAP_t
            if eval_PaCMAP:
                data_fb_fuzzy_PaCMAP_pca, data_fb_fuzzy_PaCMAP_lap = global_scores(X, TopOGraph.fb_fuzzy_PaCMAP)
                fb_fuzzy_PaCMAP_r, fb_fuzzy_PaCMAP_t = local_scores(X, TopOGraph.fb_fuzzy_PaCMAP)
                data_fb_fuzzy_PaCMAP_scores = data_fb_fuzzy_PaCMAP_pca, data_fb_fuzzy_PaCMAP_lap, fb_fuzzy_PaCMAP_r, fb_fuzzy_PaCMAP_t



    from sklearn.decomposition import PCA
    pca_emb = PCA(n_components=TopOGraph.n_eigs)
    pca_pca, pca_lap = global_scores(X, pca_emb)
    pca_r, pca_t = local_scores(X, pca_emb)

    embedding_scores = {'PCA_scores' : (pca_pca, pca_lap, pca_r, pca_t) }
    graph_scores = {}
    layout_scores = {}

    if eval_db:
        embedding_scores['DB '] = db_scores
        if eval_diff:
            graph_scores['db_diff'] = db_diff_scores
            if eval_tSNE:
                layout_scores['db_diff_tSNE'] = data_db_diff_tSNE_scores
            if eval_MAP:
                layout_scores['db_diff_MAP'] = data_db_diff_MAP_scores
            if eval_MDE:
                layout_scores['db_diff_MDE'] = data_db_diff_MDE_scores
            if eval_PaCMAP:
                layout_scores['db_diff_PaCMAP'] = data_db_diff_PaCMAP_scores
            if eval_TriMAP:
                layout_scores['db_diff_TriMAP'] = data_db_diff_TriMAP_scores
        if eval_cknn:
            graph_scores['db_cknn'] = db_cknn_scores
            if eval_tSNE:
                layout_scores['db_cknn_tSNE'] = data_db_cknn_tSNE_scores
            if eval_MAP:
                layout_scores['db_cknn_MAP'] = data_db_cknn_MAP_scores
            if eval_MDE:
                layout_scores['db_cknn_MDE'] = data_db_cknn_MDE_scores
            if eval_PaCMAP:
                layout_scores['db_cknn_PaCMAP'] = data_db_cknn_PaCMAP_scores
            if eval_TriMAP:
                layout_scores['db_cknn_TriMAP'] = data_db_cknn_TriMAP_scores
        if eval_fuzzy:
            graph_scores['db_fuzzy'] = db_fuzzy_scores
            if eval_tSNE:
                layout_scores['db_fuzzy_tSNE'] = data_db_fuzzy_tSNE_scores
            if eval_MAP:
                layout_scores['db_fuzzy_MAP'] = data_db_fuzzy_MAP_scores
            if eval_MDE:
                layout_scores['db_fuzzy_MDE'] = data_db_fuzzy_MDE_scores
            if eval_PaCMAP:
                layout_scores['db_fuzzy_PaCMAP'] = data_db_fuzzy_PaCMAP_scores
            if eval_TriMAP:
                layout_scores['db_fuzzy_TriMAP'] = data_db_fuzzy_TriMAP_scores

    if eval_cb:
        embedding_scores['CB'] = cb_scores
        if eval_diff:
            graph_scores['cb_diff'] = cb_diff_scores
            if eval_tSNE:
                layout_scores['cb_diff_tSNE'] = data_cb_diff_tSNE_scores
            if eval_MAP:
                layout_scores['cb_diff_MAP'] = data_cb_diff_MAP_scores
            if eval_MDE:
                layout_scores['cb_diff_MDE'] = data_cb_diff_MDE_scores
            if eval_PaCMAP:
                layout_scores['cb_diff_PaCMAP'] = data_cb_diff_PaCMAP_scores
            if eval_TriMAP:
                layout_scores['cb_diff_TriMAP'] = data_cb_diff_TriMAP_scores
        if eval_cknn:
            graph_scores['cb_cknn'] = cb_cknn_scores
            if eval_tSNE:
                layout_scores['cb_cknn_tSNE'] = data_cb_cknn_tSNE_scores
            if eval_MAP:
                layout_scores['cb_cknn_MAP'] = data_cb_cknn_MAP_scores
            if eval_MDE:
                layout_scores['cb_cknn_MDE'] = data_cb_cknn_MDE_scores
            if eval_PaCMAP:
                layout_scores['cb_cknn_PaCMAP'] = data_cb_cknn_PaCMAP_scores
            if eval_TriMAP:
                layout_scores['cb_cknn_TriMAP'] = data_cb_cknn_TriMAP_scores
        if eval_fuzzy:
            graph_scores['cb_fuzzy'] = cb_fuzzy_scores
            if eval_tSNE:
                layout_scores['cb_fuzzy_tSNE'] = data_cb_fuzzy_tSNE_scores
            if eval_MAP:
                layout_scores['cb_fuzzy_MAP'] = data_cb_fuzzy_MAP_scores
            if eval_MDE:
                layout_scores['cb_fuzzy_MDE'] = data_cb_fuzzy_MDE_scores
            if eval_PaCMAP:
                layout_scores['cb_fuzzy_PaCMAP'] = data_cb_fuzzy_PaCMAP_scores
            if eval_TriMAP:
                layout_scores['cb_fuzzy_TriMAP'] = data_cb_fuzzy_TriMAP_scores

    if eval_fb:
        embedding_scores['FB'] = fb_scores
        if eval_diff:
            graph_scores['fb_diff'] = fb_diff_scores
            if eval_tSNE:
                layout_scores['fb_diff_tSNE'] = data_fb_diff_tSNE_scores
            if eval_MAP:
                layout_scores['fb_diff_MAP'] = data_fb_diff_MAP_scores
            if eval_MDE:
                layout_scores['fb_diff_MDE'] = data_fb_diff_MDE_scores
            if eval_PaCMAP:
                layout_scores['fb_diff_PaCMAP'] = data_fb_diff_PaCMAP_scores
            if eval_TriMAP:
                layout_scores['fb_diff_TriMAP'] = data_fb_diff_TriMAP_scores
        if eval_cknn:
            graph_scores['fb_cknn'] = fb_cknn_scores
            if eval_tSNE:
                layout_scores['fb_cknn_tSNE'] = data_fb_cknn_tSNE_scores
            if eval_MAP:
                layout_scores['fb_cknn_MAP'] = data_fb_cknn_MAP_scores
            if eval_MDE:
                layout_scores['fb_cknn_MDE'] = data_fb_cknn_MDE_scores
            if eval_PaCMAP:
                layout_scores['fb_cknn_PaCMAP'] = data_fb_cknn_PaCMAP_scores
            if eval_TriMAP:
                layout_scores['fb_cknn_TriMAP'] = data_fb_cknn_TriMAP_scores
        if eval_fuzzy:
            graph_scores['fb_fuzzy'] = fb_fuzzy_scores
            if eval_tSNE:
                layout_scores['fb_fuzzy_tSNE'] = data_fb_fuzzy_tSNE_scores
            if eval_MAP:
                layout_scores['fb_fuzzy_MAP'] = data_fb_fuzzy_MAP_scores
            if eval_MDE:
                layout_scores['fb_fuzzy_MDE'] = data_fb_fuzzy_MDE_scores
            if eval_PaCMAP:
                layout_scores['fb_fuzzy_PaCMAP'] = data_fb_fuzzy_PaCMAP_scores
            if eval_TriMAP:
                layout_scores['fb_fuzzy_TriMAP'] = data_fb_fuzzy_TriMAP_scores

    return embedding_scores, graph_scores, layout_scores

