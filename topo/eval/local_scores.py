from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, kendalltau
from scipy.sparse.csgraph import shortest_path
import numpy as np

def geodesic_distance(data, method='J', unweighted=False):
    G = shortest_path(data, method=method, unweighted=unweighted)
    # guarantee symmetry
    G = (G + G.T) / 2
    G[(np.arange(G.shape[0]), np.arange(G.shape[0]))] = 0
    return G

def knn_spearman_r(data_graph, embedding_graph, path_method='J', subsample_idx=None, unweighted=False):
    # data_graph is a (N,N) similarity matrix from the reference high-dimensional data
    # embedding_graph is a (N,N) similarity matrix from the lower dimensional embedding
    geodesic_dist = geodesic_distance(data_graph, method=path_method, unweighted=unweighted)
    if subsample_idx is not None:
        geodesic_dist = geodesic_dist[subsample_idx, :][:, subsample_idx]
    embedded_dist = geodesic_distance(embedding_graph, method=path_method, unweighted=unweighted)
    res, _ = spearmanr(squareform(geodesic_dist), squareform(embedded_dist))
    return res

def knn_kendall_tau(data_graph, embedding_graph, path_method='J', subsample_idx=None, unweighted=False):
    geodesic_dist = geodesic_distance(data_graph, method=path_method, unweighted=unweighted)
    if subsample_idx is not None:
        geodesic_dist = geodesic_dist[subsample_idx, :][:, subsample_idx]
    embedded_dist = geodesic_distance(embedding_graph, method=path_method, unweighted=unweighted)
    res, _ = kendalltau(squareform(geodesic_dist), squareform(embedded_dist))
    return res