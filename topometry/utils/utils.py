# Some utility functions
import numpy as np
from topometry.base import ann

def hyperboloid_emb(emb):
    z = np.sqrt(1 + np.sum(emb ** 2, axis=1))
    emb[:, 2] = z

    return emb


def knn_graph(data, k=15, metric='cosine', n_jobs=10, M=30, efC=100, efS=100):
    knn = ann.NMSlibTransformer(n_neighbors=k,
                                metric=metric,
                                method='hnsw',
                                n_jobs=n_jobs,
                                M=M,
                                efC=efC,
                                efS=efS).fit(data).transform(data)
    return knn

def fast_ind_dist(data, k=15, metric='cosine', n_jobs=10, M=30, efC=100, efS=100):
    nbrs = ann.NMSlibTransformer(n_neighbors=k,
                                metric=metric,
                                method='hnsw',
                                n_jobs=n_jobs).fit(data)
    ind, dist = nbrs.ind_dist_grad(data, return_grad=False, return_graph=False)
    return ind, dist