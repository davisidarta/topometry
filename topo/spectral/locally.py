# import numpy as np
# from sklearn.neighbors import kneighbors_graph
# from scipy.sparse.csgraph import laplacian
# from scipy.sparse.linalg import eigsh

# # Load data
# X = np.loadtxt('data.csv', delimiter=',')

# # Construct neighborhood graph
# k = 10  # number of nearest neighbors
# graph = kneighbors_graph(X, k, mode='connectivity', include_self=False)

# # Compute graph Laplacian
# laplacian = laplacian(graph, normed=True)

# # Compute local intrinsic dimensionality
# n_components = 5  # number of eigenvalues to compute
# eigenvalues = np.zeros((X.shape[0], n_components))
# for i in range(X.shape[0]):
#     indices = graph.indices[graph.indptr[i]:graph.indptr[i+1]]
#     L = laplacian[indices][:, indices]
#     w, v = eigsh(L, k=n_components, which='SM')
#     eigenvalues[i] = w
# local_dim = np.mean(np.sum(eigenvalues > 1e-8, axis=1))