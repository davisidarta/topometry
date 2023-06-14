#####################################
# Author: Davi Sidarta-Oliveira
# School of Medical Sciences,University of Campinas,Brazil
# contact: davisidarta[at]fcm[dot]unicamp[dot]com
# License: MIT
######################################
# Defining graph kernels in a scikit-learn fashion

import numpy as np
from scipy.sparse import find, csr_matrix, linalg, lil_matrix, csc_matrix, kron, coo_matrix, tril, diags
from scipy.sparse.csgraph import connected_components
from scipy.spatial import procrustes
from scipy.stats import rv_discrete
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator, TransformerMixin
from topo.base.ann import kNN
from topo.base.dists import pairwise_distances
from topo.spectral._spectral import graph_laplacian, diffusion_operator
from topo.spectral._spectral import degree as compute_degree
from topo.tpgraph.cknn import cknn_graph
from topo.tpgraph.fuzzy import fuzzy_simplicial_set
from topo.utils._utils import get_indices_distances_from_sparse_matrix
import warnings
# dumb warning, suggests lilmatrix but it doesnt work
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)





def _adap_bw(K, n_neighbors):
    median_k = np.floor(n_neighbors/2).astype(int)
    adap_sd = np.zeros(K.shape[0])
    for i in np.arange(len(adap_sd)):
        adap_sd[i] = np.sort(
            K.data[K.indptr[i]: K.indptr[i + 1]])[median_k - 1]
    return adap_sd


def compute_kernel(X, metric='cosine',
                   n_neighbors=10, fuzzy=False, cknn=False, delta=1.0, pairwise=False, sigma=None, adaptive_bw=True,
                   expand_nbr_search=False, alpha_decaying=False, return_densities=False, symmetrize=True,
                   backend='nmslib', n_jobs=-1, verbose=False, **kwargs):
    """
    Compute a kernel matrix from a set of points.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features).
        The set of points to compute the kernel matrix for. Accepts np.ndarrays and scipy.sparse matrices.
        If precomputed, assumed to be a square symmetric semidefinite matrix.

    metric : string (optional, default 'cosine').
        The metric to use when computing the kernel matrix.
        Possible values are: 'cosine', 'euclidean' and others. Accepts precomputed distances.

    n_neighbors : int (optional, default 10).
        The number of neighbors to use when computing the kernel matrix. Ignored if `pairwise` set to `True`.

    fuzzy : bool (optional, default False).
        Whether to build the kernel matrix using fuzzy simplicial sets, similarly to UMAP. 
        If set to `True`, the `pairwise`, `sigma`, `adaptive_bw`, `expand_nbr_search` and `alpha_decaying` parameters are ignored.
        If set to `True` at the same time that `cknn` is set to `True`, the `cknn` parameter is ignored.

    cknn : bool (optional, default False).
        Whether to build the adjacency and affinity matrices using continuous k-nearest-neighbors. 
        If set to `True`, the `pairwise`, `sigma`, `adaptive_bw`, `expand_nbr_search` and `alpha_decaying` parameters are ignored.

    delta : float (optional, default 1.0).
        The scaling factor for the CkNN kernel. Ignored if `cknn` set to `False`.

    pairwise : bool (optional, default False).
        Whether to compute the kernel using dense pairwise distances. 
        If set to `True`, the `n_neighbors` and `backend` parameters are ignored.
        Uses `numba` for computations if available. If not, uses `sklearn`.

    sigma : float (optional, default None).
        Scaling factor if using fixed bandwidth kernel (only used if `adaptive_bw` is set to `False`).

    adaptive_bw : bool (optional, default True).
        Whether to use an adaptive bandwidth based on the distance to median k-nearest-neighbor.

    expand_nbr_search : bool (optional, default False).
        Whether to expand the neighborhood search (mitigates a choice of too small a number of k-neighbors).

    alpha_decaying : bool (optional, default False).
        Whether to use an adaptively decaying kernel.

    return_densities : bool (optional, default False).
        Whether to return the bandwidth metrics as a dictinary. If set to `True`, the function
        returns a tuple containing the kernel matrix and a dictionary containing the
        bandwidth metric.

    symmetrize : bool (optional, default True).
        Whether to symmetrize the kernel matrix after normalizations.

    backend : str (optional, default 'nmslib').
        Which backend to use for neighborhood computations. Defaults to 'nmslib'.
        Options are 'nmslib', 'hnswlib', 'faiss', 'annoy' and 'sklearn'. 

    n_jobs : int (optional, default 1).
        The number of jobs to use for parallel computations. If set to -1, all available cores are used.

    verbose : bool (optional, default False).
        Whether to print progress messages.

    **kwargs : dict, optional
        Additional arguments to be passed to the nearest-neighbors backend.

    Returns
    -------
    K : array-like, shape (n_samples, n_samples)
        The kernel matrix.

    densities : dict, optional (if `return_densities` is set to `True`)
        If `fuzzy` and `cknn` are `False`, is a dictionary containing the bandwidth metrics. 
        If `fuzzy` is set to `True`, the dictionary contains sigma and rho estimates.
        If `cknn` is set to `True`, the dictionary contains the bandwidth metric.
    """
    dens_dict = {}
    N = np.shape(X)[0]
    if n_jobs == -1:
        from joblib import cpu_count
        n_jobs = cpu_count()
    k = n_neighbors
    if metric == 'precomputed':
        K = X
        expand_nbr_search = False
    else:
        if pairwise:
            K = pairwise_distances(X, metric)
        else:
            K = kNN(X, metric=metric, n_neighbors=k,
                    backend=backend, n_jobs=n_jobs, **kwargs)
        if return_densities:
            dens_dict['knn'] = K
    if fuzzy:
        cknn = False
        W, sigmas, rhos = fuzzy_simplicial_set(K,
                                               n_neighbors=n_neighbors,
                                               metric='precomputed',
                                               set_op_mix_ratio=1.0,
                                               local_connectivity=1.0,
                                               apply_set_operations=True,
                                               return_dists=False,
                                               verbose=verbose)
        if return_densities:
            dens_dict['sigma'] = sigmas
            dens_dict['rho'] = rhos
    elif cknn:
        A, W, adap_sd = cknn_graph(K, n_neighbors=n_neighbors,
                                   delta=delta,
                                   metric='precomputed',
                                   weighted=None,
                                   include_self=False,
                                   return_densities=True,
                                   verbose=verbose)
        if return_densities:
            dens_dict['unweighted_adjacency'] = A
            dens_dict['adaptive_bw'] = adap_sd
    else:
        if adaptive_bw:
            adap_sd = _adap_bw(K, k)
            # Get an indirect measure of the local density
            pm = np.interp(adap_sd, (adap_sd.min(), adap_sd.max()), (2, k))
            if return_densities:
                dens_dict['omega'] = pm
                dens_dict['adaptive_bw'] = adap_sd
            if expand_nbr_search:
                new_k = int(k + (k - pm.max()))
                new_K = kNN(X, metric=metric, n_neighbors=new_k,
                            backend=backend, n_jobs=n_jobs, **kwargs)
                adap_sd_new = _adap_bw(new_K, new_k)
                x_new, y_new, dists_new = find(new_K)
                # Get an indirect measure of the local density
                pm_new = np.interp(
                    adap_sd_new, (adap_sd_new.min(), adap_sd_new.max()), (2, new_k))
                if return_densities:
                    dens_dict['expanded_k_neighbor'] = new_k
                    dens_dict['omega_nbr_expanded'] = adap_sd_new
                    dens_dict['adaptive_bw_nbr_expanded'] = adap_sd_new
                    dens_dict['expanded_neighborhood_graph'] = new_K
                    dens_dict['knn_expanded'] = new_K
        x, y, dists = find(K)
        # Normalize distances
        if adaptive_bw:
            # Alpha decaying: the kernel adaptively decays depending on neighborhood density
            if alpha_decaying:
                if expand_nbr_search:
                    dists = (
                        dists_new / (adap_sd_new[x] + 1e-10)) ** np.power(2, ((new_k - pm_new[x]) / pm_new[x]))
                else:
                    dists = (dists / (adap_sd[x] + 1e-10)
                             ) ** np.power(2, ((k - pm[x]) / pm[x]))
            else:
                if expand_nbr_search:
                    dists = (dists_new / (adap_sd_new[x] + 1e-10))
                else:
                    dists = (dists / (adap_sd[x] + 1e-10))
        else:
            if sigma is not None:
                if sigma == 0:
                    sigma = 1e-10
                dists = (dists / sigma) ** 2
        W = csr_matrix((np.abs(np.exp(-dists)), (x, y)), shape=[N, N])
    if symmetrize:
        W = (W + W.T) / 2
    #W[(np.arange(N), np.arange(N))] = 0
    # handle nan
    W.data = np.where(np.isnan(W.data), 1, W.data)
    if not return_densities:
        return W
    else:
        return W, dens_dict


class Kernel(BaseEstimator, TransformerMixin):
    """
    Scikit-learn flavored class for computing a kernel matrix from a set of points. Includes functions
    for computing the kernel matrix with a variety of methods (adaptive bandwidth, fuzzy simplicial sets,
    continuous k-nearest-neighbors, etc) and performing operations
    on the resulting graph, such as obtaining its Laplacian, sparsifying it, filtering and interpolating signals,
    obtaining diffusion operators, imputing missing values and computing shortest paths.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The set of points to compute the kernel matrix for. Accepts np.ndarrays and scipy.sparse matrices.
        If precomputed, assumed to be a square symmetric semidefinite matrix.

    metric : string (optional, default 'cosine')
        The metric to use when computing the kernel matrix.
        Possible values are: 'cosine', 'euclidean' and others, depending on the chosen nearest-neighbors backend. Accepts precomputed distances as 'precomputed'.

    n_neighbors : int (optional, default 10).
        The number of neighbors to use when computing the kernel matrix. Ignored if `pairwise` set to `True`.

    fuzzy : bool, optional (default False).
        Whether to build the kernel matrix using fuzzy simplicial sets, similarly to UMAP. 
        If set to `True`, the `pairwise`, `sigma`, `adaptive_bw`, `expand_nbr_search` and `alpha_decaying` parameters are ignored.
        If set to `True` at the same time that `cknn` is set to `True`, the `cknn` parameter is ignored.

    cknn : bool, optional (default False).
        Whether to build the adjacency and affinity matrices using continuous k-nearest-neighbors. 
        If set to `True`, the `pairwise`, `sigma`, `adaptive_bw`, `expand_nbr_search` and `alpha_decaying` parameters are ignored.
        If set to `True`, `laplacian_type` is automatically set to 'unnormalized'.

    pairwise : bool (optional, default False).
        Whether to compute the kernel using dense pairwise distances. 
        If set to `True`, the `n_neighbors` and `backend` parameters are ignored.
        Uses `numba` for computations if available. If not, uses `sklearn`.

    sigma : float (optional, default None)
        Scaling factor if using fixed bandwidth kernel (only used if `adaptive_bw` is set to `False`).

    adaptive_bw : bool (optional, default True).
        Whether to use an adaptive bandwidth based on the distance to median k-nearest-neighbor.

    expand_nbr_search : bool (optional, default False).
        Whether to expand the neighborhood search (mitigates a choice of too small a number of k-neighbors).

    alpha_decaying : bool (optional, default False).
        Whether to use an adaptively decaying kernel.

    laplacian_type : string (optional, default 'normalized')
        The type of Laplacian to compute. Possible values are: 'normalized', 'unnormalized', 'random_walk' and 'geometric'.

    anisotropy : float (optional, default 0).
        The anisotropy (alpha) parameter in the diffusion maps literature for kernel reweighting.

    semi_aniso : bool (optional, default False).
        Whether to use semi-anisotropic diffusion. This reweights the original kernel (not the renormalized kernel) by the renormalized degree.

    symmetrize : bool (optional, default True).
        Whether to symmetrize the kernel matrix after normalizations.

    backend : str (optional, default 'nmslib').
        Which backend to use for k-nearest-neighbor computations. Defaults to 'nmslib'.
        Options are 'nmslib', 'hnswlib', 'faiss', 'annoy' and 'sklearn'.   

    n_jobs : int (optional, default 1).
        The number of jobs to use for parallel computations. If -1, all CPUs are used.
        Parallellization (multiprocessing) is ***highly*** recommended whenever possible.

    laplacian_type : str (optional, default 'normalized').
            The type of laplacian to use. Can be 'unnormalized', 'normalized', or 'random_walk'.


    Properties
    ----------
    knn : scipy.sparse.csr_matrix, shape (n_samples, n_samples)
        The computed k-nearest-neighbors graph.

    A : scipy.sparse.csr_matrix, shape (n_samples, n_samples)
        The adjacency matrix of the kernel graph.

    K : scipy.sparse.csr_matrix, shape (n_samples, n_samples)
        The kernel matrix.

    dens_dict : dict
        Dictionary containing the density information for each point in the dataset for the computed kernel.

    L : scipy.sparse.csr_matrix, shape (n_samples, n_samples)
        The laplacian matrix of the kernel graph.

    P : scipy.sparse.csr_matrix, shape (n_samples, n_samples)
        The diffusion operator of the kernel graph.

    SP : scipy.sparse.csr_matrix, shape (n_samples, n_samples)
        The shortest-path matrix.

    degree : np.ndarray, shape (n_samples,)
        The degree of each point in the adjacency graph.

    weighted_degree : np.ndarray, shape (n_samples,)
        The weighted degree of each point in the kernel graph.

    """

    def __init__(self,
                 metric='cosine',
                 n_neighbors=10,
                 fuzzy=False,
                 cknn=False,
                 pairwise=False,
                 sigma=None,
                 adaptive_bw=True,
                 expand_nbr_search=False,
                 alpha_decaying=False,
                 symmetrize=True,
                 backend='nmslib',
                 n_jobs=1,
                 laplacian_type='normalized',
                 anisotropy=1,
                 semi_aniso=False,
                 n_landmarks=None,
                 cache_input=False,
                 verbose=False,
                 random_state=None
                 ):
        self.n_neighbors = n_neighbors
        self.fuzzy = fuzzy
        self.cknn = cknn
        self.pairwise = pairwise
        self.n_jobs = n_jobs
        self.backend = backend
        self.metric = metric
        self.sigma = sigma
        self.semi_aniso = semi_aniso
        self.adaptive_bw = adaptive_bw
        self.expand_nbr_search = expand_nbr_search
        self.alpha_decaying = alpha_decaying
        self.symmetrize = symmetrize
        self.n_landmarks = n_landmarks
        self.laplacian_type = laplacian_type
        self.cache_input = cache_input
        self.verbose = verbose
        self.random_state = random_state
        self.X = None
        self._K = None
        self.N = None
        self.M = None
        self.dens_dict = None
        self._clusters = None
        self._A = None
        self._Dd = None
        self._degree = None
        self._weighted_degree = None
        self._L = None
        self._SP = None
        self._P = None
        self._connected = None
        self.D_inv_sqrt_ = None
        self.components_ = None
        self.components_indices_ = None
        self.sigma_ = None
        self.rho_ = None
        self.adaptive_bw_ = None
        self.omega_ = None
        self.expanded_k_neighbor_ = None
        self.adaptive_bw_nbr_expanded_ = None
        self.omega_nbr_expanded_ = None
        self._sample_densities = None
        self.knn_ = None
        self.dens_dict = None
        self.anisotropy = anisotropy
        self._have_hnswlib = None
        self._have_nmslib = None
        self._have_annoy = None
        self._have_faiss = None

    def __repr__(self):
        if self._K is not None:
            if self.metric == 'precomputed':
                msg = "Kernel() estimator fitted with precomputed distance matrix"
            elif (self.N is not None) and (self.M is not None):
                msg = "Kernel() estimator fitted with %i samples and %i observations" % (
                    self.N, self.M)
            else:
                msg = "Kernel() estimator without valid fitted data."
        else:
            msg = "Kernel() estimator without any fitted data."
        if self._K is not None:
            if self.fuzzy:
                kernel_msg = " using fuzzy simplicial sets."
            elif self.cknn:
                kernel_msg = " using continuous k-nearest-neighbors."
            else:
                if not self.adaptive_bw:
                    kernel_msg = " using a kernel with fixed bandwidth sigma = %.2f" % self.sigma
                else:
                    kernel_msg = " using a kernel with adaptive bandwidth "
                    if self.alpha_decaying:
                        kernel_msg = kernel_msg + "and adaptive decay"
                    if self.expand_nbr_search:
                        kernel_msg = kernel_msg + ", with expanded neighborhood search."
            msg = msg + kernel_msg
        return msg

    def _parse_backend(self):
        try:
            import hnswlib
            self._have_hnswlib = True
        except ImportError:
            self._have_hnswlib = False
        try:
            import nmslib
            self._have_nmslib = True
        except ImportError:
            self._have_nmslib = False
        try:
            import annoy
            self._have_annoy = True
        except ImportError:
            self._have_annoy = False
        try:
            import faiss
            self._have_faiss = True
        except ImportError:
            self._have_faiss = False

        if self.backend == 'hnswlib':
            if not self._have_hnswlib:
                if self._have_nmslib:
                    self.backend = 'nmslib'
                elif self._have_annoy:
                    self.backend = 'annoy'
                elif self._have_faiss:
                    self.backend = 'faiss'
                else:
                    self.backend = 'sklearn'
        elif self.backend == 'nmslib':
            if not self._have_nmslib:
                if self._have_hnswlib:
                    self.backend = 'hnswlib'
                elif self._have_annoy:
                    self.backend = 'annoy'
                elif self._have_faiss:
                    self.backend = 'faiss'
                else:
                    self.backend = 'sklearn'
        elif self.backend == 'annoy':
            if not self._have_annoy:
                if self._have_nmslib:
                    self.backend = 'nmslib'
                elif self._have_hnswlib:
                    self.backend = 'hnswlib'
                elif self._have_faiss:
                    self.backend = 'faiss'
                else:
                    self.backend = 'sklearn'
        elif self.backend == 'faiss':
            if not self._have_faiss:
                if self._have_nmslib:
                    self.backend = 'nmslib'
                elif self._have_hnswlib:
                    self.backend = 'hnswlib'
                elif self._have_annoy:
                    self.backend = 'annoy'
                else:
                    self.backend = 'sklearn'
        else:
            print(
                "Warning: no approximate nearest neighbor library found. Using sklearn's KDTree instead.")
            self.backend == 'sklearn'

    def fit(self, X, recompute=False, **kwargs):
        """
        Fits the kernel matrix to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features).
            Input data. Takes in numpy arrays and scipy csr sparse matrices.
            Use with sparse data for top performance. You can adjust a series of
            parameters that can make the process faster and more informational depending
            on your dataset.

        recompute : bool (optional, default False).
            Whether to recompute the kernel if it has already been computed.

        **kwargs : dict, optional
            Additional arguments to be passed to the k-nearest-neighbor backend.

        Returns
        -------
            The kernel matrix.

        """
        self._parse_backend()
        self.random_state = check_random_state(self.random_state)
        if self.fuzzy:
            self.cknn = False
        if self.cache_input:
            self.X = X
        self.N, self.M = X.shape
        if self._K is None or (self._K is not None and recompute):
            self._K, self.dens_dict = compute_kernel(X, metric=self.metric, fuzzy=self.fuzzy, cknn=self.cknn, pairwise=self.pairwise,
                                                     n_neighbors=self.n_neighbors, sigma=self.sigma, adaptive_bw=self.adaptive_bw,
                                                     expand_nbr_search=self.expand_nbr_search, alpha_decaying=self.alpha_decaying, return_densities=True, symmetrize=self.symmetrize,
                                                     backend=self.backend, n_jobs=self.n_jobs, verbose=self.verbose, **kwargs)
        if self.metric != 'precomputed':
            self.knn_ = self.dens_dict['knn']
        if self.fuzzy:
            self.sigma_ = self.dens_dict['sigma']
            self.rho_ = self.dens_dict['rho']
        elif self.cknn:
            self._A = self.dens_dict['unweighted_adjacency']
            self.adaptive_bw_ = self.dens_dict['adaptive_bw']
        else:
            if self.adaptive_bw:
                self.adaptive_bw_ = self.dens_dict['adaptive_bw']
                self.omega_ = self.dens_dict['omega']
            if self.expand_nbr_search:
                self.expanded_k_neighbor_ = self.dens_dict['expanded_k_neighbor']
                self.adaptive_bw_nbr_expanded_ = self.dens_dict['adaptive_bw_nbr_expanded']
                self.omega_nbr_expanded_ = self.dens_dict['omega_nbr_expanded']
        return self

    def transform(self, X=None):
        """
        Returns the kernel matrix. Here for compability with scikit-learn only.
        """
        if self._K is None:
            raise ValueError(
                "No kernel matrix has been fitted yet. Call fit() first.")

        return self._K

    def fit_transform(self, X, y=None):
        """
        Fits the kernel matrix to the data and returns the kernel matrix.
        Here for compability with scikit-learn only.
        """
        self.fit(X)
        return self._K

    def adjacency(self):
        """
        Graph adjacency matrix (the binary version of W).

        The adjacency matrix defines which edges exist on the graph.
        It is represented as an N-by-N matrix of booleans.
        :math:`A_{i,j}` is True if :math:`W_{i,j} > 0`.
        """
        if self._K is None:
            raise ValueError(
                "No kernel matrix has been fitted yet. Call fit() first.")
        self._A = (self._K > 0).astype(int)
        self._A = self._A.astype(float)
        return self._A

    @property
    def knn(self):
        """
        Returns the k-nearest-neighbors graph.
        """
        if self.knn_ is None:
            raise ValueError(
                "No k-nearest-neighbors graph has been fitted yet or precomputed versions were used!")
        return self.knn_

    @property
    def K(self):
        """
        Kernel matrix.
        """
        if self._K is None:
            raise ValueError(
                "No kernel matrix has been fitted yet. Call fit() first.")
        return self._K

    @property
    def A(self):
        """
        Graph adjacency matrix.
        """
        if self._K is None:
            raise ValueError(
                "No kernel matrix has been fitted yet. Call fit() first.")
        if self._A is None:
            self._A = self.adjacency()
        return self._A

    @property
    def degree(self):
        """
        Returns the degree of the adjacency matrix.
        """

        if self._degree is None:
            if self._K is None:
                raise ValueError(
                    "No kernel matrix has been fitted yet. Call fit() first.")
            self._degree = compute_degree(self._A)
        return self._degree

    @property
    def weighted_degree(self):
        """
        Returns the degree of the weighted affinity matrix.
        """

        if self._weighted_degree is None:
            if self._K is None:
                raise ValueError(
                    "No kernel matrix has been fitted yet. Call fit() first.")
            self._weighted_degree = compute_degree(self._K)
        return self._weighted_degree

    def laplacian(self, laplacian_type=None):
        """
        Compute the graph Laplacian, given a adjacency or affinity graph W. For a friendly reference,
        see this material from James Melville: https://jlmelville.github.io/smallvis/spectral.html

        Parameters
        ----------
        laplacian_type : str (optional, default None).
            The type of laplacian to use. Can be 'unnormalized', 'normalized', or 'random_walk'. If not provided, will use
            the default laplacian type specified in the constructor.

        Returns
        -------
        L : scipy.sparse.csr_matrix
            The graph Laplacian.
        """
        if self._L is None:
            if laplacian_type is None:
                laplacian_type = self.laplacian_type
            self._L, self._Dd = graph_laplacian(self.K, laplacian_type=laplacian_type, return_D=True)
        return self._L

    @property
    def L(self):
        """
        Object synonym for the laplacian() function.
        """
        if self._L is None:
            return self.laplacian()
        return self._L

    def diff_op(self, anisotropy=1.0, symmetric=False):
        """
        Computes the [diffusion operator](https://doi.org/10.1016/j.acha.2006.04.006).

        Parameters
        ----------
        anisotropy : float (optional, default None).
            Anisotropy to apply. 'Alpha' in the diffusion maps literature. Defaults to the anisotropy defined in the constructor.

        symmetric : bool (optional, default False).
            Whether to use a symmetric version of the diffusion operator. This is particularly useful to yield a symmetric operator
            when using anisotropy (alpha > 1), as the diffusion operator P would be assymetric otherwise, which can be problematic
            during matrix decomposition. Eigenvalues are the same of the assymetric version, and the eigenvectors of the original assymetric
            operator can be obtained by left multiplying by Kernel.D_inv_sqrt_.

        Returns
        -------
        P : scipy.sparse.csr_matrix
            The graph diffusion operator.

        Populates the Kernel.D_inv_sqrt_ slot.

        """
        if anisotropy is None:
            anisotropy = self.anisotropy
        if self._P is None:
            if self._K is None:
                raise ValueError(
                    "No kernel matrix has been fitted yet. Call fit() first.")
            if anisotropy is None or anisotropy < 0:
                anisotropy = 0
            if anisotropy > 1:
                anisotropy = 1.0
            if symmetric:
                self._P, self.D_inv_sqrt_ = diffusion_operator(
                    self.K, alpha=anisotropy, semi_aniso=self.semi_aniso, symmetric=symmetric)
            else:
                self._P = diffusion_operator(
                    self.K, alpha=anisotropy, semi_aniso=self.semi_aniso, symmetric=symmetric)
            self._P = (self._P + self._P.T) / 2
        return self._P

    @property
    def P(self):
        """
        Object synonym for the diff_op() function.
        """
        if self._P is None:
            return self.diff_op()
        return self._P

    def shortest_paths(self, landmark=False, indices=None):
        """
        Compute the shortest paths between all pairs of nodes.
        If landmark is True, the shortest paths are computed between all pairs of landmarks,
        not all sample nodes.

        Parameters
        ----------
        landmark : bool (optional, default False).
            If True, the shortest paths are computed between all pairs of landmarks,
            not all sample nodes.
        indices : list of int (optional, default None).
            If None, the shortest paths are computed between all pairs of nodes. Else,
            the shortest paths are computed between all pairs of nodes and nodes with specified indices.

        Returns
        -------
        D : scipy.sparse.csr_matrix
            The shortest paths matrix.
        """
        if self._SP is None:
            if self._K is None:
                raise ValueError(
                    "No kernel matrix has been fitted yet. Call fit() first.")
            if landmark:
                print('Landmarks are still to be implemented.')
            from topo.eval.local_scores import geodesic_distance
            SP = geodesic_distance(self._K, method='D', unweighted=False, directed=False, indices=None, n_jobs=self.n_jobs, random_state=self.random_state)
            SP = (SP + SP.T) / 2
            SP[np.where(SP == 0)] = np.inf
            SP[(np.arange(SP.shape[0]), np.arange(SP.shape[0]))] = 0
            self._SP = SP
        return self._SP

    @property
    def SP(self):
        """
        Object synonym for the shortest_paths() function.
        """
        if self._SP is None:
            return self.shortest_paths()
        return self._SP

    def get_indices_distances(self, n_neighbors=None, kernel=True):
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if kernel:
            return get_indices_distances_from_sparse_matrix(self.K, n_neighbors)
        else:
            return get_indices_distances_from_sparse_matrix(self.knn_, n_neighbors)

    def _calculate_imputation_error(
        self, data, data_prev=None
    ):
        """
        Calculates difference before and after imputation by diffusion. 
        This is from [MAGIC](https://github.com/KrishnaswamyLab/MAGIC).

        Parameters
        ----------
        data : array-like
            current data matrix
        data_prev : array-like, optional (default: None)
            previous data matrix. If None, `data` is simply prepared for
            comparison and no error is returned

        Returns
        -------
        error : float
            Procrustes disparity value
        data_curr : array-like
            transformed data to use for the next comparison
        """
        if data_prev is not None:
            _, _, error = procrustes(data_prev, data)
        else:
            error = None
        return error, data

    def impute(self, Y=None, t=None, threshold=0.01, tmax=10):
        """
        Uses the diffusion operator computed from graph build on X to impute the input data Y.
        Although the idea behind this is far older, it was first reported in single-cell genomics
        by the Krishnaswamy lab in the MAGIC (Markov Affinity-based Graph Imputation of Cells)
        [manuscript](https://www.cell.com/cell/abstract/S0092-8674(18)30724-4)

        Parameters
        ----------
        Y : np.ndarray (optional, default None).
            The input data to impute. If None, the input data X is imputed (if it was cached by setting
            the `cache_input` parameter to True). Otherwise, you'll have to specify it as Y.

        t : int (optional, default None).
            The number of steps to perform during diffusion. The default `None` iterates until the Procrustes disparity value 
            is below the `threshold` parameter.

        threshold : float (optional, default 0.001).
            The threshold value for the Procrustes disparity when finding an optimal `t`.

        tmax : int (optional, default 30).
            The maximum number of steps to perform during diffusion when estimating an optimal `t`.

        Returns
        -------
        Y_imp : np.ndarray
            The imputed data.
        """
        if self._P is None:
            self._P = self.diff_op()
        if Y is None:
            if self.X is None:
                raise ValueError(
                    "No input data has been fitted yet. Call fit() first with the parameter `cache_input` set to True.")
            Y = self.X.copy()
        if t is None or t < 0:
            i = 1
            while i < tmax:
                P_diffused = self._P ** i
                Y_imp = np.dot(P_diffused.toarray(), Y)
                error, _ = self._calculate_imputation_error(Y_imp, Y)
                i += 1
                if error < threshold:
                    t_opt = i
                    print('Optimal t: {}'.format(t_opt))
                    break

        else:
            P_diffused = self._P ** t
            Y_imp = np.dot(P_diffused.toarray(), Y)
        return Y_imp

    def _get_landmarks(self, X, n_landmarks=None):
        if n_landmarks is None:
            if self.n_landmarks is None:
                raise ValueError(
                    "No number of landmarks has been specified. Call fit() first with the parameter `n_landmarks` set to an integer.")
            n_landmarks = self.n_landmarks

    def filter(self, signal, replicates=None,
               beta=50,
               target=None,
               filterfunc=None,
               offset=0,
               order=1,
               solver="chebyshev",
               chebyshev_order=100):
        """
        This is inspired from [MELD](https://github.com/KrishnaswamyLab/MELD) to estimate sample associated density estimation.
        However, you can naturally use this for any signal in your data, not just samples of specific conditions. In practice, this is just
        a simple [PyGSP](https://pygsp.readthedocs.io/en/stable/reference/filters.html#module-pygsp.filters) filter on a graph.
        Indeed, it calls PyGSP, so you'll need it installed to use this function.

        Parameters
        ----------
        signal: array-like
            Signal(s) to filter - usually sample labels.

        replicates: array-like, optional (default None)
            Replicate labels for each sample. If None, no replicates are assumed.

        beta : int (optional, default 50).
            Amount of smoothing to apply. Vary this parameter to get good estimates
            - this can vary widely from dataset to dataset.

        target : array-like (optional, default None).
            Similarity matrix to use for filtering. If None, uses the kernel matrix.

        filterfunc : function (optional, default None).
            Function to use for filtering. If None, the default is to use a Laplacian filter.

        offset: float (optional, default 0)
            Amount to shift the filter in the eigenvalue spectrum.
            Recommended to use an eigenvalue from the graph based on the
            spectral distribution. Should be in interval [0,1]

        order: int (optional, default 1).
            Falloff and smoothness of the filter.
            High order leads to square-like filters.

        solver : string (optional, default 'chebyshev').
            Method to solve convex problem. If 'chebyshev', uses a chebyshev polynomial approximation of the corresponding
            filter. Else, if 'exact', uses the eigenvalue solution to the problem

        chebyshev_order : int (optional, default 100).
            Order of chebyshev approximation to use.

        """
        if self._sample_densities is None:
            try:
                from pygsp import graphs, filters
            except ImportError:
                raise ImportError(
                    "pygsp is not installed. Please install it with `pip install pygsp` to use filtering functions.")
            import pandas as pd
            # try converting signal labels
            sample_labels = signal.copy()
            samples = np.unique(sample_labels)
            if hasattr(sample_labels, "index"):
                _labels_index = sample_labels.index
            else:
                _labels_index = None
            try:
                labels = sample_labels.values
            except AttributeError:
                labels = sample_labels
            if len(labels.shape) > 1:
                if labels.shape[1] == 1:
                    labels = labels.reshape(-1)
                else:
                    raise ValueError(
                        "sample_labels must be a single column. Got"
                        "shape={}".format(labels.shape)
                    )
            if samples.shape[0] == 2:
                df = pd.DataFrame(
                    [labels == samples[0], labels == samples[1]],
                    columns=_labels_index,
                ).astype(int)
                df.index = samples
                sample_indicators = df.T
            else:
                from sklearn.preprocessing import LabelBinarizer
                _LB = LabelBinarizer()
                _sample_indicators = _LB.fit_transform(sample_labels)
                sample_indicators = pd.DataFrame(
                    _sample_indicators, columns=_LB.classes_
                )
            sample_indicators = (
                sample_indicators / sample_indicators.sum(axis=0)
            )
            # convert to pygsp format
            # will need to pad 1's to diagonal for filtering
            if target is None:
                graph = graphs.Graph(self.K)
            else:
                graph = graphs.Graph(target)
            graph.estimate_lmax()
            # default to Laplacian filter
            if filterfunc is None:
                def filterfunc(x):
                    return 1 / (1 + (beta * np.abs(x / graph.lmax - offset)) ** order)
            filt = filters.Filter(graph, filterfunc)
            densities = filt.filter(
                sample_indicators, method=solver, order=chebyshev_order)
            self._sample_densities = pd.DataFrame(
                densities, index=_labels_index, columns=sample_indicators.columns
            )

        if replicates is not None:
            return self._replicate_normalize_densities(replicates)

        return self._sample_densities

    def _replicate_normalize_densities(self, replicates):
        from sklearn.preprocessing import normalize
        replicates = np.unique(replicates)
        sample_likelihoods = self._sample_densities.copy()
        for rep in replicates:
            curr_cols = self._sample_densities.columns[[
                col.endswith(rep) for col in self._sample_densities.columns]]
            sample_likelihoods[curr_cols] = normalize(
                self._sample_densities[curr_cols], norm='l1')
        return sample_likelihoods

    def is_connected(self):
        """
        Check if the graph is connected (cached).
        A graph is connected if and only if there exists a (directed) path
        between any two vertices.

        Returns
        -------
        connected : bool
            True if the graph is connected, False otherwise.

        """
        if self._connected is not None:
            return self._connected
        adjacencies = [self.A]
        for adjacency in adjacencies:
            visited = np.zeros(self.N, dtype=bool)
            stack = set([0])
            while stack:
                vertex = stack.pop()
                if visited[vertex]:
                    continue
                visited[vertex] = True
                neighbors = adjacency[vertex].nonzero()[1]
                stack.update(neighbors)
            if not np.all(visited):
                self._connected = False
                return self._connected
        self._connected = True
        return self._connected

    def _connected_components(self, target=None):
        """
        Finds the connected components of the kernel matrix by default.
        Other matrices can be specified for use with the `target` parameter.

        Parameters
        ----------
        target : array-like (optional, default None).
            The target matrix to find the connected components of. If None, uses the kernel matrix.

        Returns
        -------
        components : list of np.ndarray
            The connected components of the target matrix.

        labels : list of int
            The labels of the connected components.
        """
        if target is None:
            n_components, labels = connected_components(self.K, directed=False)
        else:
            n_components, labels = connected_components(target, directed=False)

        return n_components, labels

    def resistance_distance(self):
        """
        Uses the cached Laplacian matrix to compute the resistance distances.
        See Klein and Randic [manuscript](https://doi.org/10.1007%2FBF01164627) for details.


        Returns
        -------
        rd : sparse matrix
            Resistance distance matrix

        """

        if self.laplacian_type != 'unnormalized':
            L = self.laplacian(laplacian_type='unnormalized')
        else:
            L = self.L
        try:
            pseudo = linalg.inv(L)
        except RuntimeError:
            pseudo = lil_matrix(np.linalg.pinv(L.toarray()))

        N = np.shape(L)[0]
        d = csc_matrix(pseudo.diagonal())
        rd = kron(d, csc_matrix(np.ones((N, 1)))).T \
            + kron(d, csc_matrix(np.ones((N, 1)))) \
            - pseudo - pseudo.T

        return rd

    def sparsify(self, epsilon=0.1, maxiter=10, random_state=None):
        """
        Sparsify a graph (with the Spielman-Srivastava method). This originally only called PyGSP but now
        also has some adaptations.

        Parameters
        ----------

        epsilon : float (optional, default 0.1).
            Sparsification parameter, which must be between ``1/sqrt(N)`` and 1.

        maxiter : int (optional, default 10).
            Maximum number of iterations.

        random_state : {None, int, RandomState, Generator} (optional, default None)
            Seed for the random number generator (for reproducible sparsification).

        Returns
        -------
        sparse_graph : sparse matrix
            Sparsified graph affinity matrix.
        """
        try:
            from pygsp import graphs, reduction
        except ImportError:
            raise ImportError(
                "pygsp is not installed. Please install it with `pip install pygsp` to use filtering functions.")
        if random_state is None:
            random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            pass

        if not 1./np.sqrt(self.N) <= epsilon < 1:
            raise ValueError('Epsilon out of required range!')

        # Not sparse
        resistance_distances = self.resistance_distance().toarray()
        W = coo_matrix(self.K)
        W.data[W.data < 1e-10] = 0
        W = W.tocsc()
        W.eliminate_zeros()
        start_nodes, end_nodes, weights = find(tril(W))

        # Calculate the new weights.
        weights = np.maximum(0, weights)
        Re = np.maximum(0, resistance_distances[start_nodes, end_nodes])
        Pe = weights * Re
        Pe = Pe / np.sum(Pe)
        dist = rv_discrete(values=(np.arange(len(Pe)), Pe), seed=random_state)

        for i in range(maxiter):
            # Rudelson, 1996 Random Vectors in the Isotropic Position
            # (too hard to figure out actual C0)
            C0 = 1 / 30.
            # Rudelson and Vershynin, 2007, Thm. 3.1
            C = 4 * C0
            q = round(self.N * np.log(self.N) * 9 * C**2 / (epsilon**2))

            results = dist.rvs(size=int(q))
            values, inv = np.unique(results, return_inverse=True)
            values = values.astype(int)
            freq = np.bincount(inv).astype(int)
            spin_counts = np.array([values, freq]).T
            per_spin_weights = weights / (q * Pe)

            counts = np.zeros(np.shape(weights)[0])
            counts[spin_counts[:, 0]] = spin_counts[:, 1]
            new_weights = counts * per_spin_weights

            sparserW = csc_matrix((new_weights, (start_nodes, end_nodes)),
                                  shape=(self.N, self.N))
            sparserW = sparserW + sparserW.T
            sparserL = diags(sparserW.diagonal(), 0) - sparserW

            if self.is_connected():
                break
            elif i == maxiter - 1:
                print('Graph is disconnected. Sparsifying anyway...')
            else:
                epsilon -= (epsilon - 1/np.sqrt(self.N)) / 2.

        sparserW = diags(sparserL.diagonal(), 0) - sparserL
        sparserW = (sparserW + sparserW.T) / 2.

        return sparserW

    def interpolate(self, f_subsampled, keep_inds, target=None, order=100, reg_eps=0.005):
        """
        Interpolate a graph signal.

        Parameters
        ----------
        f_subsampled : ndarray
            A graph signal on the graph G.

        keep_inds : ndarray
            List of indices on which the signal is sampled.

        target : array-like (optional, default None).
            Similarity matrix to use for interpolation. If None, uses the kernel matrix.

        order : int
            Degree of the Chebyshev approximation (default = 100).

        reg_eps : float
            The regularized graph Laplacian is $\bar{L}=L+\epsilon I$.
            A smaller epsilon may lead to better regularization,
            but will also require a higher order Chebyshev approximation.

        Returns
        -------
        signal_interpolated : ndarray
            Interpolated graph signal on the full vertex set of G.
        """
        try:
            from pygsp import graphs, reduction
        except ImportError:
            raise ImportError(
                "pygsp is not installed. Please install it with `pip install pygsp` to use interpolating functions.")
        # convert to pygsp format
        if target is None:
            graph = graphs.Graph(self.K)
        else:
            graph = graphs.Graph(target)

        signal_interpolated = reduction.interpolate(
            graph, f_subsampled, keep_inds, order, reg_eps)
        return signal_interpolated

    def write_pkl(self, wd=None, filename='mykernel.pkl'):
        try:
            import pickle
        except ImportError:
            return (print('Pickle is needed for saving the Kernel estimator. Please install it with `pip3 install pickle`'))
        if wd is None:
            import os
            wd = os.getcwd()
        with open(wd + filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        return print('Kernel saved at ' + wd + filename)
