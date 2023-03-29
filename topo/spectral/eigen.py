#####################################
# Author: Davi Sidarta-Oliveira
# School of Medical Sciences,University of Campinas,Brazil
# contact: davisidarta[at]fcm[dot]unicamp[dot]com
# License: MIT
######################################
# Defining eigendecomposition routines for kernels in a scikit-learn fashion

from sklearn.metrics import pairwise_distances
from warnings import warn
import numpy as np
from sklearn.utils import check_random_state
from scipy.linalg import eigh
from topo.spectral import graph_laplacian, diffusion_operator, LE  # , FischerS
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse
from topo.tpgraph.kernels import Kernel
EIGEN_SOLVERS = ['dense', 'arpack', 'lobpcg']
try:
    from pyamg import smoothed_aggregation_solver
    PYAMG_LOADED = True
    EIGEN_SOLVERS.append('amg')
except ImportError:
    PYAMG_LOADED = False


def eigendecompose(G, n_components=8, eigensolver='arpack', largest=True, eigen_tol=1e-4, random_state=None, verbose=False):
    """
    Eigendecomposition of a graph matrix.

    Parameters
    ----------
    G : array-like or sparse matrix, shape (n_vertices, n_vertices).

    n_components : int (optional, default 8).
        Number of eigenpairs to compute.

    eigensolver : string (optional, default 'arpack').
        Method for computing the eigendecomposition. Can be either 'arpack', 'lobpcg', 'amg' or 'dense'.
        * 'dense' :
            use standard dense matrix operations for the eigenvalue decomposition.
            For this method, G must be an array or matrix type.
            This method should be avoided for large problems.
        * 'arpack' :
            use arnoldi iteration in shift-invert mode. For this method,
            M may be a dense matrix, sparse matrix, or general linear operator.
        * 'lobpcg' :
            Locally Optimal Block Preconditioned Conjugate Gradient Method.
            A preconditioned eigensolver for large symmetric positive definite
            (SPD) generalized eigenproblems.
        * 'amg' :
            Algebraic Multigrid solver (requires ``pyamg`` to be installed)
            It can be faster on very large, sparse problems, but may also lead
            to instabilities.

    largest : bool (optional, default True).
        If True, compute the largest eigenpairs. If False, compute the smallest eigenpairs.

    eigen_tol : float (optional, default 1e-4).
        Error tolerance for the eigenvalue solver. If 0, machine precision is used.

    random_state : int or numpy.random.RandomState() (optional, default None).
        A pseudo random number generator used for the initialization of the
        lobpcg eigen vectors decomposition when eigen_solver == 'amg'.
        By default, arpack is used.  


    Returns
    -------
    eigen_values : array, shape (n_vertices,)
        Eigenvalues of the graph matrix.
    eigen_vectors : array, shape (n_vertices, n_vertices)
        Eigenvectors of the graph matrix.
    """
    n_components = n_components+1
    N = G.shape[0]
    if eigensolver not in EIGEN_SOLVERS:
        raise ValueError('Unknown eigensolver %s' % eigensolver)
    random_state = check_random_state(random_state)
    if not sparse.issparse(G) and eigensolver != 'dense':
        if verbose:
            print('Dense input and `eigensolver` set to %s' %
                eigensolver + '.' + '\n Converting to CSR matrix.')
        G = sparse.csr_matrix(G)
    if eigensolver == 'dense':
        if sparse.issparse(G):
            if verbose:
                print('Sparse input and `eigensolver` set to %s' %
                    eigensolver + '.' + '\n Converting to dense array.')
            G = G.toarray()
        evals, evecs = eigh(G)
    if sparse.issparse(G):
        if not isinstance(G, sparse.csr_matrix):
            G.tocsr()
    G = G.astype(float)
    if eigensolver == 'arpack':
        if largest:
            which = 'LM'
        else:
            which = 'SM'
        evals, evecs = sparse.linalg.eigsh(
            G, k=n_components, which=which, tol=eigen_tol, maxiter=N * 5)
    elif eigensolver == 'lobpcg':
        evals, evecs = sparse.linalg.lobpcg(
            G, random_state.normal(size=(G.shape[0], n_components)), largest=largest, tol=eigen_tol, maxiter=N // 5)
    elif eigensolver == 'amg':
        if not PYAMG_LOADED:
            raise ImportError(
                'Using "amg" as eigensolver requires pyamg, which is not installed. Install it with pip install pyamg')
        # for numerical stability
        np.random.set_state(random_state.get_state())
        # Use AMG to get a preconditioner and speed up the eigenvalue problem.
        ml = smoothed_aggregation_solver(G)
        M = ml.aspreconditioner()
        n_find = min(N, 5 + 2*n_components)
        X = random_state.rand(N, n_find)
        X[:, 0] = (G.diagonal()).ravel()
        evals, evecs = sparse.linalg.lobpcg(G, X, M=M, largest=largest)
    evals = np.real(evals)
    evecs = np.real(evecs)
    if largest:
        idx = np.argsort(evals)[::-1]
    else:
        idx = np.argsort(evals)[::]
    evals = evals[idx]
    evecs = evecs[:, idx]
    return evals, evecs


class EigenDecomposition(BaseEstimator, TransformerMixin):
    """
    Scikit-learn flavored class for computing eigendecompositions of sparse symmetric matrices.
    and exploring the associated eigenvectors and eigenvalues.
    Takes as main input a `topo.tpgraph.Kernel()` object or a symmetric matrix, which can be either an adjacency/affinity matrix,
    a kernel, a graph laplacian, or a diffusion operator. 

    Parameters
    ----------
    n_components : int (optional, default 10).
        Number of eigenpairs to be computed.

    method : string (optional, default 'DM').
        Method for organizing the eigendecomposition. Can be either 'top', 'bottom', 'DM' or 'LE'.
        * 'top' : computes the top eigenpairs of the matrix.
        * 'bottom' : computes the bottom eigenpairs of the matrix.
        * 'DM' : computes the eigenpairs of diffusion operator on the matrix. If a `Kernel()` object is provided, will use the computed diffusion operator if available.
        * 'LE' : computes the eigenpairs of the graph laplacian on the matrix. If a `Kernel()` object is provided, will use the computed graph laplacian if available. 

    eigensolver : string (optional, default 'arpack').
        Method for computing the eigendecomposition. Can be either 'arpack', 'lobpcg', 'amg' or 'dense'.
        * 'dense' :
            use standard dense matrix operations for the eigenvalue decomposition.
            For this method, M must be an array or matrix type.
            This method should be avoided for large problems.
        * 'arpack' :
            use arnoldi iteration in shift-invert mode. For this method,
            M may be a dense matrix, sparse matrix, or general linear operator.
        * 'lobpcg' :
            Locally Optimal Block Preconditioned Conjugate Gradient Method.
            A preconditioned eigensolver for large symmetric positive definite
            (SPD) generalized eigenproblems.
        * 'amg' :
            Algebraic Multigrid solver (requires ``pyamg`` to be installed)
            It can be faster on very large, sparse problems, but requires
            setting a random seed for better reproducibility.

    eigen_tol : float (optional, default 0.0).
        Error tolerance for the eigenvalue solver. If 0, machine precision is used.

    weight : bool (optional, default True).
        Whether to weight the eigenvectors by the square root of the eigenvalues, if 'method' is 'top', 'bottom' or 'LE' ('DM' is always weighted).

    anisotropy : float (optional, default 1.0).
        Anisotropy parameter for the diffusion operator, if method is 'DM'. 'Alpha' in the diffusion maps literature.

    normalize : bool (optional, default False).
        Whether to normalize the eigenvectors, if 'method' is 'top', 'bottom' or 'LE' ('DM' is always normalized).

    t : int (optional, default 1).
        Time parameter for the diffusion operator, if 'method' is 'DM' and 'multiscale' is False. Also works with 'method' being 'LE'. 
        The diffusion operator or the graph Laplacian will be powered by t. Ignored for other methods.

    multiscale : bool (optional, default True).
        Whether to use multiscale diffusion, if 'method' is 'DM'. Ignored for other methods. Setting this to
        True will make the 't' parameter be ignored.

    return_evals : bool (optional, default False).
        Whether to return the eigenvalues along with the eigenvectors.

    random_state : int or numpy.random.RandomState() (optional, default None).
        A pseudo random number generator used for the initialization of the
        lobpcg eigen vectors decomposition when eigen_solver == 'amg'.
        By default, arpack is used.  

    """

    def __init__(self,
                 n_components=10,
                 method='DM',
                 eigensolver='arpack',
                 eigen_tol=1e-4,
                 drop_first=True,
                 normalize=True,
                 weight=True,
                 multiscale=True,
                 laplacian_type='normalized',
                 anisotropy=1,
                 t=1,
                 random_state=None,
                 return_evals=False,
                 verbose=False):
        self.n_components = n_components
        self.method = method
        self.eigensolver = eigensolver
        self.eigen_tol = eigen_tol
        self.drop_first = drop_first
        self.normalize = normalize
        self.laplacian_type = laplacian_type
        self.weight = weight
        self.multiscale = multiscale
        self.t = t
        self.anisotropy = anisotropy
        self.random_state = random_state
        self.verbose = verbose
        self.eigenvalues = None
        self.eigenvectors = None
        self.laplacian = None
        self.diffusion_operator = None
        self.dmaps = None
        self.powered_operator = None
        self.N = None
        self.D_inv_sqrt_ = None
        self.multiscaled_evecs = None
        self.return_evals = False

    def __repr__(self):
        if self.eigenvectors is not None:
            if (self.N is not None):
                msg = "EigenDecomposition() estimator fitted with %i samples" % (
                    self.N)
            else:
                msg = "EigenDecomposition() estimator without fitted data."
        else:
            msg = "EigenDecomposition() estimator without any fitted data."
        if self.eigenvectors is not None:
            if self.method == 'DM':
                if self.multiscale:
                    msg += " using multiscale Diffusion Maps."
                else:
                    msg += " using Diffusion Maps."
            else:
                if self.method == 'LE':
                    msg += " using Laplacian Eigenmaps"
                elif self.method == 'top':
                    msg += " using top eigenpairs"
                elif self.method == 'bottom':
                    msg += " using bottom eigenpairs"
                if self.normalize:
                    msg += " with normalization"
                if self.weight:
                    msg += ", weighted by the square root of the eigenvalues"
                msg += '.'
        return msg

    def fit(self, X):
        """
        Computes the eigendecomposition of the kernel matrix X following the organization set by 'method'.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_samples)
            Matrix to be decomposed. Should generally be an adjacency, affinity/kernel/similarity, Laplacian matrix or a diffusion-type operator. 

        Returns 
        -------
        self : object
            Returns the instance itself, with eigenvectors stored at EigenDecomposition.eigenvectors
            and eigenvalues stored at EigenDecomposition.eigenvalues. If method is 'DM', stores Diffusion Maps at EigenDecomposition.dmaps.
            If 'method' is 'DM' or 'LE', the diffusion operator or graph laplacian is stored at EigenDecomposition.diffusion_operator
            or EigenDecomposition.graph_laplacian, respectively.
        """
        if self.method not in ['DM', 'LE', 'top', 'bottom']:
            raise ValueError(
                "Method must be one of 'DM', 'LE', 'top', 'bottom'.")
        if self.method in ['DM', 'top']:
            largest = True
        else:
            largest = False
        if isinstance(X, Kernel):
            self.N = X.N
            if self.method == 'DM':
                self.diffusion_operator = X.P
                if X.D_inv_sqrt_ is not None:
                    self.D_inv_sqrt_ = X.D_inv_sqrt_
                    symmetric = True
                else:
                    symmetric = False
                target = self.diffusion_operator
            elif self.method == 'LE':
                self.laplacian = X.L
                target = self.laplacian
            else:
                target = X.K
        elif isinstance(X, np.ndarray) or isinstance(X, sparse.csr_matrix):
            self.N = X.shape[0]
            if self.method == 'DM':
                symmetric = False
                self.diffusion_operator = diffusion_operator(
                    X, alpha=self.anisotropy, symmetric=False, return_D_inv_sqrt=False)
                target = self.diffusion_operator
                if self.t is not None:
                    if self.t > 1:
                        if not self.multiscale:
                            self.powered_operator = np.linalg.matrix_power(
                                self.diffusion_operator, int(self.t))
                            target = self.powered_operator
            elif self.method == 'LE':
                self.laplacian = graph_laplacian(
                    X, laplacian_type='normalized')
                target = self.laplacian
            else:
                target = X
        evals, evecs = eigendecompose(target, eigensolver=self.eigensolver, n_components=self.n_components,
                                      largest=largest, eigen_tol=self.eigen_tol, random_state=self.random_state, verbose=self.verbose)
        if self.drop_first:
            evals = evals[1:]
            evecs = evecs[:, 1:]
        if self.method == 'DM':
            if symmetric:
                evecs = self.D_inv_sqrt_.dot(evecs)
            for i in range(evecs.shape[1]):
                evecs[:, i] = evecs[:, i] / np.linalg.norm(evecs[:, i])
            if self.multiscale:
                use_eigs = int(np.sum(evals > 0, axis=0))
                eigs_idx = list(range(1, int(use_eigs)))
                eig_vals = np.ravel(evals[eigs_idx])
                self.dmaps = evecs[:, eigs_idx] * (eig_vals / (1 - eig_vals))
            else:
                self.dmaps = evecs * evals
        else:
            for i in range(evecs.shape[1]):
                evecs[:, i] = evecs[:, i] / np.linalg.norm(evecs[:, i])
            if self.multiscale:
                use_eigs = int(np.sum(evals > 0, axis=0))
                eigs_idx = list(range(1, int(use_eigs)))
                eig_vals = np.ravel(evals[eigs_idx])
                self.multiscaled_evecs = evecs[:,eigs_idx] * (eig_vals / (1 - eig_vals))
            else:
                if self.weight:
                    # weight by eigenvalues
                    evecs = evecs * np.sqrt(evals + 1e-10)
        self.eigenvectors = evecs
        self.eigenvalues = evals
        return self

    def rescale(self, use_eigs=50):
        """
        If using multiscale diffusion maps, this rescales the mapping to a new number of eigenvectors.
        """
        if self.method != 'DM' or not self.multiscale:
            raise ValueError(
                "Rescaling is only available for multiscale diffusion maps.")
        if use_eigs > self.eigenvectors.shape[1]:
            raise ValueError(
                "Cannot rescale to more eigenvectors than are available.")
        eigs_idx = list(range(1, int(use_eigs)))
        eig_vals = np.ravel(self.eigenvalues[eigs_idx])
        self.dmaps = self.eigenvectors[:,
                                       eigs_idx] * (eig_vals / (1 - eig_vals))
        return self

    def results(self, return_evals=None):
        """
        Returns the results of the decomposition. If using `method='DM'`, returns the (multiscaled) diffusion maps.

        Returns
        -------
        evecs : array-like, shape (n_samples, n_components)
            Eigenvectors of the matrix.
        evals : array-like, shape (n_components, )
            Eigenvalues of the matrix. Returned only if return_evals is True.
    
        """
        if return_evals is None:
            return_evals = self.return_evals

        if self.eigenvectors is None:
            return ValueError('The estimator has not been fitted yet.')
        if self.method == 'DM':
            if return_evals:
                return self.dmaps, self.eigenvalues
            else:
                return self.dmaps
        else:
            if not self.multiscale:
                if return_evals:
                    return self.eigenvectors, self.eigenvalues
                else:
                    return self.eigenvectors
            else:
                if return_evals:
                    return self.multiscaled_evecs, self.eigenvalues
                else:
                    return self.multiscaled_evecs


    def transform(self, X=None):
        """
        Here for scikit-learn compability. Returns the eigenvectors learned during fitting.
        If using `method='DM'`, returns the (multiscaled) diffusion maps.

        Parameters
        ----------

        X : array-like, shape (n_samples, n_features)
            Data matrix. Here for scikit-learn compability.

        Returns
        -------
        evecs : array-like, shape (n_samples, n_components)
            Eigenvectors of the matrix.
        evals : array-like, shape (n_components, )
            Eigenvalues of the matrix. Returned only if return_evals is True.
        """
        if self.eigenvectors is None:
            return ValueError('The estimator has not been fitted yet.')
        if self.method == 'DM':
            if self.return_evals:
                return self.dmaps, self.eigenvalues
            else:
                return self.dmaps
        else:
            if not self.multiscale:
                if self.return_evals:
                    return self.eigenvectors, self.eigenvalues
                else:
                    return self.eigenvectors
            else:
                if self.return_evals:
                    return self.multiscaled_evecs, self.eigenvalues
                else:
                    return self.multiscaled_evecs

    def fit_transform(self, X):
        """
        Here for scikit-learn compability. Returns the eigenvectors learned during fitting.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_samples).
            Kernel matrix or Kernel() object. Ignored, here for consistency with scikit-learn API.
        Returns
        -------
        evecs : array-like, shape (n_samples, n_components)
            Eigenvectors of the matrix.
        evals : array-like, shape (n_components, )
            Eigenvalues of the matrix. Returned only if return_evals is True.
        """
        if self.eigenvectors is None:
            self.fit(X)
        return self.transform()

    def spectral_layout(self, X, laplacian_type='normalized', return_evals=False):
        """Given a graph compute the spectral embedding of the graph. This function calls specialized
        routines if the graph has several connected components.

        Parameters
        ----------
        X : sparse matrix
            The (weighted) adjacency matrix of the graph as a sparse matrix. 
        laplacian_type : string (optional, default 'normalized').
            The type of laplacian to use. Can be 'unnormalized', 'symmetric' or 'random_walk'.
        return_evals : bool
            Whether to also return the eigenvalues of the laplacian.

        Returns
        -------
        embedding: array of shape (n_vertices, dim)
            The spectral embedding of the graph.

        evals: array of shape (dim,)
            The eigenvalues of the laplacian of the graph. Only returned if return_evals is True.
        """
        n_components, labels = sparse.csgraph.connected_components(
            X, directed=False)
        if n_components > 1:
            return multi_component_layout(
                X,
                n_components,
                labels,
                self.n_components,
                laplacian_type,
                self.random_state,
                self.eigen_tol,
                return_evals
            )
        else:
            if self.eigenvectors is None:
                self.fit(X)
            if return_evals:
                return self.eigenvectors, self.eigenvalues
            else:
                return self.eigenvectors

    def plot_eigenspectrum(self):
        """
        Plots the eigenspectrum.
        """
        if self.eigenvalues is None:
            raise ValueError('The estimator has not been fitted yet.')
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError('matplotlib is required for plotting.')
        plt.plot(range(0, len(self.eigenvalues)), self.eigenvalues)
        plt.xlabel('Eigenvalue index')
        plt.ylabel('Eigenvalue')
        plt.show()



def spectral_layout(graph, dim, random_state, laplacian_type='normalized', eigen_tol=10e-4, return_evals=False):
    """Given a graph compute the spectral embedding of the graph. This is
    simply the eigenvectors of the laplacian of the graph. Here we use the
    normalized laplacian.
    Parameters
    ----------
    graph: sparse matrix
        The (weighted) adjacency matrix of the graph as a sparse matrix.
    dim: int
        The dimension of the space into which to embed.
    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.
    Returns
    -------
    embedding: array of shape (n_vertices, dim)
        The spectral embedding of the graph.
    """
    n_components, labels = sparse.csgraph.connected_components(graph, directed=False)

    if n_components > 1:
        return multi_component_layout(
            graph,
            n_components,
            labels,
            dim,
            laplacian_type,
            random_state,
            eigen_tol,
            return_evals
        )

    else:
        return LE(graph, dim, laplacian_type, eigen_tol, return_evals)


def component_layout(
        W,
        n_components,
        component_labels,
        dim,
        laplacian_type='normalized',
        eigen_tol=10e-4,
        return_evals=False
):
    """Provide a layout relating the separate connected components. This is done
    by taking the centroid of each component and then performing a spectral embedding
    of the centroids.
    Parameters
    ----------
    W: numpy.ndarray, pandas.DataFrame or scipy.sparse.csr_matrix.
         Affinity or adjacency matrix.
    n_components: int
        The number of distinct components to be layed out.
    component_labels: array of shape (n_samples)
        For each vertex in the graph the label of the component to
        which the vertex belongs.
    dim: int
        The chosen embedding dimension.
    laplacian_type: string, optional (default='normalized')
        The type of laplacian to use. Can be 'unnormalized', 'normalized' or 'random_walk'.
    Returns
    -------
    component_embedding: array of shape (n_components, dim)
        The ``dim``-dimensional embedding of the ``n_components``-many
        connected components.
    """

    # cannot compute centroids from precomputed distances
    # instead, compute centroid distances using linkage
    distance_matrix = np.zeros((n_components, n_components), dtype=np.float64)
    # use single linkage
    linkage = np.min

    for c_i in range(n_components):
        dm_i = W[component_labels == c_i]
        for c_j in range(c_i + 1, n_components):
            dist = linkage(dm_i[:, component_labels == c_j])
            distance_matrix[c_i, c_j] = dist
            distance_matrix[c_j, c_i] = dist

    affinity_matrix = np.exp(-(distance_matrix ** 2))

    component_embedding, evals = LE(
        affinity_matrix, n_eigs=dim, laplacian_type=laplacian_type, eigen_tol=eigen_tol, return_evals=True)
    component_embedding /= component_embedding.max()
    if return_evals:
        return component_embedding, evals
    else:
        return component_embedding


def multi_component_layout(
    graph,
    n_components,
    component_labels,
    dim,
    laplacian_type,
    random_state,
    eigen_tol,
    return_eval_list
):

    result = np.empty((graph.shape[0], dim), dtype=np.float32)

    if n_components > 2 * dim:
        meta_embedding = component_layout(
            graph,
            n_components,
            component_labels,
            dim,
            laplacian_type,
            return_evals=False
        )
        k = dim + 1
    else:
        k = int(np.ceil(n_components / 2.0))
        base = np.hstack([np.eye(k), np.zeros((k, dim - k))])
        meta_embedding = np.vstack([base, -base])[:n_components]

    evals_list = []
    for label in range(n_components):
        component_graph = graph.tocsr()[component_labels == label, :].tocsc()
        component_graph = component_graph[:, component_labels == label].tocoo()
        distances = pairwise_distances([meta_embedding[label]], meta_embedding)
        data_range = distances[distances > 0.0].min() / 2.0
        if component_graph.shape[0] < 2 * dim:
            result[component_labels == label] = (
                random_state.uniform(
                    low=-data_range,
                    high=data_range,
                    size=(component_graph.shape[0], dim),
                )
                + meta_embedding[label]
            )
            continue

        L = graph_laplacian(graph, laplacian_type)
        k = dim + 1
        try:
            eigenvalues, eigenvectors = sparse.linalg.eigsh(
                L,
                k,
                which="SM",
                tol=eigen_tol,
                v0=np.ones(L.shape[0]),
                maxiter=graph.shape[0] * 2,
            )
            order = np.argsort(eigenvalues)[1:k]
            component_embedding = eigenvectors[:, order]
            expansion = data_range / np.max(np.abs(component_embedding))
            component_embedding *= expansion
            result[component_labels == label] = (
                component_embedding + meta_embedding[label]
            )
            evals_list.append(eigenvalues[order])
        except sparse.linalg.ArpackError:
            warn(
                "WARNING: spectral decomposition FAILED! This is likely due to too small an eigengap. Consider\n"
                "adding some noise or jitter to your data."
            )
            return None
    if return_eval_list:
        return result, evals_list
    else:
        return result
