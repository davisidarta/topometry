#####################################
# Author: Davi Sidarta-Oliveira
# School of Medical Sciences,University of Campinas,Brazil
# contact: davisidarta[at]fcm[dot]unicamp[dot]com
# License: MIT
######################################
# Defining eigendecomposition routines for kernels in a scikit-learn fashion

import numpy as np
from scipy.linalg import eigh
from topo.spectral._spectral import DM, LE, spectral_layout, component_layout, multi_component_layout
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse

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
        Method for computing the eigendecomposition. Can be either 'top', 'bottom', 'DM' or 'LE'.
        * 'top' : computes the top eigenpairs of the matrix.
        * 'bottom' : computes the bottom eigenpairs of the matrix.
        * 'DM' : computes the eigenpairs of diffusion operator on the matrix. If a `Kernel()` object is provided, will use the computed diffusion operator if available.
        * 'LE' : computes the eigenpairs of the graph laplacian on the matrix. If a `Kernel()` object is provided, will use the computed graph laplacian if available. 
    
    eig_method : string (optional, default 'eigsh').
        Method for computing the eigendecomposition. Can be either 'eigsh' (scipy.sparse.linalg.eigsh) or 'lobpcg' (scipy.sparse.linalg.lobpcg).

    weight : bool (optional, default True).
        Whether to weight the eigenvectors by the square root of the eigenvalues, if 'method' is 'top', 'bottom' or 'LE'.

    normalize : bool (optional, default False).
        Whether to normalize the eigenvectors by the Frobenius norm.
    
    

    random_state : int or numpy.random.RandomState() (optional, default None).
        Random seed for the eigendecomposition.
    



    """
    def __init__(self, n_components=None, method='DM', normalize=False, random_state=None):
        self.n_components = n_components
        self.method = method
        self.normalize = normalize
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Computes the eigendecomposition of the kernel matrix X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_samples)
            Kernel matrix.
        y : Ignored
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if self.method == 'DM':
            self.eigenvalues_, self.eigenvectors_ = DM(X, self.n_components, self.normalize, self.random_state)
        elif self.method == 'LE':
            self.eigenvalues_, self.eigenvectors_ = LE(X, self.n_components, self.normalize, self.random_state)
        else:
            raise ValueError("Method not recognized. Please use 'DM' or 'LE'.")

        return self

    def transform(self, X, y=None):
        """
        Applies the eigendecomposition to the kernel matrix X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_samples)
            Kernel matrix.
        y : Ignored
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Transformed data.
        """
        return self.eigenvectors_.T.dot(X)

    def fit_transform(self, X, y=None):
        """
        Computes the eigendecomposition of the kernel matrix X and applies it to X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_samples)
            Kernel matrix.
        y : Ignored
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Transformed data.
        """
        return self.fit(X).transform(X)




































