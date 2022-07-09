# This Riemann metric implementation was implemented by Marina Meila, with some adaptations from Davi Sidarta-Oliveira
# The original source can be found at https://github.com/mmp2/megaman
# Author: Marina Meila <mmp@stat.washington.edu>
#         after the Matlab function rmetric.m by Dominique Perrault-Joncas
# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

import numpy as np
from scipy.sparse.csgraph import laplacian as lap_calc

def riemann_metric(Y, laplacian, n_dim=None, invert_h=False, mode_inv = 'svd'):
    n_samples = laplacian.shape[0]
    h_dual_metric = np.zeros((n_samples, n_dim, n_dim))
    n_dim_Y = Y.shape[1]
    h_dual_metric_full = np.zeros((n_samples, n_dim_Y, n_dim_Y))
    for i in range(n_dim_Y):
        for j in range(i, n_dim_Y):
            yij = Y[:,i] * Y[:,j]
            h_dual_metric_full[ :, i, j] = \
                0.5 * (laplacian.dot(yij) - \
                       Y[:, j] * laplacian.dot(Y[:, i]) - \
                       Y[:, i] * laplacian.dot(Y[:, j]))
    for j in np.arange(n_dim_Y - 1):
        for i in np.arange(j+1, n_dim_Y):
            h_dual_metric_full[:, i, j] = h_dual_metric_full[:, j, i]

    riemann_metric, h_dual_metric, Hvv, Hsvals, Gsvals = \
        compute_G_from_H(h_dual_metric_full)
    return h_dual_metric, riemann_metric, Hvv, Hsvals, Gsvals

def riemann_metric_lazy( Y, sample, laplacian, n_dim, invert_h=False, mode_inv = 'svd'):
    n_samples = laplacian.shape[0]
    laplacian = laplacian[sample,:]
    n_dim_Y = Y.shape[1]
    h_dual_metric_full = np.zeros((len(sample), n_dim_Y, n_dim_Y))
    h_dual_metric = np.zeros((len(sample), n_dim, n_dim))
    for i in np.arange(n_dim_Y):
        for j in np.arange(i, n_dim_Y):
            yij = Y[:,i]*Y[:,j]
            h_dual_metric_full[ :, i, j] = 0.5*(laplacian.dot(yij)-Y[sample,j]*laplacian.dot(Y[:,i])-Y[sample,i]*laplacian.dot(Y[:,j]))
    for j in np.arange(n_dim_Y - 1):
        for i in np.arange(j+1, n_dim_Y):
            h_dual_metric_full[ :,i,j] = h_dual_metric_full[:,j,i]

    riemann_metric, h_dual_metric, Hvv, Hsvals, Gsvals = compute_G_from_H(h_dual_metric_full)
    return h_dual_metric,riemann_metric, Hvv, Hsvals, Gsvals

def compute_G_from_H(H, mdimG=None, mode_inv="svd"):
    n_samples = H.shape[0]
    n_dim = H.shape[2]
    if mode_inv is 'svd':
        Huu, Hsvals, Hvv = np.linalg.svd(H)
        if mdimG is None or mdimG == n_dim:
            # Gsvals = 1./Hsvals
            Gsvals = np.divide(1, Hsvals, out=np.zeros_like(Hsvals), where=Hsvals != 0)
            G = np.zeros((n_samples, n_dim, n_dim))
            new_H = H
            for i in np.arange(n_samples):
                G[i,:,:] = np.dot(Huu[i,:,:], np.dot(np.diag(Gsvals[i,:]), Hvv[i,:,:]))
        elif mdimG < n_dim:
            Gsvals[:,:mdimG] = 1./Hsvals[:,:mdimG]
            Gsvals[:,mdimG:] = 0.
            # this can be redone with np.einsum() but it's barbaric
            G = np.zeros((n_samples, mdimG, mdimG))
            new_H = np.zeros((n_samples, mdimG, mdimG))
            for i in np.arange(n_samples):
                G[i,:,:mdimG] = np.dot(Huu[i,:,mdimG], np.dot( np.diag(Gsvals[i,:mdimG]), Hvv[i,:,:mdimG]))
                new_H[i, :, :mdimG] = np.dot(Huu[i,:,mdimG], np.dot( np.diag(Hsvals[i,:mdimG]), Hvv[i,:,:mdimG]))
        else:
            raise ValueError('mdimG must be <= H.shape[1]')
        return G, new_H, Hvv, Hsvals, Gsvals
    else:
        raise NotImplementedError('Not yet implemented non svd update.')
        riemann_metric = np.linalg.inv(h_dual_metric)
        return riemann_metric, None, None, None


class RiemannMetric(object):
    """
    RiemannMetric computes and stores the Riemannian metric and its dual
    associated with an embedding Y. The Riemannian metric is currently denoted
    by G, its dual by H, and the Laplacian by L. G at each point is the
    matrix inverse of H.
    For performance, the following choices have been made:
    * the class makes no defensive copies of L, Y
    * no defensive copies of the array attributes H, G, Hvv, ....
    * G is computed on request only
    In the future, this class will be extended to compute H only once,
    for mdimY dimensions, but to store multiple G's, with different dimensions.
    In the near future plans is also a "lazy" implementation, which will
    compute G (and maybe even H) only at the requested points.

    This implementation is from megaman, by Marina Meila (License simplified BSD), with
    adaptations from Davi Sidarta-Oliveira.

    Parameters
    -----------
    Y : embedding coordinates, shape = (n, mdimY)
    affinity : estimated affinity from data, ideally a diffusion operator,  shape = (n, n)
    n_dim : the manifold domension
    mod_inv : if mode_inv = svd, also returns Hvv, Hsvals,
        Gsvals the (transposed) eigenvectors of
        H and the singular values of H and G

    Returns
    ----------
    mdimG : dimension of G, H
    mdimY : dimension of Y
    H : dual Riemann metric, shape = (n, mdimY, mdimY)
    G : Riemann metric, shape = (n, mdimG, mdimG)
    Hvv : (transposed) singular vectors of H, shape = (n, mdimY, mdimY)
    Hsvals : singular values of H, shape = (n, mdimY)
    Gsvals : singular values of G, shape = (n, mdimG)
    detG : determinants of G, shape = (n,1)

    Notes
    -----
    H is always computed at full dimension self.mdimY
    G is computed at mdimG (some contradictions persist here)

    References
    ----------
    "Non-linear dimensionality reduction: Riemannian metric estimation and
    the problem of geometric discovery",
    Dominique Perraul-Joncas, Marina Meila, arXiv:1305.7255
    """
    def __init__(self, Y, affinity, n_dim=None, mode_inv='svd'):
        # input data
        self.Y = Y
        self.n, self.mdimY = Y.shape
        self.L = lap_calc(affinity)

        # input params
        if n_dim is None:
            self.mdimG = self.mdimY
        else:
            if n_dim > self.mdimY:
                raise ValueError('n_dim must be <= Y.shape[1]')
            self.mdimG = n_dim    # dimension of the riemann_metric computed
        self.mode_inv = mode_inv
        if self.mode_inv not in set(('svd', 'inv')):
            raise ValueError(("%s is not a valid value. Expected "
                              "'svd', 'inv'") % self.mode_inv)

        # results and outputs
        self.H = None
        self.G = None
        self.Hvv = None
        self.Hsvals = None
        self.Gsvals = None
        self.detG = None

    def get_dual_rmetric( self, invert_h = False, mode_inv = 'svd' ):
        """
        Compute the dual Riemannian Metric
        This is not satisfactory, because if mdimG<mdimY the shape of H
        will not be the same as the shape of G. TODO(maybe): return a (copied)
        smaller H with only the rows and columns in G.
        """
        if self.H is None:
            self.H, self.G, self.Hvv, self.Hsvals, self.Gsvals = riemann_metric(self.Y, self.L, self.mdimG, invert_h = invert_h, mode_inv = mode_inv)
        if invert_h:
            return self.H, self.G
        else:
            return self.H

    def get_rmetric( self, mode_inv = 'svd', return_svd = False ):
        """
        Compute the Reimannian Metric
        """
        if self.H is None:
            self.H, self.G, self.Hvv, self.Hsval, self.Gsvals = riemann_metric(self.Y, self.L, self.mdimG, invert_h = True, mode_inv = mode_inv)
        if mode_inv is 'svd' and return_svd:
            return self.G, self.Hvv, self.Hsvals, self.Gsvals
        else:
            return self.G

    def get_mdimG(self):
        return self.mdimG

    def get_detG(self):
        if self.G is None:
            self.H, self.G, self.Hvv, self.Hsval, self.Gsvals = riemann_metric(self.Y, self.L, self.mdimG, invert_h = True, mode_inv = self.mode_inv)
        if self.detG is None:
            if self.mdimG == self.mdimY:
                self.detG = 1./np.linalg.det(self.H)
            else:
                self.detG = 1./np.linalg.det(self.H[:,:self.mdimG,:self.mdimG])
