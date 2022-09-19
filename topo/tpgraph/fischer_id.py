# Fischer separability analysis for intrinsic dimensionality estimation
# Adapted from scikit-dimension (https://github.com/scikit-learn-contrib/scikit-dimension)
# Implemented here with performance adaptations
#
#
# BSD 3-Clause License
#
# Copyright (c) 2020, Jonathan Bac
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import numba
from scipy.sparse import issparse
from scipy.special import lambertw
try:
    from matplotlib import pyplot as plt
    _HAVE_MATPLOTLIB = True
except:
    _HAVE_MATPLOTLIB = False
import multiprocessing as mp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
import warnings
from topo.base.ann import kNN
from topo.utils._utils import get_indices_distances_from_sparse_matrix
from sklearn.neighbors import NearestNeighbors

class FisherS(BaseEstimator, TransformerMixin):
    """
    Intrinsic dimension estimation using the Fisher Separability algorithm.
    Proposed by ALbergante, Bac and Zinovyev in 2019 (https://arxiv.org/pdf/1901.06328.pdf).
    This implementation was adapted from scikit-dimension (https://github.com/scikit-learn-contrib/scikit-dimension), available under
    the BSD 3-Clause License by Jonathan Bac.

    Parameters
    ----------
    scaled: bool (optional, default True).
        A boolean value indicating whether data is already scaled (mean-centered and with unit variance).
    test_alphas: 2D np.array with dtype float (optional, default np.arange(.6,1,.02)[None]).
        A row vector of floats, with alpha range, the values must be given increasing 
        within (0,1) interval. 
    limit_maxdim: bool (optional, default False)
        Whether to cap estimated maxdim to the embedding dimension


    """

    def __init__(
        self,
        scaled=True,
        alphas=None,
        limit_maxdim=False,
        verbose=False
    ):
        self.scaled = scaled
        self.alphas = alphas
        self.limit_maxdim = limit_maxdim
        self.verbose = verbose
        self.alphas = None
        self.p_alpha_ = None
        self.separable_fraction = None
        self.py_mean = None
        self.n_single = None
        self.alpha_ind_selected = None
        self.Xp_ = None

    def fit(self, X, y=None):
        """
        A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples.
        y : dummy parameter to respect the sklearn API


        Returns
        -------
        self : object
            Returns self.
        self.dimension_: float
            The estimated intrinsic dimension
        self.n_alpha : 1D np.array, float
            Effective dimension profile as a function of alpha
        self.n_single : float
            A single estimate for the effective dimension 
        self.p_alpha : 2D np.array, float
            Distributions as a function of alpha, matrix with columns corresponding to the alpha values, and with rows corresponding to objects. 
        self.separable_fraction : 1D np.array, float
            Separable fraction of data points as a function of alpha
        self.alphas : 2D np.array, float
            Input alpha values   
        """

        X = check_array(X, ensure_min_samples=2, ensure_min_features=2)

        # test_alphas introduced to pass sklearn checks (sklearn doesn't take arrays as default parameters)
        if self.alphas is None:
            self._alphas = np.arange(0.2, 1, 0.02)[None]
        else:
            self._alphas = self.alphas

        (self.n_alpha_,
        self.dimension_,
        self.p_alpha_,
        self.alphas_,
        self.separable_fraction_,
        self.Xp_) = self._SeparabilityAnalysis(X)

        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def transform(self, X=None):
        """
        Returns the estimated global intrinsic dimensionality. Here for scikit-learn API consistency.
        """
        return self.dimension_

    @staticmethod
    @numba.njit
    def _histc(X, bins):
        map_to_bins = np.digitize(X, bins)
        r = np.zeros((len(X[0, :]), len(bins)))
        for j in range(len(map_to_bins[0, :])):
            for i in map_to_bins[:, j]:
                r[j, i - 1] += 1
        return r

    def _scale(self, X):
        """
        Fischer Separability Analysis requires data to be scaled (centered and with unit length).
        """

        # centering
        X = X - np.mean(X, axis=0)
        # #project on sphere (scale each vector to unit length)
        st = np.sqrt(np.sum(X ** 2, axis=1))
        st = np.array([st]).T
        X = X / st

        return X

    @staticmethod
    def _probability_inseparable_sphere(alphas, n):
        """ 
        %probability_inseparable_sphere calculate theoretical probability for point
        %to be inseparable for dimension n
        %
        %Inputs:
        %   alphas is 1-by-d vector of possible alphas. Must be row vector or scalar
        %   n is c-by-1 vector of dimnesions. Must be column vector or scalar.
        %
        %Outputs:
        %   p is c-by-d matrix of probabilities."""
        p = np.power((1 - np.power(alphas, 2)), (n - 1) / 2) / (
            alphas * np.sqrt(2 * np.pi * n)
        )
        return p

    def _checkSeparabilityMultipleAlpha(self, data):
        """%checkSeparabilityMultipleAlpha calculate fraction of points inseparable
        %for each alpha and fraction of points which are inseparable from each
        %point for different alpha.
        %
        %Inputs:
        %   data is data matrix to calculate separability. Each row contains one
        %       data point.
        %   alphas is array of alphas to test separability.
        %
        %Outputs:
        %   separ_fraction fraction of points inseparable from at least one point.
        %       Fraction is calculated for each alpha.
        %   py is n-by-m matrix. py(i,j) is fraction of points which are
        %       inseparable from point data(i, :) for alphas(j)."""

        # Number of points per 1 loop. 20k assumes approx 3.2GB
        nP = 2000

        alphas = self._alphas
        # Normalize alphas
        if len(alphas[:, 0]) > 1:
            alphas = alphas.T
        addedone = 0
        if max(self._alphas[0, :]) < 1:
            alphas = np.array([np.append(alphas, 1)])
            addedone = 1

        alphas = np.concatenate(
            [[float("-inf")], alphas[0, :], [float("inf")]])

        n = len(data)
        counts = np.zeros((n, len(alphas)))
        leng = np.zeros((n, 1))
        for k in range(0, n, nP):
            # print('Chunk +{}'.format(k))
            e = k + nP
            if e > n:
                e = n
            # Calculate diagonal part, divide each row by diagonal element
            xy = data[k:e, :] @ data[k:e, :].T
            leng[k:e] = np.diag(xy)[:, None]
            xy = xy - np.diag(leng[k:e].squeeze())
            xy = xy / leng[k:e]
            counts[k:e, :] = counts[k:e, :] + self._histc(xy.T, alphas)
            # Calculate nondiagonal part
            for kk in range(0, n, nP):
                # Ignore diagonal part
                if k == kk:
                    continue
                ee = kk + nP
                if ee > n:
                    ee = n

                xy = data[k:e, :] @ data[kk:ee, :].T
                xy = xy / leng[k:e]
                counts[k:e, :] = counts[k:e, :] + self._histc(xy.T, alphas)

        # Calculate cumulative sum
        counts = np.cumsum(counts[:, ::-1], axis=1)[:, ::-1]

        # print(counts)

        py = counts / (n)
        py = py.T
        if addedone:
            py = py[1:-2, :]
        else:
            py = py[1:-1, :]

        separ_fraction = sum(py == 0) / len(py[0, :])

        return separ_fraction, py

    def _dimension_uniform_sphere(self, py):
        """
        %Gives an estimation of the dimension of uniformly sampled n-sphere
        %corresponding to the average probability of being inseparable and a margin
        %value 
        %
        %Inputs:
        %   py - average fraction of data points which are INseparable.
        %   alphas - set of values (margins), must be in the range (0;1)
        % It is assumed that the length of py and alpha vectors must be of the
        % same.
        %
        %Outputs:
        %   n - effective dimension profile as a function of alpha
        %   n_single_estimate - a single estimate for the effective dimension 
        %   alfa_single_estimate is alpha for n_single_estimate.
        """

        if len(py) != len(self._alphas[0, :]):
            raise ValueError(
                "length of py (%i) and alpha (%i) does not match"
                % (len(py), len(self._alphas[0, :]))
            )

        if np.sum(self._alphas <= 0) > 0 or np.sum(self._alphas >= 1) > 0:
            raise ValueError(
                [
                    '"Alphas" must be a real vector, with alpha range, the values must be within (0,1) interval'
                ]
            )

        # Calculate dimension for each alpha
        n = np.zeros((len(self._alphas[0, :])))
        for i in range(len(self._alphas[0, :])):
            if py[i] == 0:
                # All points are separable. Nothing to do and not interesting
                n[i] = np.nan
            else:
                p = py[i]
                a2 = self._alphas[0, i] ** 2
                w = np.log(1 - a2)
                n[i] = np.real(lambertw(-(w / (2 * np.pi * p * p * a2 * (1 - a2))))) / (
                    -w
                )

        n[n == np.inf] = float("nan")
        # Find indices of alphas which are not completely separable
        inds = np.where(~np.isnan(n))[0]
        if len(inds) == 0:
            return n, np.array([np.nan]), np.nan

        # Find the maximal value of such alpha
        alpha_max = max(self._alphas[0, inds])
        # The reference alpha is the closest to 90 of maximal partially separable alpha
        alpha_ref = alpha_max * 0.9
        k = np.where(
            abs(self._alphas[0, inds] - alpha_ref)
            == min(abs(self._alphas[0, :] - alpha_ref))
        )[0]
        # Get corresponding values
        alfa_single_estimate = self._alphas[0, inds[k]]
        n_single_estimate = n[inds[k]]

        return n, n_single_estimate, alfa_single_estimate

    def point_inseparability_to_pointID(
        self, idx="all_inseparable", force_definite_dim=True, verbose=True
    ):
        """
        Turn pointwise inseparability probability into pointwise global ID
        Inputs : 
            args : same as SeparabilityAnalysis
            kwargs : 
                idx : int, string
                    int for custom alpha index
                    'all_inseparable' to choose alpha where lal points have non-zero inseparability probability
                    'selected' to keep global alpha selected
                force_definite_dim : bool
                    whether to force fully separable points to take the minimum detectable inseparability value (1/(n-1)) (i.e., maximal detectable dimension)
        """
        if idx == "all_inseparable":  # all points are inseparable
            selected_idx = np.argwhere(
                np.all(self.p_alpha_ != 0, axis=1)).max()
        elif idx == "selected":  # globally selected alpha
            selected_idx = (self.n_alpha_ ==
                            self.dimension_).tolist().index(True)
        elif type(idx) == int:
            selected_idx = idx
        else:
            raise ValueError("unknown idx parameter")

        # select palpha and corresponding alpha
        palpha_selected = self.p_alpha_[selected_idx, :]
        alpha_selected = self._alphas[0, selected_idx]

        py = palpha_selected.copy()
        _alphas = np.repeat(alpha_selected, len(palpha_selected))[None]

        if force_definite_dim:
            py[py == 0] = 1 / len(py)

        if len(py) != len(_alphas[0, :]):
            raise ValueError(
                "length of py (%i) and alpha (%i) does not match"
                % (len(py), len(_alphas[0, :]))
            )

        if np.sum(_alphas <= 0) > 0 or np.sum(_alphas >= 1) > 0:
            raise ValueError(
                [
                    '"Alphas" must be a real vector, with alpha range, the values must be within (0,1) interval'
                ]
            )

        # Calculate dimension for each alpha
        n = np.zeros((len(_alphas[0, :])))
        for i in range(len(_alphas[0, :])):
            if py[i] == 0:
                # All points are separable. Nothing to do and not interesting
                n[i] = np.nan
            else:
                p = py[i]
                a2 = _alphas[0, i] ** 2
                w = np.log(1 - a2)
                n[i] = np.real(lambertw(-(w / (2 * np.pi * p * p * a2 * (1 - a2))))) / (
                    -w
                )

        n[n == np.inf] = float("nan")

        # Find indices of alphas which are not completely separable
        inds = np.where(~np.isnan(n))[0]
        if self.verbose:
            print(
                str(len(inds)) + "/" + str(len(py)),
                "points have nonzero inseparability probability for chosen alpha = "
                + str(round(alpha_selected, 2))
                + f", force_definite_dim = {force_definite_dim}",
            )
        return n, inds

    def getSeparabilityGraph(self, idx="all_inseparable", top_edges=10000):
        data = self.Xp_
        if idx == "all_inseparable":  # all points are inseparable
            selected_idx = np.argwhere(
                np.all(self.p_alpha_ != 0, axis=1)).max()
        elif idx == "selected":  # globally selected alpha
            selected_idx = (self.n_alpha_ ==
                            self.dimension_).tolist().index(True)
        elif type(idx) == int:
            selected_idx = idx
        else:
            raise ValueError("unknown idx parameter")
        alpha_selected = self._alphas[0, selected_idx]
        return self.buildSeparabilityGraph(data, alpha_selected, top_edges=top_edges)

    @staticmethod
    def plotSeparabilityGraph(x, y, edges, alpha=0.3):
        if not _HAVE_MATPLOTLIB:
            return (print('Matplotlib required for plotting.'))
        for i in range(len(edges)):
            ii = edges[i][0]
            jj = edges[i][1]
            plt.plot([x[ii], x[jj]], [y[ii], y[jj]], "k-", alpha=alpha)

    @staticmethod
    def buildSeparabilityGraph(data, alpha, top_edges=10000):
        """weighted directed separability graph, represented by a list of tuples (point i, point j) and an array of weights
        each tuple is the observation that point i is inseparable from j, the weight is <x_i,x_j>/<xi,xi>-alpha
        data is a matrix of data which is assumed to be properly normalized
        alpha parameter is a signle value in this case
        top_edges is the number of edges to return. if top_edges is negative then all edges will be returned
        """

        # Number of points per 1 loop. 20k assumes approx 3.2GB
        nP = np.min([2000, len(data)])
        n = len(data)
        leng = np.zeros((n, 1))

        # globalxy = np.zeros((n,n))

        insep_edges = []
        weights = []
        symmetric_graph = True
        symmetry_message = False

        for k in range(0, n, nP):
            e = k + nP
            if e > n:
                e = n
            # Calculate diagonal part, divide each row by diagonal element
            xy = data[k:e, :] @ data[k:e, :].T
            leng[k:e] = np.diag(xy)[:, None]
            xy = xy - np.diag(leng[k:e].squeeze())
            xy = xy / leng[k:e]
            # if skdim.lid.FisherS.check_symmetric(xy):
            if np.allclose(xy, xy.T, rtol=1e-05, atol=1e-08):
                # globalxy[k:e,k:e] = np.triu(xy)
                for i in range(len(xy)):
                    for j in range(i + 1, len(xy)):
                        if xy[i, j] > alpha:
                            insep_edges.append((k + i, k + j))
                            weights.append(xy[i, j] - alpha)
            else:
                symmetric_graph = False
                # globalxy[k:e,k:e] = xy
                for i in range(len(xy)):
                    for j in range(i + 1, len(xy)):
                        if xy[i, j] > alpha:
                            insep_edges.append((k + i, k + j))
                            weights.append(xy[i, j] - alpha)
            # Calculate nondiagonal part
            startpoint = 0
            if symmetric_graph:
                startpoint = k
                if not symmetry_message:
                    print(
                        "Graph is symmetric, only upper triangle of the separability matrix will be used"
                    )
                    symmetry_message = True
            for kk in range(startpoint, n, nP):
                # Ignore diagonal part
                if not k == kk:
                    ee = kk + nP
                    if ee > n:
                        ee = n
                    xy = data[k:e, :] @ data[kk:ee, :].T
                    xy = xy / leng[k:e]
                    # globalxy[k:e,kk:ee] = xy
                    for i in range(ee - kk):
                        for j in range(ee - kk):
                            if xy[i, j] > alpha:
                                insep_edges.append((k + i, kk + j))
                                weights.append(xy[i, j] - alpha)

        weights = np.array(weights)

        if top_edges > 0:
            if top_edges < len(insep_edges):
                weights_sorted = np.sort(weights)
                weights_sorted = weights_sorted[::-1]
                thresh = weights_sorted[top_edges]
                insep_edges_filtered = []
                weights_filtered = []
                for i, w in enumerate(weights):
                    if w > thresh:
                        insep_edges_filtered.append(insep_edges[i])
                        weights_filtered.append(w)
                weights = np.array(weights_filtered)
                insep_edges = insep_edges_filtered

        return insep_edges, weights

    @staticmethod
    def check_symmetric(a, rtol=1e-05, atol=1e-08):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

    def _SeparabilityAnalysis(self, X):
        self.N = len(X[:, 0])
        # Preprocess data
        if not self.scaled:
            Xp = self._scale(X)
        else:
            Xp = X.copy()
        # Check separability
        self.separable_fraction, self.p_alpha = self._checkSeparabilityMultipleAlpha(
            Xp)
        # Calculate mean fraction of separable points for each alpha.
        self.py_mean = np.mean(self.p_alpha, axis=1)
        self.n_alpha, self.n_single, alpha_single = self._dimension_uniform_sphere(
            self.py_mean)
        if np.nan in self.n_single:
            print('WARNING: some alphas are not separable (results are not reliable!). Consider increasing k for kNN search or using unscaled data...')
        self.alpha_ind_selected = np.where(self.n_single == self.n_alpha)[0]

        if self.limit_maxdim:
            self.n_single = np.clip(self.n_single, None, X.shape[1])

        if len(self.n_single) > 1:
            print(
                "FisherS selected several dimensions as equally probable. Taking the maximum"
            )
            self.n_single = self.n_single.max()

        return (
            self.n_alpha,
            self.n_single,
            self.p_alpha,
            self._alphas,
            self.separable_fraction,
            self.Xp_,
        )

    def generate_plots(self):
        if not _HAVE_MATPLOTLIB:
            return (print('Matplotlib required for plotting.'))
        # Define the minimal and maximal dimensions for theoretical graph with
        # two dimensions in each side
        n_min = np.floor(min(self.n_alpha)) - 2
        n_max = np.floor(max(self.n_alpha) + 0.8) + 2
        if n_min < 1:
            n_min = 1

        ns = np.arange(n_min, n_max + 1)

        plt.figure()
        plt.plot(self._alphas[0, :], self.n_alpha, "ko-")
        plt.plot(
            self._alphas[0, self.alpha_ind_selected], self.n_single, "rx", markersize=16,
        )
        plt.xlabel("\u03B1", fontsize=16)
        plt.ylabel("Effective dimension", fontsize=16)
        locs, labels = plt.xticks()
        plt.show()
        nbins = int(round(np.floor(self.N / 200)))

        if nbins < 20:
            nbins = 20

        plt.figure()
        plt.hist(self.p_alpha[self.alpha_ind_selected, :][0], bins=nbins)
        plt.xlabel(
            "Inseparability prob. for \u03B1=%2.2f"
            % (self._alphas[0, self.alpha_ind_selected]),
            fontsize=16,
        )
        plt.ylabel("Number of values")
        plt.show()

        plt.figure()
        plt.xticks(locs, labels)
        pteor = np.zeros((len(ns), len(self._alphas[0, :])))
        for k in range(len(ns)):
            for j in range(len(self._alphas[0, :])):
                pteor[k, j] = self._probability_inseparable_sphere(
                    self._alphas[0, j], ns[k]
                )

        for i in range(len(pteor[:, 0])):
            plt.semilogy(self._alphas[0, :], pteor[i, :], "-", color="r")
        plt.xlim(min(self._alphas[0, :]), 1)
        if True in np.isnan(self.n_alpha):
            plt.semilogy(
                self._alphas[0, : np.where(np.isnan(self.n_alpha))[0][0]],
                self.py_mean[: np.where(np.isnan(self.n_alpha))[0][0]],
                "bo-",
                "LineWidth",
                3,
            )
        else:
            plt.semilogy(self._alphas[0, :],
                         self.py_mean, "bo-", "LineWidth", 3)

        plt.xlabel("\u03B1")
        plt.ylabel("Mean inseparability prob.", fontsize=16)
        plt.title("Theor.curves for n=%i:%i" % (n_min, n_max))
        plt.show()

    def get_nn(self, X, k, n_jobs=-1, **kwargs):
        """Compute the k-nearest neighbors of a dataset np.array (n_samples x n_dims)"""

        knn_graph = kNN(X, n_neighbors=k, n_jobs=n_jobs, **kwargs)
        inds, dists = get_indices_distances_from_sparse_matrix(knn_graph, n_neighbors=k)
        return dists, inds

    def fit_pw(self, X, precomputed_knn=None, smooth=False, n_neighbors=30, n_jobs=1, **kwargs):
        """Creates an array of pointwise ID estimates (self.dimension_pw_) by fitting the estimator in kNN of each point.
        Parameters
        ----------
        X: np.array (n_samples x n_neighbors)
            Dataset to fit
        precomputed_knn: np.array (n_samples x n_dims)
            An array of precomputed (sorted) nearest neighbor indices
        n_neighbors:
            Number of nearest neighbors to use (ignored when using precomputed_knn)
        n_jobs: int
            Number of processes
        smooth: bool, default = False
            Additionally computes a smoothed version of pointwise estimates by 
            taking the ID of a point as the average ID of each point in its neighborhood (self.dimension_pw_)
           smooth_ 
        kwargs: dict, optional
            Keyword arguments for `topo.base.ann.kNN` for k-nearest-neighbor search.

        Returns
        -------
        self : object
            Returns self
        """
        if issparse(X):
            X_use = X.toarray()
        else:
            X_use = X
        X_use = check_array(X_use, ensure_min_samples=n_neighbors +
                        1, ensure_min_features=2)

        if precomputed_knn is not None:
            knnidx = precomputed_knn
        else:
            _, knnidx = self.get_nn(X_use, k=n_neighbors, n_jobs=n_jobs)
        if n_jobs > 1:
            pool = mp.Pool(n_jobs)
            results = pool.map(self.fit, [X_use[i, :] for i in knnidx])
            pool.close()
            self.dimension_pw_ = np.array([r.dimension_ for r in results])
        else:
            self.dimension_pw_ = np.array(
                [self.fit(X_use[i, :]).dimension_ for i in knnidx]
            )
        if smooth:
            self.dimension_pw_smooth_ = np.zeros(len(knnidx))
            for i, point_nn in enumerate(knnidx):
                self.dimension_pw_smooth_[i] = np.mean(
                    np.append(self.dimension_pw_[
                              i], self.dimension_pw_[point_nn])
                )
        return self

    def transform_pw(self, X=None):
        """ Return an array of pointwise ID estimates after a previous call to self.fit_pw
        Parameters
        ----------
        X : Dummy parameter
        Returns
        -------
        dimension_pw_ : np.array with dtype {int, float}
            Pointwise ID estimates
        dimension_pw_smooth_ : np.array with dtype float
            Smoothed pointwise ID estimates returned if self.fit_pw(smooth=True)
        """

        check_is_fitted(
            self,
            "dimension_pw_",
            msg=(
                "This class instance is not fitted yet. Call 'fit_pw' with "
                "appropriate arguments before using this method."
            ),
        )

        if hasattr(self, "dimension_pw_smooth_"):
            return self.dimension_pw_, self.dimension_pw_smooth_
        else:
            return self.dimension_pw_

    def fit_transform_pw(self, X, precomputed_knn=None, smooth=False, n_neighbors=10, n_jobs=1):
        """
        Returns the estimated local intrinsic dimensionality for each sample. 
        """
        self.fit_pw(X, precomputed_knn=precomputed_knn,
                    smooth=smooth, n_neighbors=n_neighbors, n_jobs=n_jobs)
        return self.transform_pw()
