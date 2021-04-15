# This is an adaptation of the original implementation of PaCMAP
# Originally implemented by Yingfan Wang and Haiyang Huang at https://github.com/YingfanWang/PaCMAP
# under the terms of the Apache License. Their algorithm is brilliantly described in the
# following manuscript, along with an excellent discussion on dimensional reduction:
#
#
# https://arxiv.org/abs/2012.04456
#
# Note that this means that the Apache license and -the following copyright applies if you also use the PaCMAP
# embedding for your work.
#
#    Copyright 2020 Yingfan Wang, Haiyang Huang, Cynthia Rudin, Yaron Shaposhnik
#
# Apache License
#
# Version 2.0, January 2004
#
# http://www.apache.org/licenses/
#
# TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
#
# 1. Definitions.
#
# "License" shall mean the terms and conditions for use, reproduction, and distribution as defined by Sections 1 through 9 of this document.
#
# "Licensor" shall mean the copyright owner or entity authorized by the copyright owner that is granting the License.
#
# "Legal Entity" shall mean the union of the acting entity and all other entities that control, are controlled by, or are under common control with that entity. For the purposes of this definition, "control" means (i) the power, direct or indirect, to cause the direction or management of such entity, whether by contract or otherwise, or (ii) ownership of fifty percent (50%) or more of the outstanding shares, or (iii) beneficial ownership of such entity.
#
# "You" (or "Your") shall mean an individual or Legal Entity exercising permissions granted by this License.
#
# "Source" form shall mean the preferred form for making modifications, including but not limited to software source code, documentation source, and configuration files.
#
# "Object" form shall mean any form resulting from mechanical transformation or translation of a Source form, including but not limited to compiled object code, generated documentation, and conversions to other media types.
#
# "Work" shall mean the work of authorship, whether in Source or Object form, made available under the License, as indicated by a copyright notice that is included in or attached to the work (an example is provided in the Appendix below).
#
# "Derivative Works" shall mean any work, whether in Source or Object form, that is based on (or derived from) the Work and for which the editorial revisions, annotations, elaborations, or other modifications represent, as a whole, an original work of authorship. For the purposes of this License, Derivative Works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work and Derivative Works thereof.
#
# "Contribution" shall mean any work of authorship, including the original version of the Work and any modifications or additions to that Work or Derivative Works thereof, that is intentionally submitted to Licensor for inclusion in the Work by the copyright owner or by an individual or Legal Entity authorized to submit on behalf of the copyright owner. For the purposes of this definition, "submitted" means any form of electronic, verbal, or written communication sent to the Licensor or its representatives, including but not limited to communication on electronic mailing lists, source code control systems, and issue tracking systems that are managed by, or on behalf of, the Licensor for the purpose of discussing and improving the Work, but excluding communication that is conspicuously marked or otherwise designated in writing by the copyright owner as "Not a Contribution."
#
# "Contributor" shall mean Licensor and any individual or Legal Entity on behalf of whom a Contribution has been received by Licensor and subsequently incorporated within the Work.
#
# 2. Grant of Copyright License. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute the Work and such Derivative Works in Source or Object form.
#
# 3. Grant of Patent License. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this section) patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work, where such license applies only to those patent claims licensable by such Contributor that are necessarily infringed by their Contribution(s) alone or by combination of their Contribution(s) with the Work to which such Contribution(s) was submitted. If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Work or a Contribution incorporated within the Work constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for that Work shall terminate as of the date such litigation is filed.
#
# 4. Redistribution. You may reproduce and distribute copies of the Work or Derivative Works thereof in any medium, with or without modifications, and in Source or Object form, provided that You meet the following conditions:
#
# You must give any other recipients of the Work or Derivative Works a copy of this License; and
# You must cause any modified files to carry prominent notices stating that You changed the files; and
# You must retain, in the Source form of any Derivative Works that You distribute, all copyright, patent, trademark, and attribution notices from the Source form of the Work, excluding those notices that do not pertain to any part of the Derivative Works; and
# If the Work includes a "NOTICE" text file as part of its distribution, then any Derivative Works that You distribute must include a readable copy of the attribution notices contained within such NOTICE file, excluding those notices that do not pertain to any part of the Derivative Works, in at least one of the following places: within a NOTICE text file distributed as part of the Derivative Works; within the Source form or documentation, if provided along with the Derivative Works; or, within a display generated by the Derivative Works, if and wherever such third-party notices normally appear. The contents of the NOTICE file are for informational purposes only and do not modify the License. You may add Your own attribution notices within Derivative Works that You distribute, alongside or as an addendum to the NOTICE text from the Work, provided that such additional attribution notices cannot be construed as modifying the License.
#
# You may add Your own copyright statement to Your modifications and may provide additional or different license terms and conditions for use, reproduction, or distribution of Your modifications, or for any such Derivative Works as a whole, provided Your use, reproduction, and distribution of the Work otherwise complies with the conditions stated in this License.
# 5. Submission of Contributions. Unless You explicitly state otherwise, any Contribution intentionally submitted for inclusion in the Work by You to the Licensor shall be under the terms and conditions of this License, without any additional terms or conditions. Notwithstanding the above, nothing herein shall supersede or modify the terms of any separate license agreement you may have executed with Licensor regarding such Contributions.
#
# 6. Trademarks. This License does not grant permission to use the trade names, trademarks, service marks, or product names of the Licensor, except as required for reasonable and customary use in describing the origin of the Work and reproducing the content of the NOTICE file.
#
# 7. Disclaimer of Warranty. Unless required by applicable law or agreed to in writing, Licensor provides the Work (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.
#
# 8. Limitation of Liability. In no event and under no legal theory, whether in tort (including negligence), contract, or otherwise, unless required by applicable law (such as deliberate and grossly negligent acts) or agreed to in writing, shall any Contributor be liable to You for damages, including any direct, indirect, special, incidental, or consequential damages of any character arising as a result of this License or out of the use or inability to use the Work (including but not limited to damages for loss of goodwill, work stoppage, computer failure or malfunction, or any and all other commercial damages or losses), even if such Contributor has been advised of the possibility of such damages.
#
# 9. Accepting Warranty or Additional Liability. While redistributing the Work or Derivative Works thereof, You may choose to offer, and charge a fee for, acceptance of support, warranty, indemnity, or other liability obligations and/or rights consistent with this License. However, in accepting such obligations, You may act only on Your own behalf and on Your sole responsibility, not on behalf of any other Contributor, and only if You agree to indemnify, defend, and hold each Contributor harmless for any liability incurred by, or claims asserted against, such Contributor by reason of your accepting any such warranty or additional liability.
#
# END OF TERMS AND CONDITIONS
#


import numpy as np
from sklearn.base import BaseEstimator
import numba
from annoy import AnnoyIndex
import time
import math
import datetime
from topometry.base.dists import *
from sklearn import preprocessing

@numba.jit(nopython=False, fastmath=True)
def calculate_dist(x1, x2, metric):
    metric = named_distances[metric]
    return metric(x1, x2)

@numba.njit("i4[:](i4,i4,i4[:])", nogil=True)
def sample_FP(n_samples, maximum, reject_ind):
    result = np.empty(n_samples, dtype=np.int32)
    for i in range(n_samples):
        reject_sample = True
        while reject_sample:
            j = np.random.randint(maximum)
            for k in range(i):
                if j == result[k]:
                    break
            for k in range(reject_ind.shape[0]):
                if j == reject_ind[k]:
                    break
            else:
                reject_sample = False
        result[i] = j
    return result

@numba.njit("i4[:,:](f4[:,:],f4[:,:],i4[:,:],i4)", parallel=True, nogil=True)
def sample_neighbors_pair(X, scaled_dist, nbrs, n_neighbors):
    n = X.shape[0]
    pair_neighbors = np.empty((n*n_neighbors, 2), dtype=np.int32)

    for i in numba.prange(n):
        scaled_sort = np.argsort(scaled_dist[i])
        for j in numba.prange(n_neighbors):
            pair_neighbors[i*n_neighbors + j][0] = i
            pair_neighbors[i*n_neighbors + j][1] = nbrs[i][scaled_sort[j]]
    return pair_neighbors

@numba.njit("i4[:,:](f4[:,:],i4)", nogil=True)
def sample_MN_pair(X, n_MN):
    n = X.shape[0]
    pair_MN = np.empty((n*n_MN, 2), dtype=np.int32)
    for i in numba.prange(n):
        for jj in range(n_MN):
            sampled = np.random.randint(0, n, 6)
            dist_list = np.empty((6), dtype=np.float32)
            for t in range(sampled.shape[0]):
                dist_list[t] = euclidean(X[i], X[sampled[t]])
            min_dic = np.argmin(dist_list)
            dist_list = np.delete(dist_list, [min_dic])
            sampled = np.delete(sampled, [min_dic])
            picked = sampled[np.argmin(dist_list)]
            pair_MN[i*n_MN + jj][0] = i
            pair_MN[i*n_MN + jj][1] = picked
    return pair_MN


@numba.njit("i4[:,:](f4[:,:],i4[:,:],i4,i4)", parallel=True, nogil=True)
def sample_FP_pair(X, pair_neighbors, n_neighbors, n_FP):
    n = X.shape[0]
    pair_FP = np.empty((n * n_FP, 2), dtype=np.int32)
    for i in numba.prange(n):
        for k in numba.prange(n_FP):
            FP_index = sample_FP(n_FP, n, pair_neighbors[i*n_neighbors: i*n_neighbors + n_neighbors][1])
            pair_FP[i*n_FP + k][0] = i
            pair_FP[i*n_FP + k][1] = FP_index[k]
    return pair_FP

@numba.njit("f4[:,:](f4[:,:],f4[:],i4[:,:])", parallel=True, nogil=True)
def scale_dist(knn_distance, sig, nbrs):
    n, num_neighbors = knn_distance.shape
    scaled_dist = np.zeros((n, num_neighbors), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(num_neighbors):
            scaled_dist[i, j] = knn_distance[i, j] ** 2 / sig[i] / sig[nbrs[i, j]]
    return scaled_dist

def generate_pair(
        X,
        n_neighbors,
        n_MN,
        n_FP,
        distance='euclidean',
        verbose=True
):
    metric=distance
    n, dim = X.shape
    n_neighbors_extra = min(n_neighbors + 50, n)
    tree = AnnoyIndex(dim, metric=distance)
    for i in range(n):
        tree.add_item(i, X[i, :])
    tree.build(20)

    nbrs = np.zeros((n, n_neighbors_extra), dtype=np.int32)
    knn_distances = np.empty((n, n_neighbors_extra), dtype=np.float32)

    for i in range(n):
        nbrs_ = tree.get_nns_by_item(i, n_neighbors_extra+1)
        nbrs[i, :] = nbrs_[1:]
        for j in range(n_neighbors_extra):
            knn_distances[i, j] = tree.get_distance(i, nbrs[i, j])
    if verbose:
        print("found nearest neighbor")
    sig = np.maximum(knn_distances[:, n_neighbors], 1e-10)
    # TopOMetry adaptation: change sig from 3:6 mean to distance to the k-th NN
    if verbose:
        print("found sig")
    scaled_dist = scale_dist(knn_distances, sig, nbrs)
    if verbose:
        print("found scaled dist")
    pair_neighbors = sample_neighbors_pair(X, scaled_dist, nbrs, n_neighbors)
    pair_MN = sample_MN_pair(X, n_MN, metric)
    pair_FP = sample_FP_pair(X, pair_neighbors, n_neighbors, n_FP)
    return pair_neighbors, pair_MN, pair_FP

@numba.njit("void(f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4,f4,f4,i4)", parallel=True, nogil=True)
def update_embedding_adam(Y, grad, m, v, beta1, beta2, lr, itr):
    n, dim = Y.shape
    lr_t = lr * math.sqrt(1.0 - beta2**(itr+1)) / (1.0 - beta1**(itr+1))
    for i in numba.prange(n):
        for d in numba.prange(dim):
            m[i][d] += (1 - beta1) * (grad[i][d] - m[i][d])
            v[i][d] += (1 - beta2) * (grad[i][d]**2 - v[i][d])
            Y[i][d] -= lr_t * m[i][d]/(math.sqrt(v[i][d]) + 1e-7)

@numba.njit("f4[:,:](f4[:,:],i4[:,:],i4[:,:],i4[:,:],f4,f4,f4)", parallel=True, nogil=True)
def pacmap_grad(Y, pair_neighbors, pair_MN, pair_FP, w_neighbors, w_MN, w_FP):
    n, dim = Y.shape
    grad = np.zeros((n+1, dim), dtype=np.float32)
    y_ij = np.empty(dim, dtype=np.float32)
    loss = np.zeros(3, dtype=np.float32)
    for t in range(pair_neighbors.shape[0]):
        i = pair_neighbors[t, 0]
        j = pair_neighbors[t, 1]
        d_ij = 1.0
        for d in range(dim):
            y_ij[d] = Y[i, d] - Y[j, d]
            d_ij += y_ij[d] ** 2
        loss[0] += w_neighbors * (d_ij/(10. + d_ij))
        w1 = w_neighbors * (20./(10. + d_ij) ** 2)
        for d in range(dim):
            grad[i, d] += w1 * y_ij[d]
            grad[j, d] -= w1 * y_ij[d]
    for tt in range(pair_MN.shape[0]):
        i = pair_MN[tt, 0]
        j = pair_MN[tt, 1]
        d_ij = 1.0
        for d in range(dim):
            y_ij[d] = Y[i][d] - Y[j][d]
            d_ij += y_ij[d] ** 2
        loss[1] += w_MN * d_ij/(10000. + d_ij)
        w = w_MN * 20000./(10000. + d_ij) ** 2
        for d in range(dim):
            grad[i, d] += w * y_ij[d]
            grad[j, d] -= w * y_ij[d]
    for ttt in range(pair_FP.shape[0]):
        i = pair_FP[ttt, 0]
        j = pair_FP[ttt, 1]
        d_ij = 1.0
        for d in range(dim):
            y_ij[d] = Y[i, d] - Y[j, d]
            d_ij += y_ij[d] ** 2
        loss[2] += w_FP * 1./(1. + d_ij)
        w1 = w_FP * 2./(1. + d_ij) ** 2
        for d in range(dim):
            grad[i, d] -= w1 * y_ij[d]
            grad[j, d] += w1 * y_ij[d]
    grad[-1, 0] = loss.sum()
    return grad

def pacmap(
        X,
        n_dims,
        n_neighbors,
        n_MN,
        n_FP,
        pair_neighbors,
        pair_MN,
        pair_FP,
        distance,
        lr,
        num_iters,
        Yinit,
        verbose,
        intermediate
):
    start_time = time.time()
    n, high_dim = X.shape

    if intermediate:
        itr_dic = [0, 10, 30, 60, 100, 120, 140, 170, 200, 250, 300, 350, 450]
        intermediate_states = np.empty((13, n, 2), dtype=np.float32)
    else:
        intermediate_states = None

    pca_solution = False
    if pair_neighbors is None:
        if verbose:
            print("finding pairs!")
        if distance != "hamming":
                X -= np.min(X)
                X /= np.max(X)
                X -= np.mean(X, axis=0)
        pair_neighbors, pair_MN, pair_FP = generate_pair(
            X, n_neighbors, n_MN, n_FP, distance, verbose
        )
        if verbose:
            print("sampled pairs")
    else:
        if verbose:
            print("using stored pairs")

    if Yinit is None:
        print('No initialisation provided, falling back to random...')
        init = 'random'
    elif Yinit == "random":
        Y = np.random.normal(size=[n, n_dims]).astype(np.float32) * 0.0001
    else: # user_supplied matrix
        print('Using user-supplied initialisation...')
        Yinit = Yinit.astype(np.float32)
        scaler = preprocessing.StandardScaler().fit(Yinit)
        Y = scaler.transform(Yinit) * 0.0001

    w_MN_init = 1000.
    beta1 = 0.9
    beta2 = 0.999
    m = np.zeros_like(Y, dtype=np.float32)
    v = np.zeros_like(Y, dtype=np.float32)

    if intermediate:
        itr_ind = 1
        intermediate_states[0, :, :] = Y

    for itr in range(num_iters):
        if itr < 100:
            w_MN = (1 - itr/100) * w_MN_init + itr/100 * 3.0
            w_neighbors = 2.0
            w_FP = 1.0
        elif itr < 200:
            w_MN = 3.0
            w_neighbors = 3
            w_FP = 1
        else:
            w_MN = 0.0
            w_neighbors = 1.
            w_FP = 1.

        grad = pacmap_grad(Y, pair_neighbors, pair_MN, pair_FP, w_neighbors, w_MN, w_FP)
        C = grad[-1, 0]
        update_embedding_adam(Y, grad, m, v, beta1, beta2, lr, itr)

        if intermediate:
            if (itr+1) == itr_dic[itr_ind]:
                intermediate_states[itr_ind, :, :] = Y
                itr_ind += 1
                if itr_ind > 12:
                    itr_ind -= 1
        if verbose:
            if (itr + 1) % 10 == 0:
                print("Iteration: %4d, Loss: %f" % (itr + 1, C))

    if verbose:
        elapsed = str(datetime.timedelta(seconds=time.time() - start_time))
        print("Elapsed time: %s" % (elapsed))
    return Y, intermediate_states, pair_neighbors, pair_MN, pair_FP

class PaCMAP(BaseEstimator):
    def __init__(self,
        n_dims=2,
        n_neighbors=None,
        MN_ratio=0.5,
        FP_ratio=2.0,
        pair_neighbors = None,
        pair_MN=None,
        pair_FP = None,
        distance="euclidean",
        lr=1.0,
        num_iters=450,
        init='spectral',
        verbose=False,
        intermediate=False
    ):
        self.n_dims = n_dims
        self.n_neighbors = n_neighbors
        self.MN_ratio = MN_ratio
        self.FP_ratio = FP_ratio
        self.pair_neighbors = pair_neighbors
        self.pair_MN = pair_MN
        self.pair_FP = pair_FP
        self.distance = distance
        self.lr = lr
        self.num_iters = num_iters
        self.init = init
        self.verbose = verbose
        self.intermediate = intermediate

        if self.n_dims < 2:
            raise ValueError("The number of projection dimensions must be at least 2")
        if self.lr <= 0:
            raise ValueError("The learning rate must be larger than 0")

    def fit(self, X, init=None):
        X = X.astype(np.float32)
        n, dim = X.shape
        if n <= 0:
            raise ValueError("The sample size must be larger than 0")
        if self.n_neighbors is None:
            if n <= 10000:
                self.n_neighbors = 10
            else:
                self.n_neighbors = int(round(10 + 15 * (np.log10(n) - 4)))
        self.n_MN = int(round(self.n_neighbors * self.MN_ratio))
        self.n_FP = int(round(self.n_neighbors * self.FP_ratio))
        if self.n_neighbors < 1:
            raise ValueError("The number of nearest neighbors can't be less than 1")
        if self.n_FP < 1:
            raise ValueError("The number of further points can't be less than 1")
        if self.verbose:
            print(
                "PaCMAP(n_neighbors={}, n_MN={}, n_FP={}, distance={},"
                "lr={}, n_iters={}, init={}, opt_method='adam', verbose={}, intermediate={})".format(
                    self.n_neighbors,
                    self.n_MN,
                    self.n_FP,
                    self.distance,
                    self.lr,
                    self.num_iters,
                    self.init,
                    self.verbose,
                    self.intermediate
                )
            )
        self.embedding_, self.intermediate_states, self.pair_neighbors, self.pair_MN, self.pair_FP = pacmap(
            X,
            self.n_dims,
            self.n_neighbors,
            self.n_MN,
            self.n_FP,
            self.pair_neighbors,
            self.pair_MN,
            self.pair_FP,
            self.distance,
            self.lr,
            self.num_iters,
            self.init,
            self.verbose,
            self.intermediate
        )
        return self

    def fit_transform(self, X, init="random"):
        self.fit(X, init)
        if self.intermediate:
            return self.intermediate_states
        else:
            return self.embedding_

    def sample_pairs(self, X):
        if self.verbose:
            print("sampling pairs")
        X = X.astype(np.float32)
        n, dim = X.shape
        if n <= 0:
            raise ValueError("The sample size must be larger than 0")
        if self.n_neighbors is None:
            if n <= 10000:
                self.n_neighbors = 10
            else:
                self.n_neighbors = int(round(10 + 15 * (np.log10(n) - 4)))
        self.n_MN = int(round(self.n_neighbors * self.MN_ratio))
        self.n_FP = int(round(self.n_neighbors * self.FP_ratio))
        if self.n_neighbors < 1:
            raise ValueError("The number of nearest neighbors can't be less than 1")
        if self.n_FP < 1:
            raise ValueError("The number of further points can't be less than 1")
        if self.distance != "hamming":
                X -= np.min(X)
                X /= np.max(X)
                X -= np.mean(X, axis=0)
        self.pair_neighbors, self.pair_MN, self.pair_FP = generate_pair(
            X,
            self.n_neighbors,
            self.n_MN,
            self.n_FP,
            self.distance,
            self.verbose
                )
        if self.verbose:
            print("sampled pairs")

        return self

    def del_pairs(self):
        self.pair_neighbors = None,
        self.pair_MN = None,
        self.pair_FP = None,
        return self
