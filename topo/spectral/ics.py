# Functions for independent eigencoordinate selection (IES) by 
# the brilliant Yu-Chia Chen
# See the manuscript at https://arxiv.org/pdf/1907.01651.pdf
# and the original source code at https://github.com/yuchaz/independent_coordinate_search
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)
# 
# Reproduced here with modifications and adaptations by Davi Sidarta-Oliveira

import numpy as np
from topo.eval.rmetric import RiemannMetric
from itertools import combinations
from scipy.spatial import procrustes

def find_independent_coordinates(emb, emb_evals, laplacian, intrisic_dim, greedy=False, score=False, data=None):
    embedding_dim = emb.shape[1]
    evecs = compute_tangent_plane(emb, laplacian)
    zeta_chosen, plotting_dict = zeta_search(evecs, emb_evals, intrisic_dim, embedding_dim)
    if not greedy:
        proj_vol, all_comb = projected_volume(evecs, intrisic_dim, embedding_dim, emb_evals, zeta_chosen)
        chosen_axis = all_comb[proj_vol.mean(1).argmax()]
        if score:
            if data is None:
                raise ValueError('Data must be provided to calculate the procrustes scores')
            proc_scores = calc_m2_score(data, emb, all_comb[proj_vol.mean(1).argsort()[::-1]])
    else:
        if score:
            return_records = True
            chosen_axis, ratio_records, remaining_axes_records = greedy_coordinate_search(evecs, intrisic_dim, eigen_values=emb_evals, zeta=zeta_chosen, return_records=return_records)
        else:
            return_records = False
            chosen_axis = greedy_coordinate_search(evecs, intrisic_dim, eigen_values=emb_evals, zeta=zeta_chosen, return_records=return_records)
        if score:
            if data is None:
                raise ValueError('Data must be provided to calculate the procrustes scores')
            proc_scores = calc_m2_score(data, emb, all_comb[ratio_records[0].mean(1).argsort()[::-1]])
    if score:
        return chosen_axis, proc_scores
    else:
        return chosen_axis

def calc_m2_score(clean_data, emb, ranked_axes, max_rank=1000):
    return np.array([procrustes(clean_data[:, :axis.shape[0]],
                                emb[:, axis])[-1]
                     for _, axis in zip(range(max_rank), ranked_axes)])

def compute_tangent_plane(embedding, laplacian):
    # From https://github.com/yuchaz/independent_coordinate_search
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    rmetric = RiemannMetric(embedding, laplacian)
    rmetric.get_dual_rmetric()
    HH = rmetric.H
    evals, evects = map(np.array, zip(*[eigsorted(HHi) for HHi in HH]))
    return evects


def projected_volume(principal_space, intrinsic_dim, embedding_dim=None,
                     eigen_values=None, zeta=1):
    # From https://github.com/yuchaz/independent_coordinate_search
    candidate_dim = principal_space.shape[1]
    embedding_dim = intrinsic_dim if embedding_dim is None else embedding_dim

    all_axes = np.array(list(combinations(
        range(1, candidate_dim), embedding_dim-1)))
    all_axes = np.hstack([
        np.zeros((all_axes.shape[0], 1), dtype=all_axes.dtype), all_axes])

    proj_volume = []
    for proj_axis in all_axes:
        proj_vol = _comp_projected_volume(principal_space, proj_axis,
                                          intrinsic_dim, embedding_dim,
                                          eigen_values, zeta)
        proj_volume.append(proj_vol)

    proj_volume = np.array(proj_volume)
    return proj_volume, all_axes


def _comp_projected_volume(principal_space, proj_axis, intrinsic_dim,
                           embedding_dim, eigen_values=None, zeta=1):
    basis = principal_space[:, proj_axis, :min(intrinsic_dim, embedding_dim)]
    basis = basis / np.linalg.norm(basis, axis=1)[:, None, :]
    try:
        vol_sq = np.linalg.det(np.einsum(
            'ijk,ijl->ikl', basis, basis))
        parallelepipe_vol = np.sqrt(vol_sq)
    except Exception as e:
        print(vol_sq[vol_sq<0])
        parallelepipe_vol = np.sqrt(np.abs(vol_sq))

    regu_term = _calc_regularizer(eigen_values, proj_axis, zeta)
    return np.log(parallelepipe_vol) - regu_term


def _calc_regularizer(eigen_values, proj_axis, zeta=1):
    if eigen_values is None:
        return 0
    eigen_values = np.abs(eigen_values[proj_axis])
    regu_term = np.sum(eigen_values) * zeta
    return regu_term


def ies_range_search(principal_space, intrinsic_dim, embedding_dim,
                     eigen_values, zeta_range):
    proj_volume_loss_no_regu, all_comb = projected_volume(
        principal_space, intrinsic_dim, embedding_dim)
    lambdas_comb = (np.abs(eigen_values)[all_comb]).sum(1)

    R_mean = proj_volume_loss_no_regu.mean(1)
    return R_mean[:, None] - lambdas_comb[:, None] * zeta_range[None, :], all_comb

def average_no_regu_gain(proj_vol, candidate_set_ix):
    average_gain_all = []
    n_points = proj_vol.shape[1]
    for iset in candidate_set_ix:
        sum_all = proj_vol.sum(1)
        average_gain = []
        for ii in range(n_points):
            test_val = proj_vol[:, ii]
            residual_val = (sum_all - test_val) / (n_points - 1)
            test_max_set = test_val.argmax()
            average_gain.append(residual_val[iset] - residual_val[test_max_set])
        average_gain_all.append(average_gain)

    return np.array(average_gain_all)


def zeta_search(evects, lambdas, intrinsic_dim, embedding_dim, alpha=75,
                low=1e-2, high=1e5, sep=300):
    zeta_range = np.logspace(np.log10(low), np.log10(high), sep)
    R_lambdas, all_comb = ies_range_search(
        evects, intrinsic_dim, embedding_dim, lambdas, zeta_range)
    rloss, __ = projected_volume(evects, intrinsic_dim, embedding_dim)

    all_candidate_set_ix = R_lambdas.argmax(0)
    mid_x_all = []
    candiate_set = []
    last_zeta_same_set = zeta_range[0]
    for idx_, (start_ix, end_ix, start_zeta, end_zeta) in enumerate(zip(
        all_candidate_set_ix[:-1], all_candidate_set_ix[1:],
        zeta_range[:-1], zeta_range[1:])):

        if start_ix != end_ix:
            mid_x_all.append(np.exp(0.5 * (np.log(last_zeta_same_set) + np.log(start_zeta))))
            last_zeta_same_set = end_zeta
            candiate_set.append(start_ix)
        elif idx_ == zeta_range.shape[0]-2:
            mid_x_all.append(np.exp(0.5 * (np.log(last_zeta_same_set) + np.log(end_zeta))))
            candiate_set.append(start_ix)
    candiate_set, mid_x_all = map(np.array, [candiate_set, mid_x_all])

    average_gain_all = average_no_regu_gain(rloss, candiate_set)
    ptile = np.percentile(-average_gain_all, alpha, axis=1)
    last_ix = np.where(ptile <= 1e-10)[0][-1]
    zeta_chosen = mid_x_all[last_ix]

    plotting_dicts=dict(zeta_range=zeta_range, R_lambdas=R_lambdas,
                        average_gain_all=average_gain_all,
                        all_comb=all_comb, mid_x_all=mid_x_all)

    return zeta_chosen, plotting_dicts



def greedy_coordinate_search(principal_space, intrinsic_dim, eigen_values=None,
                             zeta=1, return_records=False):
    candidate_dim = principal_space.shape[1]
    proj_vol, all_comb = projected_volume(
        principal_space, intrinsic_dim, eigen_values, zeta)

    argmax_proj_vol = proj_vol.mean(1).argmax()
    opt_proj_axis = list(all_comb[argmax_proj_vol])
    remaining_axes = [ii for ii in range(candidate_dim)
                      if ii not in opt_proj_axis]

    ratio_records = [proj_vol]
    remaining_axes_records = [np.array(remaining_axes)]
    for embedding_dim in range(intrinsic_dim+1, candidate_dim+1):
        proj_vols = np.array([_comp_projected_volume(
            principal_space, np.array(opt_proj_axis + [k]),
            intrinsic_dim, embedding_dim, eigen_values, zeta)
            for k in remaining_axes])

        if return_records:
            ratio_records.append(proj_vols)
            remaining_axes_records.append(np.array(remaining_axes))

        k_opt_ix_ = np.argmax(proj_vols.mean(1))
        k_opt = remaining_axes[k_opt_ix_]
        opt_proj_axis.append(k_opt)
        remaining_axes.pop(k_opt_ix_)

    if return_records:
        return opt_proj_axis, ratio_records, remaining_axes_records
    else:
        return opt_proj_axis