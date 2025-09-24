#!/usr/bin/env python3

import numpy as np
import ot
from matplotlib import pyplot as plt

x = np.arange(10)
distrib_rouge, distrib_vert = np.zeros(10), np.zeros(10)
distrib_rouge[:5], distrib_vert[5:] = 1/5, 1/5

def get_segregation(distrib_A, distrib_B, alpha=-0.01):
	ref_distrib  = distrib_A + distrib_B
	ref_distrib /= np.sum(ref_distrib)

	distance_matrix = np.zeros((len(distrib_A), len(distrib_A)))
	for i,j in np.ndindex(distance_matrix.shape):
		distance_matrix[i, j] = np.abs(i - j)
	distance_matrix_alpha = np.power(distance_matrix, 1+alpha)

	ot_dist_contribution_local_per_candidate = np.zeros((len(ref_distrib), 2))
	ot_dist_contribution_local               = np.zeros(len(ref_distrib))
	ot_dist_contribution_candidate           = np.zeros(2)
	total_ot_dist                            = 0
	ot_matrix                                = np.zeros((len(ref_distrib), len(ref_distrib), 2))

	total_voting_population = np.sum(distrib_A) + np.sum(distrib_B)
	for i in range(2):
		total_vote_candidate = np.sum(distrib_A if i == 0 else distrib_B)

		ot_matrix[:, :, i] = ot.emd(
				ref_distrib,
				(distrib_A if i == 0 else distrib_B) / total_vote_candidate,
				distance_matrix_alpha
			)
		ot_matrix_candidate = ot_matrix[:, :, i] * distance_matrix

		ot_dist_contribution_local_per_candidate[:, i]  = (ot_matrix_candidate.sum(axis=0) + ot_matrix_candidate.sum(axis=1)) / 2 / ref_distrib
		ot_dist_contribution_candidate[             i]  = np.sum(ot_dist_contribution_local_per_candidate[:, i] * ref_distrib)
		ot_dist_contribution_local                     += ot_dist_contribution_local_per_candidate[:, i] * total_vote_candidate / total_voting_population
		total_ot_dist                                  += ot_dist_contribution_candidate[i]              * total_vote_candidate / total_voting_population

	return (ot_dist_contribution_local_per_candidate, ot_dist_contribution_candidate, ot_dist_contribution_local, total_ot_dist, ot_matrix)

ot_dist_contribution_local_per_candidate, ot_dist_contribution_candidate, ot_dist_contribution_local, total_ot_dist, candidate_ot_mat = get_segregation(distrib_rouge, distrib_vert)

print(candidate_ot_mat[:, :, 0])
print(candidate_ot_mat[:, :, 1])
print(ot_dist_contribution_local_per_candidate[:, 0])
print(ot_dist_contribution_local_per_candidate[:, 1])

ot_dist_contribution_local_per_candidate, ot_dist_contribution_candidate, ot_dist_contribution_local, total_ot_dist, candidate_ot_mat = get_segregation(distrib_rouge, distrib_vert, 0.01)

print(candidate_ot_mat[:, :, 0])
print(candidate_ot_mat[:, :, 1])
print(ot_dist_contribution_local_per_candidate[:, 0])
print(ot_dist_contribution_local_per_candidate[:, 1])