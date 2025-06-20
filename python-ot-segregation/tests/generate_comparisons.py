#!/usr/bin/env python3

import numpy as np
import ot
from matplotlib import pyplot as plt
from os import path

distrib_3d = np.zeros((40, 40, 3))
for i in range(40):
	for j in range(40):
		if np.sqrt((i-19.5)**2 + (j-19.5)**2) <= 20:
			if np.sqrt((i-10)**2 + (j-10)**2) <= 10:
				distrib_3d[i, j, 0] = 0.5
				distrib_3d[i, j, 1] = 0.3
				distrib_3d[i, j, 2] = 0.2
			elif np.sqrt((i-35)**2 + (j-35)**2) <= 17:
				distrib_3d[i, j, 0] = 0.5
				distrib_3d[i, j, 1] = 0.05
				distrib_3d[i, j, 2] = 0.45
			else:
				distrib_3d[i, j, 0] = 0.6
				distrib_3d[i, j, 1] = 0.1
				distrib_3d[i, j, 2] = 0.3
		distrib_3d[i, j, :] *= np.exp(-((i - 19.5)**2 + (j - 19.5)**2) / (15**2) / 2) / (np.sqrt(2 * np.pi) * 15)

total_voting_population = np.sum(distrib_3d)

ot_distrib = distrib_3d.reshape(-1, distrib_3d.shape[2])


distance_matrix          = np.zeros((ot_distrib.shape[0], ot_distrib.shape[0]))
for index0 in np.ndindex(distrib_3d.shape[:2]):
	i0, j0 = index0
	idx0 = i0*distrib_3d.shape[1] + j0
	for index1 in np.ndindex(distrib_3d.shape[:2]):
		i1, j1 = index1
		idx1 = i1*distrib_3d.shape[1] + j1

		distance_matrix[idx0, idx1] = np.sqrt((i0 - i1)**2 + (j0 - j1)**2) / np.sqrt(np.prod(ot_distrib.shape[:2]))


reference_distrib  = np.sum(distrib_3d, axis=2).flatten()
reference_distrib /= np.sum(reference_distrib)


fig, axes = plt.subplots(2, 2, figsize=(5*2, 5*2))

x = np.arange(distrib_3d.shape[0])
y = np.arange(distrib_3d.shape[1])
X, Y = np.meshgrid(x, y)


for i_alpha, alpha in enumerate([-0.01, 0.01]):
	ot_dist_contribution_local_per_candidate = np.zeros(ot_distrib.shape)
	ot_dist_contribution_local               = np.zeros(ot_distrib.shape[0])
	ot_dist_contribution_candidate           = np.zeros(ot_distrib.shape[1])
	total_ot_dist                            = 0
	
	distance_matrix_alpha = np.power(distance_matrix, 1+alpha)

	for i in range(distrib_3d.shape[2]):
		total_vote_candidate = np.sum(distrib_3d[:, :, i])

		candidate_ot_mat = ot.emd(reference_distrib, ot_distrib[:, i] / total_vote_candidate, distance_matrix_alpha)*distance_matrix

		ot_dist_contribution_local_per_candidate[:, i]  = (candidate_ot_mat.sum(axis=0) + candidate_ot_mat.sum(axis=1)) / 2 / reference_distrib
		ot_dist_contribution_candidate[             i]  = np.sum(ot_dist_contribution_local_per_candidate[:, i] * reference_distrib)
		ot_dist_contribution_local                     += ot_dist_contribution_local_per_candidate[:, i] * total_vote_candidate / total_voting_population
		total_ot_dist                                  += ot_dist_contribution_candidate[i]              * total_vote_candidate / total_voting_population

	cax = axes[0, i_alpha].contourf(X, Y, ot_dist_contribution_local.reshape(distrib_3d.shape[:2]))
	cb = fig.colorbar(cax)

	axes[0, i_alpha].set_title(f"Local heterogeneity index, { ["concave", "convex"][i_alpha] }")

	for i in range(min(3, distrib_3d.shape[2])):
		cb.ax.plot((i + 1) / (min(3, distrib_3d.shape[2]) + 1), ot_dist_contribution_candidate[i],
			markerfacecolor=['r', 'g', 'b'][i], marker='.', markersize=12,
			markeredgecolor='w', markeredgewidth=0.2)
	cb.ax.plot(0.5, total_ot_dist,
		markerfacecolor='w', markeredgecolor='w', marker='x', markersize=10)

Kl_divergence = np.zeros(ot_distrib.shape[:1])
for i in range(distrib_3d.shape[2]):
	total_vote_proportion_candidate = np.sum(distrib_3d[:, :, i]) / total_voting_population
	candidate_distrib_3d               = np.array(ot_distrib[:, i]) / (reference_distrib * total_voting_population)
	Kl_divergence                  += total_vote_proportion_candidate * np.log(total_vote_proportion_candidate / np.maximum(candidate_distrib_3d, 1e-5))

cax = axes[1, 0].contourf(X, Y, Kl_divergence.reshape(distrib_3d.shape[:2]))
cb = fig.colorbar(cax)

axes[1, 0].set_title("KL-divergence to the global average")

idx_matrix        = np.argsort(distance_matrix, axis=1)
vote_trajectories = np.zeros((ot_distrib.shape[1], ot_distrib.shape[0], ot_distrib.shape[0]))
Kl_trajectories   = np.zeros((                     ot_distrib.shape[0], ot_distrib.shape[0]))
focal_distances   = np.zeros((                     ot_distrib.shape[0], ot_distrib.shape[0]))
for i in range(ot_distrib.shape[1]):
	vote_trajectories[i, :, :] = np.cumsum(ot_distrib[:, i][idx_matrix], axis=1)
population_trajectory = np.maximum(vote_trajectories.sum(axis=0), 1e-7)
for i in range(ot_distrib.shape[1]):
	vote_trajectories[i, :, :] /= population_trajectory
	vote_trajectories[i, :, :]  = np.maximum(vote_trajectories[i, :, :], 1e-7)
normalisation_factor = np.sum(vote_trajectories, axis=0)
for i in range(ot_distrib.shape[1]):
	total_vote_proportion_candidate = np.sum(distrib_3d[:, :, i]) / total_voting_population
	vote_trajectories[i, :, :] /= normalisation_factor
	Kl_trajectories            += total_vote_proportion_candidate * np.log(total_vote_proportion_candidate / vote_trajectories[i, :, :])
for j in reversed(range(ot_distrib.shape[0])):
	focal_distances[:, j] = np.max(Kl_trajectories[:, j:j+2], axis=1)
integration_coef         = population_trajectory.copy()
integration_coef[:, 1:] -= population_trajectory[:, :-1]
distort_coef             = np.sum(np.multiply(focal_distances, integration_coef), axis=1)

distort_coef[reference_distrib == 0] = np.nan

cax = axes[1, 1].contourf(X, Y, distort_coef.reshape(distrib_3d.shape[:2]))
cb = fig.colorbar(cax)

axes[1, 1].set_title("Multiscalar heterogeneity index")

fig.savefig("index_comparison_map.png")
