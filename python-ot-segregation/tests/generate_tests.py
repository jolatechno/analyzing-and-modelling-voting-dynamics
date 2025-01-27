#!/usr/bin/env python3

import numpy as np
import ot
from matplotlib import pyplot as plt

def compute_and_plot_segregation(distrib_3d_, alpha=-0.01, plot_direction=False):
	distrib_3d = distrib_3d_.copy()
	reference_distrib  = np.sum(distrib_3d, axis=2).flatten()
	reference_distrib /= np.sum(reference_distrib)

	ot_distrib = distrib_3d.reshape(-1, distrib_3d.shape[2])
	for i in range(distrib_3d.shape[2]):
		ot_distrib[:, i] /= np.sum(distrib_3d[:, :, i])

	ot_dist_contribution_local_per_candidate = np.zeros(ot_distrib.shape)
	ot_dist_contribution_local               = np.zeros(ot_distrib.shape[:1])
	ot_dist_contribution_candidate           = np.zeros(ot_distrib.shape[1])
	total_ot_dist                            = 0
	ot_direction_per_candidate               = np.zeros((ot_distrib.shape[0], 2, ot_distrib.shape[1]))
	ot_direction                             = np.zeros((ot_distrib.shape[0], 2))

	unitary_direction_matrix = np.zeros((ot_distrib.shape[0], ot_distrib.shape[0], 2))
	distance_matrix          = np.zeros((ot_distrib.shape[0], ot_distrib.shape[0]))
	for index0 in np.ndindex(distrib_3d.shape[:2]):
		i0, j0 = index0
		idx0 = i0*distrib_3d.shape[1] + j0
		for index1 in np.ndindex(distrib_3d.shape[:2]):
			i1, j1 = index1
			idx1 = i1*distrib_3d.shape[1] + j1

			distance_matrix[idx0, idx1] = np.sqrt((i0 - i1)**2 + (j0 - j1)**2)
			if distance_matrix[idx0, idx1] != 0:
				unitary_direction_matrix[idx0, idx1, 0] = (i0 - i1)/distance_matrix[idx0, idx1]
				unitary_direction_matrix[idx0, idx1, 1] = (j0 - j1)/distance_matrix[idx0, idx1]
	distance_matrix_alpha = np.power(distance_matrix, 1+alpha)

	total_voting_population = np.sum(distrib_3d)
	for i in range(distrib_3d.shape[2]):
		total_vote_candidate = np.sum(distrib_3d[:, :, i])

		candidate_ot_mat = ot.emd(reference_distrib, ot_distrib[:, i], distance_matrix_alpha)*distance_matrix

		ot_dist_contribution_local_per_candidate[:, i]  = (candidate_ot_mat.sum(axis=0) + candidate_ot_mat.sum(axis=1)) / 2 / reference_distrib
		ot_dist_contribution_candidate[             i]  = np.sum(ot_dist_contribution_local_per_candidate[:, i] * reference_distrib)
		ot_dist_contribution_local                     += ot_dist_contribution_local_per_candidate[:, i] * total_vote_candidate / total_voting_population
		total_ot_dist                                  += ot_dist_contribution_candidate[i]              * total_vote_candidate / total_voting_population
		ot_direction_per_candidate[:, 0,            i]  = ((unitary_direction_matrix[:, :, 0]*candidate_ot_mat).sum(axis=0) + (unitary_direction_matrix[:, :, 0].T*candidate_ot_mat).sum(axis=1)) / 2 / reference_distrib
		ot_direction_per_candidate[:, 1,            i]  = ((unitary_direction_matrix[:, :, 1]*candidate_ot_mat).sum(axis=0) + (unitary_direction_matrix[:, :, 1].T*candidate_ot_mat).sum(axis=1)) / 2 / reference_distrib
		ot_direction                                   += ot_direction_per_candidate[:, :, i]            * total_vote_candidate / total_voting_population

	fig, axes = plt.subplots(1, 2 + plot_direction, figsize=(5*(2 + plot_direction), 5))
	ax0, ax1 = axes[0], axes[1]

	disrib_image = np.zeros((distrib_3d.shape[0], distrib_3d.shape[1], 3))
	if distrib_3d.shape[2] >= 3:
		disrib_image[:, :, :] = distrib_3d[:, :, :3]
	else:
		disrib_image[:, :, :distrib_3d.shape[2]] = distrib_3d[:, :, :]
	for i in range(min(3, distrib_3d.shape[2])):
		disrib_image[:, :, i] /= np.max(disrib_image[:, :, i])
		disrib_image[:, :, i] = np.round(disrib_image[:, :, i] * 255)
	disrib_image = disrib_image.astype(int)

	ax0.imshow(disrib_image, origin="lower")

	x = np.arange(distrib_3d.shape[0])
	y = np.arange(distrib_3d.shape[1])
	X, Y = np.meshgrid(x, y)

	cax = ax1.contourf(X, Y, ot_dist_contribution_local.reshape(distrib_3d.shape[:2]))
	
	cb = fig.colorbar(cax)
	for i in range(min(3, distrib_3d.shape[2])):
		cb.ax.plot((i + 1) / (min(3, distrib_3d.shape[2]) + 1), ot_dist_contribution_candidate[i],
			markerfacecolor=['r', 'g', 'b'][i], marker='.', markersize=12,
			markeredgecolor='w', markeredgewidth=0.2)
	cb.ax.plot(0.5, total_ot_dist,
		markerfacecolor='w', markeredgecolor='w', marker='x', markersize=10)

	if plot_direction:
		ax2 = axes[2]

		ax2.quiver(X, Y, ot_direction[:, 1].reshape(distrib_3d.shape[:2]), ot_direction[:, 0].reshape(distrib_3d.shape[:2]))

	fig.tight_layout(pad=1.0)
	return fig


distrib = np.zeros((40, 40, 2))
distrib[:, :20,  0] = 1 
distrib[:,   :,  1] = 1 - distrib[:,   :,   0]

fig = compute_and_plot_segregation(distrib, 0.1)
fig.savefig("two-side_alphaPOS.png")

fig = compute_and_plot_segregation(distrib, -0.1)
fig.savefig("two-side_alphaNEG.png")


distrib = np.zeros((40, 40, 2))
distrib[:10,   :,  0] = 1
distrib[20:30, :,  0] = 1
distrib[:,     :,  1] = 1 - distrib[:,   :,   0]

fig = compute_and_plot_segregation(distrib, 0.1)
fig.savefig("stripes-4_alphaPOS.png")

fig = compute_and_plot_segregation(distrib, -0.1)
fig.savefig("stripes-4_alphaNEG.png")


distrib = np.zeros((40, 40, 2))
distrib[:20, :20, 0] = 1 
distrib[20:, 20:, 0] = 1 
distrib[:, :,     1] = 1 - distrib[:, :, 0]

fig = compute_and_plot_segregation(distrib, 0.1, True)
fig.savefig("checkerboard-2_alphaPOS.png")

fig = compute_and_plot_segregation(distrib, -0.1, True)
fig.savefig("checkerboard-2_alphaNEG.png")


distrib = np.zeros((40, 40, 2))
distrib[:10,   :10,   0] = 1 
distrib[10:20, 10:20, 0] = 1 
distrib[20:,   :20,   0] = distrib[:20, :20, 0] 
distrib[:,     20:,   0] = distrib[:,   :20, 0] 
distrib[:, :,     1] = 1 - distrib[:,   :,   0]

fig = compute_and_plot_segregation(distrib, 0.1, True)
fig.savefig("checkerboard-4_alphaPOS.png")

fig = compute_and_plot_segregation(distrib, -0.1, True)
fig.savefig("checkerboard-4_alphaNEG.png")


distrib = np.zeros((40, 40, 2))
distrib[:20, :20, 0] = 1
distrib[20:, 20:, 0] = 1
distrib[:, :,     1] = 1 - distrib[:, :, 0]
distrib[20:, :20, 0] = 0.75

fig = compute_and_plot_segregation(distrib, 0.01, True)
fig.savefig("checkerboard-2-less-segregated_alphaPOS.png")

fig = compute_and_plot_segregation(distrib, -0.01, True)
fig.savefig("checkerboard-2-less-segregated_alphaNEG.png")