#!/usr/bin/env python3

import numpy as np
import ot
from matplotlib import pyplot as plt
from os import path

overwrite = True

def compute_and_plot_segregation(distrib_3d_, alpha=-0.01):
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

	ot_dissimilarity                         = np.zeros(ot_distrib.shape)

	unitary_direction_matrix = np.zeros((ot_distrib.shape[0], ot_distrib.shape[0], 2))
	distance_matrix          = np.zeros((ot_distrib.shape[0], ot_distrib.shape[0]))
	for index0 in np.ndindex(distrib_3d.shape[:2]):
		i0, j0 = index0
		idx0 = i0*distrib_3d.shape[1] + j0
		for index1 in np.ndindex(distrib_3d.shape[:2]):
			i1, j1 = index1
			idx1 = i1*distrib_3d.shape[1] + j1

			distance_matrix[idx0, idx1] = np.sqrt((i0 - i1)**2 + (j0 - j1)**2) / np.sqrt(np.prod(ot_distrib.shape[:2]))
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

		ot_dissimilarity[:,                         i] += (candidate_ot_mat.sum(axis=0) - candidate_ot_mat.sum(axis=1)) / 2 / reference_distrib

		ot_direction_per_candidate[:, 0,            i]  = ((unitary_direction_matrix[:, :, 0]*candidate_ot_mat).sum(axis=0) + (unitary_direction_matrix[:, :, 0].T*candidate_ot_mat).sum(axis=1)) / 2 / reference_distrib
		ot_direction_per_candidate[:, 1,            i]  = ((unitary_direction_matrix[:, :, 1]*candidate_ot_mat).sum(axis=0) + (unitary_direction_matrix[:, :, 1].T*candidate_ot_mat).sum(axis=1)) / 2 / reference_distrib
		ot_direction                                   += ot_direction_per_candidate[:, :, i]            * total_vote_candidate / total_voting_population

	fig, axes = plt.subplots(2, 3, figsize=(5*3, 5*2))
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

	axes[0, 0].imshow(disrib_image, origin="lower")

	axes[0, 0].set_title("Repartion of the two candidate (red and green)")

	x = np.arange(distrib_3d.shape[0])
	y = np.arange(distrib_3d.shape[1])
	X, Y = np.meshgrid(x, y)

	cax = axes[0, 1].contourf(X, Y, ot_dist_contribution_local.reshape(distrib_3d.shape[:2]))
	cb = fig.colorbar(cax)

	axes[0, 1].set_title("Local segregation index")
	
	for i in range(min(3, distrib_3d.shape[2])):
		cb.ax.plot((i + 1) / (min(3, distrib_3d.shape[2]) + 1), ot_dist_contribution_candidate[i],
			markerfacecolor=['r', 'g', 'b'][i], marker='.', markersize=12,
			markeredgecolor='w', markeredgewidth=0.2)
	cb.ax.plot(0.5, total_ot_dist,
		markerfacecolor='w', markeredgecolor='w', marker='x', markersize=10)

	axes[0, 2].quiver(
		X, Y,
		(ot_direction[:, 1] / ot_dist_contribution_local).reshape(distrib_3d.shape[:2]),
		(ot_direction[:, 0] / ot_dist_contribution_local).reshape(distrib_3d.shape[:2])
	)

	axes[0, 2].set_title("Directionality")

	cax = axes[1, 0].contourf(X, Y, ot_dissimilarity[:, 0].reshape(distrib_3d.shape[:2]))
	cb = fig.colorbar(cax)

	axes[1, 0].set_title("Dissimilarity for candidate 0 (red)")

	cax = axes[1, 1].contourf(X, Y, ot_dist_contribution_local_per_candidate[:, 0].reshape(distrib_3d.shape[:2]))
	cb = fig.colorbar(cax)

	axes[1, 1].set_title("Local contribution to segregation for candidate 0 (red)")

	cax = axes[1, 2].contourf(X, Y, ot_dist_contribution_local_per_candidate[:, 1].reshape(distrib_3d.shape[:2]))
	cb = fig.colorbar(cax)

	axes[1, 2].set_title("Local contribution to segregation for candidate 1 (green)")

	fig.tight_layout(pad=1.0)
	return fig, total_ot_dist


distrib = np.zeros((40, 40, 2))
distrib[:, :20,  0] = 1 
distrib[:,   :,  1] = 1 - distrib[:,   :,   0]

if overwrite or not path.exists("two-side_alphaPOS.png"):
	fig, _ = compute_and_plot_segregation(distrib, 0.1)
	fig.savefig("two-side_alphaPOS.png")
	plt.close(fig)

if overwrite or not path.exists("two-side_alphaNEG.png"):
	fig, _ = compute_and_plot_segregation(distrib, -0.1)
	fig.savefig("two-side_alphaNEG.png")
	plt.close(fig)

distrib = np.zeros((40, 40, 2))
distrib[:10,   :,  0] = 1
distrib[20:30, :,  0] = 1
distrib[:,     :,  1] = 1 - distrib[:,   :,   0]

if overwrite or not path.exists("stripes-4_alphaPOS.png"):
	fig, _ = compute_and_plot_segregation(distrib, 0.1)
	fig.savefig("stripes-4_alphaPOS.png")
	plt.close(fig)

if overwrite or not path.exists("stripes-4_alphaNEG.png"):
	fig, _ = compute_and_plot_segregation(distrib, -0.1)
	fig.savefig("stripes-4_alphaNEG.png")
	plt.close(fig)

size_list = [2, 4, 8, 16, 32, 64]
seg_list  = np.zeros(len(size_list))
for i,size in enumerate(size_list):
	distrib = np.zeros((size, size, 2))
	distrib[:size//2, :size//2,  0] = 1 
	distrib[ size//2:, size//2:, 0] = 1 
	distrib[:,     :,  1] = 1 - distrib[:,   :,   0]

	if overwrite or not path.exists(f"checkerboard-2-{ size }_alphaPOS.png"):
		fig, seg_list[i] = compute_and_plot_segregation(distrib, 0.1)
		fig.savefig(f"checkerboard-2-{ size }_alphaPOS.png")
		plt.close(fig)

	if overwrite or not path.exists(f"checkerboard-2-{ size }_alphaNEG.png"):
		fig, seg_list[i] = compute_and_plot_segregation(distrib, -0.1)
		fig.savefig(f"checkerboard-2-{ size }_alphaNEG.png")
		plt.close(fig)

if sum(seg_list != 0) == len(size_list) and (overwrite or not path.exists(f"checkerboard-2-segregation_evolution.png")):
	fig, ax = plt.subplots(1, 1, figsize=(5, 5))

	ax.plot(size_list, seg_list, "+-")

	ax.set_title("Evolution of segregation index vs the subdivion of the square")
	ax.set_xlabel("Number of subdivision per side of the square")
	ax.set_ylabel("Global segregation index")

	fig.savefig(f"checkerboard-2-segregation_evolution.png")
	plt.close(fig)

distrib = np.zeros((40, 40, 2))
distrib[:20, :20, 0] = 1 
distrib[20:, 20:, 0] = 1 
distrib[:, :,     1] = 1 - distrib[:, :, 0]

if overwrite or not path.exists("checkerboard-2_alphaPOS.png"):
	fig, _ = compute_and_plot_segregation(distrib, 0.1)
	fig.savefig("checkerboard-2_alphaPOS.png")
	plt.close(fig)

if overwrite or not path.exists("checkerboard-2_alphaNEG.png"):
	fig, _ = compute_and_plot_segregation(distrib, -0.1)
	fig.savefig("checkerboard-2_alphaNEG.png")
	plt.close(fig)


distrib = np.zeros((40, 40, 2))
distrib[:10,   :10,   0] = 1 
distrib[10:20, 10:20, 0] = 1 
distrib[20:,   :20,   0] = distrib[:20, :20, 0] 
distrib[:,     20:,   0] = distrib[:,   :20, 0] 
distrib[:, :,     1] = 1 - distrib[:,   :,   0]

if overwrite or not path.exists("checkerboard-4_alphaPOS.png"):
	fig, _ = compute_and_plot_segregation(distrib, 0.1)
	fig.savefig("checkerboard-4_alphaPOS.png")

if overwrite or not path.exists("checkerboard-4_alphaNEG.png"):
	fig, _ = compute_and_plot_segregation(distrib, -0.1)
	fig.savefig("checkerboard-4_alphaNEG.png")
	plt.close(fig)


distrib = np.zeros((40, 40, 2))
distrib[ 0:5,    0:5,  0] = 1 
distrib[ 5:10,   5:10, 0] = 1 
distrib[10:20,   0:10, 0] = distrib[ 0:10, 0:10, 0]
distrib[ 0:20,  10:20, 0] = distrib[ 0:20, 0:10, 0]
distrib[20:,     0:20, 0] = distrib[:20, :20, 0] 
distrib[:,      20:,   0] = distrib[:,   :20, 0] 
distrib[:,        :,   1] = 1 - distrib[:,   :,   0]

if overwrite or not path.exists("checkerboard-8_alphaPOS.png"):
	fig, _ = compute_and_plot_segregation(distrib, 0.01)
	fig.savefig("checkerboard-8_alphaPOS.png")
	plt.close(fig)

if overwrite or not path.exists("checkerboard-8_alphaNEG.png"):
	fig, _ = compute_and_plot_segregation(distrib, -0.01)
	fig.savefig("checkerboard-8_alphaNEG.png")
	plt.close(fig)


distrib = np.zeros((40, 40, 2))
distrib[:20, :20, 0] = 1
distrib[20:, 20:, 0] = 1
distrib[:, :,     1] = 1 - distrib[:, :, 0]
distrib[20:, :20, 0] = 0.75

if overwrite or not path.exists("checkerboard-2-less-segregated_alphaPOS.png"):
	fig, _ = compute_and_plot_segregation(distrib, 0.01)
	fig.savefig("checkerboard-2-less-segregated_alphaPOS.png")
	plt.close(fig)

if overwrite or not path.exists("checkerboard-2-less-segregated_alphaNEG.png"):
	fig, _ = compute_and_plot_segregation(distrib, -0.01)
	fig.savefig("checkerboard-2-less-segregated_alphaNEG.png")
	plt.close(fig)


distrib = np.zeros((40, 40, 2))
distrib[20:30,   :10, 0] = 1 
distrib[30:,   10:20, 0] = 1 
distrib[:,       :,   1] = 1 - distrib[:,   :,   0]

if overwrite or not path.exists("corner-checkerboard-2_alphaPOS.png"):
	fig, _ = compute_and_plot_segregation(distrib, 0.01)
	fig.savefig("corner-checkerboard-2_alphaPOS.png")
	plt.close(fig)

if overwrite or not path.exists("corner-checkerboard-2_alphaNEG.png"):
	fig, _ = compute_and_plot_segregation(distrib, -0.01)
	fig.savefig("corner-checkerboard-2_alphaNEG.png")
	plt.close(fig)

distrib = np.zeros((40, 40, 2))
distrib[20:25,   :5,  0] = 1 
distrib[25:30,  5:10, 0] = 1 
distrib[30:40,   :10, 0] = distrib[20:30, :10, 0] 
distrib[20:,   10:20, 0] = distrib[20:,   :10, 0] 
distrib[:,       :,   1] = 1 - distrib[:,   :,   0]

if overwrite or not path.exists("corner-checkerboard-4_alphaPOS.png"):
	fig, _ = compute_and_plot_segregation(distrib, 0.01)
	fig.savefig("corner-checkerboard-4_alphaPOS.png")
	plt.close(fig)

if overwrite or not path.exists("corner-checkerboard-4_alphaNEG.png"):
	fig, _ = compute_and_plot_segregation(distrib, -0.01)
	fig.savefig("corner-checkerboard-4_alphaNEG.png")
	plt.close(fig)

distrib = np.zeros((40, 40, 2))
for i in range(0, 20, 4):
	distrib[20:22, i:i+2, 0] = 1
for i in range(22, 40, 2):
	distrib[i:i+2, 0:20,  0] = 1 - distrib[i-2:i, :20 ,0]
distrib[:, :, 1] = 1 - distrib[:, :, 0]

if overwrite or not path.exists("corner-checkerboard-10_alphaPOS.png"):
	fig, _ = compute_and_plot_segregation(distrib, 0.01)
	fig.savefig("corner-checkerboard-10_alphaPOS.png")
	plt.close(fig)

if overwrite or not path.exists("corner-checkerboard-10_alphaNEG.png"):
	fig, _ = compute_and_plot_segregation(distrib, -0.01)
	fig.savefig("corner-checkerboard-10_alphaNEG.png")
	plt.close(fig)