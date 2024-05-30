#!/usr/bin/python3

from util.plot import *

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import copy
import h5py
import json
import sys

base_path   = "../computation/output/"
config_file = "config.json"

candidates = [
	"ARTHAUD",
	"POUTOU",
	"MÉLENCHON",
	"ROUSSEL",
	"HIDALGO",
	"JADOT",
	"LASSALLE",
	"MACRON",
	"PÉCRESSE",
	"DUPONT_AIGNAN",
	"LE_PEN",
	"ZEMMOUR"
]
N_candidates = len(candidates)


#########################################################
#########################################################
#########################################################


def get_config(filename):
	with open(filename) as raw_json:
	    json_file = json.load(raw_json)

	    if len(sys.argv) > 1:
	    	return json_file[sys.argv[1]]
	    else:
	    	return json_file

config = get_config(base_path + config_file)

output_file = config["output_file_segregation"]

base_path_figure = "figures/" + config["segregation"]["postprocessing"]["base_filename"]

N_full_analyze    = config["segregation"]["N_full_analyze"]
N_thresh          = config["segregation"]["N_thresh"]
with h5py.File(base_path + output_file, "r") as file:
	N_total_nodes = len(file["geo_data"]["lat"])


#########################################################
#########################################################
#########################################################


longitude, latitude = np.zeros(N_total_nodes), np.zeros(N_total_nodes)
populations         = np.zeros(N_total_nodes)

dist_coef_idx  = np.zeros(N_total_nodes)
dist_coef_pop  = np.zeros(N_total_nodes)
dist_coef_dist = np.zeros(N_total_nodes)

convergence_thresholds = np.zeros(N_thresh)
vote_traj              = np.zeros((N_candidates, N_full_analyze, N_total_nodes))
KL_div_traj            = np.zeros((N_full_analyze, N_total_nodes))
focal_distances        = np.zeros((N_full_analyze, N_thresh))

distances                  = np.zeros((N_full_analyze, N_total_nodes))
accumulated_trajectory_pop = np.zeros((N_full_analyze, N_total_nodes))

worst_Xvalues_pop      = np.zeros(N_total_nodes)
worst_Xvalues_dist     = np.zeros(N_total_nodes)
worst_KLdiv_trajectory = np.zeros(N_total_nodes)
worst_focal_distances  = np.zeros(N_thresh)


with h5py.File(base_path + output_file, "r") as file:
	latitude [:]    = file["geo_data"]["lat"]
	longitude[:]    = file["geo_data"]["lon"]
	#populations[:] = file["full_analysis"]["voter_population"]


	distances_begin_end = file["partial_analysis"]["distances_begin_end_idx"]
	for i, (begin,end) in enumerate(zip(distances_begin_end[:-1], distances_begin_end[1:])):
		distances[i] = file["partial_analysis"]["distances"][begin:end]

	accumulated_trajectory_pop_begin_end = file["partial_analysis"]["accumulated_trajectory_pop_begin_end_idx"]
	for i, (begin,end) in enumerate(zip(accumulated_trajectory_pop_begin_end[:-1], accumulated_trajectory_pop_begin_end[1:])):
		accumulated_trajectory_pop[i] = file["partial_analysis"]["accumulated_trajectory_pop"][begin:end]


	worst_Xvalues_pop     [:] = file["full_analysis"]["normalization_factors"]["worst_Xvalues_pop"]
	worst_Xvalues_dist    [:] = file["full_analysis"]["normalization_factors"]["worst_Xvalues_dist"]
	worst_KLdiv_trajectory[:] = file["full_analysis"]["normalization_factors"]["worst_KLdiv_trajectory"]
	worst_focal_distances [:] = file["full_analysis"]["normalization_factors"]["worst_focal_distances"]


	convergence_thresholds[:] = file["partial_analysis"]["convergence_thresholds"]

	dist_coef_idx [:] = file["full_analysis"]["normalized_distortion_coefs"]
	dist_coef_pop [:] = file["full_analysis"]["normalized_distortion_coefs_pop"]
	dist_coef_dist[:] = file["full_analysis"]["normalized_distortion_coefs_dist"]

	for i,candidate in enumerate(candidates):
		field_name = "vote_trajectory_" + candidate
		vote_traj_begin_end = file["partial_analysis"][field_name + "_begin_end_idx"]
		for j, (begin,end) in enumerate(zip(vote_traj_begin_end[:-1], vote_traj_begin_end[1:])):
			vote_traj[i][j] = file["partial_analysis"][field_name][begin:end]

	KL_div_traj_begin_end = file["partial_analysis"]["KLdiv_trajectories_begin_end_idx"]
	for i, (begin,end) in enumerate(zip(KL_div_traj_begin_end[:-1], KL_div_traj_begin_end[1:])):
		KL_div_traj[i] = file["partial_analysis"]["KLdiv_trajectories"][begin:end]

	focal_distances_begin_end = file["partial_analysis"]["focal_distances_idxes_begin_end_idx"]
	for i, (begin,end) in enumerate(zip(focal_distances_begin_end[:-1], focal_distances_begin_end[1:])):
		focal_distances[i] = file["partial_analysis"]["focal_distances_idxes"][begin:end]


#########################################################
#########################################################
#########################################################


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,5))


ax1.scatter(longitude, latitude, c=dist_coef_idx, s=30, alpha=0.6)
ax1.scatter(longitude, latitude, c=dist_coef_idx, s=10, alpha=0.6)
pl = ax1.scatter(longitude, latitude, c=dist_coef_idx, s=1)

cbar = fig.colorbar(pl, label="distortion coefficient")

#ax1.set_aspect('equal', adjustable='box')
ax1.set_title("map of the distortion coefficient based\non the number of voting bureau")

""" -------------------------------------------------------
-----------------------------------------------------------
------------------------------------------------------- """

ax2.scatter(longitude, latitude, c=dist_coef_pop, s=30, alpha=0.6)
ax2.scatter(longitude, latitude, c=dist_coef_pop, s=10, alpha=0.6)
pl = ax2.scatter(longitude, latitude, c=dist_coef_pop, s=1)

cbar = fig.colorbar(pl, label="distortion coefficient")

#ax2.set_aspect('equal', adjustable='box')
ax2.set_title("map of the distortion coefficient based\non the agreagated population")

""" -------------------------------------------------------
-----------------------------------------------------------
------------------------------------------------------- """

ax3.scatter(longitude, latitude, c=dist_coef_dist, s=30, alpha=0.6)
ax3.scatter(longitude, latitude, c=dist_coef_dist, s=10, alpha=0.6)
pl = ax3.scatter(longitude, latitude, c=dist_coef_dist, s=1)

cbar = fig.colorbar(pl, label="distortion coefficient")

#ax3.set_aspect('equal', adjustable='box')
ax3.set_title("map of the distortion coefficient based\non distances")


fig.tight_layout(pad=2.0)
fig.savefig(base_path_figure + "map.png", dpi=200)


#########################################################
#########################################################
#########################################################


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))


ax1.hist(dist_coef_idx, density=True, bins=30)

ax1.set_title("normalized distortion coefficient\nbased on number of voting bureau")
ax1.set_ylabel("density")
ax1.set_xlabel("distortion coefficient")

""" -------------------------------------------------------
-----------------------------------------------------------
------------------------------------------------------- """

ax2.hist(dist_coef_pop, density=True, bins=30)

ax2.set_title("normalized distortion coefficient\nbased on agregated population")
ax2.set_ylabel("density")
ax2.set_xlabel("distortion coefficient")

""" -------------------------------------------------------
-----------------------------------------------------------
------------------------------------------------------- """

ax3.hist(dist_coef_dist, density=True, bins=30)

ax3.set_title("normalized distortion coefficient\nbased on distance")
ax3.set_ylabel("density")
ax3.set_xlabel("distortion coefficient")


fig.savefig(base_path_figure + "histograms.png", dpi=200)


#########################################################
#########################################################
#########################################################


fig, axes = plt.subplots(1, 3, figsize=(15,5))
interesting_candidates = [2, 7, 10]
for i_ax in range(3):
	for i in range(N_full_analyze):
		axes[i_ax].plot(vote_traj[interesting_candidates[i_ax]][i], "-")
		axes[i_ax].set_title(f"Voting trajectory for { candidates[interesting_candidates[i_ax]] }")
		axes[i_ax].set_ylabel("vote proportion")
		axes[i_ax].set_xlabel("number of voting bureau")

fig.savefig(base_path_figure + "vote_trajectory.png", dpi=200)


#########################################################
#########################################################
#########################################################


fig, ax = plt.subplots(1, 1, figsize=(7,7))


for i in range(N_full_analyze):
	ax.plot(KL_div_traj[i], "-")

ax.set_title("KL-divergence trajectories")
ax.set_ylabel("KL-divergence")
ax.set_xlabel("number of voting bureau")
ax.set_ylim([0, 0.5])


fig.savefig(base_path_figure + "KL-traj.png", dpi=200)


#########################################################
#########################################################
#########################################################


fig, ax = plt.subplots(1, 1, figsize=(7,7))


for i in range(N_full_analyze):
	ax.plot(convergence_thresholds, focal_distances[i], "-")

ax.set_title("Focal-distance based on the number of voting bureau")
ax.set_ylabel("focal distance [number of voting bureau]")
ax.set_xlabel("convergence threshold")
ax.set_xlim([0, 0.5])


fig.savefig(base_path_figure + "focal_distances.png", dpi=200)