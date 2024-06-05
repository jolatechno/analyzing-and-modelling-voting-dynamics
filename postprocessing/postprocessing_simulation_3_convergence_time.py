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

output_file = config["output_file_convergence_time"]

base_path_figure = "figures/" + config["convergence_time"]["postprocessing"]["base_filename"]

interesting_candidates  = config["convergence_time"]["postprocessing"]["interesting_candidates"]
kldiv_clip_percentile   = config["convergence_time"]["postprocessing"]["kl-div_clip_percentile"]
dist_coef_clip_lims     = config["convergence_time"]["postprocessing"]["dist_coef_clip_lims"]
dist_coef_map_clip_lims = config["convergence_time"]["postprocessing"]["dist_coef_map_clip_lims"]

N_try      = config["convergence_time"]["N_try"]
N_it       = config["convergence_time"]["N_it"]
n_save     = config["convergence_time"]["n_save"]
N_thresh   = config["convergence_time"]["N_thresh"]
with h5py.File(base_path + output_file, "r") as file:
	N_nodes = len(file["geo_data"]["lat"])

iterations_saved = np.arange(0, N_it, n_save)


#########################################################
#########################################################
#########################################################


longitude, latitude = np.zeros(N_nodes), np.zeros(N_nodes)
populations         = np.zeros(N_nodes)

convergence_thresholds = np.zeros(N_thresh)

simulation_data  = np.zeros((N_try, N_candidates, N_nodes, N_it//n_save+1))
KL_trajectories  = np.zeros((N_try,               N_nodes, N_it//n_save+1))
focal_distances  = np.zeros((N_try,               N_nodes, N_thresh))
distortion_coefs = np.zeros((N_try,               N_nodes))

neighbors = []

with h5py.File(base_path + output_file, "r") as file:
	latitude [:]   = file["geo_data"]["lat"]
	longitude[:]   = file["geo_data"]["lon"]
	populations[:] = file["initial_state"]["population"]

	neighbors_begin_end = file["network"]["neighbors_begin_end_idx"]
	for begin,end in zip(neighbors_begin_end[:-1], neighbors_begin_end[1:]):
		neighbors.append(file["network"]["neighbors"][begin:end])

	convergence_thresholds[:] = file["analysis_0"]["convergence_thresholds"]

	for i in range(N_try):
		for k in range(N_candidates):
			field_name = "proportions_" + str(k)
			simulation_data[i, k, :, 0] = file["initial_state"][field_name]

		state_name = "analysis_" + str(i)

		for k in range(N_candidates):
			field_name = "vote_traj_" + candidates[k]
			trajectory_begin_end = file[state_name][field_name + "_begin_end_idx"]
			for j,(begin,end) in enumerate(zip(trajectory_begin_end[:-1], trajectory_begin_end[1:])):
				simulation_data[i, k, j, :] = file[state_name][field_name][begin:end]

		field_name = "KLdiv_trajectories"
		kl_trajectory_begin_end = file[state_name][field_name + "_begin_end_idx"]
		for j,(begin,end) in enumerate(zip(kl_trajectory_begin_end[:-1], kl_trajectory_begin_end[1:])):
			KL_trajectories[i, j, :] = file[state_name][field_name][begin:end]

		field_name = "focal_distances"
		focal_distances_begin_end = file[state_name][field_name + "_begin_end_idx"]
		for j,(begin,end) in enumerate(zip(focal_distances_begin_end[:-1], focal_distances_begin_end[1:])):
			focal_distances[i, j, :] = file[state_name][field_name][begin:end]

		distortion_coefs[i, :] = np.clip(file[state_name]["distortion_coefs"], *dist_coef_clip_lims)

""" -------------------------------------------------------
-----------------------------------------------------------
------------------------------------------------------- """

nodes = np.arange(N_nodes)
np.random.shuffle(nodes)


#########################################################
#########################################################
#########################################################

##################################################################################################################
##################################################################################################################
##################################################################################################################

#########################################################
#########################################################
#########################################################


fig, ax = plt.subplots(1, 1, figsize=(8,8))

plot_graph_from_scratch(neighbors, longitude, latitude, ax=ax)

fig.savefig(base_path_figure + "network.png", dpi=200)


#########################################################
#########################################################
#########################################################


fig, axes = plt.subplots(1, 3, figsize=(18,5))


for i_ax in range(3):
	for i,node in enumerate(nodes):
		if N_nodes-i <= 10:
			axes[i_ax].plot(iterations_saved, simulation_data[0, interesting_candidates[i_ax], node, :],
			                "k--", alpha=1, linewidth=1.1)
		else:
			axes[i_ax].plot(iterations_saved, simulation_data[0, interesting_candidates[i_ax], node, :],
				            "-", alpha=0.3, linewidth=0.3)

	axes[i_ax].set_title(f"Voting trajectory for { candidates[interesting_candidates[i_ax]] }")
	axes[i_ax].set_ylabel("vote proportion")
	axes[i_ax].set_xlabel("number of iterations")


fig.tight_layout(pad=2.0)
fig.savefig(base_path_figure + "vote_trajectory.png", dpi=200)

fig, ax = plt.subplots(1, 1, figsize=(8,8))


#########################################################
#########################################################
#########################################################


fig, ax = plt.subplots(1, 1, figsize=(8,8))

for i,node in enumerate(nodes):
	if N_nodes-i <= 10:
		ax.plot(iterations_saved[:-1], KL_trajectories[0, node, :-1],
		        "k--", alpha=1, linewidth=1.1)
	else:
		ax.plot(iterations_saved[:-1], KL_trajectories[0, node, :-1],
		        "-", alpha=0.2, linewidth=0.3)

ax.set_title("KL-divergence trajectory between the\ni-th state and the final state")
ax.set_ylabel("KL-divergence")
ax.set_xlabel("number of steps")
ax.set_yscale("log")
ax.set_ylim([
		np.percentile(KL_trajectories[0, :, -2], kldiv_clip_percentile[0])*0.8,
		np.percentile(KL_trajectories[0, :,  0], kldiv_clip_percentile[1])/0.8,
	])

fig.savefig(base_path_figure + "kl-div_trajectory.png", dpi=200)


#########################################################
#########################################################
#########################################################


fig, ax = plt.subplots(1, 1, figsize=(8,8))


for i,node in enumerate(nodes):
	if N_nodes-i <= 10:
		ax.plot(convergence_thresholds, iterations_saved[focal_distances[0, node, :].astype(np.int32)],
		        "k--", alpha=1, linewidth=1.1)
	else:
		ax.plot(convergence_thresholds, iterations_saved[focal_distances[0, node, :].astype(np.int32)],
		        "-", alpha=0.3, linewidth=0.4)

ax.set_title("Focal time trajectory")
ax.set_ylabel("focal time")
ax.set_xlabel("convergence thresholds")


fig.savefig(base_path_figure + "focal_time_trajectory.png", dpi=200)


#########################################################
#########################################################
#########################################################


fig, ax = plt.subplots(1, 1, figsize=(8,8))


ax.hist(np.mean(distortion_coefs, axis=0),
	density=True, label="mean distortion coefs per node over tries")
ax.hist(distortion_coefs.flatten(),
	density=True, label="distortion coefs", alpha=0.3)

ax.legend()
ax.set_title("Pseudo-distortion coefficient distribution")
ax.set_ylabel("Number of nodes")
ax.set_xlabel("Pseudo-distortion coefficient")


fig.savefig(base_path_figure + "histograms.png", dpi=200)


#########################################################
#########################################################
#########################################################


fig, ax = plt.subplots(1, 1, figsize=(9,8))

pl = ax.scatter(longitude, latitude,
	c=np.clip(np.mean(distortion_coefs, axis=0), *dist_coef_map_clip_lims),
	s=40, alpha=0.5)
pl = ax.scatter(longitude, latitude,
	c=np.clip(np.mean(distortion_coefs, axis=0), *dist_coef_map_clip_lims),
	s=20)

cbar = fig.colorbar(pl, label="pseudo-distortion coefficient")

ax.set_title("distortion coeffecient map")

fig.savefig(base_path_figure + "map.png", dpi=200)


#########################################################
#########################################################
#########################################################

##################################################################################################################
##################################################################################################################
##################################################################################################################

#########################################################
#########################################################
#########################################################


output_file = config["output_file_segregation"]
dist_coef      = np.zeros(N_nodes)
dist_coef_pop  = np.zeros(N_nodes)
dist_coef_dist = np.zeros(N_nodes)

with h5py.File(base_path + output_file, "r") as file:
	dist_coef     [:] = file["full_analysis"]["normalized_distortion_coefs"]
	dist_coef_pop [:] = file["full_analysis"]["normalized_distortion_coefs_pop"]
	dist_coef_dist[:] = file["full_analysis"]["normalized_distortion_coefs_dist"]


#########################################################
#########################################################
#########################################################


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,5))


for itry in range(N_try):
	ax1.plot(dist_coef, distortion_coefs[itry, :],
		"xC1", alpha=1.5/N_try, markersize=2,
		label="raw value" if itry==0 else None)
ax1.plot(dist_coef, np.mean(distortion_coefs, axis=0),
	"+C0", label="average per node over tries")

reg_time_vs_idx = LinearRegression().fit(
	np.expand_dims(dist_coef,                         1),
	np.expand_dims(np.mean(distortion_coefs, axis=0), 1))
score_time_vs_idx = reg_time_vs_idx.score(
	np.expand_dims(dist_coef,                         1),
	np.expand_dims(np.mean(distortion_coefs, axis=0), 1))
X = np.linspace(min(dist_coef), max(dist_coef), 100)
ax1.plot(X, reg_time_vs_idx.intercept_ + X*reg_time_vs_idx.coef_[0][0], "--r",
	 	 label=f"linear fit, R²={ round(score_time_vs_idx, 3) },\ncoef = { round(reg_time_vs_idx.coef_[0][0], 3) }")

ax1.legend(prop={'size': 8})
ax1.set_title("normalized distortion coefficient\nbased on number of voting bureau versus\nbased on number of voting bureau")
ax1.set_ylabel("distortion coefficient [based on convegrence time]")
ax1.set_xlabel("distortion coefficient [based on number of voting bureau]")

""" -------------------------------------------------------
-----------------------------------------------------------
------------------------------------------------------- """


for itry in range(N_try):
	ax2.plot(dist_coef_pop, distortion_coefs[itry, :],
		"xC1", alpha=1.5/N_try, markersize=2,
		label="raw value" if itry==0 else None)
ax2.plot(dist_coef_pop, np.mean(distortion_coefs, axis=0),
	"+C0", label="average per node over tries")

reg_time_vs_pop = LinearRegression().fit(
	np.expand_dims(dist_coef_pop,                         1),
	np.expand_dims(np.mean(distortion_coefs, axis=0), 1))
score_time_vs_pop = reg_time_vs_pop.score(
	np.expand_dims(dist_coef_pop,                         1),
	np.expand_dims(np.mean(distortion_coefs, axis=0), 1))
X = np.linspace(min(dist_coef_pop), max(dist_coef_pop), 100)
ax2.plot(X, reg_time_vs_pop.intercept_ + X*reg_time_vs_pop.coef_[0][0], "--r",
	 	 label=f"linear fit, R²={ round(score_time_vs_pop, 3) },\ncoef = { round(reg_time_vs_pop.coef_[0][0], 3) }")


ax2.legend(prop={'size': 8})
ax2.set_title("normalized distortion coefficient\nbased on distance versus\nbased on agregated population")
ax2.set_ylabel("distortion coefficient [based on convegrence time]")
ax2.set_xlabel("distortion coefficient [based on agregated population]")

""" -------------------------------------------------------
-----------------------------------------------------------
------------------------------------------------------- """

for itry in range(N_try):
	ax3.plot(dist_coef_dist, distortion_coefs[itry, :],
		"xC1", alpha=1.5/N_try, markersize=2,
		label="raw value" if itry==0 else None)
ax3.plot(dist_coef_dist, np.mean(distortion_coefs, axis=0),
	"+C0", label="average per node over tries")

reg_time_vs_dist = LinearRegression().fit(
	np.expand_dims(dist_coef_dist,                         1),
	np.expand_dims(np.mean(distortion_coefs, axis=0), 1))
score_time_vs_dist = reg_time_vs_dist.score(
	np.expand_dims(dist_coef_dist,                         1),
	np.expand_dims(np.mean(distortion_coefs, axis=0), 1))
X = np.linspace(min(dist_coef_dist), max(dist_coef_dist), 100)
ax3.plot(X, reg_time_vs_dist.intercept_ + X*reg_time_vs_dist.coef_[0][0], "--r",
	 	 label=f"linear fit, R²={ round(score_time_vs_dist, 3) },\ncoef = { round(reg_time_vs_dist.coef_[0][0], 3) }")

ax3.legend(prop={'size': 8})
ax3.set_title("normalized distortion coefficient\nbased on distance versus\nbased on distance")
ax3.set_ylabel("distortion coefficient [based on convegrence time]")
ax3.set_xlabel("distortion coefficient [based on distance]")


fig.tight_layout(pad=2.0)
fig.savefig(base_path_figure + "distortion_coef_comparison.png", dpi=200)

