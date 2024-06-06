#!/usr/bin/python3

from util.util import *

import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib import patches
from sklearn.linear_model import LinearRegression
import copy
import h5py


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


config = get_config(base_path + config_file)

output_file = config["output_file_simulation"]

base_path_figure = "figures/" + config["simulation"]["postprocessing"]["base_filename"]

interesting_candidates = config["simulation"]["postprocessing"]["interesting_candidates"]

N_counties = config["simulation"]["N_counties"]
N_try      = config["simulation"]["N_try"]
N_it       = config["simulation"]["N_it"]
n_election = config["simulation"]["n_election"]
n_save     = config["simulation"]["n_save"]
with h5py.File(base_path + output_file, "r") as file:
	N_nodes = len(file["geo_data"]["lat"])

iterations_saved     = np.arange(0, N_it-1, n_save)
iterations_elections = np.arange(0, N_it-1, n_election)


#########################################################
#########################################################
#########################################################


longitude, latitude = np.zeros(N_nodes), np.zeros(N_nodes)
populations         = np.zeros(N_nodes)

simulation_data     = np.zeros((N_try, N_it//n_save, 2*N_candidates, N_nodes))
stubborn_equilibrium = np.zeros((N_candidates, N_nodes))

counties  = []
neighbors = []

general_election_results      = np.zeros((N_try, N_it//n_election))
general_election_proportions  = np.zeros((N_try, N_it//n_election, N_candidates))
counties_election_results     = np.zeros((N_try, N_it//n_election, N_counties))
counties_election_proportions = np.zeros((N_try, N_it//n_election, N_counties, N_candidates))

normalized_distortion_coefs = np.zeros((N_nodes, N_it//n_save))

with h5py.File(base_path + output_file, "r") as file:
	latitude [:]   = file["geo_data"]["lat"]
	longitude[:]   = file["geo_data"]["lon"]
	populations[:] = file["initial_state"]["population"]

	counties_begin_end = file["counties"]["counties_begin_end_idx"]
	for begin,end in zip(counties_begin_end[:-1], counties_begin_end[1:]):
		counties.append(file["counties"]["counties"][begin:end])

	neighbors_begin_end = file["network"]["neighbors_begin_end_idx"]
	for begin,end in zip(neighbors_begin_end[:-1], neighbors_begin_end[1:]):
		neighbors.append(file["network"]["neighbors"][begin:end])

	for k in range(N_candidates):
		field_name = "stubborn_equilibrium_" + str(k)
		stubborn_equilibrium[k, :] = file["initial_state"][field_name]
	
	normalized_distortion_coefs[:, 0] = file["segregation_initial_state"]["normalized_distortion_coefs"]

	for i in range(N_try):
		for k in range(2*N_candidates):
			field_name = "proportions_" + str(k)
			simulation_data[i, 0, k, :] = file["initial_state"][field_name]

		for j in range(1, N_it//n_save):
			it = j*n_save
			state_name = "states_" + str(i) + "_" + str(it)
			for k in range(2*N_candidates):
				field_name = "proportions_" + str(k)
				simulation_data[i, j, k, :] = file[state_name][field_name]

			segregation_state_name = "segregation_state_" + str(i) + "_" + str(it)
			normalized_distortion_coefs[:, j] = file[segregation_state_name]["normalized_distortion_coefs"]

		for j in range(0, N_it//n_election):
			it = j*n_election

			general_election_name  = "general_election_result_" + str(i) + "_" + str(it)
			counties_election_name = "counties_election_result_" + str(i) + "_" + str(it)

			general_election_results[i, j] = file[general_election_name]["result"][0]
			for l in range(N_candidates):
				field_name = "proportion_" + str(l)
				general_election_proportions[i, j, l] = file[general_election_name][field_name][0]

			for k in range(N_counties):
				county_name = "election_result_" + str(k)

				counties_election_results[i, j, k] = file[counties_election_name][county_name]["result"][0]
				for l in range(N_candidates):
					field_name = "proportion_" + str(l)
					counties_election_proportions[i, j, k, l] = file[counties_election_name][county_name][field_name][0]

""" -------------------------------------------------------
-----------------------------------------------------------
------------------------------------------------------- """

nodes = np.arange(N_nodes)
np.random.shuffle(nodes)

map_ratio = get_map_ratio(longitude, latitude)


#########################################################
#########################################################
#########################################################

##################################################################################################################
##################################################################################################################
##################################################################################################################

#########################################################
#########################################################
#########################################################


fig, ax = plt.subplots(1, 1, figsize=(8, 8/map_ratio))

color_0, color_1 = 0, 1
colors = np.empty(N_nodes)
for node in range(N_nodes):
	colors[node] = color_0 if node in counties[0] else color_1

plot_graph_from_scratch(neighbors, longitude, latitude, colors, ax)


fig.tight_layout(pad=1.0)
fig.savefig(base_path_figure + "county_map.png", dpi=120)


#########################################################
#########################################################
#########################################################


fig, ax = plt.subplots(1, 1, figsize=(8, 8))

for i,node in enumerate(nodes):
	if N_nodes-i <= 10:
		ax.plot(iterations_saved, np.sum(simulation_data[0, :, N_candidates:, node], axis=1),
		        "k--", alpha=1, linewidth=1.1)
	else:
		ax.plot(iterations_saved, np.sum(simulation_data[0, :, N_candidates:, node], axis=1),
		        "-", alpha=0.2, linewidth=0.3)

ax.set_title("Trajectories of the total vote\nstubborness for each node (try 0)")
ax.set_xlabel("number of steps")
ax.set_ylabel("stubborn proportion")


fig.tight_layout(pad=1.0)
fig.savefig(base_path_figure + "total_stubborness_proportion.png", dpi=120)


#########################################################
#########################################################
#########################################################


total_accumulated_states = np.zeros(2*N_candidates)
for node in range(N_nodes):
	total_accumulated_states += np.sum(np.sum(simulation_data[:, :, :, node], axis=0), axis=0)

total_accumulated_states /= np.sum(total_accumulated_states)

stubborn   = total_accumulated_states[N_candidates:]
unstubborn = total_accumulated_states[:N_candidates]

proprtion = stubborn/(stubborn + unstubborn)
proprtion[stubborn == 0] = 0

""" -------------------------------------------------------
-----------------------------------------------------------
------------------------------------------------------- """

fig, ax = plt.subplots(1, 1, figsize=(8,8))

ax.bar(range(N_candidates), total_accumulated_states[:N_candidates] + total_accumulated_states[N_candidates:],
	color='C0', width=0.8, edgecolor="k",
	label='total voter proportion')
ax.bar(range(N_candidates), total_accumulated_states[N_candidates:],
	width=0.8, edgecolor="k", hatch="XXX",
	label='struborn voter proportion')
ax.bar(range(N_candidates), proprtion,
	color='C1', width=0.4, edgecolor="k",
	label='struborn voter relative proportion')

ax.set_xticks(range(N_candidates), labels=candidates)
ax.tick_params(axis='x', labelrotation=45)
ax.legend()


fig.tight_layout(pad=1.0)
fig.savefig(base_path_figure + "average_voter_state.png", dpi=120)


#########################################################
#########################################################
#########################################################


fig, axes = plt.subplots(1, 3, figsize=(6*3, 5))


for i_ax in range(3):
	for i,node in enumerate(nodes):
		stubborn   = simulation_data[0, :, N_candidates+interesting_candidates[i_ax], node]
		unstubborn = simulation_data[0, :,              interesting_candidates[i_ax], node]

		proprtion = np.divide(stubborn, unstubborn + stubborn)
		proprtion[stubborn == 0] = 0

		if N_nodes-i <= 10:
			axes[i_ax].plot(iterations_saved, proprtion,
			         "k--", alpha=1, linewidth=1.1)
		else:
			axes[i_ax].plot(iterations_saved, proprtion,
			         "-", alpha=0.2, linewidth=0.3)

	axes[i_ax].set_title(f"Trajectories of the proportion of stubborn\n voter for each node for { candidates[interesting_candidates[i_ax]] } (try 0)")
	axes[i_ax].set_xlabel("number of steps")
	axes[i_ax].set_ylabel("stubborn proportion")
	axes[i_ax].set_ylim([0, 0.4])


fig.tight_layout(pad=1.0)
fig.savefig(base_path_figure + "stubborness_proportion.png", dpi=120)


#########################################################
#########################################################
#########################################################


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6*2, 5))

ax1b = ax1.twinx()
general_votes = np.sum(simulation_data[:, :, :N_candidates, :] + simulation_data[:, :, N_candidates:, :], axis=-1)
counties_votes = [np.sum(simulation_data[:, :, :N_candidates, county] + simulation_data[:, :, N_candidates:, county], axis=-1) for county in counties]
it_start = [0]
for i in range(N_it//n_election):
	for j in range(N_it//n_save):
		if iterations_saved[j] >= iterations_elections[i]:
			it_start.append(j)
			break

for i in range(N_try):
	for j in range(N_it//n_election):
		ax1.plot(iterations_saved[it_start[j]:it_start[j+1]], general_votes[i, it_start[j]:it_start[j+1], int(general_election_results[i, j])],
		         "--b", alpha=0.5, label="up-vote proportion" if i==0 else None)

ax1.set_title("Voting results for national election")
ax1.set_xlabel("iterations")
ax1.set_ylabel("Vote intention for current elected candidate")

""" -------------------------------------------------------
-----------------------------------------------------------
------------------------------------------------------- """

#ax1.legend()
for i in range(N_try):
	ax1b.plot(iterations_elections, general_election_results[i, :],
	          "-k", alpha=0.5, label="Election results" if i==0 else None)

ax1b.set_ylabel("Winning candidate index")

""" -------------------------------------------------------
-----------------------------------------------------------
------------------------------------------------------- """

ax2b = ax2.twinx()
for i in range(N_try):
	for k in range(N_counties):
		for j in range(N_it//n_election):
			ax2.plot(iterations_saved[it_start[j]:it_start[j+1]], counties_votes[k][i, it_start[j]:it_start[j+1], int(counties_election_results[i, j, k])],
			        "--b", alpha=0.4, label="up-vote proportion" if i==0 else None)

ax2.set_title("Voting results for county elections")
ax2.set_xlabel("iterations")
ax2.set_ylabel("Vote intention for current elected candidate")

""" -------------------------------------------------------
-----------------------------------------------------------
------------------------------------------------------- """

for i in range(N_try):
	for j in range(N_counties):
		ax2b.plot(iterations_elections, counties_election_results[i, :, j],
		          "k-", alpha=0.1, label="county result" if i==0 and j==0 else None)

ax2b.set_ylabel("Winning candidate index")
ax2b.legend()

fig.tight_layout(pad=1.0)
fig.savefig(base_path_figure + "election_results.png", dpi=120)


#########################################################
#########################################################
#########################################################


fig, ax = plt.subplots(1, 1, figsize=(8, 8))

for i,node in enumerate(nodes):
	if N_nodes-i <= 10:
		ax.plot(iterations_saved, normalized_distortion_coefs[node, :],
		         "k--", alpha=1, linewidth=1.1)
	else:
		ax.plot(iterations_saved, normalized_distortion_coefs[node, :],
		         "-", alpha=0.2, linewidth=0.3)

ax.set_title("Trajectories of the normalized distortion coeffecients")
ax.set_ylabel("normalized distortion coeffecients")
ax.set_xlabel("number of steps")
ax.set_yscale("log")


fig.tight_layout(pad=1.0)
fig.savefig(base_path_figure + "normalized_distortion_coefs.png", dpi=120)