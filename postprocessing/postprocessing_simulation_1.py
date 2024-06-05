#!/usr/bin/python3

from util.plot import *

import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import copy
import h5py
import json
import sys

base_path = "../computation/output/"

input_filename = "output-simulation.h5"

N_nodes    = 800
N_counties = 3
N_try      = 10
N_it       = 3001
n_election = 500
n_save     = 10

iterations_saved     = np.arange(0, N_it-1, n_save)
iterations_elections = np.arange(0, N_it-1, n_election)


#########################################################
#########################################################
#########################################################


simulation_data = np.zeros((N_try, N_it//n_save, 4, N_nodes))
populations     = np.zeros(N_nodes)

stubborn_equilibrium = np.zeros((2, N_nodes))
neighbors = []
counties  = []

general_election_results      = np.zeros((N_try, N_it//n_election))
general_election_proportions  = np.zeros((N_try, N_it//n_election))
counties_election_results     = np.zeros((N_try, N_it//n_election, N_counties))
counties_election_proportions = np.zeros((N_try, N_it//n_election, N_counties))

with h5py.File(base_path + input_filename, "r") as file:
	counties_begin_end = file["counties"]["counties_begin_end_idx"]
	for begin,end in zip(counties_begin_end[:-1], counties_begin_end[1:]):
		counties.append(file["counties"]["counties"][begin:end])
	
	populations[:] = file["initial_state"]["population"]

	neighbors_begin_end = file["network"]["neighbors_begin_end_idx"]
	for begin,end in zip(neighbors_begin_end[:-1], neighbors_begin_end[1:]):
		neighbors.append(file["network"]["neighbors"][begin:end])

	stubborn_equilibrium[0, :] = file["initial_state"]["stubborn_equilibrium_false"]
	stubborn_equilibrium[1, :] = file["initial_state"]["stubborn_equilibrium_true"]

	for i in range(N_try):
		for k in range(4):
			field_name = "proportions_" + str(k)
			simulation_data[i, 0, k, :] = file["initial_state"][field_name]

		for j in range(1, N_it//n_save):
			it = j*n_save
			state_name = "states_" + str(i) + "_" + str(it)
			for k in range(4):
				field_name = "proportions_" + str(k)
				simulation_data[i, j, k, :] = file[state_name][field_name]

		for j in range(0, N_it//n_election):
			it = j*n_election
			general_election_name  = "general_election_result_" + str(i) + "_" + str(it)
			counties_election_name = "counties_election_result_" + str(i) + "_" + str(it)

			general_election_results[i, j]     = file[general_election_name]["result"][0]
			general_election_proportions[i, j] = file[general_election_name]["proportion"][0]

			for k in range(N_counties):
				county_name = "election_result_" + str(k)
				counties_election_results[i, j, k]     = file[counties_election_name][county_name]["result"][0]
				counties_election_proportions[i, j, k] = file[counties_election_name][county_name]["proportion"][0]

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


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10))

X_eq_vs_vote = []
Y_eq_vs_vote = []
for i in range(N_try):
	ax1.plot(stubborn_equilibrium[1] - stubborn_equilibrium[0],
			 simulation_data[i, -1, 1, :] + simulation_data[i, -1, 3, :], "+c",)

	X_eq_vs_vote.extend(stubborn_equilibrium[1] - stubborn_equilibrium[0])
	Y_eq_vs_vote.extend(simulation_data[i, -1, 1, :] + simulation_data[i, -1, 3, :])

reg_eq_vs_vote = LinearRegression().fit(
	np.expand_dims(X_eq_vs_vote, 1),
	np.expand_dims(Y_eq_vs_vote, 1))
score_eq_vs_vote = reg_eq_vs_vote.score(
	np.expand_dims(X_eq_vs_vote, 1),
	np.expand_dims(Y_eq_vs_vote, 1))
X = np.linspace(min(X_eq_vs_vote), max(X_eq_vs_vote), 100)
ax1.plot(X, reg_eq_vs_vote.intercept_ + X*reg_eq_vs_vote.coef_[0][0], "--r",
	 	 label=f"linear fit, R²={ round(score_eq_vs_vote, 3) },\ncoef = { round(reg_eq_vs_vote.coef_[0][0], 3) }")

ax1.set_title("Up-vote proportion versus\nthe political bias equilibrium")
ax1.set_xlabel("up-eq. - down-eq.")
ax1.set_ylabel("up-vote porportion")
ax1.legend()

""" -------------------------------------------------------
-----------------------------------------------------------
------------------------------------------------------- """

X_eq_vs_bias = []
Y_eq_vs_bias = []
for i in range(N_try):
	y = np.divide(
			simulation_data[i, -1, 3, :],
			simulation_data[i, -1, 3, :] + simulation_data[i, -1, 1, :]
		) - np.divide(
			simulation_data[i, -1, 2, :],
			simulation_data[i, -1, 2, :] + simulation_data[i, -1, 0, :]
		)

	ax2.plot(stubborn_equilibrium[1] - stubborn_equilibrium[0], y, "+c")
	X_eq_vs_bias.extend(stubborn_equilibrium[1] - stubborn_equilibrium[0])
	Y_eq_vs_bias.extend(y)

reg_eq_vs_bias = LinearRegression().fit(
	np.expand_dims(X_eq_vs_bias, 1),
	np.expand_dims(Y_eq_vs_bias, 1))
score_eq_vs_bias = reg_eq_vs_bias.score(
	np.expand_dims(X_eq_vs_bias, 1),
	np.expand_dims(Y_eq_vs_bias, 1))
ax2.plot(X, reg_eq_vs_bias.intercept_ + X*reg_eq_vs_bias.coef_[0][0], "--r",
         label=f"linear fit, R²={ round(score_eq_vs_bias, 3) },\ncoef = { round(reg_eq_vs_bias.coef_[0][0], 3) }")

ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.set_title("Polititical bias versus the\npolitical bias equilibrium")
ax2.set_xlabel("up eq. - down eq.")
ax2.set_ylabel("up-stubborn proportion - down-stubborn prop.")
ax2.legend()

""" -------------------------------------------------------
-----------------------------------------------------------
------------------------------------------------------- """

X_eq_vs_neighbors = []
Y_eq_vs_neighbors = []
for i in range(N_try):
	neighbor_mean_true_vote = np.zeros(N_nodes)
	for node in range(N_nodes):
		neighbor_mean_true_vote[node] = np.mean(
		simulation_data[i, -1, 1, neighbors[node]] +
		simulation_data[i, -1, 3, neighbors[node]])

	ax3.plot(neighbor_mean_true_vote,
	         simulation_data[i, -1, 1, :] + simulation_data[i, -1, 3, :], "+c")
	X_eq_vs_neighbors.extend(neighbor_mean_true_vote)
	Y_eq_vs_neighbors.extend(simulation_data[i, -1, 1, :] + simulation_data[i, -1, 3, :])

reg_eq_vs_neighbors = LinearRegression().fit(
	np.expand_dims(X_eq_vs_neighbors, 1),
	np.expand_dims(Y_eq_vs_neighbors, 1))
score_eq_vs_neighbors = reg_eq_vs_neighbors.score(
	np.expand_dims(X_eq_vs_neighbors, 1),
	np.expand_dims(Y_eq_vs_neighbors, 1))
X = np.linspace(min(X_eq_vs_neighbors), max(X_eq_vs_neighbors), 100)
ax3.plot(X, reg_eq_vs_neighbors.intercept_ + X*reg_eq_vs_neighbors.coef_[0][0], "--r",
         label=f"linear fit, R²={ round(score_eq_vs_neighbors, 3) },\ncoef = { round(reg_eq_vs_neighbors.coef_[0][0], 3) }")

ax3.set_title("Up-vote proportion versus mean\nneighbors up-vote proportion")
ax3.set_xlabel("mean neighbors up-vote proportion")
ax3.set_ylabel("up-vote proportion")
ax3.legend()

""" -------------------------------------------------------
-----------------------------------------------------------
------------------------------------------------------- """

X_eq_vs_election_results = []
Y_eq_vs_election_results = []
for i in range(N_try):
	election_results = np.ones(N_nodes)
	election_results *= 2*general_election_results[i, -1] - 1
	for icounty in range(N_counties):
		election_results[counties[icounty]] += counties_election_results[i, -1, icounty] - 0.5

	ax4.plot(election_results,
		     simulation_data[i, -1, 1, :] + simulation_data[i, -1, 3, :], "+c")

	X_eq_vs_election_results.extend(election_results)
	Y_eq_vs_election_results.extend(simulation_data[i, -1, 1, :] + simulation_data[i, -1, 3, :])

for i,x_target in enumerate([-1.5, -0.5, 0.5, 1.5]):
	y_mean = np.mean([y for (x, y) in zip(X_eq_vs_election_results, Y_eq_vs_election_results) if x == x_target])
	ax4.plot([x_target,], [y_mean,],
		"or", label="mean proportion" if i==0 else None)

reg_eq_vs_election_results = LinearRegression().fit(
	np.expand_dims(X_eq_vs_election_results, 1),
	np.expand_dims(Y_eq_vs_election_results, 1))
score_eq_vs_election_results = reg_eq_vs_election_results.score(
	np.expand_dims(X_eq_vs_election_results, 1),
	np.expand_dims(Y_eq_vs_election_results, 1))
X = np.linspace(-1.6, 1.6, 100)
ax4.plot(X, reg_eq_vs_election_results.intercept_ + X*reg_eq_vs_election_results.coef_[0][0], "--r",
         label=f"linear fit, R²={ round(score_eq_vs_election_results, 3) },\ncoef = { round(reg_eq_vs_election_results.coef_[0][0], 3) }")
ax4.yaxis.set_label_position("right")
ax4.yaxis.tick_right()
ax4.set_title("Up-vote proportion versus\nelections results")
ax4.set_xlabel("election results (1*general + 0.5*counties)")
ax4.set_ylabel("up-vote proportion")
ax4.legend()

""" -------------------------------------------------------
-----------------------------------------------------------
------------------------------------------------------- """

fig.tight_layout(pad=1.0)
fig.savefig("figures/simulation/scatter.png", dpi=120)


#########################################################
#########################################################
#########################################################


fig, axes = plt.subplots(1, 2, figsize=(10,5))


for i_ax in range(2):
	for i,node in enumerate(nodes):
		stubborn   = simulation_data[0, :, 2+i_ax, node]
		unstubborn = simulation_data[0, :,   i_ax, node]

		proprtion = np.divide(stubborn, unstubborn + stubborn)
		proprtion[stubborn == 0] = 0
		
		if N_nodes-i <= 10:
			axes[i_ax].plot(iterations_saved, proprtion,
			         "k--", alpha=1, linewidth=1.1)
		else:
			axes[i_ax].plot(iterations_saved, proprtion,
			         "-", alpha=0.2, linewidth=0.3)

	axes[i_ax].set_title(f"Trajectories of the { "up" if i_ax==1 else "down" }-vote\nstubborness for each node (try 0)")
	axes[i_ax].set_xlabel("number of steps")
	axes[i_ax].set_ylabel("stubborn proportion")
	axes[i_ax].set_ylim([0, 0.6])

axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()


fig.tight_layout(pad=1.0)
fig.savefig("figures/simulation/trajectories.png", dpi=120)


#########################################################
#########################################################
#########################################################


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

ax1b = ax1.twinx()
up_votes = simulation_data[:, :, 1, :] + simulation_data[:, :, 3, :]

for node in range(N_nodes):
	up_votes[:, :, node] *= populations[node]
general_upvotes = np.sum(up_votes, axis=2)/np.sum(populations)
counties_upvotes = np.array([np.sum(up_votes[:, :, county], axis=2)/np.sum(populations[county]) for county in counties])

for i in range(N_try):
	ax1.plot(iterations_saved, general_upvotes[i, :], "--b",
	         label="up-vote proportion" if i==0 else None)

ax1.set_title("Voting results for national election")
ax1.set_xlabel("iterations")
ax1.set_ylabel("up-vote proportion")
#ax1.legend()

for i in range(N_try):
	ax1b.plot(iterations_elections, general_election_results[i, :], "-k",
	          label="up-vote win" if i==0 else None)

ax1b.set_ylabel("up-vote wins")
#ax1b.legend()

""" -------------------------------------------------------
-----------------------------------------------------------
------------------------------------------------------- """

ax2b = ax2.twinx()
for i in range(N_try):
	for j in range(N_counties):
		ax2.plot(iterations_saved, counties_upvotes[j, i, :], "--b", alpha=0.4,
		         label="county up-vote proportion" if i==0 and j==0 else None)

ax2.set_title("Voting results for county elections")
ax2.set_xlabel("iterations")
ax2.set_ylabel("up-vote proportion")
#ax2.legend()

for i in range(N_try):
	for j in range(N_counties):
		ax2b.plot(iterations_elections, counties_election_results[i, :, j], "--r", alpha=0.1,
		          label="county result" if i==0 and j==0 else None)

	mean_result = np.mean(counties_election_results[i, :, :], axis=1)
	ax2b.plot(iterations_elections, mean_result, "-k", alpha=0.7,
	          label="up-vote win-rate on all counties" if i==0 else None)
	ax2b.set_ylabel("up-vote wins/win-rate")
	#ax2b.legend()


fig.tight_layout(pad=1.0)
fig.savefig("figures/simulation/election_results.png", dpi=120)