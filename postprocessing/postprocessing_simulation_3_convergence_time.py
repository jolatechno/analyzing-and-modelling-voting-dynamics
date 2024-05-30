#!/usr/bin/python3

from util.plot import *

import networkx as nx
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

N_try      = config["convergence_time"]["N_try"]
N_it       = config["convergence_time"]["N_it"]
n_save     = config["convergence_time"]["n_save"]
with h5py.File(base_path + output_file, "r") as file:
	N_nodes = len(file["geo_data"]["lat"])

iterations_saved = np.arange(0, N_it-1, n_save)


#########################################################
#########################################################
#########################################################


longitude, latitude = np.zeros(N_nodes), np.zeros(N_nodes)
populations         = np.zeros(N_nodes)

simulation_data = np.zeros((N_try, N_it//n_save, N_candidates, N_nodes))

neighbors = []

with h5py.File(base_path + output_file, "r") as file:
	latitude [:]    = file["geo_data"]["lat"]
	longitude[:]    = file["geo_data"]["lon"]
	#populations[:] = file["full_analysis"]["voter_population"]

	neighbors_begin_end = file["network"]["neighbors_begin_end_idx"]
	for begin,end in zip(neighbors_begin_end[:-1], neighbors_begin_end[1:]):
		neighbors.append(file["network"]["neighbors"][begin:end])

	for i in range(N_try):
		for k in range(N_candidates):
			field_name = "proportions_" + str(k)
			simulation_data[i, 0, k, :] = file["initial_state"][field_name]

		for j in range(1, N_it//n_save):
			it = j*n_save
			state_name = "states_" + str(i) + "_" + str(it)
			for k in range(N_candidates):
				field_name = "proportions_" + str(k)
				simulation_data[i, j, k, :] = file[state_name][field_name]

""" -------------------------------------------------------
-----------------------------------------------------------
------------------------------------------------------- """

graph = nx.Graph()
for node in range(N_nodes):
	graph.add_node(node)

	for neighbor in neighbors[node]:
		graph.add_edge(node, neighbor)


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

nodes = np.arange(N_nodes)
np.random.shuffle(nodes)

for i,node in enumerate(nodes):
	if N_nodes-i <= 10:
		ax.plot(iterations_saved, simulation_data[0, :, 1, node],
		        "k--", alpha=1, linewidth=1.1)
	else:
		ax.plot(iterations_saved, simulation_data[0, :, 1, node],
		        "-", alpha=0.2, linewidth=0.3)

ax.set_title("Trajectory of the up-vote proportion")
ax.set_ylabel("up-vote proportion")
ax.set_xlabel("number of steps")
ax.set_ylim([0, 0.1])

fig.savefig(base_path_figure + "vote_trajectory.png", dpi=200)


#########################################################
#########################################################
#########################################################

def KLdiv_1D(Q, P):
	epsilon = 1e-20
	return P*np.log2(np.clip(P, epsilon, 1)/np.clip(Q, epsilon, 1)) + \
		(1-P)*np.log2(np.clip(1-P, epsilon, 1)/np.clip(1-Q, epsilon, 1))

def KLdiv(Q, P):
	epsilon = 1e-20
	return np.sum(P *np.log2(np.clip(P, epsilon, 1)/np.clip(Q, epsilon, 1)), axis=1)

KL_div = np.zeros((N_try, N_it//n_save, N_nodes))

for itry in range(N_try):
	final_state = simulation_data[itry, -1, :, :].T
	for i in range(N_it//n_save):
		KL_div[itry, i, :] = KLdiv(simulation_data[itry, i, :, :].T, final_state)

""" -------------------------------------------------------
-----------------------------------------------------------
------------------------------------------------------- """

fig, ax = plt.subplots(1, 1, figsize=(8,8))

for i,node in enumerate(nodes):
	if N_nodes-i <= 10:
		ax.plot(iterations_saved, KL_div[0, :, node],
		        "k--", alpha=1, linewidth=1.1)
	else:
		ax.plot(iterations_saved, KL_div[0, :, node],
		        "-", alpha=0.2, linewidth=0.3)

ax.set_title("KL-divergence trajectory between the\ni-th state and the final state")
ax.set_ylabel("KL-divergence")
ax.set_xlabel("number of steps")
ax.set_ylim([0, 2])

fig.savefig(base_path_figure + "kl-div_trajectory.png", dpi=200)


#########################################################
#########################################################
#########################################################

convergence_thresholds = np.linspace(0, 1, 300)
focal_times = np.zeros((N_try, len(convergence_thresholds), N_nodes))
for itry in range(N_try):
	for node in range(N_nodes):
		idx = N_it//n_save - 1

		for i,thresh in enumerate(convergence_thresholds):
			while KL_div[itry, idx, node] < thresh and idx > 0:
				idx -= 1
			focal_times[itry, i, node] = iterations_saved[idx]

""" -------------------------------------------------------
-----------------------------------------------------------
------------------------------------------------------- """

fig, ax = plt.subplots(1, 1, figsize=(8,8))
for i,node in enumerate(nodes):
	if N_nodes-i <= 10:
		ax.plot(convergence_thresholds, focal_times[0, :, node],
		        "k--", alpha=1, linewidth=1.1)
	else:
		ax.plot(convergence_thresholds, focal_times[0, :, node],
		        "-", alpha=0.2, linewidth=0.3)

ax.set_title("Focal time trajectory")
ax.set_ylabel("focal time")
ax.set_xlabel("convergence thresholds")

fig.savefig(base_path_figure + "focal_time_trajectory.png", dpi=200)


#########################################################
#########################################################
#########################################################


distortion_coefficients = np.zeros(N_nodes)

for i in range(0, len(convergence_thresholds)-1):
	for node in range(N_nodes):
		for itry in range(N_try):
			distortion_coefficients[node] += (convergence_thresholds[i+1]-convergence_thresholds[i])* \
				(focal_times[itry, i, node]+focal_times[itry, i+1, node])/2

""" -------------------------------------------------------
-----------------------------------------------------------
------------------------------------------------------- """

fig, ax = plt.subplots(1, 1, figsize=(8,8))

ax.hist(distortion_coefficients)

ax.set_title("Pseudo-distortion coefficient distribution")
ax.set_ylabel("Number of nodes")
ax.set_xlabel("Pseudo-distortion coefficient")

fig.savefig(base_path_figure + "histograms.png", dpi=200)


#########################################################
#########################################################
#########################################################


fig, ax = plt.subplots(1, 1, figsize=(8,8))

pl = ax.scatter(longitude, latitude, c=np.clip(distortion_coefficients, 2000, 15000), s=4)

cbar = fig.colorbar(pl, label="pseudo-distortion coefficient")

ax.set_aspect('equal', adjustable='box')
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


ax1.plot(dist_coef, distortion_coefficients, "+")

ax1.set_title("normalized distortion coefficient\nbased on number of voting bureau versus\nbased on agregated population")
ax1.set_ylabel("distortion coefficient [based on convegrence time]")
ax1.set_xlabel("distortion coefficient [based on number of voting bureau]")

""" -------------------------------------------------------
-----------------------------------------------------------
------------------------------------------------------- """

ax2.plot(dist_coef_pop, distortion_coefficients, "+")

ax2.set_title("normalized distortion coefficient\nbased on distance versus\nbased on agregated population")
ax2.set_ylabel("distortion coefficient [based on convegrence time]")
ax2.set_xlabel("distortion coefficient [based on agregated population]")

""" -------------------------------------------------------
-----------------------------------------------------------
------------------------------------------------------- """

ax3.plot(dist_coef_dist, distortion_coefficients, "+")

ax3.set_title("normalized distortion coefficient\nbased on distance versus\nbased on agregated population")
ax3.set_ylabel("distortion coefficient [based on convegrence time]")
ax3.set_xlabel("distortion coefficient [based on distance]")


fig.tight_layout(pad=2.0)
fig.savefig(base_path_figure + "distortion_coef_comparison.png", dpi=200)

