#!/usr/bin/env python3

from util.util import *

import pandas as pd
import numpy as np
import ot
from matplotlib import pyplot as plt
from scipy import interpolate

election_id  = "france_pres_tour1_2022"
departements = ["34", "74"]

interesting_candidate_idx = [4, 4]
candidate_color           = ["tab:brown", "tab:brown"]

input_file_names = {
	"france_pres_tour1_2022" : "data/france_pres_tour1_2022_preprocessed.csv"
}

fig_file_name = [
	["results/fig_1a.png", "results/fig_1b.png"],
	["results/fig_2a.png", "results/fig_2b.png"],
	["results/fig_3a.png", "results/fig_3b.png"]
]

epsilon = -0.02

def compute_distance(lon1, lat1, lon2, lat2):
	R = 6371e3
	phi1         = lat1 * np.pi/180
	phi2         = lat2 * np.pi/180
	delta_phi    = (lat2-lat1) * np.pi/180
	delta_lambda = (lon2-lon1) * np.pi/180

	a = np.sin(delta_phi/2)    * np.sin(delta_phi/2) + \
		np.cos(phi1)           * np.cos(phi2) * \
		np.sin(delta_lambda/2) ** 2
	c = 2 * np.atan2(np.sqrt(a), np.sqrt(1-a))

	return R * c

""" ##############################################
##################################################
read field from the preprocessed election database
##################################################
############################################## """

print(f"Reading data from \"{ input_file_names[election_id] }\"")
election_database = pd.read_csv(input_file_names[election_id], low_memory=False)

""" #####################
apply geographical filter
##################### """

for filter_idx,geographical_filter in enumerate(departements):
	geographical_mask          = election_database["code_commune"].str[0:2] == geographical_filter
	filtered_election_database = election_database[geographical_mask]
	filtered_election_database = filtered_election_database.dropna(subset=["longitude", "latitude"]).reset_index(drop=True)

	""" ############
	################
	compute distance
	################
	############ """

	num_nodes       = len(filtered_election_database["longitude"])
	distance_matrix = np.zeros((num_nodes, num_nodes))

	lon = np.array(filtered_election_database["longitude"])
	lat = np.array(filtered_election_database["latitude" ])
	for i in range(1, num_nodes):
		distance_matrix[i, :i] = np.maximum(compute_distance(lon[:i], lat[:i], lon[i], lat[i]), 10)
		distance_matrix[:i, i] = distance_matrix[i, :i]
	distance_matrix_alpha = np.pow(distance_matrix, 1 + epsilon)

	unitary_direction_matrix = np.zeros((distance_matrix.shape[0], distance_matrix.shape[0], 2))
	
	for i in range(1, num_nodes):		
		distances_line = distance_matrix[i, :i]
		unitary_direction_matrix[i, :i, 0] = (lon[i] - lon[:i]) / distances_line
		unitary_direction_matrix[:i, i, 0] = -unitary_direction_matrix[i, :i, 0]
		unitary_direction_matrix[i, :i, 1] = (lat[i] - lat[:i]) / distances_line
		unitary_direction_matrix[:i, i, 1] = -unitary_direction_matrix[i, :i, 1]

	""" ##########################
	##############################
	compute/read optimal transport
	##############################
	########################## """

	## TO BE READ FROM THE FILE
	candidate_list            = ['ARTHAUD', 'ROUSSEL', 'MACRON', 'LASSALLE', 'LE PEN', 'ZEMMOUR', 'MÉLENCHON', 'HIDALGO', 'JADOT', 'PÉCRESSE', 'POUTOU', 'DUPONT-AIGNAN']

	""" #####################
	compute optimal transport
	##################### """

	ot_dist_contribution            = np.zeros(                      len(filtered_election_database["Votants"]))
	ot_dist_contribution_candidates = np.zeros((len(candidate_list), len(filtered_election_database["Votants"])))

	ot_direction_per_candidate = np.zeros((len(filtered_election_database["Votants"]), 2, len(candidate_list)))
	ot_direction               = np.zeros((len(filtered_election_database["Votants"]), 2))

	total_voting_population = np.sum(  filtered_election_database["Votants"])
	reference_distrib       = np.array(filtered_election_database["Votants"]) / total_voting_population

	candidate_padding_length = max([len(x) for x in candidate_list])
	for i,candidate in enumerate(candidate_list):
		total_vote_candidate = np.sum(  filtered_election_database[candidate + " Voix"])
		candidate_distrib    = np.array(filtered_election_database[candidate + " Voix"]) / total_vote_candidate

		candidate_ot_mat = ot.emd(reference_distrib, candidate_distrib, distance_matrix_alpha)*distance_matrix

		ot_dist_contribution_candidates[i]  = (candidate_ot_mat.sum(axis=0) + candidate_ot_mat.sum(axis=1)) / 2 / reference_distrib
		ot_dist_contribution               += ot_dist_contribution_candidates[i] * total_vote_candidate / total_voting_population
		
		ot_direction_per_candidate[:, 0,            i]  = ((unitary_direction_matrix[:, :, 0]*candidate_ot_mat).sum(axis=0) + (unitary_direction_matrix[:, :, 0].T*candidate_ot_mat).sum(axis=1)) / 2 / reference_distrib
		ot_direction_per_candidate[:, 1,            i]  = ((unitary_direction_matrix[:, :, 1]*candidate_ot_mat).sum(axis=0) + (unitary_direction_matrix[:, :, 1].T*candidate_ot_mat).sum(axis=1)) / 2 / reference_distrib
		ot_direction                                   += ot_direction_per_candidate[:, :, i] * total_vote_candidate / total_voting_population

	map_ratio = get_map_ratio(lon, lat)

	""" ######
	##########
	plot votes
	##########
	###### """

	fig, ax = plt.subplots(1, 1, figsize=(6 + 2, 6/map_ratio + 2))

	vote_distrib_candidate    = np.array(filtered_election_database[candidate_list[interesting_candidate_idx[filter_idx]] + " Voix"])
	vote_proportion_candidate = vote_distrib_candidate / np.array(filtered_election_database["Votants"])

	ax.scatter(lon, lat, c=vote_proportion_candidate, s=30, alpha=0.6)
	ax.scatter(lon, lat, c=vote_proportion_candidate, s=10, alpha=0.6)
	pl = ax.scatter(lon, lat, c=vote_proportion_candidate, s=1)

	cbar = fig.colorbar(pl, label="proportion of votes")

	ax.set_aspect(map_ratio)
	ax.set_title(f"Vote proportion for { candidate_list[interesting_candidate_idx[filter_idx]] }\nduring the 2022 presidencial elections")

	fig.tight_layout(pad=1.0)
	fig.savefig(fig_file_name[0][filter_idx])

	""" ##################
	######################
	plot optimal transport
	######################
	################## """

	fig, ax = plt.subplots(1, 1, figsize=(6 + 2, 6/map_ratio + 2))

	x, y = np.linspace(min(lon), max(lon) ,1000), np.linspace(min(lat), max(lat), 1000)
	X, Y = np.meshgrid(x,y)

	Ti = interpolate.griddata((lon, lat), ot_dist_contribution, (X, Y), method="cubic")
	pl = ax.contourf(X, Y, Ti)

	#ax.scatter(lon, lat, c=ot_dist_contribution, s=30, alpha=0.6)
	#ax.scatter(lon, lat, c=ot_dist_contribution, s=10, alpha=0.6)
	#pl = ax.scatter(lon, lat, c=ot_dist_contribution, s=1)

	cbar = fig.colorbar(pl, label="local contribution [m]")

	ax.set_aspect(map_ratio)
	ax.set_title("Local segregation index in Paris\nduring the 2022 presidencial elections")

	fig.tight_layout(pad=1.0)
	fig.savefig(fig_file_name[1][filter_idx])

	""" ##########
	##############
	plot direction
	##############
	########## """

	fig, ax = plt.subplots(1, 1, figsize=(6, 6/map_ratio + 1))

	relative_direction = np.power(ot_direction[:, 0], 2) + np.power(ot_direction[:, 1], 2) / ot_dist_contribution

	ax.quiver(
		lon, lat,
		ot_direction[:, 0] / ot_dist_contribution,
		ot_direction[:, 1] / ot_dist_contribution
	)

	ax.set_aspect(map_ratio)
	ax.set_title("Direction of segregation in Paris\nduring the 2022 presidencial elections")

	fig.savefig(fig_file_name[2][filter_idx])