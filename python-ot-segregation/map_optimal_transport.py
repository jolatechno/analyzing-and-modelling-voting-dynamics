#!/usr/bin/env python3

from util.util import *

import pandas as pd
import numpy as np
import ot
from matplotlib import pyplot as plt

election_id            = "france_pres_tour1_2022"
geographical_filter_id = "Paris"

input_file_names = {
	"france_pres_tour1_2022" : "data/france_pres_tour1_2022_preprocessed.csv"
}
distance_file_names = {
	"Paris"             : "data/distances_paris.csv",
	"Petite_couronne"   : "data/distances_petite_couronne.csv",
	"Region_parisienne" : "data/distances_region_parisienne.csv",
	"Metropole"         : "data/distances_metropole.csv"
}

save_file_name = {
	"france_pres_tour1_2022" : {
		"Paris"             : "data/france_pres_tour1_2022_paris_ot_map.csv",
		"Petite_couronne"   : "data/france_pres_tour1_2022_petite_couronne_ot_map.csv",
		"Region_parisienne" : "data/france_pres_tour1_2022_region_parisienne_ot_map.csv",
		"Metropole"         : "data/france_pres_tour1_2022_metropole_ot_map.csv"
	}
}
fig_file_name = {
	"france_pres_tour1_2022" : {
		"Paris"             : "results/france_pres_tour1_2022_paris_ot_map.png",
		"Petite_couronne"   : "results/france_pres_tour1_2022_petite_couronne_ot_map.png",
		"Region_parisienne" : "results/france_pres_tour1_2022_region_parisienne_ot_map.png",
		"Metropole"         : "results/france_pres_tour1_2022_metropole_ot_map.png"
	}
}

geographical_filter_departement_list = {
	"Paris"             : ["75"],
	"Petite_couronne"   : ["75", "92", "93", "94"],
	"Region_parisienne" : ["75", "92", "93", "94", "77", "78", "91", "95"],
	"Metropole"         : [str(idx).zfill(2) for idx in range(1, 95+1)]
}

plot_direction = {
	"Paris"             : True,
	"Petite_couronne"   : False,
	"Region_parisienne" : False,
	"Metropole"         : False
}
direction_file_names = {
	"Paris"             : {
		"x" : "data/direction_paris_x.csv",
		"y" : "data/direction_paris_y.csv"
	}
}

epsilon = -0.02

""" ##############################################
##################################################
read field from the preprocessed election database
##################################################
############################################## """

print(f"Reading data from \"{ input_file_names[election_id] }\"")
election_database = pd.read_csv(input_file_names[election_id], low_memory=False)

geographical_mask          = election_database["code_commune"].str[0:2].isin(geographical_filter_departement_list[geographical_filter_id])
filtered_election_database = election_database[geographical_mask]
filtered_election_database = filtered_election_database.dropna(subset=["longitude", "latitude"]).reset_index(drop=True)

""" ##########
##############
read distances
##############
########## """

print(f"Read distances from \"{ distance_file_names[geographical_filter_id] }\"")
distance_matrix       = np.genfromtxt(distance_file_names[geographical_filter_id], delimiter=',')
distance_matrix_alpha = np.pow(distance_matrix, 1 + epsilon)

if plot_direction[geographical_filter_id]:
	unitary_direction_matrix = np.zeros((distance_matrix.shape[0], distance_matrix.shape[0], 2))
	unitary_direction_matrix[:, :, 0] = np.genfromtxt(direction_file_names[geographical_filter_id]["x"], delimiter=',')
	unitary_direction_matrix[:, :, 1] = np.genfromtxt(direction_file_names[geographical_filter_id]["y"], delimiter=',')

""" ##########################
##############################
compute/read optimal transport
##############################
########################## """

## TO BE READ FROM THE FILE
candidate_list            = ['ARTHAUD', 'ROUSSEL', 'MACRON', 'LASSALLE', 'LE PEN', 'ZEMMOUR', 'MÉLENCHON', 'HIDALGO', 'JADOT', 'PÉCRESSE', 'POUTOU', 'DUPONT-AIGNAN']
interesting_candidate_idx = [2, 4, 5, 6, 7]

try:

	""" ###################
	read optimal transport
	################### """

	print(f"Reading data from \"{ save_file_name[election_id][geographical_filter_id] }\"")
	data = pd.read_csv(save_file_name[election_id][geographical_filter_id], low_memory=False)

except	:

	""" #####################
	compute optimal transport
	##################### """

	ot_dist_contribution            = np.zeros(                      len(filtered_election_database["Votants"]))
	ot_dist_contribution_candidates = np.zeros((len(candidate_list), len(filtered_election_database["Votants"])))
	if plot_direction[geographical_filter_id]:
		ot_direction_per_candidate = np.zeros((len(filtered_election_database["Votants"]), 2, len(candidate_list)))
		ot_direction               = np.zeros((len(filtered_election_database["Votants"]), 2))

	total_voting_population = np.sum(  filtered_election_database["Votants"])
	reference_distrib       = np.array(filtered_election_database["Votants"]) / total_voting_population

	print()
	candidate_padding_length = max([len(x) for x in candidate_list])
	for i,candidate in enumerate(candidate_list):
		total_vote_candidate = np.sum(  filtered_election_database[candidate + " Voix"])
		candidate_distrib    = np.array(filtered_election_database[candidate + " Voix"]) / total_vote_candidate

		candidate_ot_mat = ot.emd(reference_distrib, candidate_distrib, distance_matrix_alpha)*distance_matrix

		ot_dist_contribution_candidates[i]  = (candidate_ot_mat.sum(axis=0) + candidate_ot_mat.sum(axis=1)) / 2 / reference_distrib
		ot_dist_contribution               += ot_dist_contribution_candidates[i] * total_vote_candidate / total_voting_population
		if plot_direction[geographical_filter_id]:
			ot_direction_per_candidate[:, 0,            i]  = ((unitary_direction_matrix[:, :, 0]*candidate_ot_mat).sum(axis=0) + (unitary_direction_matrix[:, :, 0].T*candidate_ot_mat).sum(axis=1)) / 2 / reference_distrib
			ot_direction_per_candidate[:, 1,            i]  = ((unitary_direction_matrix[:, :, 1]*candidate_ot_mat).sum(axis=0) + (unitary_direction_matrix[:, :, 1].T*candidate_ot_mat).sum(axis=1)) / 2 / reference_distrib
			ot_direction                                   += ot_direction_per_candidate[:, :, i] * total_vote_candidate / total_voting_population


	""" ###################
	write optimal transport
	################### """

	data = {
		"optimal_transport_contribution" : ot_dist_contribution,
		"ot_direction_x"                 : ot_direction[:, 0],
		"ot_direction_y"                 : ot_direction[:, 1],
		"latitude"                       : filtered_election_database["latitude"],
		"longitude"                      : filtered_election_database["longitude"]
	}
	for candidate_idx,name in enumerate(candidate_list):
		data["optimal_transport_contribution_" + name] = ot_dist_contribution_candidates[candidate_idx, :]
		data["ot_direction_" + name + "_x"]            = ot_direction_per_candidate[:, 0, candidate_idx]
		data["ot_direction_" + name + "_y"]            = ot_direction_per_candidate[:, 1, candidate_idx]

	print(f"Write to \"{ save_file_name[election_id][geographical_filter_id] }\"")
	pd.DataFrame(data).to_csv(save_file_name[election_id][geographical_filter_id], index=False)

""" ###################
#######################
print optimal transport
#######################
################### """

total_voting_population = np.sum(  filtered_election_database["Votants"])
reference_distrib       = np.array(filtered_election_database["Votants"]) / total_voting_population

candidate_padding_length = max([len(x) for x in candidate_list])
print()
for i,candidate in enumerate(candidate_list):
	total_vote_candidate = np.sum(  filtered_election_database[candidate + " Voix"])

	ot_dist_candidate = np.sum(data["optimal_transport_contribution_" + candidate] * reference_distrib)
	print(f"Optimal transport distance for { candidate.ljust(candidate_padding_length , " ") } electors : { str(round(ot_dist_candidate)).rjust(4, " ") }m with { str(round(total_vote_candidate / total_voting_population * 100)).rjust(2, " ") }% of electors")

ot_dist = np.sum(data["optimal_transport_contribution"] * reference_distrib)
print(f"\nOptimal transport average distance : { round(ot_dist) }m")

""" ##################
######################
plot optimal transport
######################
################## """

map_ratio = get_map_ratio(data["longitude"], data["latitude"])

fig, axes = plt.subplots(3, 2, figsize=(6*2, 5/map_ratio*3))
axes = axes.flatten()

axes[0].scatter(data["longitude"], data["latitude"], c=data["optimal_transport_contribution"], s=30, alpha=0.6)
axes[0].scatter(data["longitude"], data["latitude"], c=data["optimal_transport_contribution"], s=10, alpha=0.6)
pl = axes[0].scatter(data["longitude"], data["latitude"], c=data["optimal_transport_contribution"], s=1)

cbar = fig.colorbar(pl, label="local contribution [m]")

axes[0].set_aspect(map_ratio)
axes[0].set_title("Local segregation index and direction\nof segregation in Paris\nduring the 2022 presidencial elections")

if plot_direction[geographical_filter_id]:
	relative_direction             = np.power(data["ot_direction_x"], 2) + np.power(data["ot_direction_y"], 2) / data["optimal_transport_contribution"]
	"""relative_direction_limit_value = np.percentile(relative_direction, 6)
	percentile_filter              = relative_direction < relative_direction_limit_value

	axes[0].plot(data["longitude"][percentile_filter], data["latitude"][percentile_filter], 'xr', markersize=6)"""

	axes[0].quiver(
		data["longitude"], data["latitude"],
		data["ot_direction_x"] / data["optimal_transport_contribution"],
		data["ot_direction_y"] / data["optimal_transport_contribution"]
	)

for ax,candidate_idx in zip(axes[1:],interesting_candidate_idx):
	ax.scatter(data["longitude"], data["latitude"], c=data["optimal_transport_contribution_" + candidate_list[candidate_idx]], s=30, alpha=0.6)
	ax.scatter(data["longitude"], data["latitude"], c=data["optimal_transport_contribution_" + candidate_list[candidate_idx]], s=10, alpha=0.6)
	pl = ax.scatter(data["longitude"], data["latitude"], c=data["optimal_transport_contribution_" + candidate_list[candidate_idx]], s=1)

	cbar = fig.colorbar(pl, label="local contribution [m]")

	ax.set_aspect(map_ratio)
	ax.set_title(f"map of the local contribution for { candidate_list[candidate_idx] }")

	if plot_direction[geographical_filter_id]:
		relative_direction             = np.power(data["ot_direction_" + candidate_list[candidate_idx] + "_x"], 2) + np.power(data["ot_direction_" + candidate_list[candidate_idx] + "_y"], 2) / data["optimal_transport_contribution"]
		"""relative_direction_limit_value = np.percentile(relative_direction, 6)
		percentile_filter              = relative_direction < relative_direction_limit_value

		ax.plot(data["longitude"][percentile_filter], data["latitude"][percentile_filter], 'xr', markersize=6)"""

		ax.quiver(
			data["longitude"], data["latitude"],
			data["ot_direction_" + candidate_list[candidate_idx] + "_x"] / data["optimal_transport_contribution_" + candidate_list[candidate_idx]],
			data["ot_direction_" + candidate_list[candidate_idx] + "_y"] / data["optimal_transport_contribution_" + candidate_list[candidate_idx]]
		)

fig.tight_layout(pad=1.0)
fig.savefig(fig_file_name[election_id][geographical_filter_id])