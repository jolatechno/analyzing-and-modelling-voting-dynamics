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
		"Paris"             : "data/france_pres_tour1_2022_paris_ot_distrib.csv",
		"Petite_couronne"   : "data/france_pres_tour1_2022_petite_couronne_ot_distrib.csv",
		"Region_parisienne" : "data/france_pres_tour1_2022_region_parisienne_ot_distrib.csv",
		"Metropole"         : "data/france_pres_tour1_2022_metropole_ot_distrib.csv"
	}
}
fig_file_name = {
	"france_pres_tour1_2022" : {
		"Paris"             : "results/france_pres_tour1_2022_paris_ot_distrib.png",
		"Petite_couronne"   : "results/france_pres_tour1_2022_petite_couronne_ot_distrib.png",
		"Region_parisienne" : "results/france_pres_tour1_2022_region_parisienne_ot_distrib.png",
		"Metropole"         : "results/france_pres_tour1_2022_metropole_ot_distrib.png"
	}
}

geographical_filter_departement_list = {
	"Paris"             : ["75"],
	"Petite_couronne"   : ["75", "92", "93", "94"],
	"Region_parisienne" : ["75", "92", "93", "94", "77", "78", "91", "95"],
	"Metropole"         : [str(idx).zfill(2) for idx in range(1, 95+1)]
}

epsilon_conc = -0.02
epsilon_conv =  0.02

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
distance_matrix            = np.genfromtxt(distance_file_names[geographical_filter_id], delimiter=',')
distance_matrix_alpha_conc = np.pow(distance_matrix, 1 + epsilon_conc)
distance_matrix_alpha_conv = np.pow(distance_matrix, 1 + epsilon_conv)

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

	total_voting_population = np.sum(  filtered_election_database["Votants"])
	reference_distrib       = np.array(filtered_election_database["Votants"]) / total_voting_population

	""" ###############################
	compute optimal transport (concave)
	############################### """

	ot_weight_conc           = np.zeros(                       len(filtered_election_database["Votants"])*len(filtered_election_database["Votants"]))
	ot_weight_candidates_conc = np.zeros((len(candidate_list), len(filtered_election_database["Votants"])*len(filtered_election_database["Votants"])))

	print()
	candidate_padding_length = max([len(x) for x in candidate_list])
	for i,candidate in enumerate(candidate_list):
		total_vote_candidate = np.sum(  filtered_election_database[candidate + " Voix"])
		candidate_distrib    = np.array(filtered_election_database[candidate + " Voix"]) / total_vote_candidate

		candidate_ot_mat = ot.emd(reference_distrib, candidate_distrib, distance_matrix_alpha_conc)

		ot_weight_candidates_conc[i]  = candidate_ot_mat.flatten()
		ot_weight_conc               += ot_weight_candidates_conc[i] * total_vote_candidate / total_voting_population

	""" ##############################
	compute optimal transport (convex)
	############################## """

	ot_weight_conv            = np.zeros(                      len(filtered_election_database["Votants"])*len(filtered_election_database["Votants"]))
	ot_weight_candidates_conv = np.zeros((len(candidate_list), len(filtered_election_database["Votants"])*len(filtered_election_database["Votants"])))

	print()
	candidate_padding_length = max([len(x) for x in candidate_list])
	for i,candidate in enumerate(candidate_list):
		total_vote_candidate = np.sum(  filtered_election_database[candidate + " Voix"])
		candidate_distrib    = np.array(filtered_election_database[candidate + " Voix"]) / total_vote_candidate

		candidate_ot_mat = ot.emd(reference_distrib, candidate_distrib, distance_matrix_alpha_conv)

		ot_weight_candidates_conv[i]  = candidate_ot_mat.flatten()
		ot_weight_conv               += ot_weight_candidates_conv[i] * total_vote_candidate / total_voting_population

	""" ###################
	write optimal transport
	################### """

	data = {
		"optimal_transport_flux_concave" : ot_weight_conc,
		"optimal_transport_flux_convex"  : ot_weight_conv,
	}
	for candidate_idx,name in enumerate(candidate_list):
		data["optimal_transport_flux_concave_" + name] = ot_weight_candidates_conc[candidate_idx, :]
		data["optimal_transport_flux_convex_"  + name] = ot_weight_candidates_conv[candidate_idx, :]

	print(f"Write to \"{ save_file_name[election_id][geographical_filter_id] }\"")
	pd.DataFrame(data).to_csv(save_file_name[election_id][geographical_filter_id], index=False)

""" ##################
######################
plot optimal transport
######################
################## """

print(f"{ round(np.sum(data["optimal_transport_flux_convex" ][distance_matrix.flatten() == 0] / np.sum(data["optimal_transport_flux_convex" ]) * 100)) }% of people not moving in the concave case")
print(f"{ round(np.sum(data["optimal_transport_flux_concave"][distance_matrix.flatten() == 0] / np.sum(data["optimal_transport_flux_concave"]) * 100)) }% of people not moving in the convex case")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6*2, 5*1))

ax1.set_xlabel("distances")
ax1.set_title("distances cumulative distribution\nfor a convex cost")

ax2.set_xlabel("distances")
ax2.set_title("distances cumulative distribution\nfor a concave cost")

ax1.hist(distance_matrix.flatten(), weights=data["optimal_transport_flux_convex" ],
	density=True, log=True, bins=20) #, cumulative=True)
ax2.hist(distance_matrix.flatten(), weights=data["optimal_transport_flux_concave"],
	density=True, log=True, bins=20) #, cumulative=True)

fig.tight_layout(pad=1.0)
fig.savefig(fig_file_name[election_id][geographical_filter_id])