#!/usr/bin/env python3

import pandas as pd
import numpy as np
import ot
from matplotlib import pyplot as plt

election_id            = "france_pres_tour1_2022"
geographical_filter_id = "Petite_couronne"
trajectory_step        = 5

input_file_names = {
	"france_pres_tour1_2022" : "data/france_pres_tour1_2022_preprocessed.csv"
}
distance_file_names = {
	"Paris"             : "data/distances_paris.csv",
	"Petite_couronne"   : "data/distances_petite_couronne.csv",
	"Region_parisienne" : "data/distances_region_parisienne.csv",
	"Metropole"         : "data/distances_metropole.csv"
}

geographical_filter_departement_list = {
	"Paris"             : ["75"],
	"Petite_couronne"   : ["75", "92", "93", "94"],
	"Region_parisienne" : ["75", "92", "93", "94", "77", "78", "91", "95"],
	"Metropole"         : [str(idx).zfill(2) for idx in range(1, 95+1)]
}

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
distance_matrix = np.genfromtxt(distance_file_names[geographical_filter_id], delimiter=',')

""" #####################
#########################
compute optimal transport
#########################
##################### """

## TO BE READ FROM THE FILE
candidate_list = ['ARTHAUD', 'ROUSSEL', 'MACRON', 'LASSALLE', 'LE PEN', 'ZEMMOUR', 'MÉLENCHON', 'HIDALGO', 'JADOT', 'PÉCRESSE', 'POUTOU', 'DUPONT-AIGNAN']

ot_dist = 0

total_voting_population = np.sum(  filtered_election_database["Votants"])
reference_distrib       = np.array(filtered_election_database["Votants"]) / total_voting_population

print()
candidate_padding_length = max([len(x) for x in candidate_list])
for candidate in candidate_list:
	total_vote_candidate = np.sum(  filtered_election_database[candidate + " Voix"])
	candidate_distrib    = np.array(filtered_election_database[candidate + " Voix"]) / total_vote_candidate

	candidate_ot_dist = ot.emd2(reference_distrib, candidate_distrib, distance_matrix)
	ot_dist          += candidate_ot_dist * total_vote_candidate / total_voting_population

	print(f"Optimal transport distance for { candidate.ljust(candidate_padding_length , " ") } electors : { str(round(candidate_ot_dist)).rjust(4, " ") }m with { str(round(total_vote_candidate / total_voting_population * 100)).rjust(2, " ") }% of electors")

print(f"\nOptimal transport average distance : { round(ot_dist) }m")

center_idx = np.argmin(np.mean(distance_matrix, axis=1))
idx_order  = np.argsort(distance_matrix[center_idx])

print()
distance_list              = np.concat((distance_matrix[center_idx, idx_order][1:-1:trajectory_step], [distance_matrix[center_idx, idx_order[-1]]]))
optimal_transport_list     = np.zeros_like(distance_list)
optimal_transport_list[-1] = ot_dist
for i,idx in enumerate(range(1, distance_matrix.shape[0]-1, trajectory_step)):
	print(f"{ i }/{ len(idx_order) - 2 }")
	total_voting_population = np.sum(  filtered_election_database["Votants"][idx_order[:idx+1]])
	reference_distrib       = np.array(filtered_election_database["Votants"][idx_order[:idx+1]]) / total_voting_population

	this_ot_dist = 0
	for candidate in candidate_list:
		total_vote_candidate = np.sum(  filtered_election_database[candidate + " Voix"][idx_order[:idx+1]])
		candidate_distrib    = np.array(filtered_election_database[candidate + " Voix"][idx_order[:idx+1]]) / total_vote_candidate

		candidate_ot_dist = ot.emd2(reference_distrib, candidate_distrib, distance_matrix[idx_order[:idx+1], :][:, idx_order[:idx+1]])
		this_ot_dist     += candidate_ot_dist * total_vote_candidate / total_voting_population

	optimal_transport_list[i] = this_ot_dist

fig, ax = plt.subplots()

ax.set_title("Optimal transport distance versus integration distance\n(around a voting bureau at the center of Paris)")
ax.set_xlabel("Integration distance [Km]")
ax.set_ylabel("Absolute optimal transport distance [Km]", color='r')

lns1 = ax.plot(distance_list / 1000, optimal_transport_list / 1000, "-r", label="OT absolute dist.")

ax2 = ax.twinx()
ax2.set_ylabel("Relative optimal transport ratio distance", color='b')

lns2 = ax2.plot(distance_list[1:] / 1000, optimal_transport_list[1:] / distance_list[1:], "--b", label="OT relative dist.")

lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

fig.savefig("results/optimal_transport_trajectory.png")
