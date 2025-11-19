#!/usr/bin/env python3

import pandas as pd
import numpy as np
import ot
from matplotlib import pyplot as plt

election_id            = "france_pres_tour1_2022"
trajectory_step        = 5

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

input_file_names = {
	"france_pres_tour1_2022" : "../data/france_pres_tour1_2022_preprocessed.csv"
}

geographical_filter_departement_list = [str(idx).zfill(2) for idx in range(1, 95+1)]
exluded_departement_list             = []
departement_of_interest_list         = ["34", "74"]

""" ##############################################
##################################################
read field from the preprocessed election database
##################################################
############################################## """

print(f"Reading data from \"{ input_file_names[election_id] }\"")
election_database = pd.read_csv(input_file_names[election_id], low_memory=False)
print()

""" #####################
apply geographical filter
##################### """

results = {}
for geographical_filter in geographical_filter_departement_list:
	if geographical_filter in exluded_departement_list:
		continue

	geographical_mask          = election_database["code_commune"].str[0:2] == geographical_filter
	filtered_election_database = election_database[geographical_mask]
	filtered_election_database = filtered_election_database.dropna(subset=["longitude", "latitude"]).reset_index(drop=True)

	if len(filtered_election_database) == 0:
		continue

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

	candidate_padding_length = max([len(x) for x in candidate_list])
	for idx_candidate,candidate in enumerate(candidate_list):
		total_vote_candidate = np.sum(  filtered_election_database[candidate + " Voix"])
		candidate_distrib    = np.array(filtered_election_database[candidate + " Voix"]) / total_vote_candidate

		candidate_ot_dist = ot.emd2(reference_distrib, candidate_distrib, distance_matrix)
		ot_dist          += candidate_ot_dist * total_vote_candidate / total_voting_population
		
		if geographical_filter in departement_of_interest_list :
			print(f"Optimal transport distance in departement { geographical_filter } for { candidate.ljust(candidate_padding_length , " ") } (idx { idx_candidate }) electors : { str(round(candidate_ot_dist)).rjust(4, " ") }m with { str(round(total_vote_candidate / total_voting_population * 100)).rjust(2, " ") }% of electors")

	results[geographical_filter] = ot_dist

	if geographical_filter in departement_of_interest_list :
		print(f"\nOptimal transport average distance for departement { geographical_filter } : { round(ot_dist) }m")
		print()

results = dict(sorted(results.items(), key=lambda item: item[1]))

for departement, segregation in results.items():
	print(f"Optimal transport average distance for departement { departement } : { round(segregation) }m")
