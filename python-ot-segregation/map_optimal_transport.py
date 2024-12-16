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
