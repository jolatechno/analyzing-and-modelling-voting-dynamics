#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os.path
import networkx as nx
import osmnx as ox

election_id            = "france_pres_tour1_2022"
geographical_filter_id = "Paris"

distance = "vol d'oiseau"

osm_source_region = {
	"Paris"             : "Paris, France",
	"Petite_couronne"   : "file",
	"Region_parisienne" : "file",
	"Metropole"         : "file"
}
osm_file_name     = "data/france.osm"
osm_pbf_file_name = "data/france-latest.osm.pbf"
osm_pbf_file_url  = "https://www.data.gouv.fr/fr/datasets/r/01fdab09-ed86-4259-b863-69913a3e04d1"

input_file_names = {
	"france_pres_tour1_2022" : "../data/france_pres_tour1_2022_preprocessed.csv"
}
distance_file_names = {
	"Paris"             : "../data/distances_paris.csv",
	"Petite_couronne"   : "../data/distances_petite_couronne.csv",
	"Region_parisienne" : "../data/distances_region_parisienne.csv",
	"Metropole"         : "../data/distances_metropole.csv"
}

geographical_filter_departement_list = {
	"Paris"             : ["75"],
	"Petite_couronne"   : ["75", "92", "93", "94"],
	"Region_parisienne" : ["75", "92", "93", "94", "77", "78", "91", "95"],
	"Metropole"         : [str(idx).zfill(2) for idx in range(1, 95+1)]
}

compute_direction = {
	"Paris"             : True,
	"Petite_couronne"   : False,
	"Region_parisienne" : False,
	"Metropole"         : False
}
direction_file_names = {
	"Paris"             : {
		"x" : "../data/direction_paris_x.csv",
		"y" : "../data/direction_paris_y.csv"
	}
}

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

geographical_mask          = election_database["code_commune"].str[0:2].isin(geographical_filter_departement_list[geographical_filter_id])
filtered_election_database = election_database[geographical_mask]
filtered_election_database = filtered_election_database.dropna(subset=["longitude", "latitude"]).reset_index(drop=True)

""" ############
################
compute distance
################
############ """

num_nodes       = len(filtered_election_database["longitude"])
distance_matrix = np.zeros((num_nodes, num_nodes))

if distance == "vol d'oiseau":
	""" ##########################
	compute straigth-line distance
	########################## """

	lon = np.array(filtered_election_database["longitude"])
	lat = np.array(filtered_election_database["latitude" ])
	for i in range(1, num_nodes):
		distance_matrix[i, :i] = np.maximum(compute_distance(lon[:i], lat[:i], lon[i], lat[i]), 10)
		distance_matrix[:i, i] = distance_matrix[i, :i]
else:
	""" ###########################
	compute distance based on route
	Read file :
	########################### """

	if osm_source_region[geographical_filter_id] == "file":
		if not os.path.isfile(osm_file_name):
			if not os.path.isfile(osm_pbf_file_name):
				print(f"\"{ osm_pbf_file_name }\" not found, downloading from { osm_pbf_file_url }")
				# TODO : downlaod .pbf

			print(f"\"{ osm_file_name }\" not found, converting from \"{ osm_pbf_file_name }\"")
			# TODO : convert .pbf to .osm

		print(f"Reading geographical data from \"{ osm_file_name }\"")
		graph = ox.graph_from_xml(osm_file_name)
	else:
		print(f"Querrying geographical for region \"{ osm_source_region[geographical_filter_id] }\"")
		graph = ox.graph_from_place(osm_source_region[geographical_filter_id], network_type="drive")

	""" ############
	compute distance
	############ """

	print("Computing distance matrix")
	for i in range(num_nodes):
		for j in range(i):
			dist = 10
			try:
				orig_node = ox.nearest_nodes(graph, filtered_election_database["longitude"][i], filtered_election_database["latitude"][i])
				dest_node = ox.nearest_nodes(graph, filtered_election_database["longitude"][j], filtered_election_database["latitude"][j])

				if orig_node != dest_node:
					route = ox.shortest_path(       graph, orig_node, dest_node, weight="length")
					gdf   = ox.routing.route_to_gdf(graph, route,                weight="length")
					dist  = int(gdf["length"].sum())
			except:
				dist = 10
			distance_matrix[i, j] = dist
			distance_matrix[j, i] = distance_matrix[i, j]

if compute_direction[geographical_filter_id]:
	print("Computing distance matrix")
	unitary_direction_matrix = np.zeros((distance_matrix.shape[0], distance_matrix.shape[0], 2))
	
	for i in range(1, num_nodes):		
		if distance == "vol d'oiseau":
			distances_line = distance_matrix[i, :i]
		else:
			distances_line = np.maximum(compute_distance(lon[:i], lat[:i], lon[i], lat[i]), 10)
		unitary_direction_matrix[i, :i, 0] = (lon[i] - lon[:i]) / distances_line
		unitary_direction_matrix[:i, i, 0] = -unitary_direction_matrix[i, :i, 0]
		unitary_direction_matrix[i, :i, 1] = (lat[i] - lat[:i]) / distances_line
		unitary_direction_matrix[:i, i, 1] = -unitary_direction_matrix[i, :i, 1]

""" #########
write to file
######### """

print(f"Write to \"{ distance_file_names[geographical_filter_id] }\"")
np.savetxt(distance_file_names[geographical_filter_id], distance_matrix, delimiter=',', header=",".join(filtered_election_database["id_brut_bv_reu"]))

if compute_direction[geographical_filter_id]:
	print(f"Write to \"{ direction_file_names[geographical_filter_id]["x"] }\"")
	np.savetxt(direction_file_names[geographical_filter_id]["x"], unitary_direction_matrix[:, :, 0], delimiter=',', header=",".join(filtered_election_database["id_brut_bv_reu"]))

	print(f"Write to \"{ direction_file_names[geographical_filter_id]["y"] }\"")
	np.savetxt(direction_file_names[geographical_filter_id]["y"], unitary_direction_matrix[:, :, 1], delimiter=',', header=",".join(filtered_election_database["id_brut_bv_reu"]))