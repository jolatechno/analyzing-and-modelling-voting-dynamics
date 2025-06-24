#!/usr/bin/env python3

from util.util import *

import pandas as pd
import numpy as np
import ot
import sys
from matplotlib import pyplot as plt
from scipy import interpolate

election_id  = "france_pres_tour1_2022"
commune = [
	#["Lyon"],
	#["Toulouse"],
	#["Marseille"],
	["Paris"]
]
clip_segregation = None # [200, 3200]
interesting_candidates = [
	#["LE PEN", "MACRON", "MÉLENCHON"],
	#["LE PEN", "MACRON", "MÉLENCHON"],
	#["LE PEN", "MACRON", "MÉLENCHON"],
	["LE PEN", "MACRON", "MÉLENCHON", "ZEMMOUR"]
]
index_comparison = [
	#False,
	#False,
	#False,
	True
]
comparison_percetiles = [10, 90]

candidate_color = ["tab:brown", "tab:brown"]

input_file_names = {
	"france_pres_tour1_2022" : "data/france_pres_tour1_2022_preprocessed.csv"
}
bvote_position_file_name  = "data/table-adresses-preprocessed.csv"

fig_file_name = [
	[
		[f"results/article/{ geo[0] }/fig_{ geo[0] }_votes_{ candidate.replace(" ", "_") }.png"         for candidate in interesting_candidates[i]],
		 f"results/article/{ geo[0] }/fig_{ geo[0] }_segregation.png",
		[f"results/article/{ geo[0] }/fig_{ geo[0] }_segregation_{ candidate.replace(" ", "_") }.png"   for candidate in interesting_candidates[i]],
		[f"results/article/{ geo[0] }/fig_{ geo[0] }_dissimilarity_{ candidate.replace(" ", "_") }.png" for candidate in interesting_candidates[i]],
		 f"results/article/{ geo[0] }/fig_{ geo[0] }_direction.png"
	] for i,geo in enumerate(commune)
]
for i,comparison in enumerate(index_comparison):
	if index_comparison:
		fig_file_name[i].append([
			 f"results/article/{ commune[i][0] }/comparison/fig_{ commune[i][0] }_comparison_KL.png",
			 f"results/article/{ commune[i][0] }/comparison/fig_{ commune[i][0] }_comparison_KL_map.png",
			 f"results/article/{ commune[i][0] }/comparison/fig_{ commune[i][0] }_comparison_multiscalar.png",
			 f"results/article/{ commune[i][0] }/comparison/fig_{ commune[i][0] }_comparison_multiscalar_map.png",
			[f"results/article/{ commune[i][0] }/comparison/fig_{ commune[i][0] }_comparison_dissimilarity_{ candidate.replace(" ", "_") }.png"     for candidate in interesting_candidates[i]],
			[f"results/article/{ commune[i][0] }/comparison/fig_{ commune[i][0] }_comparison_dissimilarity_map_{ candidate.replace(" ", "_") }.png" for candidate in interesting_candidates[i]]
		])

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

def plot_geo_data(position_database, data, id_field, id_field_name="id_brut_bv_reu", clip=None):
	dat, lon, lat = [], [], []
	data_to_use = data if clip is None else np.clip(data, *clip)
	for id_,value in zip(id_field,data_to_use):
		mask = position_database[id_field_name] == id_
		lon.extend(np.array(position_database[mask]["longitude"]))
		lat.extend(np.array(position_database[mask]["latitude"]))
		dat.extend([value] * np.sum(mask))

	return ax.scatter(lon, lat, c=dat, s=0.7, alpha=1)

def plot_categories(position_database, categories, colors, id_field, id_field_name="id_brut_bv_reu", labels=None):
	cat, lon, lat = [], [], []
	for id_,value in zip(id_field,categories):
		mask = position_database[id_field_name] == id_
		lon.extend(np.array(position_database[mask]["longitude"]))
		lat.extend(np.array(position_database[mask]["latitude"]))
		cat.extend([value] * np.sum(mask))
	cat, lon, lat = np.array(cat), np.array(lon), np.array(lat)

	label_is_none = labels is None
	if label_is_none:
		label = [None] * len(colors)

	for category,color,label in zip(np.sort(np.unique(categories)),colors,labels):
		filter_ = cat == category
		ax.scatter(lon[filter_], lat[filter_], c=color, s=0.7, alpha=1, label=label)

	if not label_is_none:
		fig = ax.get_figure()
		fig.tight_layout(pad=1.0)
		fig.legend()

""" ##############################################
##################################################
read field from the preprocessed election database
##################################################
############################################## """

print(f"Reading data from \"{ input_file_names[election_id] }\"", file=sys.stderr)
election_database = pd.read_csv(input_file_names[election_id], low_memory=False)

print(f"Reading data from \"{ bvote_position_file_name }\"", file=sys.stderr)
bvote_position_database = pd.read_csv(bvote_position_file_name, low_memory=False)

""" #####################
apply geographical filter
##################### """

for filter_idx,geographical_filter in enumerate(commune):
	geographical_mask = np.isin(election_database["Libellé de la commune"], geographical_filter)
	filtered_election_database = election_database[geographical_mask]
	filtered_election_database = filtered_election_database.dropna(subset=["longitude", "latitude"]).reset_index(drop=True)
	filtered_bvote_position_database = bvote_position_database[np.isin(
		bvote_position_database["id_brut_bv_reu"].str[:4], 
		np.unique(filtered_election_database["id_brut_bv_reu"].str[:4]))]

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
	candidate_list = [
		'ARTHAUD', 'ROUSSEL', 'MACRON',
		'LASSALLE', 'LE PEN', 'ZEMMOUR',
		'MÉLENCHON', 'HIDALGO', 'JADOT',
		'PÉCRESSE', 'POUTOU', 'DUPONT-AIGNAN'
	]

	""" #####################
	compute optimal transport
	##################### """

	total_ot_dist         = 0
	ot_dist_candidates    = np.zeros(len(candidate_list))
	total_vote_candidates = np.zeros(len(candidate_list))

	ot_dist_contribution            = np.zeros(                      len(filtered_election_database["Votants"]))
	ot_dist_contribution_candidates = np.zeros((len(candidate_list), len(filtered_election_database["Votants"])))
	ot_dist_dissimilarity           = np.zeros((len(candidate_list), len(filtered_election_database["Votants"])))

	ot_direction_per_candidate = np.zeros((len(filtered_election_database["Votants"]), 2, len(candidate_list)))
	ot_direction               = np.zeros((len(filtered_election_database["Votants"]), 2))

	total_voting_population = np.sum(  filtered_election_database["Votants"])
	reference_distrib       = np.array(filtered_election_database["Votants"]) / total_voting_population

	candidate_padding_length = max([len(x) for x in candidate_list])
	for i,candidate in enumerate(candidate_list):
		total_vote_candidates[i] = np.sum(filtered_election_database[candidate + " Voix"])
		candidate_distrib        = np.array(filtered_election_database[candidate + " Voix"]) / total_vote_candidates[i]

		candidate_ot_mat = ot.emd(reference_distrib, candidate_distrib, distance_matrix_alpha)*distance_matrix

		ot_dist_contribution_candidates[i, :]  = (candidate_ot_mat.sum(axis=0) + candidate_ot_mat.sum(axis=1)) / 2 / reference_distrib
		ot_dist_dissimilarity[          i, :]  = (candidate_ot_mat.sum(axis=0) - candidate_ot_mat.sum(axis=1)) / 2 / reference_distrib
		ot_dist_contribution                  += ot_dist_contribution_candidates[i] * total_vote_candidates[i] / total_voting_population
		ot_dist_candidates[             i   ]  = np.sum(ot_dist_contribution_candidates[i] * reference_distrib)
		total_ot_dist                         += ot_dist_candidates[i] * total_vote_candidates[i] / total_voting_population
		
		ot_direction_per_candidate[:, 0,            i]  = ((unitary_direction_matrix[:, :, 0]*candidate_ot_mat).sum(axis=0) + (unitary_direction_matrix[:, :, 0].T*candidate_ot_mat).sum(axis=1)) / 2 / reference_distrib
		ot_direction_per_candidate[:, 1,            i]  = ((unitary_direction_matrix[:, :, 1]*candidate_ot_mat).sum(axis=0) + (unitary_direction_matrix[:, :, 1].T*candidate_ot_mat).sum(axis=1)) / 2 / reference_distrib
		ot_direction                                   += ot_direction_per_candidate[:, :, i] * total_vote_candidates[i] / total_voting_population

	total_ot_dist                    = np.sum(ot_dist_contribution * reference_distrib)
	total_vote_proportion_candidate  = total_vote_candidates / total_voting_population
	total_vote_proportion_candidate /= np.sum(total_vote_proportion_candidate)

	map_ratio = get_map_ratio(lon, lat)

	""" ######
	##########
	plot votes
	##########
	###### """

	for interesting_candidate_idx,interesting_candidate in enumerate(interesting_candidates[filter_idx]):
		candidate_idx = candidate_list.index(interesting_candidate)
		fig, ax = plt.subplots(1, 1, figsize=(6 + 2, 6/map_ratio + 2))

		vote_distrib_candidate    = np.array(filtered_election_database[interesting_candidate + " Voix"])
		vote_proportion_candidate = vote_distrib_candidate / np.array(filtered_election_database["Votants"])

		pl = plot_geo_data(filtered_bvote_position_database, vote_proportion_candidate, filtered_election_database["id_brut_bv_reu"])

		cbar = fig.colorbar(pl, label="proportion of votes")

		line = cbar.ax.plot(0.5, total_vote_proportion_candidate[candidate_idx],
			markerfacecolor='w', markeredgecolor='w', marker='x', markersize=10)[0]
		cbar.ax.text(-1.1, line.get_ydata()-20, "avg.")

		ax.set_aspect(map_ratio)
		ax.set_title(f"Vote proportion for { interesting_candidate }\nduring the 2022 presidencial elections")

		fig.tight_layout(pad=1.0)
		fig.savefig(fig_file_name[filter_idx][0][interesting_candidate_idx])
		plt.close(fig)

	""" ##################
	######################
	plot optimal transport
	######################
	################## """

	fig, ax = plt.subplots(1, 1, figsize=(6 + 2, 6/map_ratio + 2))

	pl = plot_geo_data(filtered_bvote_position_database, ot_dist_contribution, filtered_election_database["id_brut_bv_reu"], clip=clip_segregation)

	cbar = fig.colorbar(pl, label="local contribution [m]")

	"""for candidate in interesting_candidates.keys() :
		candidate_idx = candidate_list.index(candidate)
		line = cbar.ax.plot(0.5, ot_dist_candidates[candidate_idx],
			markerfacecolor='w', markeredgecolor='w', marker='+', markersize=10)[0]
		cbar.ax.text(-1.1, line.get_ydata()-20, interesting_candidates[candidate]["abv"])"""
	line = cbar.ax.plot(0.5, total_ot_dist,
		markerfacecolor='w', markeredgecolor='w', marker='x', markersize=10)[0]
	cbar.ax.text(-1.1, line.get_ydata()-20, "avg.")

	ax.set_aspect(map_ratio)
	ax.set_title("Local segregation index in Paris\nduring the 2022 presidencial elections")

	fig.tight_layout(pad=1.0)
	fig.savefig(fig_file_name[filter_idx][1])
	plt.close(fig)

	for interesting_candidate_idx,interesting_candidate in enumerate(interesting_candidates[filter_idx]):
		candidate_idx = candidate_list.index(interesting_candidate)

		fig, ax = plt.subplots(1, 1, figsize=(6 + 2, 6/map_ratio + 2))

		pl = plot_geo_data(filtered_bvote_position_database, ot_dist_contribution_candidates[candidate_idx, :], filtered_election_database["id_brut_bv_reu"], clip=clip_segregation)

		cbar = fig.colorbar(pl, label="local contribution [m]")

		line = cbar.ax.plot(0.5, ot_dist_candidates[candidate_idx],
			markerfacecolor='w', markeredgecolor='w', marker='x', markersize=10)[0]
		cbar.ax.text(-1.1, line.get_ydata()-20, "avg.")

		print(f"{ round(ot_dist_candidates[candidate_idx]) }m for { interesting_candidate } for {  commune[filter_idx][0] }")

		ax.set_aspect(map_ratio)
		ax.set_title(f"Local segregation index in Paris for { interesting_candidate }\nduring the 2022 presidencial elections")

		fig.tight_layout(pad=1.0)
		fig.savefig(fig_file_name[filter_idx][2][interesting_candidate_idx])
		plt.close(fig)

	""" ##################
	######################
	plot dissimilarity
	######################
	################## """

	for interesting_candidate_idx,interesting_candidate in enumerate(interesting_candidates[filter_idx]):
		candidate_idx = candidate_list.index(interesting_candidate)

		fig, ax = plt.subplots(1, 1, figsize=(6 + 2, 6/map_ratio + 2))

		pl = plot_geo_data(filtered_bvote_position_database, ot_dist_dissimilarity[candidate_idx, :], filtered_election_database["id_brut_bv_reu"])

		cbar = fig.colorbar(pl, label="dissimilarity [m]")

		ax.set_aspect(map_ratio)
		ax.set_title(f"Dissimilarity in Paris for { interesting_candidate }\nduring the 2022 presidencial elections")

		fig.tight_layout(pad=1.0)
		fig.savefig(fig_file_name[filter_idx][3][interesting_candidate_idx])
		plt.close(fig)

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

	fig.savefig(fig_file_name[filter_idx][4])
	plt.close(fig)

	if index_comparison[filter_idx]:
		""" ###################################
		#######################################
		comparison with the KL divergence index
		#######################################
		################################### """

		Kl_divergence = np.zeros(len(filtered_election_database["Votants"]))
		for i,candidate in enumerate(candidate_list):
			candidate_distrib  = np.array(filtered_election_database[candidate + " Voix"]) / (reference_distrib * total_voting_population)
			Kl_divergence     += total_vote_proportion_candidate[i] * np.log(total_vote_proportion_candidate[i] / np.maximum(candidate_distrib, 1e-5))

		Kl_divergence_over_ot_segregation = Kl_divergence / ot_dist_contribution
		Kl_upper_lim, Kl_lower_lim        = np.percentile(Kl_divergence_over_ot_segregation, comparison_percetiles[1]), np.percentile(Kl_divergence_over_ot_segregation, comparison_percetiles[0])
		Kl_is_upper, Kl_is_lower          = Kl_divergence_over_ot_segregation > Kl_upper_lim, Kl_divergence_over_ot_segregation < Kl_lower_lim
		Kl_is_middle                      = np.logical_and(np.logical_not(Kl_is_upper), np.logical_not(Kl_is_lower))

		""" ########################################
		plot comparison with the KL divergence index
		######################################## """

		fig, ax = plt.subplots(1, 1, figsize=(8, 8))

		ax.plot(ot_dist_contribution[Kl_is_middle], Kl_divergence[Kl_is_middle], "+k", label=None)
		ax.plot(ot_dist_contribution[Kl_is_upper],  Kl_divergence[Kl_is_upper],  "+r", label=f"Upper { 100 - comparison_percetiles[1] }% of ratio of indeces")
		ax.plot(ot_dist_contribution[Kl_is_lower],  Kl_divergence[Kl_is_lower],  "+b", label=f"Lower { comparison_percetiles[0] }% of ratio of indeces")

		ax.set_xlim(np.percentile(ot_dist_contribution, [1, 99]) * np.array([0.9, 1.1]))
		ax.set_ylim(np.percentile(Kl_divergence,        [1, 99]) * np.array([0.9, 1.1]))

		ax.set_xscale("log")
		ax.set_yscale("log")

		ax.set_title("Comparison of our segregation index to the KL-divergence")
		ax.set_xlabel("Our optimal-transport based index")
		ax.set_ylabel("KL-divergence")

		fig.tight_layout(pad=1.0)
		fig.legend(loc="lower right", bbox_to_anchor=[0.9, 0.1])
		fig.savefig(fig_file_name[filter_idx][5][0])
		plt.close(fig)

		""" ##########################################################
		ploting the map of the comparison with the KL divergence index
		########################################################## """

		fig, ax = plt.subplots(1, 1, figsize=(6, 6/map_ratio + 1))

		pl = plot_categories(filtered_bvote_position_database, Kl_is_upper + Kl_is_lower * 2, ["k", "b", "r"], filtered_election_database["id_brut_bv_reu"],
			labels=[None, f"Lower { comparison_percetiles[0] }% of ratio of indeces", f"Upper { 100 - comparison_percetiles[1] }% of ratio of indeces"])

		ax.set_aspect(map_ratio)
		ax.set_title("map of the comparison of our segregation\nindex to the KL-divergence")

		fig.savefig(fig_file_name[filter_idx][5][1])
		plt.close(fig)

		""" #################################
		#####################################
		comparison with the multiscalar index
		#####################################
		################################# """

		idx_matrix        = np.argsort(distance_matrix, axis=1)
		vote_trajectories = np.zeros((len(candidate_list), len(filtered_election_database["Votants"]), len(filtered_election_database["Votants"])))
		Kl_trajectories   = np.zeros((                     len(filtered_election_database["Votants"]), len(filtered_election_database["Votants"])))
		focal_distances   = np.zeros((                     len(filtered_election_database["Votants"]), len(filtered_election_database["Votants"])))
		for i,candidate in enumerate(candidate_list):
			vote_trajectories[i, :, :] = np.cumsum(np.array(filtered_election_database[candidate + " Voix"])[idx_matrix], axis=1)
		population_trajectory = np.maximum(vote_trajectories.sum(axis=0), 1e-7)
		for i in range(len(candidate_list)):
			vote_trajectories[i, :, :] /= population_trajectory
			vote_trajectories[i, :, :]  = np.maximum(vote_trajectories[i, :, :], 1e-7)
		normalisation_factor = np.sum(vote_trajectories, axis=0)
		for i in range(len(candidate_list)):
			vote_trajectories[i, :, :] /= normalisation_factor
			Kl_trajectories            += total_vote_proportion_candidate[i] * np.log(total_vote_proportion_candidate[i] / vote_trajectories[i, :, :])
		for j in reversed(range(len(filtered_election_database["Votants"]))):
			focal_distances[:, j] = np.max(Kl_trajectories[:, j:j+2], axis=1)
		integration_coef         = population_trajectory.copy()
		integration_coef[:, 1:] -= population_trajectory[:, :-1]
		distort_coef             = np.sum(np.multiply(focal_distances, integration_coef), axis=1)

		distort_coef_over_ot_segregation = distort_coef / ot_dist_contribution
		dist_upper_lim, dist_lower_lim   = np.percentile(distort_coef_over_ot_segregation, comparison_percetiles[1]), np.percentile(distort_coef_over_ot_segregation, comparison_percetiles[0])
		dist_is_upper, dist_is_lower     = distort_coef_over_ot_segregation > dist_upper_lim, distort_coef_over_ot_segregation < dist_lower_lim
		dist_is_middle                   = np.logical_and(np.logical_not(dist_is_upper), np.logical_not(dist_is_lower))

		""" ######################################
		plot comparison with the multiscalar index
		###################################### """

		fig, ax = plt.subplots(1, 1, figsize=(8, 8))

		ax.plot(ot_dist_contribution[dist_is_middle], distort_coef[dist_is_middle], "+k", label=None)
		ax.plot(ot_dist_contribution[dist_is_upper],  distort_coef[dist_is_upper],  "+r", label=f"Upper { 100 - comparison_percetiles[1] }% of ratio of indeces")
		ax.plot(ot_dist_contribution[dist_is_lower],  distort_coef[dist_is_lower],  "+b", label=f"Lower { comparison_percetiles[0] }% of ratio of indeces")

		ax.set_xlim(np.percentile(ot_dist_contribution, [1, 99]) * np.array([0.9, 1.1]))
		ax.set_ylim(np.percentile(distort_coef,         [1, 99]) * np.array([0.9, 1.1]))

		ax.set_xscale("log")
		ax.set_yscale("log")

		ax.set_title("Comparison of our segregation index to\nthe multiscalar segregation index")
		ax.set_xlabel("Our optimal-transport based index")
		ax.set_ylabel("Multiscalar segregation index (non-normalized)")

		fig.tight_layout(pad=1.0)
		fig.legend(loc="lower right", bbox_to_anchor=[0.9, 0.1])
		fig.savefig(fig_file_name[filter_idx][5][2])
		plt.close(fig)

		""" ########################################################
		ploting the map of the comparison with the multiscalar index
		######################################################## """

		fig, ax = plt.subplots(1, 1, figsize=(6, 6/map_ratio + 1))

		pl = plot_categories(filtered_bvote_position_database, dist_is_upper + dist_is_lower * 2, ["k", "b", "r"], filtered_election_database["id_brut_bv_reu"],
			labels=[None, f"Lower { comparison_percetiles[0] }% of ratio of indeces", f"Upper { 100 - comparison_percetiles[1] }% of ratio of indeces"])

		ax.set_aspect(map_ratio)
		ax.set_title("map of the comparison of our segregation index\nto the multiscalar segregation index")

		fig.savefig(fig_file_name[filter_idx][5][3])
		plt.close(fig)

		""" ###########################################
		###############################################
		comparison between dissimilarity and difference
		###############################################
		########################################### """

		for interesting_candidate_idx,interesting_candidate in enumerate(interesting_candidates[filter_idx]):
			candidate_idx = candidate_list.index(interesting_candidate)

			vote_distrib_candidate    = np.array(filtered_election_database[interesting_candidate + " Voix"])
			vote_proportion_candidate = vote_distrib_candidate / np.array(filtered_election_database["Votants"])
			candidate_vote_difference = vote_proportion_candidate - total_vote_proportion_candidate[candidate_idx]

			diff_over_dissimilarity          = candidate_vote_difference / ot_dist_dissimilarity[candidate_idx, :]
			diff_upper_lim, diff_lower_lim   = np.percentile(diff_over_dissimilarity[diff_over_dissimilarity > 0], comparison_percetiles[1]), np.percentile(diff_over_dissimilarity[diff_over_dissimilarity > 0], comparison_percetiles[0])
			diff_is_upper, diff_is_lower     = diff_over_dissimilarity > diff_upper_lim, np.logical_and(diff_over_dissimilarity < diff_lower_lim, diff_over_dissimilarity > 0)
			diff_is_middle                   = np.logical_and(np.logical_not(diff_is_upper), np.logical_not(diff_is_lower))

			""" ########################################
			plot comparison dissimilarity and difference
			######################################## """

			fig, ax = plt.subplots(1, 1, figsize=(8, 8))

			ax.plot(ot_dist_dissimilarity[candidate_idx, :][diff_is_middle], candidate_vote_difference[diff_is_middle], "+k", label=None)
			ax.plot(ot_dist_dissimilarity[candidate_idx, :][diff_is_upper],  candidate_vote_difference[diff_is_upper],  "+r", label=f"Upper { 100 - comparison_percetiles[1] }% of ratio of indeces")
			ax.plot(ot_dist_dissimilarity[candidate_idx, :][diff_is_lower],  candidate_vote_difference[diff_is_lower],  "+b", label=f"Lower { comparison_percetiles[0] }% of ratio of indeces")

			ax.set_xlim(np.percentile(ot_dist_dissimilarity[candidate_idx, :], [1, 99]) * 1.1)
			ax.set_ylim(np.percentile(candidate_vote_difference,               [1, 99]) * 1.1)

			ax.set_title(f"Comparison of our dissimilarity\nindex to the vote excess/deficit\nfor { interesting_candidate }")
			ax.set_xlabel("Our optimal-transport based dissimilarity")
			ax.set_ylabel("vote excess/deficit")

			fig.tight_layout(pad=1.0)
			fig.legend(loc="lower right", bbox_to_anchor=[0.9, 0.1])
			fig.savefig(fig_file_name[filter_idx][5][4][interesting_candidate_idx])
			plt.close(fig)

			""" ##########################################################
			ploting the map of the comparison dissimilarity and difference
			########################################################## """

			fig, ax = plt.subplots(1, 1, figsize=(6, 6/map_ratio + 1))

			pl = plot_categories(filtered_bvote_position_database, diff_is_upper + diff_is_lower * 2, ["k", "b", "r"], filtered_election_database["id_brut_bv_reu"],
				labels=[None, f"Lower { comparison_percetiles[0] }% of ratio of indeces", f"Upper { 100 - comparison_percetiles[1] }% of ratio of indeces"])

			ax.set_aspect(map_ratio)
			ax.set_title(f"map of the comparison of our dissimilarity\nindex to the vote excess/deficit\nfor { interesting_candidate }")

			fig.savefig(fig_file_name[filter_idx][5][5][interesting_candidate_idx])
			plt.close(fig)
		

