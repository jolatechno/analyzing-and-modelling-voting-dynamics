#!/usr/bin/env python3

from util.util import *

import pandas as pd
import numpy as np
import ot
import sys
from scipy import interpolate
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

import oterogeneity as oth
from oterogeneity import utils

def get_lambda_line_filter(long0, lat0, long1, lat1, select_top=False):
	a = (lat1 - lat0) / (long1 - long0)
	b = lat0 - a * long0

	return lambda long_, lat_ : lat_ > long_ * a + b if select_top else lat_ < long_ * a + b

election_id  = "france_pres_tour1_2022"
commune = [
	#["Lyon"],
	#["Toulouse"],
	#["Marseille"],
	["Paris"]
]
dont_show_filter = [
	#[],
	#[],
	#[],
	[
		get_lambda_line_filter(2.250, 48.845, 2.280, 48.885, True),
		get_lambda_line_filter(2.385, 48.820, 2.430, 48.845, False)
	]
]
if_clip_segregation = True
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
			[f"results/article/{ commune[i][0] }/comparison/fig_{ commune[i][0] }_comparison_dissimilarity_map_{ candidate.replace(" ", "_") }.png" for candidate in interesting_candidates[i]],
			 f"results/article/{ commune[i][0] }/comparison/convex/fig_{ commune[i][0] }_convex_map.png",
			 f"results/article/{ commune[i][0] }/comparison/fig_{ commune[i][0] }_comparison_convex_concave.png",
			 f"results/article/{ commune[i][0] }/comparison/fig_{ commune[i][0] }_comparison_convex_concave_map.png",
			[f"results/article/{ commune[i][0] }/comparison/convex/fig_{ commune[i][0] }_convex_map_{ candidate.replace(" ", "_") }.png"     for candidate in interesting_candidates[i]],
			[f"results/article/{ commune[i][0] }/comparison/fig_{ commune[i][0] }_comparison_convex_concave_{ candidate.replace(" ", "_") }.png"     for candidate in interesting_candidates[i]],
			[f"results/article/{ commune[i][0] }/comparison/fig_{ commune[i][0] }_comparison_convex_concave_map_{ candidate.replace(" ", "_") }.png" for candidate in interesting_candidates[i]]
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

def plot_geo_data(position_database, data, id_field, id_field_name="id_brut_bv_reu", clip=None, filters=[], norm="linear"):
	dat, lon, lat = [], [], []
	data_to_use = data if clip is None else np.clip(data, *clip)
	for id_,value in zip(id_field, data_to_use):
		mask = position_database[id_field_name] == id_
		long_toadd, lat_toadd = np.array(position_database[mask]["longitude"]), np.array(position_database[mask]["latitude"])
		show = np.full(long_toadd.shape, True)
		for filter_ in filters:
			for j in range(len(long_toadd)):
				if show[j]:
					if filter_(long_toadd[j], lat_toadd[j]):
						show[j] = False
		lon.extend(long_toadd[show])
		lat.extend(lat_toadd[show])
		dat.extend([value] * np.sum(show))

	return ax.scatter(lon, lat, c=dat, s=0.7, alpha=1, norm=norm)

def plot_hatching(position_database, data, id_field, id_field_name="id_brut_bv_reu", hatches_limit=0, filters=[], n_points=100, hatches_strength=3):
	dat, lon, lat = [], [], []
	for id_,value in zip(id_field, data):
		mask = position_database[id_field_name] == id_
		long_toadd, lat_toadd = np.array(position_database[mask]["longitude"]), np.array(position_database[mask]["latitude"])
		show = np.full(long_toadd.shape, True)
		for filter_ in filters:
			for j in range(len(long_toadd)):
				if show[j]:
					if filter_(long_toadd[j], lat_toadd[j]):
						show[j] = False
		lon.extend(long_toadd[show])
		lat.extend(lat_toadd[show])
		dat.extend([value] * np.sum(show))

	lon_plot_1d = np.linspace(np.min(lon), np.max(lon), n_points)
	lat_plot_1d = np.linspace(np.min(lat), np.max(lat), n_points)
	lon_plot, lat_plot = np.meshgrid(lon_plot_1d, lat_plot_1d)
	dat_plot  = np.zeros((n_points, n_points))
	dat_count = np.zeros((n_points, n_points))

	for lon_,lat_,val_ in zip(lon, lat, dat):
		idx_lon = np.searchsorted(lon_plot_1d, lon_)
		idx_lat = np.searchsorted(lat_plot_1d, lat_)
		dat_count[idx_lat, idx_lon] += 1
		dat_plot[ idx_lat, idx_lon] += val_

	dat_plot[dat_count == 0] = -np.inf
	dat_plot /= np.maximum(dat_count, 1)

	return ax.contourf(lon_plot, lat_plot, dat_plot, levels=[-np.inf, hatches_limit, np.inf], hatches=['', '/'*hatches_strength])

def plot_categories(position_database, categories, colors, id_field, id_field_name="id_brut_bv_reu", labels=None, filters=[]):
	cat, lon, lat = [], [], []
	for id_,value in zip(id_field,categories):
		mask = position_database[id_field_name] == id_
		long_toadd, lat_toadd = np.array(position_database[mask]["longitude"]), np.array(position_database[mask]["latitude"])
		show = np.full(long_toadd.shape, True)
		for filter_ in filters:
			for j in range(len(long_toadd)):
				if show[j]:
					if filter_(long_toadd[j], lat_toadd[j]):
						show[j] = False
		lon.extend(long_toadd[show])
		lat.extend(lat_toadd[show])
		cat.extend([value] * np.sum(show))
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

	lon = np.array(filtered_election_database["longitude"])
	lat = np.array(filtered_election_database["latitude" ])

	unitary_direction_matrix, distance_matrix = utils.compute_unitary_direction_matrix_polar(lat, lon, unit="deg")

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
	
	distrib_canidates        = np.array([np.array(filtered_election_database[candidate + " Voix"]) for candidate in candidate_list])
	total_vote_candidates    = np.sum(distrib_canidates, axis=1)
	total_voting_population  = np.sum(total_vote_candidates)
	reference_distrib        = np.clip(np.sum(distrib_canidates, axis=0) / total_voting_population, 1e-6, np.inf)

	results = oth.ot_heterogeneity_populations(distrib_canidates, distance_matrix, unitary_direction_matrix)
	
	total_ot_dist      = results.global_heterogeneity
	ot_dist_candidates = results.global_heterogeneity_per_category

	ot_dist_contribution            = results.local_heterogeneity
	ot_dist_contribution_candidates = results.local_heterogeneity_per_category
	ot_dist_dissimilarity           = results.local_signed_heterogeneity

	ot_direction_per_candidate = np.swapaxes(results.direction_per_category, 0, 2)
	ot_direction               = np.swapaxes(results.direction, 0, 1)

	total_vote_proportion_candidate  = total_vote_candidates / total_voting_population
	total_vote_proportion_candidate /= np.sum(total_vote_proportion_candidate)

	clip_segregation, clip_dissimilarity, clip_votes = None, None, None
	if if_clip_segregation:
		clip_votes = [0, 0]
		for interesting_candidate in interesting_candidates[filter_idx]:
			candidate_idx     = candidate_list.index(interesting_candidate)
			candidate_distrib = np.array(filtered_election_database[interesting_candidate + " Voix"]) / (reference_distrib * total_voting_population)
			clip_votes[1]     = max(    clip_votes[1], np.percentile(candidate_distrib, 93))
			clip_votes[0]     = max(min(clip_votes[0], np.percentile(candidate_distrib, 12)), 0.01)
		clip_segregation   = [
			min(np.percentile(ot_dist_contribution_candidates, 12),  np.percentile(ot_dist_contribution, 12)),
			max(np.percentile(ot_dist_contribution_candidates, 93), np.percentile(ot_dist_contribution, 93))
		]
		#clip_dissimilarity = [np.percentile(ot_dist_dissimilarity, 4), np.percentile(ot_dist_dissimilarity, 96)]
		clip_dissimilarity = [np.percentile(np.abs(ot_dist_dissimilarity), 4), np.percentile(np.abs(ot_dist_dissimilarity), 96)]

	map_ratio = get_map_ratio(lon, lat)

	""" ######
	##########
	plot votes
	##########
	###### """

	for interesting_candidate_idx,interesting_candidate in enumerate(interesting_candidates[filter_idx]):
		candidate_idx = candidate_list.index(interesting_candidate)
		fig, ax = plt.subplots(1, 1, figsize=(6 + 1, 6/map_ratio + 0.5))

		vote_distrib_candidate    = np.array(filtered_election_database[interesting_candidate + " Voix"])
		vote_proportion_candidate = vote_distrib_candidate / np.array(filtered_election_database["Votants"])

		pl = plot_geo_data(filtered_bvote_position_database, vote_proportion_candidate, filtered_election_database["id_brut_bv_reu"],
			clip=clip_votes, filters=dont_show_filter[filter_idx], norm=matplotlib.colors.SymLogNorm(linthresh=0.001, base=2))

		cbar = fig.colorbar(pl, label="proportion of votes", format='%1.2f')
		cbar.mappable.set_clim(*clip_votes)
		cbar.mappable.set_cmap("viridis")

		""" ###################################################
		code to show the segregation index for each candidate :
		#######################################################
		line = cbar.ax.plot(0.5, total_vote_proportion_candidate[candidate_idx],
			markerfacecolor='w', markeredgecolor='w', marker='x', markersize=10)[0]
		cbar.ax.text(-1.1, line.get_ydata()-20, "avg.")
		################################################### """

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

	fig, ax = plt.subplots(1, 1, figsize=(6 + 1, 6/map_ratio + 0.5))

	pl = plot_geo_data(filtered_bvote_position_database, ot_dist_contribution, filtered_election_database["id_brut_bv_reu"],
		clip=clip_segregation, filters=dont_show_filter[filter_idx], norm=matplotlib.colors.SymLogNorm(linthresh=0.001, base=2))

	""" ########################################
	code to show a grid to get coordinats easily
	############################################
	ax.xaxis.set_major_locator(MaxNLocator(nbins=20))
	ax.yaxis.set_major_locator(MaxNLocator(nbins=20))
	ax.tick_params(axis='x', labelrotation=90)
	ax.grid()
	######################################## """

	cbar = fig.colorbar(pl, label="local contribution [m]", format='%1.0f')
	cbar.mappable.set_clim(*clip_segregation)
	cbar.mappable.set_cmap("viridis")

	""" ###################################################
	code to show the segregation index for each candidate :
	#######################################################
	for candidate in interesting_candidates.keys() :
		candidate_idx = candidate_list.index(candidate)
		line = cbar.ax.plot(0.5, ot_dist_candidates[candidate_idx],
			markerfacecolor='w', markeredgecolor='w', marker='+', markersize=10)[0]
		cbar.ax.text(-1.1, line.get_ydata()-20, interesting_candidates[candidate]["abv"])
	line = cbar.ax.plot(0.5, total_ot_dist,
		markerfacecolor='w', markeredgecolor='w', marker='x', markersize=10)[0]
	cbar.ax.text(-1.1, line.get_ydata()-20, "avg.")
	################################################### """

	ax.set_aspect(map_ratio)
	ax.set_title("Local heteogeneity index\nduring the 2022 presidencial elections")

	fig.tight_layout(pad=1.0)
	fig.savefig(fig_file_name[filter_idx][1])
	plt.close(fig)

	for interesting_candidate_idx,interesting_candidate in enumerate(interesting_candidates[filter_idx]):
		candidate_idx = candidate_list.index(interesting_candidate)

		fig, ax = plt.subplots(1, 1, figsize=(6 + 1.5, 6/map_ratio + 0.5))

		pl = plot_geo_data(filtered_bvote_position_database, ot_dist_contribution_candidates[candidate_idx, :], filtered_election_database["id_brut_bv_reu"],
			clip=clip_segregation, filters=dont_show_filter[filter_idx], norm=matplotlib.colors.SymLogNorm(linthresh=0.001, base=2))

		cbar = fig.colorbar(pl, label="local contribution [m]", format='%1.0f')
		cbar.mappable.set_clim(*clip_segregation)
		cbar.mappable.set_cmap("viridis")

		""" ###################################################
		code to show the segregation index for each candidate :
		#######################################################
		line = cbar.ax.plot(0.5, ot_dist_candidates[candidate_idx],
			markerfacecolor='w', markeredgecolor='w', marker='x', markersize=10)[0]
		cbar.ax.text(-1.1, line.get_ydata()-20, "avg.")
		################################################### """

		print(f"{ round(ot_dist_candidates[candidate_idx]) }m for { interesting_candidate } for {  commune[filter_idx][0] }")

		ax.set_aspect(map_ratio)
		ax.set_title(f"Local heteogeneity index for { interesting_candidate }\nduring the 2022 presidencial elections")

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

		fig, ax = plt.subplots(1, 1, figsize=(6 + 1.5, 6/map_ratio + 0.5))

		#pl = plot_geo_data(filtered_bvote_position_database, ot_dist_dissimilarity[candidate_idx, :], filtered_election_database["id_brut_bv_reu"],
		pl = plot_geo_data(filtered_bvote_position_database, np.abs(ot_dist_dissimilarity[candidate_idx, :]), filtered_election_database["id_brut_bv_reu"],
			clip=clip_dissimilarity, filters=dont_show_filter[filter_idx], norm=matplotlib.colors.SymLogNorm(linthresh=0.001, base=2))

		plot_hatching(filtered_bvote_position_database, (ot_dist_dissimilarity[candidate_idx, :] < 0) - 0.5, filtered_election_database["id_brut_bv_reu"],
			filters=dont_show_filter[filter_idx])

		cbar = fig.colorbar(pl, label="signed heterogeneity [m]", format='%1.2f')
		cbar.mappable.set_clim(*clip_dissimilarity)
		#cbar.mappable.set_cmap("managua")
		cbar.mappable.set_cmap("viridis")

		ax.set_aspect(map_ratio)
		ax.set_title(f"signed heterogeneity index for { interesting_candidate }\nduring the 2022 presidencial elections")

		fig.tight_layout(pad=1.0)
		fig.savefig(fig_file_name[filter_idx][3][interesting_candidate_idx])
		plt.close(fig)

	""" ##########
	##############
	plot direction
	##############
	########## """

	fig, ax = plt.subplots(1, 1, figsize=(6, 6/map_ratio + 1))

	ax.quiver(
		lon, lat,
		(ot_direction[:, 1] / np.clip(ot_dist_contribution, max(1e-6, np.percentile(ot_dist_contribution, 0.5)), np.inf)),
		(ot_direction[:, 0] / np.clip(ot_dist_contribution, max(1e-6, np.percentile(ot_dist_contribution, 0.5)), np.inf))
	)

	ax.set_aspect(map_ratio)
	ax.set_title("Direction of heteogeneity during\nthe 2022 presidencial elections")

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

		Kl_divergence_over_ot_segregation = Kl_divergence / np.clip(ot_dist_contribution, max(1e-6, np.percentile(ot_dist_contribution, 0.5)), np.inf)
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

		ax.set_title("Comparison of our heteogeneity index to the KL-divergence")
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

		pl = plot_categories(filtered_bvote_position_database, (Kl_is_upper + Kl_is_lower * 2), ["k", "b", "r"], filtered_election_database["id_brut_bv_reu"],
			filters=dont_show_filter[filter_idx],
			labels=[None, f"Lower { comparison_percetiles[0] }% of ratio of indeces", f"Upper { 100 - comparison_percetiles[1] }% of ratio of indeces"])

		ax.set_aspect(map_ratio)
		ax.set_title("map of the comparison of our heteogeneity\nindex to the KL-divergence")

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

		distort_coef_over_ot_segregation = distort_coef / np.clip(ot_dist_contribution, max(1e-6, np.percentile(ot_dist_contribution, 0.5)), np.inf)
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

		ax.set_title("Comparison of our heteogeneity index to\nthe multiscalar heteogeneity index")
		ax.set_xlabel("Our optimal-transport based index")
		ax.set_ylabel("Multiscalar heteogeneity index (non-normalized)")

		fig.tight_layout(pad=1.0)
		fig.legend(loc="lower right", bbox_to_anchor=[0.9, 0.1])
		fig.savefig(fig_file_name[filter_idx][5][2])
		plt.close(fig)

		""" ########################################################
		ploting the map of the comparison with the multiscalar index
		######################################################## """

		fig, ax = plt.subplots(1, 1, figsize=(6, 6/map_ratio + 1))

		pl = plot_categories(filtered_bvote_position_database, (dist_is_upper + dist_is_lower * 2), ["k", "b", "r"], filtered_election_database["id_brut_bv_reu"],
			filters=dont_show_filter[filter_idx],
			labels=[None, f"Lower { comparison_percetiles[0] }% of ratio of indeces", f"Upper { 100 - comparison_percetiles[1] }% of ratio of indeces"])

		ax.set_aspect(map_ratio)
		ax.set_title("map of the comparison of our heteogeneity index\nto the multiscalar heteogeneity index")

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

			ot_dist_dissimilarity_is_negative   = ot_dist_dissimilarity[candidate_idx, :] < 0
			ot_dist_dissimilarity_abs           = np.abs(ot_dist_dissimilarity[candidate_idx, :])
			ot_dist_dissimilarity_clipped_limit = max(1e-6, np.percentile(ot_dist_dissimilarity_abs, 0.5))
			ot_dist_dissimilarity_clipped       = np.clip(ot_dist_dissimilarity_abs, ot_dist_dissimilarity_clipped_limit, np.inf)
			ot_dist_dissimilarity_clipped[ot_dist_dissimilarity_is_negative] *= -1

			diff_over_dissimilarity          = candidate_vote_difference / ot_dist_dissimilarity_clipped
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

			pl = plot_categories(filtered_bvote_position_database, (diff_is_upper + diff_is_lower * 2), ["k", "b", "r"], filtered_election_database["id_brut_bv_reu"],
				filters=dont_show_filter[filter_idx],
				labels=[None, f"Lower { comparison_percetiles[0] }% of ratio of indeces", f"Upper { 100 - comparison_percetiles[1] }% of ratio of indeces"])

			ax.set_aspect(map_ratio)
			ax.set_title(f"map of the comparison of our dissimilarity\nindex to the vote excess/deficit\nfor { interesting_candidate }")

			fig.savefig(fig_file_name[filter_idx][5][5][interesting_candidate_idx])
			plt.close(fig)

		""" ############################################
		################################################
		comparison with between concave and convex index
		################################################
		############################################ """

		results_convex = oth.ot_heterogeneity_populations(distrib_canidates, distance_matrix, unitary_direction_matrix, epsilon_exponent=1e-3)

		ot_dist_contribution_convex            = results_convex.local_heterogeneity
		ot_dist_contribution_candidates_convex = results_convex.local_heterogeneity_per_category

		convex_over_concave_segregation    = ot_dist_contribution_convex / np.clip(ot_dist_contribution, max(1e-6, np.percentile(ot_dist_contribution, 0.5)), np.inf)
		convex_upper_lim, convex_lower_lim = np.percentile(convex_over_concave_segregation, comparison_percetiles[1]), np.percentile(convex_over_concave_segregation, comparison_percetiles[0])
		convex_is_upper, convex_is_lower   = convex_over_concave_segregation > convex_upper_lim, convex_over_concave_segregation < convex_lower_lim
		convex_is_middle                   = np.logical_and(np.logical_not(convex_is_upper), np.logical_not(convex_is_lower))

		""" ############################
		plotting the map for convex cost
		############################ """

		fig, ax = plt.subplots(1, 1, figsize=(6 + 1, 6/map_ratio + 0.5))

		pl = plot_geo_data(filtered_bvote_position_database, ot_dist_contribution_convex, filtered_election_database["id_brut_bv_reu"],
			clip=clip_segregation, filters=dont_show_filter[filter_idx], norm=matplotlib.colors.SymLogNorm(linthresh=0.001, base=2))

		cbar = fig.colorbar(pl, label="local contribution [m]", format='%1.0f')
		cbar.mappable.set_clim(*clip_segregation)
		cbar.mappable.set_cmap("viridis")

		ax.set_aspect(map_ratio)
		ax.set_title("Local heteogeneity (convex) index\nduring the 2022 presidencial elections")

		fig.tight_layout(pad=1.0)
		fig.savefig(fig_file_name[filter_idx][5][6])
		plt.close(fig)

		""" ############################################
		plot comparison between concave and convex index
		############################################ """

		fig, ax = plt.subplots(1, 1, figsize=(8, 8))

		ax.plot(ot_dist_contribution[convex_is_middle], ot_dist_contribution_convex[convex_is_middle], "+k", label=None)
		ax.plot(ot_dist_contribution[convex_is_upper],  ot_dist_contribution_convex[convex_is_upper],  "+r", label=f"Upper { 100 - comparison_percetiles[1] }% of ratio of indeces")
		ax.plot(ot_dist_contribution[convex_is_lower],  ot_dist_contribution_convex[convex_is_lower],  "+b", label=f"Lower { comparison_percetiles[0] }% of ratio of indeces")

		ax.set_xlim(np.percentile(ot_dist_contribution,        [1, 99]) * np.array([0.9, 1.1]))
		ax.set_ylim(np.percentile(ot_dist_contribution_convex, [1, 99]) * np.array([0.9, 1.1]))

		ax.set_xscale("log")
		ax.set_yscale("log")

		ax.set_title("Comparison of our heteogeneity index to\nbetween convex and concave")
		ax.set_xlabel("Concave index")
		ax.set_ylabel("Convex index")

		fig.tight_layout(pad=1.0)
		fig.legend(loc="lower right", bbox_to_anchor=[0.9, 0.1])
		fig.savefig(fig_file_name[filter_idx][5][7])
		plt.close(fig)

		""" ###################################################################
		ploting the map of the comparison with between concave and convex index
		################################################################### """

		fig, ax = plt.subplots(1, 1, figsize=(6, 6/map_ratio + 1))

		pl = plot_categories(filtered_bvote_position_database, (convex_is_upper + convex_is_lower * 2), ["k", "b", "r"], filtered_election_database["id_brut_bv_reu"],
			filters=dont_show_filter[filter_idx],
			labels=[None, f"Lower { comparison_percetiles[0] }% of ratio of indeces", f"Upper { 100 - comparison_percetiles[1] }% of ratio of indeces"])

		ax.set_aspect(map_ratio)
		ax.set_title("map of the comparison of our heteogeneity index\nbetween convex and concave")

		fig.savefig(fig_file_name[filter_idx][5][8])
		plt.close(fig)
		
		for interesting_candidate_idx,interesting_candidate in enumerate(interesting_candidates[filter_idx]):
			candidate_idx = candidate_list.index(interesting_candidate)

			convex_over_concave                = ot_dist_contribution_candidates_convex[candidate_idx, :] / np.clip(ot_dist_contribution_candidates[candidate_idx, :], max(1e-6, np.percentile(ot_dist_contribution_candidates[candidate_idx, :], 0.5)), np.inf)
			convex_upper_lim, convex_lower_lim = np.percentile(convex_over_concave[convex_over_concave > 0], comparison_percetiles[1]), np.percentile(convex_over_concave[convex_over_concave > 0], comparison_percetiles[0])
			convex_is_upper, convex_is_lower   = convex_over_concave > convex_upper_lim, np.logical_and(convex_over_concave < convex_lower_lim, convex_over_concave > 0)
			convex_is_middle                   = np.logical_and(np.logical_not(convex_is_upper), np.logical_not(convex_is_lower))

			""" ############################
			plotting the map for convex cost
			############################ """

			fig, ax = plt.subplots(1, 1, figsize=(6 + 1, 6/map_ratio + 0.5))

			pl = plot_geo_data(filtered_bvote_position_database, ot_dist_contribution_candidates_convex[candidate_idx, :], filtered_election_database["id_brut_bv_reu"],
				clip=clip_segregation, filters=dont_show_filter[filter_idx], norm=matplotlib.colors.SymLogNorm(linthresh=0.001, base=2))

			cbar = fig.colorbar(pl, label="local contribution [m]", format='%1.0f')
			cbar.mappable.set_clim(*clip_segregation)
			cbar.mappable.set_cmap("viridis")

			ax.set_aspect(map_ratio)
			ax.set_title("Local heteogeneity (convex) index\nduring the 2022 presidencial elections\nfor { interesting_candidate }")

			fig.tight_layout(pad=1.0)
			fig.savefig(fig_file_name[filter_idx][5][9][interesting_candidate_idx])
			plt.close(fig)

			""" #########################################################################
			plot comparison between concave and convex index for candidates per candidate
			######################################################################### """

			fig, ax = plt.subplots(1, 1, figsize=(8, 8))

			ax.plot(ot_dist_contribution_candidates[candidate_idx, :][convex_is_middle], ot_dist_contribution_candidates_convex[candidate_idx, :][convex_is_middle], "+k", label=None)
			ax.plot(ot_dist_contribution_candidates[candidate_idx, :][convex_is_upper],  ot_dist_contribution_candidates_convex[candidate_idx, :][convex_is_upper],  "+r", label=f"Upper { 100 - comparison_percetiles[1] }% of ratio of indeces")
			ax.plot(ot_dist_contribution_candidates[candidate_idx, :][convex_is_lower],  ot_dist_contribution_candidates_convex[candidate_idx, :][convex_is_lower],  "+b", label=f"Lower { comparison_percetiles[0] }% of ratio of indeces")

			ax.set_xlim(np.percentile(ot_dist_contribution_candidates[candidate_idx, :],        [1, 99]) * 1.1)
			ax.set_ylim(np.percentile(ot_dist_contribution_candidates_convex[candidate_idx, :], [1, 99]) * 1.1)

			ax.set_title(f"Comparison of our index\nbetween convex and concave\nfor { interesting_candidate }")
			ax.set_xlabel("Concave index")
			ax.set_ylabel("Convex index")

			fig.tight_layout(pad=1.0)
			fig.legend(loc="lower right", bbox_to_anchor=[0.9, 0.1])
			fig.savefig(fig_file_name[filter_idx][5][10][interesting_candidate_idx])
			plt.close(fig)

			""" #############################################################################
			ploting the map of the comparison between concave and convex index for candidates
			############################################################################# """

			fig, ax = plt.subplots(1, 1, figsize=(6, 6/map_ratio + 1))

			pl = plot_categories(filtered_bvote_position_database, (convex_is_upper + convex_is_lower * 2), ["k", "b", "r"], filtered_election_database["id_brut_bv_reu"],
				filters=dont_show_filter[filter_idx],
				labels=[None, f"Lower { comparison_percetiles[0] }% of ratio of indeces", f"Upper { 100 - comparison_percetiles[1] }% of ratio of indeces"])

			ax.set_aspect(map_ratio)
			ax.set_title(f"map of the comparison of our index\nbetween convex and concave\nfor { interesting_candidate }")

			fig.savefig(fig_file_name[filter_idx][5][11][interesting_candidate_idx])
			plt.close(fig)

