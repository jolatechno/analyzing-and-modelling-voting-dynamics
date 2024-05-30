#include <iostream>

#include "modular_election_simulation_framework/src/core/segregation/multiscalar.hpp"
#include "modular_election_simulation_framework/src/core/segregation/multiscalar_util.hpp"
#include "modular_election_simulation_framework/src/core/segregation/map_util.hpp"

#include "modular_election_simulation_framework/src/util/json_util.hpp"
#include "modular_election_simulation_framework/src/util/hdf5_util.hpp"
#include "modular_election_simulation_framework/src/util/util.hpp"


const std::vector<std::string> candidates_from_left_to_right = {
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
};
const int N_candidates = 12;

const std::string root        = "output/";
const std::string config_file = "config.json";


int main(int argc, char *argv[]) {
	std::string config_name = util::get_first_cmd_arg(argc, argv);
	auto config             = util::json::read_config((root + config_file).c_str(), config_name);

	const std::string input_file_name  = root + std::string(config["preprocessed_file"      ].asString());
	const std::string output_file_name = root + std::string(config["output_file_segregation"].asString());

	const int N_full_analyze = config["segregation"]["N_full_analyze"].asInt();
	const int N_thresh       = config["segregation"]["N_thresh"      ].asInt();

	const auto convergence_thresholds = util::math::logspace<double>(1e-7d, 9.d, N_thresh);


	H5::H5File output_file(output_file_name.c_str(), H5F_ACC_TRUNC);
	H5::H5File input_file( input_file_name .c_str(), H5F_ACC_RDWR);


	std::vector<float> lat, lon;
	H5::Group geo_data = input_file.openGroup("geo_data");
	util::hdf5io::H5ReadVector(geo_data, lat, "lat");
	util::hdf5io::H5ReadVector(geo_data, lon, "lon");

	H5::Group output_geo_data = output_file.createGroup("geo_data");
	util::hdf5io::H5WriteVector(output_geo_data, lat, "lat");
	util::hdf5io::H5WriteVector(output_geo_data, lon, "lon");


	std::vector<double> populations;
	H5::Group demo_data = input_file.openGroup("demo_data");
	util::hdf5io::H5ReadVector(demo_data, populations, "voter_population");

	std::vector<std::vector<double>> votes(N_candidates);
	H5::Group vote_data = input_file.openGroup("vote_data");
	for (int icandidate = 0; icandidate < N_candidates; ++icandidate) {
		std::string field_name = "Voix_" + candidates_from_left_to_right[icandidate];
		util::hdf5io::H5ReadVector(vote_data, votes[icandidate], field_name.c_str());
	}


	H5::Group full_analysis = output_file.createGroup("full_analysis");

	double normalization_factor = 1.d, normalization_factor_pop = 1.d, normalization_factor_dist = 1.d;
	{
		auto distances = segregation::map::util::get_distances(lat, lon);

		util::hdf5io::H5WriteIrregular2DVector(output_geo_data, distances, "distances");


		auto worst_population_trajectory = segregation::multiscalar::util::get_worst_population_trajectory(votes);
		auto worst_KLdiv_trajectory      = segregation::multiscalar::util::get_worst_KLdiv_trajectory(votes);
		auto worst_focal_distances       = segregation::multiscalar::get_focal_distance_indexes(std::vector<std::vector<double>>{worst_KLdiv_trajectory}, convergence_thresholds)[0];

		H5::Group normalization_factors = full_analysis.createGroup("normalization_factors");
		util::hdf5io::H5WriteVector(normalization_factors, worst_population_trajectory, "worst_population_trajectory");
		util::hdf5io::H5WriteVector(normalization_factors, worst_KLdiv_trajectory,      "worst_KLdiv_trajectory");
		util::hdf5io::H5WriteVector(normalization_factors, worst_focal_distances,       "worst_focal_distances");
		util::hdf5io::H5WriteVector(normalization_factors, convergence_thresholds,      "convergence_thresholds");


		{
			     normalization_factor = segregation::multiscalar::get_normalization_factor(votes);
			auto worst_Xvalues        = segregation::multiscalar::util::get_worst_Xvalues({}, votes[0].size());

			util::hdf5io::H5WriteSingle(normalization_factors, normalization_factor, "normalization_factor");
			util::hdf5io::H5WriteVector(normalization_factors, worst_Xvalues,        "worst_Xvalues");
		}


		{
			auto worst_Xvalues = segregation::multiscalar::util::get_worst_population_trajectory(votes);

			normalization_factor_pop = segregation::multiscalar::get_normalization_factor_pop(votes);

			util::hdf5io::H5WriteSingle(normalization_factors, normalization_factor_pop, "normalization_factor_pop");
			util::hdf5io::H5WriteVector(normalization_factors, worst_Xvalues,            "worst_Xvalues_pop");
		}


		{
			for (size_t i = 0; i < distances.size(); ++i) {
				std::sort(distances[i].begin(), distances[i].end());
			}
			auto worst_Xvalues = segregation::multiscalar::util::get_worst_Xvalues(distances);

			normalization_factor_dist = segregation::multiscalar::get_normalization_factor(votes, distances);

			util::hdf5io::H5WriteSingle(normalization_factors, normalization_factor_dist, "normalization_factor_dist");
			util::hdf5io::H5WriteVector(normalization_factors, worst_Xvalues,             "worst_Xvalues_dist");
		}
	}


	{
		std::vector<size_t> full_analyze_idxs(lat.size());
		std::iota(full_analyze_idxs.begin(), full_analyze_idxs.end(), 0);
		std::shuffle(full_analyze_idxs.begin(), full_analyze_idxs.end(), util::get_random_generator());
		full_analyze_idxs.resize(N_full_analyze);
		std::sort(full_analyze_idxs.begin(), full_analyze_idxs.end());

		std::cout << "Computing partial analysis...\n";


		auto distances_slice       = segregation::map::util::get_distances(lat, lon, full_analyze_idxs);
		auto traj_idxes_slice      = segregation::multiscalar::get_closest_neighbors(distances_slice);
		auto accumulated_pop_slice = segregation::multiscalar::util::get_accumulated_trajectory(votes, traj_idxes_slice);
		auto vote_trajectories     = segregation::multiscalar::get_trajectories(votes, traj_idxes_slice);
		auto KLdiv_trajectories    = segregation::multiscalar::get_KLdiv_trajectories(vote_trajectories);
		auto focal_distances_idxes = segregation::multiscalar::get_focal_distance_indexes(KLdiv_trajectories, convergence_thresholds);
		auto distortion_coefs      = segregation::multiscalar::get_distortion_coefs_from_KLdiv(KLdiv_trajectories, std::vector<std::vector<float>>{}, normalization_factor);


		H5::Group partial_analysis = output_file.createGroup("partial_analysis");
		util::hdf5io::H5WriteVector(           partial_analysis, convergence_thresholds, "convergence_thresholds");
		util::hdf5io::H5WriteIrregular2DVector(partial_analysis, distances_slice,        "distances");
		util::hdf5io::H5WriteIrregular2DVector(partial_analysis, accumulated_pop_slice,  "accumulated_trajectory_pop");
		util::hdf5io::H5WriteVector(           partial_analysis, full_analyze_idxs,      "full_analyze_idxs");
		util::hdf5io::H5WriteIrregular2DVector(partial_analysis, traj_idxes_slice,       "knn");

		for (int icandidate = 0; icandidate < N_candidates; ++icandidate) {
			std::string field_name = "vote_trajectory_" + candidates_from_left_to_right[icandidate];
			util::hdf5io::H5WriteIrregular2DVector(partial_analysis, vote_trajectories[icandidate], field_name.c_str());
		}
		util::hdf5io::H5WriteIrregular2DVector(partial_analysis, KLdiv_trajectories,    "KLdiv_trajectories");
		util::hdf5io::H5WriteIrregular2DVector(partial_analysis, focal_distances_idxes, "focal_distances_idxes");
		util::hdf5io::H5WriteVector(           partial_analysis, distortion_coefs,      "distortion_coefs");


		std::vector<float> partial_lat(N_full_analyze), partial_lon(N_full_analyze);
		for (size_t i = 0; i < N_full_analyze; ++i) {
			partial_lat[i] = lat[full_analyze_idxs[i]];
			partial_lon[i] = lon[full_analyze_idxs[i]];
		}

		H5::Group partial_analysis_geo_data = partial_analysis.createGroup("geo_data");
		util::hdf5io::H5WriteVector(partial_analysis_geo_data, partial_lat, "lat");
		util::hdf5io::H5WriteVector(partial_analysis_geo_data, partial_lon, "lon");
	}

	std::cout << "Computing full analysis...\n";

	util::hdf5io::H5WriteVector(full_analysis, populations, "voter_population");

	{
		auto normalized_distortion_coefs = segregation::multiscalar::get_distortion_coefs_fast(votes,
			(std::function<std::pair<std::vector<size_t>, std::vector<double>>(size_t)>) [&lat, &lon](size_t i) {
				auto distances_slice  = segregation::map::util::get_distances(lat, lon, std::vector<size_t>{i});
				auto traj_idxes_slice = segregation::multiscalar::get_closest_neighbors(distances_slice);

				return std::pair<std::vector<size_t>, std::vector<double>>(traj_idxes_slice[0], {});
			}, normalization_factor);
		std::cout << "\nnormalized distortion coefs: " << normalized_distortion_coefs;
		std::cout << "  <<  (" << *std::max_element(normalized_distortion_coefs.begin(), normalized_distortion_coefs.end()) << ")\n";
		
		util::hdf5io::H5WriteVector(full_analysis, normalized_distortion_coefs,      "normalized_distortion_coefs");
	}


	{
		auto normalized_distortion_coefs_pop = segregation::multiscalar::get_distortion_coefs_fast(votes,
			(std::function<std::pair<std::vector<size_t>, std::vector<double>>(size_t)>) [&votes, &lat, &lon](size_t i) {
				auto distances_slice  = segregation::map::util::get_distances(lat, lon, std::vector<size_t>{i});
				auto traj_idxes_slice = segregation::multiscalar::get_closest_neighbors(distances_slice);

				auto accumulated_trajectory_pop = segregation::multiscalar::util::get_accumulated_trajectory(votes, traj_idxes_slice);

				return std::pair<std::vector<size_t>, std::vector<double>>(traj_idxes_slice[0], accumulated_trajectory_pop[0]);
			}, normalization_factor_pop);
		std::cout << "\nnormalized distortion coefs (pop): " << normalized_distortion_coefs_pop;
		std::cout << "  <<  (" << *std::max_element(normalized_distortion_coefs_pop.begin(), normalized_distortion_coefs_pop.end()) << ")\n";
		
		util::hdf5io::H5WriteVector(full_analysis, normalized_distortion_coefs_pop,  "normalized_distortion_coefs_pop");
	}

	{
		auto normalized_distortion_coefs_dist = segregation::multiscalar::get_distortion_coefs_fast(votes,
			(std::function<std::pair<std::vector<size_t>, std::vector<float>>(size_t)>) [&lat, &lon](size_t i) {
				auto distances_slice  = segregation::map::util::get_distances(lat, lon, std::vector<size_t>{i});
				auto traj_idxes_slice = segregation::multiscalar::get_closest_neighbors(distances_slice);

				std::sort(distances_slice[0].begin(), distances_slice[0].end());

				return std::pair<std::vector<size_t>, std::vector<float>>(traj_idxes_slice[0], distances_slice[0]);
			}, normalization_factor_dist);
		std::cout << "\nnormalized distortion coefs (dist): " << normalized_distortion_coefs_dist;
		std::cout << "  <<  (" << *std::max_element(normalized_distortion_coefs_dist.begin(), normalized_distortion_coefs_dist.end()) << ")\n";

		util::hdf5io::H5WriteVector(full_analysis, normalized_distortion_coefs_dist, "normalized_distortion_coefs_dist");
	}
}