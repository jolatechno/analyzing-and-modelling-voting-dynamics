#include <iostream>

#include "modular_election_simulation_framework/src/core/network.hpp"
#include "modular_election_simulation_framework/src/core/networks/network_file_io.hpp"
#include "modular_election_simulation_framework/src/core/networks/network_generator.hpp"
#include "modular_election_simulation_framework/src/core/networks/network_partition.hpp"
#include "modular_election_simulation_framework/src/core/networks/network_util.hpp"
#include "modular_election_simulation_framework/src/core/agent_population/agent_population.hpp"
#include "modular_election_simulation_framework/src/implementations/Nvoter_model.hpp"
#include "modular_election_simulation_framework/src/implementations/population_Nvoter_model.hpp"

#include "modular_election_simulation_framework/src/util/json_util.hpp"
#include "modular_election_simulation_framework/src/util/hdf5_util.hpp"
#include "modular_election_simulation_framework/src/util/util.hpp"

#include "modular_election_simulation_framework/src/core/segregation/convergence_time.hpp"
#include "modular_election_simulation_framework/src/core/segregation/map_util.hpp"


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


template<class Type>
double get_median(std::vector<Type> vec) {
	std::sort(vec.begin(), vec.end()); 

	size_t median_element = vec.size()/2;
	return vec[median_element];
} 


int main(int argc, char *argv[]) {
	std::string config_name = util::get_first_cmd_arg(argc, argv);
	auto config             = util::json::read_config((root + config_file).c_str(), config_name);

	const std::string input_file_name  = root + std::string(config["preprocessed_file"    ].asString());
	const std::string output_file_name = root + std::string(config["output_file_convergence_time"].asString());

	const bool   parallel = config["convergence_time"]["parallel"].asBool();
	const size_t N_select = config["convergence_time"]["N_select"].asInt();

	      size_t N_nodes;
	const int    N_try  = config["convergence_time"]["N_try" ].asInt();
	const int    N_it   = config["convergence_time"]["N_it"  ].asInt();
	const int    n_save = config["convergence_time"]["n_save"].asInt();

	const bool read_network_from_file = config["convergence_time"]["read_network_from_file"].asBool();
	const int  n_attachment           = config["convergence_time"]["n_attachment"          ].asInt();

	const int N_thresh = config["convergence_time"]["N_thresh"].asInt();

	const auto convergence_thresholds = util::math::logspace<double>(1e-7d, 1.5d, N_thresh);


	H5::H5File output_file(output_file_name, H5F_ACC_TRUNC);
	H5::H5File input_file( input_file_name,  H5F_ACC_RDWR);

	auto *interaction      = new BPsimulation::implem::population_Nvoter_interaction_function<N_candidates>(N_select);
	auto *renormalize      = new BPsimulation::core::agent::population::PopulationRenormalizeProportions<BPsimulation::implem::Nvoter<N_candidates>>();
	auto *agent_serializer = new BPsimulation::core::agent::population::AgentPopulationSerializer<BPsimulation::implem::Nvoter<N_candidates>>();


	std::vector<float> lat, lon;
	H5::Group geo_data = input_file.openGroup("geo_data");
	util::hdf5io::H5ReadVector(geo_data, lat, "lat");
	util::hdf5io::H5ReadVector(geo_data, lon, "lon");
	N_nodes = lat.size();

	H5::Group output_geo_data = output_file.createGroup("geo_data");
	util::hdf5io::H5WriteVector(output_geo_data, lat, "lat");
	util::hdf5io::H5WriteVector(output_geo_data, lon, "lon");


	auto *network = new BPsimulation::SocialNetwork<BPsimulation::core::agent::population::AgentPopulation<BPsimulation::implem::Nvoter<N_candidates>>>(N_nodes);
	if (read_network_from_file) {
		BPsimulation::io::read_network_from_file(network, input_file);
	} else {
		auto distances = segregation::map::util::get_distances(lat, lon);
		BPsimulation::random::closest_neighbor_limited_attachment(network, distances, n_attachment);
	}
	BPsimulation::io::write_network_to_file(network, output_file);


	std::vector<double> populations;
	H5::Group demo_data = input_file.openGroup("demo_data");
	util::hdf5io::H5ReadVector(demo_data, populations,  "voter_population");

	std::vector<std::vector<double>> votes(N_candidates);
	H5::Group vote_data = input_file.openGroup("vote_data");
	for (int icandidate = 0; icandidate < N_candidates; ++icandidate) {
		std::string field_name = "PROP_Voix_" + candidates_from_left_to_right[icandidate];
		util::hdf5io::H5ReadVector(vote_data, votes[icandidate], field_name.c_str());
	}

	for (size_t node = 0; node < N_nodes; ++node) {

		for (int icandidate = 0; icandidate < N_candidates; ++icandidate) {
			(*network)[node].proportions[icandidate] = votes[icandidate][node];
		}

		(*network)[node].population = (size_t)populations[node];
	}
	network->update_agentwise(renormalize);



	std::vector<std::vector<std::vector<float>>> trajectories(N_candidates,
			std::vector<std::vector<float>>(                  N_nodes,
				std::vector<float>(                           N_it/n_save+1)));


	BPsimulation::io::write_agent_states_to_file(network, agent_serializer, output_file, "/initial_state");
	util::hdf5io::H5flush_and_clean(output_file, true);
	for (size_t node = 0; node < N_nodes; ++node) {
		for (int icandidate = 0; icandidate < N_candidates; ++icandidate) {
			trajectories[icandidate][node][0] = (*network)[node].proportions[icandidate];
		}
	}


	BPsimulation::implem::Nvoter_majority_election_result<N_candidates>* general_election_results;
	for (int itry = 0; itry < N_try; ++itry) {
		if (itry > 0) {
			BPsimulation::io::read_agent_states_from_file(network, agent_serializer, output_file, "/initial_state");
		}

		std::cout << "try " << itry+1 << "/" << N_try << "\n";

		for (int it = 0; it < N_it; ++it) {
			if (it%n_save == 0 && it > 0) {
				for (size_t node = 0; node < N_nodes; ++node) {
					for (int icandidate = 0; icandidate < N_candidates; ++icandidate) {
						trajectories[icandidate][node][it/n_save] = (*network)[node].proportions[icandidate];
					}
				}
			}

			network->interact(interaction, parallel);
			network->update_agentwise(renormalize);
		}

		{
			std::string dir_name = "/analysis_" + std::to_string(itry);
			H5::Group analysis = output_file.createGroup(dir_name);

			auto KLdiv_trajectories    = segregation::convergence_time::get_KLdiv_trajectories_versus_trajectory_end(trajectories);
			auto focal_distances_idxes = segregation::convergence_time::get_focal_distance_indexes(KLdiv_trajectories, convergence_thresholds);
			auto distortion_coefs      = segregation::convergence_time::get_distortion_coefs_from_KLdiv(KLdiv_trajectories);

			for (int icandidate = 0; icandidate < N_candidates; ++icandidate) {
				std::string field_name = "vote_traj_" + candidates_from_left_to_right[icandidate];
				util::hdf5io::H5WriteIrregular2DVector(analysis, trajectories[icandidate], field_name.c_str());
			}
			util::hdf5io::H5WriteIrregular2DVector(analysis, KLdiv_trajectories,    "KLdiv_trajectories");
			util::hdf5io::H5WriteIrregular2DVector(analysis, focal_distances_idxes, "focal_distances");
			util::hdf5io::H5WriteVector(           analysis, distortion_coefs,      "distortion_coefs");

			util::hdf5io::H5WriteVector(analysis, convergence_thresholds, "convergence_thresholds");
			util::hdf5io::H5flush_and_clean(output_file, true);
		}
	}
}