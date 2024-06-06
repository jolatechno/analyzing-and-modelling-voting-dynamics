#include <iostream>

#include "modular_election_simulation_framework/src/core/network.hpp"
#include "modular_election_simulation_framework/src/core/networks/network_file_io.hpp"
#include "modular_election_simulation_framework/src/core/networks/network_generator.hpp"
#include "modular_election_simulation_framework/src/core/networks/network_partition.hpp"
#include "modular_election_simulation_framework/src/core/networks/network_util.hpp"
#include "modular_election_simulation_framework/src/core/agent_population/agent_population.hpp"
#include "modular_election_simulation_framework/src/core/agent_population/agent_population_util.hpp"

#include "modular_election_simulation_framework/src/implementations/voter_model.hpp"
#include "modular_election_simulation_framework/src/implementations/voter_model_stubborn.hpp"
#include "modular_election_simulation_framework/src/implementations/population_voter_model.hpp"
#include "modular_election_simulation_framework/src/implementations/population_voter_model_stubborn.hpp"

#include "modular_election_simulation_framework/src/implementations/Nvoter_stubborn_model.hpp"
#include "modular_election_simulation_framework/src/implementations/population_Nvoter_stubborn_model.hpp"

#include "modular_election_simulation_framework/src/util/json_util.hpp"
#include "modular_election_simulation_framework/src/util/hdf5_util.hpp"
#include "modular_election_simulation_framework/src/util/util.hpp"

#include "modular_election_simulation_framework/src/core/segregation/multiscalar.hpp"
#include "modular_election_simulation_framework/src/core/segregation/multiscalar_util.hpp"
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
	std::vector<Type> vec_copy(vec.begin(), vec.end());
	std::sort(vec_copy.begin(), vec_copy.end()); 

	size_t median_element = vec_copy.size()/2;
	return vec_copy[median_element];
} 


int main(int argc, char *argv[]) {
	std::string config_name = util::get_first_cmd_arg(argc, argv);
	auto config             = util::json::read_config((root + config_file).c_str(), config_name);

	const std::string input_file_name  = root + std::string(config["preprocessed_file"     ].asString());
	const std::string output_file_name = root + std::string(config["output_file_simulation"].asString());


	const bool   parallel                           = config["simulation"]["parallel"                          ].asBool();
	const size_t N_select                           = config["simulation"]["N_select"                          ].asInt();
	const double dt                                 = config["simulation"]["dt"                                ].asDouble();
	const double overtoon_multiplier                = config["simulation"]["overtoon_multiplier"               ].asDouble();
	const double overtoon_radicalization_multiplier = config["simulation"]["overtoon_radicalization_multiplier"].asDouble();
	const double overtoon_radius                    = config["simulation"]["overtoon_radius"                   ].asDouble();
	const double frustration_multiplier             = config["simulation"]["frustration_multiplier"            ].asDouble();

	const double initial_radicalization_multiplier = config["simulation"]["initial_radicalization_multiplier"].asDouble();

	      size_t N_nodes;
	const int    N_counties = config["simulation"]["N_counties"].asInt();
	const int    N_try      = config["simulation"]["N_try"     ].asInt();
	const int    N_it       = config["simulation"]["N_it"      ].asInt();
	const int    n_election = config["simulation"]["n_election"].asInt();
	const int    n_save     = config["simulation"]["n_save"    ].asInt();

	const bool read_network_from_file = config["simulation"]["read_network_from_file"].asBool();
	const int  n_attachment           = config["simulation"]["n_attachment"          ].asInt();

	const bool   random_seed = config["simulation"]["random_seed"].asBool();
	const size_t seed        = config["simulation"]["seed"       ].asInt();
	if (!random_seed) {
		util::set_generator_seed(seed);
	}


	H5::H5File output_file = util::hdf5io::open_truncate_if_needed(output_file_name.c_str());
	H5::H5File input_file(input_file_name .c_str(), H5F_ACC_RDONLY);


	auto *election = new BPsimulation::core::agent::population::PopulationElection<BPsimulation::implem::Nvoter_stubborn<N_candidates>>(new BPsimulation::implem::Nvoter_majority_election<N_candidates, BPsimulation::implem::Nvoter_stubborn<N_candidates>>());

	auto *interaction = new BPsimulation::implem::population_Nvoter_stubborn_interaction_function<N_candidates>(N_select);
	auto *agentwise   = new BPsimulation::implem::Nvoter_stubborn_equilibirum_function<           N_candidates>(dt);
	auto *overton     = new BPsimulation::implem::Nvoter_stubborn_overtoon_effect<                N_candidates>(dt, overtoon_multiplier, overtoon_radicalization_multiplier, overtoon_radius);
	auto *frustration = new BPsimulation::implem::Nvoter_stubborn_frustration_effect<             N_candidates>(dt, frustration_multiplier);

	auto *renormalize = new BPsimulation::core::agent::population::PopulationRenormalizeProportions<BPsimulation::implem::Nvoter_stubborn<N_candidates>>();

	auto *agent_full_serializer    = new BPsimulation::implem::AgentPopulationNVoterstubbornSerializer<N_candidates>();
	auto *agent_partial_serializer = new BPsimulation::core::agent::population::AgentPopulationSerializer<BPsimulation::implem::Nvoter_stubborn<N_candidates>>();
	auto *election_serializer      = new BPsimulation::implem::NVoterMajorityElectionSerializer<N_candidates>();


	std::vector<float> lat, lon;
	H5::Group geo_data = input_file.openGroup("geo_data");
	util::hdf5io::H5ReadVector(geo_data, lat, "lat");
	util::hdf5io::H5ReadVector(geo_data, lon, "lon");
	geo_data.close();
	N_nodes = lat.size();

	H5::Group output_geo_data = output_file.createGroup("geo_data");
	util::hdf5io::H5WriteVector(output_geo_data, lat, "lat");
	util::hdf5io::H5WriteVector(output_geo_data, lon, "lon");
	output_geo_data.close();


	auto *network = new BPsimulation::SocialNetwork<BPsimulation::implem::AgentPopulationNVoterstubborn<N_candidates>>(N_nodes);
	if (read_network_from_file) {
		BPsimulation::io::read_network_from_file(network, input_file);
	} else {
		auto distances = segregation::map::util::get_distances(lat, lon);
		BPsimulation::random::closest_neighbor_limited_attachment(network, distances, n_attachment);
	}
	BPsimulation::io::write_network_to_file(network, output_file);


	std::vector<std::vector<size_t>> counties = {{}, {}};
	float median = get_median(lat);
	for (size_t node = 0; node < N_nodes; ++node) {
		int group = lat[node] < median;
		counties[group].push_back(node);
	}
	BPsimulation::io::write_counties_to_file(counties, output_file);


	std::vector<double> populations;
	H5::Group demo_data = input_file.openGroup("demo_data");
	util::hdf5io::H5ReadVector(demo_data, populations,  "voter_population");
	demo_data.close();


	std::vector<std::vector<double>> votes(N_candidates);
	H5::Group vote_data = input_file.openGroup("vote_data");
	for (int icandidate = 0; icandidate < N_candidates; ++icandidate) {
		std::string field_name = "PROP_Voix_" + candidates_from_left_to_right[icandidate];
		util::hdf5io::H5ReadVector(vote_data, votes[icandidate], field_name.c_str());
	}
	vote_data.close();


	double normalization_factor_pop = segregation::multiscalar::get_normalization_factor_pop(votes);


	for (size_t node = 0; node < N_nodes; ++node) {

		for (int icandidate = 0; icandidate < N_candidates; ++icandidate) {
			(*network)[node].proportions[icandidate]         = votes[icandidate][node];
			(*network)[node].stubborn_equilibrium[icandidate] = votes[icandidate][node]*initial_radicalization_multiplier;
		}

		(*network)[node].population = (size_t)populations[node];
	}
	network->update_agentwise(renormalize);

	BPsimulation::io::write_agent_states_to_file(network, agent_full_serializer, output_file, "/initial_state");
	util::hdf5io::H5flush_and_clean(output_file);

	{
		std::string segregation_dir_name = "/segregation_initial_state";
		H5::Group segregation = output_file.createGroup(segregation_dir_name);

		auto vote_proportions = BPsimulation::core::agent::population::util::get_vote_proportions<BPsimulation::implem::Nvoter_stubborn<N_candidates>>(network);
		BPsimulation::implem::util::accumulate_stubborn_votes(vote_proportions);

		auto normalized_distortion_coefs = segregation::multiscalar::get_distortion_coefs_fast(vote_proportions,
			(std::function<std::pair<std::vector<size_t>, std::vector<double>>(size_t)>) [&votes, &lat, &lon](size_t i) {
				auto distances_slice  = segregation::map::util::get_distances(lat, lon, std::vector<size_t>{i});
				auto traj_idxes_slice = segregation::multiscalar::get_closest_neighbors(distances_slice);

				auto accumulated_trajectory_pop = segregation::multiscalar::util::get_accumulated_trajectory(votes, traj_idxes_slice);

				return std::pair<std::vector<size_t>, std::vector<double>>(traj_idxes_slice[0], accumulated_trajectory_pop[0]);
			}, normalization_factor_pop);
		
		util::hdf5io::H5WriteVector(segregation, normalized_distortion_coefs, "normalized_distortion_coefs");
		segregation.close();
		util::hdf5io::H5flush_and_clean(output_file, true);
	}


	BPsimulation::implem::Nvoter_majority_election_result<N_candidates>* general_election_results;
	std::vector<BPsimulation::core::election::ElectionResultTemplate*> counties_election_results, stubborness_results;
	for (int itry = 0; itry < N_try; ++itry) {
		if (itry > 0) {
			BPsimulation::io::read_agent_states_from_file(network, agent_full_serializer, output_file, "/initial_state");
		}

		for (int it = 0; it < N_it; ++it) {
			if (it%n_save == 0 && it > 0) {
				std::string dir_name = "/states_" + std::to_string(itry) + "_" + std::to_string(it);
				BPsimulation::io::write_agent_states_to_file(network, agent_partial_serializer, output_file, dir_name.c_str());
			

				std::string segregation_dir_name = "/segregation_state_" + std::to_string(itry) + "_" + std::to_string(it);
				H5::Group segregation = output_file.createGroup(segregation_dir_name);

				auto vote_proportions            = BPsimulation::core::agent::population::util::get_vote_proportions<BPsimulation::implem::Nvoter_stubborn<N_candidates>>(network);
				auto normalized_distortion_coefs = segregation::multiscalar::get_distortion_coefs_fast(vote_proportions,
					(std::function<std::pair<std::vector<size_t>, std::vector<double>>(size_t)>) [&votes, &lat, &lon](size_t i) {
						auto distances_slice  = segregation::map::util::get_distances(lat, lon, std::vector<size_t>{i});
						auto traj_idxes_slice = segregation::multiscalar::get_closest_neighbors(distances_slice);

				auto accumulated_trajectory_pop = segregation::multiscalar::util::get_accumulated_trajectory(votes, traj_idxes_slice);

				return std::pair<std::vector<size_t>, std::vector<double>>(traj_idxes_slice[0], accumulated_trajectory_pop[0]);
					}, normalization_factor_pop);
				
				util::hdf5io::H5WriteVector(segregation, normalized_distortion_coefs, "normalized_distortion_coefs");
				segregation.close();
				util::hdf5io::H5flush_and_clean(output_file);
			}

			if (it%n_election == 0) {
				general_election_results  = (BPsimulation::implem::Nvoter_majority_election_result<N_candidates>*)network->get_election_results(election);
				counties_election_results = network->get_election_results(counties, election);

				std::string dir_name_general  = "/general_election_result_"  + std::to_string(itry) + "_" + std::to_string(it);
				std::string dir_name_counties = "/counties_election_result_" + std::to_string(itry) + "_" + std::to_string(it);

				BPsimulation::io::write_election_result_to_file( general_election_results,  election_serializer, output_file, dir_name_general.c_str());
				BPsimulation::io::write_election_results_to_file(counties_election_results, election_serializer, output_file, dir_name_counties.c_str());
				util::hdf5io::H5flush_and_clean(output_file);

				std::cout << "\n\ntry " << itry+1 << "/" << N_try << ", it " << it << "/" << N_it-1 << ":\n\n";
				std::cout << "network->get_election_results(...) = " << general_election_results->result << " (" << general_election_results->proportions << ")\n";
				std::cout << "network->get_election_results(counties, ...): \n";
				for (int couty = 0; couty < counties.size(); couty++) {
					BPsimulation::implem::Nvoter_majority_election_result<N_candidates> *county_result = (BPsimulation::implem::Nvoter_majority_election_result<N_candidates>*)counties_election_results[couty];
					std::cout << "\t" << county_result->result  << " (" << county_result->proportions << ") for county: " << counties[couty] << "\n";
				}
				std::cout << "\n";
			}

			network->interact(interaction, parallel);
			network->update_agentwise(agentwise);
			network->election_retroinfluence(general_election_results, overton);
			network->election_retroinfluence(general_election_results, frustration);
			network->election_retroinfluence(counties, counties_election_results, overton);
			network->election_retroinfluence(counties, counties_election_results, frustration);
			network->update_agentwise(renormalize);
		}

		util::hdf5io::H5flush_and_clean(output_file, true);
	}
}