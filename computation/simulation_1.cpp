#include <iostream>

#include "modular_election_simulation_framework/src/core/network.hpp"
#include "modular_election_simulation_framework/src/core/networks/network_file_io.hpp"
#include "modular_election_simulation_framework/src/core/networks/network_generator.hpp"
#include "modular_election_simulation_framework/src/core/networks/network_partition.hpp"
#include "modular_election_simulation_framework/src/core/networks/network_util.hpp"
#include "modular_election_simulation_framework/src/core/agent_population/agent_population.hpp"
#include "modular_election_simulation_framework/src/implementations/voter_model.hpp"
#include "modular_election_simulation_framework/src/implementations/voter_model_stubborn.hpp"
#include "modular_election_simulation_framework/src/implementations/population_voter_model.hpp"
#include "modular_election_simulation_framework/src/implementations/population_voter_model_stubborn.hpp"
#include "modular_election_simulation_framework/src/util/util.hpp"


const size_t N_select               = 15;
const double dt                     = 0.2;
const double overtoon_multiplier    = 0.1;
const double frustration_multiplier = 0.01;

const bool   parallel   = false;
const size_t N_nodes    = 800;
const int    n_con      = 3;
const int    N_counties = 3;
const int    N_try      = 10;
const int    N_it       = 3001;
const int    n_election = 500;
const int    n_save     = 10;

const bool   random_seed = false;
const size_t seed        = 1;

const char* file_name = "output/output-simulation.h5";


int main() {
	H5::H5File file(file_name, H5F_ACC_TRUNC);


	auto *election            = new BPsimulation::core::agent::population::PopulationElection<BPsimulation::implem::voter_stubborn>(new BPsimulation::implem::voter_majority_election<BPsimulation::implem::voter_stubborn>());
	auto *stubborness_election = new BPsimulation::core::agent::population::PopulationElection<BPsimulation::implem::voter_stubborn>(new BPsimulation::implem::voter_stubborness_election());

	auto *interaction = new BPsimulation::implem::population_voter_stubborn_interaction_function(N_select);
	auto *agentwise   = new BPsimulation::implem::voter_stubborn_equilibirum_function(dt);
	auto *overton     = new BPsimulation::implem::voter_stubborn_overtoon_effect(     dt, overtoon_multiplier);
	auto *frustration = new BPsimulation::implem::voter_stubborn_frustration_effect(  dt, frustration_multiplier);

	auto *renormalize = new BPsimulation::core::agent::population::PopulationRenormalizeProportions<BPsimulation::implem::voter_stubborn>();

	auto *agent_full_serializer    = new BPsimulation::implem::AgentPopulationVoterstubbornSerializer();
	auto *agent_partial_serializer = new BPsimulation::core::agent::population::AgentPopulationSerializer<BPsimulation::implem::voter_stubborn>();
	auto *election_serializer      = new BPsimulation::implem::VoterMajorityElectionSerializer();


	if (!random_seed) {
		util::set_generator_seed(seed);
	}


	auto *network = new BPsimulation::SocialNetwork<BPsimulation::implem::AgentPopulationVoterstubborn>(N_nodes);
	BPsimulation::random::preferential_attachment(network, N_counties);
	BPsimulation::io::write_network_to_file(network, file);


	std::vector<std::vector<size_t>> counties = BPsimulation::random::random_graphAgnostic_partition_graph(network, n_con);
	BPsimulation::io::write_counties_to_file(counties, file);

	BPsimulation::random::network_randomize_agent_states_county(network, counties[0], 0.1,  0.25, 150, 50,  std::vector<double>({0.3, 0.7, 0.15, 0.15}));
	BPsimulation::random::network_randomize_agent_states_county(network, counties[1], 0.25, 0.1,  150, 75,  std::vector<double>({0.7, 0.3, 0.15, 0.15}));
	BPsimulation::random::network_randomize_agent_states_county(network, counties[2], 0.14, 0.15, 200, 100, std::vector<double>({0.5, 0.5, 0.1,  0.1}));
	BPsimulation::io::write_agent_states_to_file(network, agent_full_serializer, file, "/initial_state");


	BPsimulation::implem::voter_majority_election_result* general_election_results;
	std::vector<BPsimulation::core::election::ElectionResultTemplate*> counties_election_results, stubborness_results;
	for (int itry = 0; itry < N_try; ++itry) {
		if (itry > 0) {
			BPsimulation::io::read_agent_states_from_file(network, agent_full_serializer, file, "/initial_state");
		}

		for (int it = 0; it < N_it; ++it) {
			if (it%n_save == 0 && it > 0) {
				std::string dir_name = "/states_" + std::to_string(itry) + "_" + std::to_string(it);
				BPsimulation::io::write_agent_states_to_file(network, agent_partial_serializer, file, dir_name.c_str());
				util::hdf5io::H5flush_and_clean(file);
			}

			if (it%n_election == 0) {
				general_election_results  = (BPsimulation::implem::voter_majority_election_result*)network->get_election_results(election);
				counties_election_results = network->get_election_results(counties, election);
				stubborness_results        = network->get_election_results(counties, stubborness_election);

				std::string dir_name_general  = "/general_election_result_"  + std::to_string(itry) + "_" + std::to_string(it);
				std::string dir_name_counties = "/counties_election_result_" + std::to_string(itry) + "_" + std::to_string(it);
				BPsimulation::io::write_election_result_to_file( general_election_results,  election_serializer, file, dir_name_general.c_str());
				BPsimulation::io::write_election_results_to_file(counties_election_results, election_serializer, file, dir_name_counties.c_str());
				util::hdf5io::H5flush_and_clean(file);

				std::cout << "\n\ntry " << itry+1 << "/" << N_try << ", it " << it << "/" << N_it-1 << ":\n\n";
				std::cout << "network->get_election_results(...) = " << general_election_results->result << " (" << int(general_election_results->proportion*100) << "%)\n";
				std::cout << "network->get_election_results(counties, ...): \n";
				for (int couty = 0; couty < counties.size(); couty++) {
					BPsimulation::implem::voter_majority_election_result *county_result = (BPsimulation::implem::voter_majority_election_result*)counties_election_results[couty];
					std::cout << "\t" << county_result->result  << " (" << int(county_result->proportion*100) << "%) for county: " << counties[couty] << "\n";

					BPsimulation::implem::voter_stubborness_result *county_stubborness = (BPsimulation::implem::voter_stubborness_result*)stubborness_results[couty];
					std::cout << "\t\tstubborness distribution: " << county_stubborness->proportions[0] << ", " << county_stubborness->proportions[1] << ", " << county_stubborness->proportions[2] << ", " << county_stubborness->proportions[3] << "\n";
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

		util::hdf5io::H5flush_and_clean(file, true);
	}
}