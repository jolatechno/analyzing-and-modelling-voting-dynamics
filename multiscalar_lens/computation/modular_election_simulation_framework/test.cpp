#include <iostream>

#include "src/core/network.hpp"
#include "src/core/networks/network_generator.hpp"
#include "src/core/networks/network_partition.hpp"
#include "src/core/networks/network_util.hpp"
#include "src/core/agent_population/agent_population.hpp"
#include "src/implementations/voter_model.hpp"
#include "src/implementations/voter_model_stubborn.hpp"
#include "src/implementations/Nvoter_model.hpp"
#include "src/implementations/Nvoter_stubborn_model.hpp"
#include "src/implementations/population_voter_model.hpp"
#include "src/implementations/population_voter_model_stubborn.hpp"
#include "src/implementations/population_Nvoter_model.hpp"
#include "src/implementations/population_Nvoter_stubborn_model.hpp"
#include "src/util/util.hpp"


int main() {
	std::cout << "VOTER MODEL:\n\n";

	{
		auto *test = new BPsimulation::SocialNetwork<BPsimulation::implem::voter>(20);

		BPsimulation::random::preferential_attachment(test, 2);
		BPsimulation::random::network_randomize_agent_states(test, 0.5);

		test->add_connection(1, 2);
		test->add_connection(1, 0);

		(*test)[1].candidate = false;
		(*test)[2].candidate = true;

		std::cout << "network.num_nodes() = " << test->num_nodes() << "\n";
		std::cout << "network.degrees() = " << test->degrees() << "\n";
		std::cout << "network[1] = " << (*test)[1].candidate << "\n";
		std::cout << "network.are_neighbors(1, 2) = " << test->are_neighbors(1, 2) << "\n";
		std::cout << "network.are_neighbors(0, 2) = " << test->are_neighbors(0, 2) << "\n";
		std::cout << "network.neighbors(1) = " << test->neighbors(1) << "\n";

		BPsimulation::implem::voter_majority_election<BPsimulation::implem::voter> *election = new BPsimulation::implem::voter_majority_election<BPsimulation::implem::voter>();

		BPsimulation::implem::voter_majority_election_result *result = (BPsimulation::implem::voter_majority_election_result*)test->get_election_results(election);
		std::cout << "\nnetwork->get_election_results(...) = " << result->result << " (" << int(result->proportion*100) << "%)\n";

		std::vector<std::vector<size_t>> counties = BPsimulation::random::random_graphAgnostic_partition_graph(test, 3);
		auto results = test->get_election_results(counties, election);
		std::cout << "network->get_election_results(counties, ...): \n";
		for (size_t i = 0; i < counties.size(); i++) {
			BPsimulation::implem::voter_majority_election_result *county_result = (BPsimulation::implem::voter_majority_election_result*)results[i];
			std::cout << "\t" << county_result->result  << " (" << int(county_result->proportion*100) << "%) for county: " << counties[i] << "\n";
		}
		std::cout << "\n";

		BPsimulation::implem::voter_interaction_function *interaction = new BPsimulation::implem::voter_interaction_function();
		for (int i = 0; i < 10; ++i) {
			if (i%1 == 0) {
				BPsimulation::implem::voter_majority_election_result *result = (BPsimulation::implem::voter_majority_election_result*)test->get_election_results(election);
				std::cout << "network->get_election_results(...) = " << result->result << " (" << int(result->proportion*100) << "%)\n";

			}

			test->interact(interaction);
		}
	}

	std::cout << "\n\n\nstubborn VOTER MODEL:\n\n";

	{
		auto *test = new BPsimulation::SocialNetwork<BPsimulation::implem::voter_stubborn>(20);

		BPsimulation::random::preferential_attachment(test, 2);
		BPsimulation::random::network_randomize_agent_states(test, 0.5, 0.05);

		test->add_connection(1, 2);
		test->add_connection(1, 0);

		(*test)[1].candidate = false;
		(*test)[2].candidate = true;
		(*test)[3].stubborn   = true;

		std::cout << "network.num_nodes() = " << test->num_nodes() << "\n";
		std::cout << "network.degrees() = " << test->degrees() << "\n";
		std::cout << "network[1] = " << (*test)[1].candidate << "\n";
		std::cout << "network[3].stubborn = " << (*test)[3].stubborn << "\n";
		std::cout << "network.are_neighbors(1, 2) = " << test->are_neighbors(1, 2) << "\n";
		std::cout << "network.are_neighbors(0, 2) = " << test->are_neighbors(0, 2) << "\n";
		std::cout << "network.neighbors(1) = " << test->neighbors(1) << "\n";

		BPsimulation::implem::voter_majority_election<BPsimulation::implem::voter_stubborn> *election = new BPsimulation::implem::voter_majority_election<BPsimulation::implem::voter_stubborn>();

		BPsimulation::implem::voter_majority_election_result *result = (BPsimulation::implem::voter_majority_election_result*)test->get_election_results(election);
		std::cout << "\nnetwork->get_election_results(...) = " << result->result << " (" << int(result->proportion*100) << "%)\n";

		std::vector<std::vector<size_t>> counties = BPsimulation::random::random_graphAgnostic_partition_graph(test, 3);
		auto results = test->get_election_results(counties, election);
		std::cout << "network->get_election_results(counties, ...): \n";
		for (size_t i = 0; i < counties.size(); i++) {
			BPsimulation::implem::voter_majority_election_result *county_result = (BPsimulation::implem::voter_majority_election_result*)results[i];
			std::cout << "\t" << county_result->result  << " (" << int(county_result->proportion*100) << "%) for county: " << counties[i] << "\n";
		}
		std::cout << "\n";

		BPsimulation::implem::voter_stubborn_interaction_function *interaction = new BPsimulation::implem::voter_stubborn_interaction_function();
		for (int i = 0; i < 10; ++i) {
			if (i%1 == 0) {
				BPsimulation::implem::voter_majority_election_result *result = (BPsimulation::implem::voter_majority_election_result*)test->get_election_results(election);
				std::cout << "network->get_election_results(...) = " << result->result << " (" << int(result->proportion*100) << "%)\n";

			}

			test->interact(interaction);
		}
	}

	std::cout << "\n\n\nN VOTER MODEL:\n\n";

	{
		const int N_candidates = 3;
		auto *test = new BPsimulation::SocialNetwork<BPsimulation::implem::Nvoter<N_candidates>>(50);

		BPsimulation::random::preferential_attachment(test, 2);

		std::vector<std::vector<size_t>> counties = BPsimulation::random::random_graphAgnostic_partition_graph(test, 3);
		BPsimulation::random::network_randomize_agent_states_county(test, counties[0], std::vector<double>{0.5, 0.2, 0.3});
		BPsimulation::random::network_randomize_agent_states_county(test, counties[1], std::vector<double>{0.3, 0.5, 0.2});
		BPsimulation::random::network_randomize_agent_states_county(test, counties[2], std::vector<double>{0.2, 0.3, 0.5});

		test->add_connection(1, 2);
		test->add_connection(1, 0);

		(*test)[1].candidate = 0;
		(*test)[2].candidate = 2;

		std::cout << "network.num_nodes() = " << test->num_nodes() << "\n";
		std::cout << "network.degrees() = " << test->degrees() << "\n";
		std::cout << "network[1] = " << (*test)[1].candidate << "\n";
		std::cout << "network[2] = " << (*test)[2].candidate << "\n";
		std::cout << "network.are_neighbors(1, 2) = " << test->are_neighbors(1, 2) << "\n";
		std::cout << "network.are_neighbors(0, 2) = " << test->are_neighbors(0, 2) << "\n";
		std::cout << "network.neighbors(1) = " << test->neighbors(1) << "\n";

		BPsimulation::implem::Nvoter_majority_election<N_candidates, BPsimulation::implem::Nvoter<N_candidates>> *election = new BPsimulation::implem::Nvoter_majority_election<N_candidates, BPsimulation::implem::Nvoter<N_candidates>>();

		BPsimulation::implem::Nvoter_majority_election_result<N_candidates> *result = (BPsimulation::implem::Nvoter_majority_election_result<N_candidates>*)test->get_election_results(election);
		std::cout << "\nnetwork->get_election_results(...) = " << result->result << " (" << result->proportions << "%)\n";

		auto results = test->get_election_results(counties, election);
		std::cout << "network->get_election_results(counties, ...): \n";
		for (size_t i = 0; i < counties.size(); i++) {
			BPsimulation::implem::Nvoter_majority_election_result<N_candidates> *county_result = (BPsimulation::implem::Nvoter_majority_election_result<N_candidates>*)results[i];
			std::cout << "\t" << county_result->result  << " (" << county_result->proportions << ") for county: " << counties[i] << "\n";
		}
		std::cout << "\n";

		BPsimulation::implem::Nvoter_interaction_function<N_candidates> *interaction = new BPsimulation::implem::Nvoter_interaction_function<N_candidates>();
		for (int i = 0; i < 10; ++i) {
			if (i%1 == 0) {
				BPsimulation::implem::Nvoter_majority_election_result<N_candidates> *result = (BPsimulation::implem::Nvoter_majority_election_result<N_candidates>*)test->get_election_results(election);
				std::cout << "network->get_election_results(...) = " << result->result << " (" << result->proportions << ")\n";

			}

			test->interact(interaction);
		}
	}

	std::cout << "\n\n\nPOPULATION DYNAMIC:\n\n";

	{
		auto *test = new BPsimulation::SocialNetwork<BPsimulation::core::agent::population::AgentPopulation<BPsimulation::implem::voter>>(100);

		BPsimulation::random::preferential_attachment(test, 3);
		std::vector<std::vector<size_t>> counties = BPsimulation::random::random_graphAgnostic_partition_graph(test, 3);

		BPsimulation::random::network_randomize_agent_states_county(test, counties[0], 150, 50,  std::vector<double>({0.6, 0.4}));
		BPsimulation::random::network_randomize_agent_states_county(test, counties[1], 150, 100, std::vector<double>({0.4, 0.6}));
		BPsimulation::random::network_randomize_agent_states_county(test, counties[2], 400, 100, std::vector<double>({0.5, 0.5}));


		std::cout << "network.num_nodes() = " << test->num_nodes() << "\n";
		std::cout << "network.degrees() = " << test->degrees() << "\n";
		std::cout << "network.neighbors(0) = " << test->neighbors(0) << "\n";
		std::cout << "network[1].population = " << (*test)[1].population << "\n";
		std::cout << "network[2].proportions = " << (*test)[1].proportions << "\n";

		BPsimulation::core::agent::population::PopulationElection<BPsimulation::implem::voter> *election = new BPsimulation::core::agent::population::PopulationElection<BPsimulation::implem::voter>(new BPsimulation::implem::voter_majority_election<BPsimulation::implem::voter>());

		BPsimulation::implem::voter_majority_election_result *result = (BPsimulation::implem::voter_majority_election_result*)test->get_election_results(election);
		std::cout << "\nnetwork->get_election_results(...) = " << result->result << " (" << int(result->proportion*100) << "%)\n";

		auto results = test->get_election_results(counties, election);
		std::cout << "network->get_election_results(counties, ...): \n";
		for (int i = 0; i < counties.size(); i++) {
			BPsimulation::implem::voter_majority_election_result *county_result = (BPsimulation::implem::voter_majority_election_result*)results[i];
			std::cout << "\t" << county_result->result  << " (" << int(county_result->proportion*100) << "%) for county: " << counties[i] << "\n";
		}
		std::cout << "\n";


		BPsimulation::implem::population_voter_interaction_function *interaction = new BPsimulation::implem::population_voter_interaction_function(20);

		BPsimulation::core::agent::population::PopulationRenormalizeProportions<BPsimulation::implem::voter> *renormalize = new BPsimulation::core::agent::population::PopulationRenormalizeProportions<BPsimulation::implem::voter>();

		for (int i = 0; i < 51; ++i) {
			if (i%10 == 0) {
				BPsimulation::implem::voter_majority_election_result *result = (BPsimulation::implem::voter_majority_election_result*)test->get_election_results(election);
				std::cout << "\nnetwork->get_election_results(...) = " << result->result << " (" << int(result->proportion*100) << "%)\n";

				auto results = test->get_election_results(counties, election);
				std::cout << "network->get_election_results(counties, ...): \n";
				for (int i = 0; i < counties.size(); i++) {
					BPsimulation::implem::voter_majority_election_result *county_result = (BPsimulation::implem::voter_majority_election_result*)results[i];
					std::cout << "\t" << county_result->result  << " (" << int(county_result->proportion*100) << "%) for county: " << counties[i] << "\n";
				}
				std::cout << "\n";
			}

			test->interact(interaction);
			test->update_agentwise(renormalize);
		}
	}

	std::cout << "\n\n\nstubborn VOTERS POPULATION DYNAMIC:\n\n";

	{
		auto *test = new BPsimulation::SocialNetwork<BPsimulation::implem::AgentPopulationVoterstubborn>(800);

		BPsimulation::random::preferential_attachment(test, 3);
		std::vector<std::vector<size_t>> counties = BPsimulation::random::random_graphAgnostic_partition_graph(test, 3);

		BPsimulation::random::network_randomize_agent_states_county(test, counties[0], 0.2, 0.2, 150, 50,  std::vector<double>({0.6, 0.4, 0.1, 0.2}));
		BPsimulation::random::network_randomize_agent_states_county(test, counties[1], 0.2, 0.1, 150, 100, std::vector<double>({0.4, 0.6, 0.1, 0.2}));
		BPsimulation::random::network_randomize_agent_states_county(test, counties[2], 0.1, 0.2, 400, 100, std::vector<double>({0.5, 0.5, 0.1, 0.1}));


		std::cout << "network.num_nodes() = " << test->num_nodes() << "\n";
		std::cout << "network.degrees() = " << test->degrees() << "\n";
		std::cout << "network.neighbors(0) = " << test->neighbors(0) << "\n";
		std::cout << "network[1].population = " << (*test)[1].population << "\n";
		std::cout << "network[2].proportions = " << (*test)[1].proportions << "\n";
		std::cout << "network[3].stubborn_equilibrium[0] = " << (*test)[3].stubborn_equilibrium[0] << "\n";

		BPsimulation::core::agent::population::PopulationElection<BPsimulation::implem::voter_stubborn> *election            = new BPsimulation::core::agent::population::PopulationElection<BPsimulation::implem::voter_stubborn>(new BPsimulation::implem::voter_majority_election<BPsimulation::implem::voter_stubborn>());
		BPsimulation::core::agent::population::PopulationElection<BPsimulation::implem::voter_stubborn> *stubborness_election = new BPsimulation::core::agent::population::PopulationElection<BPsimulation::implem::voter_stubborn>(new BPsimulation::implem::voter_stubborness_election());

		BPsimulation::implem::voter_majority_election_result *result = (BPsimulation::implem::voter_majority_election_result*)test->get_election_results(election);
		std::cout << "\nnetwork->get_election_results(...) = " << result->result << " (" << int(result->proportion*100) << "%)\n";

		auto results            = test->get_election_results(counties, election);
		auto stubborness_results = test->get_election_results(counties, stubborness_election);

		std::cout << "network->get_election_results(counties, ...): \n";
		for (int i = 0; i < counties.size(); i++) {
			BPsimulation::implem::voter_majority_election_result *county_result = (BPsimulation::implem::voter_majority_election_result*)results[i];
			std::cout << "\t" << county_result->result  << " (" << int(county_result->proportion*100) << "%) for county: " << counties[i] << "\n";

			BPsimulation::implem::voter_stubborness_result *county_stubborness = (BPsimulation::implem::voter_stubborness_result*)stubborness_results[i];
			std::cout << "\t\tstubborness distribution: " << county_stubborness->proportions[0] << ", " << county_stubborness->proportions[1] << ", " << county_stubborness->proportions[2] << ", " << county_stubborness->proportions[3];
			std::cout << " (" << county_stubborness->proportions[0]+county_stubborness->proportions[1]+county_stubborness->proportions[2]+county_stubborness->proportions[3] << ")\n";
		}
		std::cout << "\n";


		double dt = 0.2;
		BPsimulation::implem::population_voter_stubborn_interaction_function *interaction = new BPsimulation::implem::population_voter_stubborn_interaction_function(10);
		BPsimulation::implem::voter_stubborn_equilibirum_function            *agentwise   = new BPsimulation::implem::voter_stubborn_equilibirum_function(dt);
		BPsimulation::implem::voter_stubborn_overtoon_effect                 *overton     = new BPsimulation::implem::voter_stubborn_overtoon_effect(     dt, 0.015);
		BPsimulation::implem::voter_stubborn_frustration_effect              *frustration = new BPsimulation::implem::voter_stubborn_frustration_effect(  dt, 0.02);

		BPsimulation::core::agent::population::PopulationRenormalizeProportions<BPsimulation::implem::voter_stubborn> *renormalize = new BPsimulation::core::agent::population::PopulationRenormalizeProportions<BPsimulation::implem::voter_stubborn>();

		for (int i = 0; i < 2001; ++i) {
			if (i%400 == 0) {
				result = (BPsimulation::implem::voter_majority_election_result*)test->get_election_results(election);
				std::cout << "\nnetwork->get_election_results(...) = " << result->result << " (" << int(result->proportion*100) << "%)\n";

				results = test->get_election_results(counties, election);
				auto stubborness_results = test->get_election_results(counties, stubborness_election);

				std::cout << "network->get_election_results(counties, ...): \n";
				for (int i = 0; i < counties.size(); i++) {
					BPsimulation::implem::voter_majority_election_result *county_result = (BPsimulation::implem::voter_majority_election_result*)results[i];
					std::cout << "\t" << county_result->result  << " (" << int(county_result->proportion*100) << "%) for county: " << counties[i] << "\n";

					BPsimulation::implem::voter_stubborness_result *county_stubborness = (BPsimulation::implem::voter_stubborness_result*)stubborness_results[i];
					std::cout << "\t\tstubborness distribution: " << county_stubborness->proportions[0] << ", " << county_stubborness->proportions[1] << ", " << county_stubborness->proportions[2] << ", " << county_stubborness->proportions[3];
					std::cout << " (" << county_stubborness->proportions[0]+county_stubborness->proportions[1]+county_stubborness->proportions[2]+county_stubborness->proportions[3] << ")\n";
				}
				std::cout << "\n";
			}

			test->interact(interaction);
			test->update_agentwise(agentwise);
			test->election_retroinfluence(result, overton);
			test->election_retroinfluence(result, frustration);
			test->election_retroinfluence(counties, results, overton);
			test->election_retroinfluence(counties, results, frustration);
			test->update_agentwise(renormalize);
		}
	}

	std::cout << "\n\n\nN CANDIDATES POPULATION DYNAMIC:\n\n";

	{
		const int N_candidates = 3;
		auto *test = new BPsimulation::SocialNetwork<BPsimulation::core::agent::population::AgentPopulation<BPsimulation::implem::Nvoter<N_candidates>>>(100);

		BPsimulation::random::preferential_attachment(test, 3);
		std::vector<std::vector<size_t>> counties = BPsimulation::random::random_graphAgnostic_partition_graph(test, 3);

		BPsimulation::random::network_randomize_agent_states_county(test, counties[0], 150, 50,  std::vector<double>({0.5, 0.3, 0.2}));
		BPsimulation::random::network_randomize_agent_states_county(test, counties[1], 150, 100, std::vector<double>({0.2, 0.5, 0.3}));
		BPsimulation::random::network_randomize_agent_states_county(test, counties[2], 400, 100, std::vector<double>({0.3, 0.2, 0.5}));


		std::cout << "network.num_nodes() = " << test->num_nodes() << "\n";
		std::cout << "network.degrees() = " << test->degrees() << "\n";
		std::cout << "network.neighbors(0) = " << test->neighbors(0) << "\n";
		std::cout << "network[1].population = " << (*test)[1].population << "\n";
		std::cout << "network[2].proportions = " << (*test)[1].proportions << "\n";

		BPsimulation::core::agent::population::PopulationElection<BPsimulation::implem::Nvoter<N_candidates>> *election = new BPsimulation::core::agent::population::PopulationElection<BPsimulation::implem::Nvoter<N_candidates>>(new BPsimulation::implem::Nvoter_majority_election<N_candidates, BPsimulation::implem::Nvoter<N_candidates>>());

		BPsimulation::implem::Nvoter_majority_election_result<N_candidates> *result = (BPsimulation::implem::Nvoter_majority_election_result<N_candidates>*)test->get_election_results(election);
		std::cout << "\nnetwork->get_election_results(...) = " << result->result << " (" << result->proportions << ")\n";

		auto results = test->get_election_results(counties, election);
		std::cout << "network->get_election_results(counties, ...): \n";
		for (int i = 0; i < counties.size(); i++) {
			BPsimulation::implem::Nvoter_majority_election_result<N_candidates> *county_result = (BPsimulation::implem::Nvoter_majority_election_result<N_candidates>*)results[i];
			std::cout << "\t" << county_result->result  << " (" << county_result->proportions << ") for county: " << counties[i] << "\n";
		}
		std::cout << "\n";


		BPsimulation::implem::population_Nvoter_interaction_function<N_candidates> *interaction = new BPsimulation::implem::population_Nvoter_interaction_function<N_candidates>(20);

		BPsimulation::core::agent::population::PopulationRenormalizeProportions<BPsimulation::implem::Nvoter<N_candidates>> *renormalize = new BPsimulation::core::agent::population::PopulationRenormalizeProportions<BPsimulation::implem::Nvoter<N_candidates>>();

		for (int i = 0; i < 51; ++i) {
			if (i%10 == 0) {
				BPsimulation::implem::Nvoter_majority_election_result<N_candidates> *result = (BPsimulation::implem::Nvoter_majority_election_result<N_candidates>*)test->get_election_results(election);
				std::cout << "\nnetwork->get_election_results(...) = " << result->result << " (" << result->proportions << ")\n";

				auto results = test->get_election_results(counties, election);
				std::cout << "network->get_election_results(counties, ...): \n";
				for (int i = 0; i < counties.size(); i++) {
					BPsimulation::implem::Nvoter_majority_election_result<N_candidates> *county_result = (BPsimulation::implem::Nvoter_majority_election_result<N_candidates>*)results[i];
					std::cout << "\t" << county_result->result  << " (" << county_result->proportions << ") for county: " << counties[i] << "\n";
				}
				std::cout << "\n";
			}

			test->interact(interaction);
			test->update_agentwise(renormalize);
		}
	}

	std::cout << "\n\n\nN CANDIDATES stubborn VOTERS POPULATION DYNAMIC:\n\n";

	{
		const int N_candidates = 3;
		auto *test = new BPsimulation::SocialNetwork<BPsimulation::implem::AgentPopulationNVoterstubborn<N_candidates>>(200);

		BPsimulation::random::preferential_attachment(test, 3);
		std::vector<std::vector<size_t>> counties = BPsimulation::random::random_graphAgnostic_partition_graph(test, 3);

		BPsimulation::random::network_randomize_agent_states_county(test, counties[0], std::vector<double>({0.07, 0.03, 0.04}), 150, 50,  std::vector<double>({0.5, 0.2, 0.3, 0.2, 0.1, 0.1}));
		BPsimulation::random::network_randomize_agent_states_county(test, counties[1], std::vector<double>({0.04, 0.07, 0.03}), 150, 100, std::vector<double>({0.3, 0.5, 0.2, 0.1, 0.2, 0.1}));
		BPsimulation::random::network_randomize_agent_states_county(test, counties[2], std::vector<double>({0.03, 0.04, 0.07}), 400, 100, std::vector<double>({0.2, 0.3, 0.5, 0.1, 0.1, 0.2}));


		std::cout << "network.num_nodes() = " << test->num_nodes() << "\n";
		std::cout << "network.degrees() = " << test->degrees() << "\n";
		std::cout << "network.neighbors(0) = " << test->neighbors(0) << "\n";
		std::cout << "network[1].population = " << (*test)[1].population << "\n";
		std::cout << "network[2].proportions = " << (*test)[1].proportions << "\n";
		std::cout << "network[3].stubborn_equilibrium[0] = " << (*test)[3].stubborn_equilibrium[0] << "\n";

		BPsimulation::core::agent::population::PopulationElection<BPsimulation::implem::Nvoter_stubborn<N_candidates>> *election = new BPsimulation::core::agent::population::PopulationElection<BPsimulation::implem::Nvoter_stubborn<N_candidates>>(new BPsimulation::implem::Nvoter_majority_election<N_candidates, BPsimulation::implem::Nvoter_stubborn<N_candidates>>());

		BPsimulation::implem::Nvoter_majority_election_result<N_candidates> *result = (BPsimulation::implem::Nvoter_majority_election_result<N_candidates>*)test->get_election_results(election);
		std::cout << "\nnetwork->get_election_results(...) = " << result->result << " (" << result->proportions << ")\n";

		auto results = test->get_election_results(counties, election);

		std::cout << "network->get_election_results(counties, ...): \n";
		for (int i = 0; i < counties.size(); i++) {
			BPsimulation::implem::Nvoter_majority_election_result<N_candidates> *county_result = (BPsimulation::implem::Nvoter_majority_election_result<N_candidates>*)results[i];
			std::cout << "\t" << county_result->result  << " (" << county_result->proportions << ") for county: " << counties[i] << "\n";

		}
		std::cout << "\n";


		double dt = 0.2;
		BPsimulation::implem::population_Nvoter_stubborn_interaction_function<N_candidates> *interaction = new BPsimulation::implem::population_Nvoter_stubborn_interaction_function<N_candidates>(10);
		BPsimulation::implem::Nvoter_stubborn_equilibirum_function<N_candidates>            *agentwise   = new BPsimulation::implem::Nvoter_stubborn_equilibirum_function           <N_candidates>(dt);
		BPsimulation::implem::Nvoter_stubborn_overtoon_effect<N_candidates>                 *overton     = new BPsimulation::implem::Nvoter_stubborn_overtoon_effect                <N_candidates>(dt, 0.015);
		BPsimulation::implem::Nvoter_stubborn_frustration_effect<N_candidates>              *frustration = new BPsimulation::implem::Nvoter_stubborn_frustration_effect             <N_candidates>(dt, 0.02);

		BPsimulation::core::agent::population::PopulationRenormalizeProportions<BPsimulation::implem::Nvoter_stubborn<N_candidates>> *renormalize = new BPsimulation::core::agent::population::PopulationRenormalizeProportions<BPsimulation::implem::Nvoter_stubborn<N_candidates>>();

		for (int i = 0; i < 2001; ++i) {
			if (i%400 == 0) {
				result = (BPsimulation::implem::Nvoter_majority_election_result<N_candidates>*)test->get_election_results(election);
				std::cout << "\nnetwork->get_election_results(...) = " << result->result << " (" << result->proportions << ")\n";

				results = test->get_election_results(counties, election);

				std::cout << "network->get_election_results(counties, ...): \n";
				for (int i = 0; i < counties.size(); i++) {
					BPsimulation::implem::Nvoter_majority_election_result<N_candidates> *county_result = (BPsimulation::implem::Nvoter_majority_election_result<N_candidates>*)results[i];
					std::cout << "\t" << county_result->result  << " (" << county_result->proportions << ") for county: " << counties[i] << "\n";
				}
				std::cout << "\n";
			}

			test->interact(interaction);
			test->update_agentwise(agentwise);
			test->election_retroinfluence(result, overton);
			test->election_retroinfluence(result, frustration);
			test->election_retroinfluence(counties, results, overton);
			test->election_retroinfluence(counties, results, frustration);
			test->update_agentwise(renormalize);
		}
	}
}